import inspect
import logging

import dolfinx

from ..utils import extract_submesh, transfer_to_parent
from . import cylinder, lv_ellipsoid, slab, utils
from .utils import Microstructure

__all__ = ["lv_ellipsoid", "slab", "cylinder", "utils", "Microstructure"]

logger = logging.getLogger(__name__)


supported_mesh_types = ("slab", "cylinder", "lv_ellipsoid", "biv_ellipsoid", "ukb", "lv", "biv")


# Arguments to `ldrb.dolfinx_doste` with other names in `Geometry.info` or `generate_fibers`
_DOSTE_ARGUMENTS = {
    "fiber_angle_endo": "alpha_endo_lv",
    "fiber_angle_epi": "alpha_epi_lv",
    "alpha_endo": "alpha_endo_lv",
    "alpha_epi": "alpha_epi_lv",
    "function_space": "fiber_space",
}


def ldrb_fibers(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags | None,
    markers: dict[str, tuple[int, int]],
    cfun: dolfinx.mesh.MeshTags | None = None,
    clipped: bool = False,
    fiber_method: str = "ldrb",
    **kwargs,
):
    """Create fibers with a Laplace-Dirichlet rule-based method from fenicsx-ldrb.

    With `fiber_method` "ldrb" the method of Bayer et al. (`ldrb.dolfinx_ldrb`)
    is used, and with "doste" the method of Doste et al. for geometries with
    outflow tracts (`ldrb.dolfinx_doste`), which needs the valves of the UKB
    geometries to be marked separately.

    For geometries where the cavities are meshed as well (marked with
    ``*_CAVITY``, e.g. the closed UKB geometries), the fibers are computed on
    the wall (marked ``Wall``) only, and are zero in the cavities.

    Returns
    -------
    tuple[ldrb.io.LDRBOutput | ldrb.io.DosteOutput, Microstructure]
        The output from fenicsx-ldrb, on the mesh it was computed on (the wall
        for geometries with cavities), and the fibers on `mesh`.
    """
    try:
        import ldrb
    except ImportError as ex:
        msg = (
            "To create fibers you need to install the ldrb package "
            "which you can install with pip install fenicsx-ldrb"
        )
        raise ImportError(msg) from ex

    from ..mesh import doste_markers, transform_markers

    if fiber_method == "ldrb":
        method_markers = transform_markers(markers, clipped=clipped)

        def compute(mesh, ffun):
            return ldrb.dolfinx_ldrb(mesh=mesh, ffun=ffun, markers=method_markers, **kwargs)

    elif fiber_method == "doste":
        method_markers = doste_markers(markers)
        # Skip arguments that only the method of Bayer et al. or other mesh types take
        parameters = inspect.signature(ldrb.dolfinx_doste).parameters
        doste_kwargs = {}
        for name, value in kwargs.items():
            name = _DOSTE_ARGUMENTS.get(name, name)
            if name in parameters:
                doste_kwargs[name] = value

        def compute(mesh, ffun):
            return ldrb.dolfinx_doste(mesh=mesh, ffun=ffun, markers=method_markers, **doste_kwargs)

    else:
        raise ValueError(f"Unknown fiber method {fiber_method!r}, expected 'ldrb' or 'doste'")

    has_cavities = any(name.endswith("_CAVITY") for name in markers)
    if not has_cavities or cfun is None or ffun is None:
        system = compute(mesh, ffun)
        return system, Microstructure(f0=system.f0, s0=system.s0, n0=system.n0)

    wall, wall_ffun, parent_cells = extract_submesh(mesh, cfun, ffun, [markers["Wall"][0]])
    system = compute(wall, wall_ffun)
    V = dolfinx.fem.functionspace(mesh, system.f0.function_space.ufl_element())
    return system, Microstructure(
        f0=transfer_to_parent(system.f0, V, parent_cells),
        s0=transfer_to_parent(system.s0, V, parent_cells),
        n0=transfer_to_parent(system.n0, V, parent_cells),
    )


def generate_fibers_ldrb(mesh: dolfinx.mesh.Mesh, **kwargs) -> Microstructure:
    _, microstructure = ldrb_fibers(mesh, **kwargs)
    return microstructure


def generate_fibers(
    mesh_type: str, mesh: dolfinx.mesh.Mesh, force_ldrb: bool = False, **kwargs
) -> Microstructure:
    """Generate fibers based on the mesh type."""

    kwargs = kwargs.copy()
    # Only used by LDRB, to compute the fibers on the wall of geometries with meshed cavities
    cfun = kwargs.pop("cfun", None)
    if force_ldrb:
        return generate_fibers_ldrb(mesh, cfun=cfun, **kwargs)

    # Map fiber_angle_endo and fiber_angle_epi to alpha_endo and alpha_epi
    if "fiber_angle_endo" in kwargs:
        kwargs["alpha_endo"] = kwargs.pop("fiber_angle_endo")
    if "fiber_angle_epi" in kwargs:
        kwargs["alpha_epi"] = kwargs.pop("fiber_angle_epi")
    if "fiber_space" in kwargs:
        kwargs["function_space"] = kwargs.pop("fiber_space")

    if mesh_type == "slab":
        return slab.create_microstructure(mesh, **kwargs)
    elif mesh_type == "cylinder":
        return cylinder.create_microstructure(mesh, **kwargs)
    elif mesh_type == "lv_ellipsoid":
        return lv_ellipsoid.create_microstructure(mesh, **kwargs)
    else:
        if mesh_type not in supported_mesh_types:
            logger.warning(
                f"Mesh type {mesh_type!r} is not recognized. "
                f"Supported mesh types are: {supported_mesh_types!r}. "
                "Lets try with LDRB algorithm to generate fibers.",
            )
        return generate_fibers_ldrb(mesh, cfun=cfun, **kwargs)
