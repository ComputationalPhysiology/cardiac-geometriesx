import logging

import dolfinx

from . import cylinder, lv_ellipsoid, slab, utils
from .utils import Microstructure

__all__ = ["lv_ellipsoid", "slab", "cylinder", "utils", "Microstructure"]

logger = logging.getLogger(__name__)


supported_mesh_types = ("slab", "cylinder", "lv_ellipsoid", "biv_ellipsoid", "ukb", "lv", "biv")


def generate_fibers_ldrb(mesh: dolfinx.mesh.Mesh, **kwargs) -> Microstructure:
    # Try with LDRB
    try:
        import ldrb
    except ImportError as ex:
        msg = (
            "To create fibers you need to install the ldrb package "
            "which you can install with pip install fenicsx-ldrb"
        )
        raise ImportError(msg) from ex

    markers = kwargs.pop("markers", None)
    clipped = kwargs.pop("clipped", False)

    from ..mesh import transform_markers

    markers = transform_markers(markers, clipped=clipped)

    system = ldrb.dolfinx_ldrb(mesh=mesh, markers=markers, **kwargs)
    return Microstructure(
        f0=system.f0,
        s0=system.s0,
        n0=system.n0,
    )


def generate_fibers(
    mesh_type: str, mesh: dolfinx.mesh.Mesh, force_ldrb: bool = False, **kwargs
) -> Microstructure:
    """Generate fibers based on the mesh type."""

    kwargs = kwargs.copy()
    if force_ldrb:
        return generate_fibers_ldrb(mesh, **kwargs)

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
        return generate_fibers_ldrb(mesh, **kwargs)
