from pathlib import Path

import dolfinx
import numpy as np

from ..utils import space_from_string
from . import utils


def compute_system(
    t_func: dolfinx.fem.Function,
    alpha_endo: float = -60,
    alpha_epi: float = 60,
    endo_epi_axis="y",
) -> utils.Microstructure:
    """Compute ldrb system for slab, assuming linear
    angle between endo and epi

    Parameters
    ----------
    t_func : dolfin.Function
        Solution to laplace equation with 0 on endo
        and 1 on epi
    alpha_endo : float, optional
        Angle on endocardium, by default -60
    alpha_epi : float, optional
        Angle on epicardium, by default 60

    Returns
    -------
    Microstructure
        Tuple with fiber, sheet and sheet normal
    """

    V = t_func.function_space
    element = V.ufl_element()
    mesh = V.mesh

    alpha = lambda x: (alpha_endo + (alpha_epi - alpha_endo) * x) * (np.pi / 180)

    t = t_func.x.array

    if endo_epi_axis == "y":
        f0 = np.array(
            [
                np.cos(alpha(t)),
                np.zeros_like(t),
                np.sin(alpha(t)),
            ],
        )
        n0 = np.array(
            [
                np.zeros_like(t),
                np.ones_like(t),
                np.zeros_like(t),
            ],
        )
    elif endo_epi_axis == "z":
        f0 = np.array(
            [
                np.cos(alpha(t)),
                np.sin(alpha(t)),
                np.zeros_like(t),
            ],
        )
        n0 = np.array(
            [
                np.zeros_like(t),
                np.zeros_like(t),
                np.ones_like(t),
            ],
        )
    elif endo_epi_axis == "x":
        f0 = np.array(
            [
                np.zeros_like(t),
                np.cos(alpha(t)),
                np.sin(alpha(t)),
            ],
        )
        n0 = np.array(
            [
                np.ones_like(t),
                np.zeros_like(t),
                np.zeros_like(t),
            ],
        )
    else:
        raise ValueError(f"Unknown endo_epi_axis: {endo_epi_axis}")

    s0 = np.cross(f0, n0, axis=0)
    s0 = utils.normalize(s0)

    Vv = space_from_string(
        space_string=f"{element.family_name}_{element.degree}",
        mesh=mesh,
        dim=mesh.geometry.dim,
        discontinuous=element.discontinuous,
    )

    fiber = dolfinx.fem.Function(Vv)
    norm_f = np.linalg.norm(f0, axis=0)
    fiber.x.array[:] = (f0 / norm_f).T.reshape(-1)
    fiber.name = "f0"

    sheet = dolfinx.fem.Function(Vv)
    sheet.x.array[:] = s0.T.reshape(-1)
    sheet.name = "s0"

    sheet_normal = dolfinx.fem.Function(Vv)
    sheet_normal.x.array[:] = n0.T.reshape(-1)
    sheet_normal.name = "n0"

    return utils.Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)


def create_microstructure(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    markers: dict[str, tuple[int, int]],
    alpha_endo: float,
    alpha_epi: float,
    function_space: str = "P_1",
    outdir: str | Path | None = None,
    **kwargs: dict,
) -> utils.Microstructure:
    """Generate microstructure for slab using LDRB algorithm

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        A slab mesh
    ffun : dolfinx.mesh.MeshTags
        Facet function defining the boundaries
    markers: Dict[str, Tuple[int, int]]
        Markers with keys Y0 and Y1 representing the endo and
        epi planes respectively. The values should be a tuple
        whose first value is the value of the marker (corresponding
        to ffun) and the second value is the dimension
    alpha_endo : float
        Angle on the endocardium
    alpha_epi : float
        Angle on the epicardium
    function_space : str
        Function space to interpolate the fibers, by default P_1
    outdir : Optional[Union[str, Path]], optional
        Output directory to store the results, by default None.
        If no output directory is specified the results will not be stored,
        but only returned.

    Returns
    -------
    Microstructure
        Tuple with fiber, sheet and sheet normal
    """

    endo_marker = markers["Y0"][0]
    epi_marker = markers["Y1"][0]

    t = utils.laplace(
        mesh,
        ffun,
        endo_marker=endo_marker,
        epi_marker=epi_marker,
        function_space=function_space,
    )
    if outdir is not None:
        try:
            with dolfinx.io.VTXWriter(
                mesh.comm, Path(outdir) / "laplace.bp", [t], engine="BP4"
            ) as file:
                file.write(0.0)
        except RuntimeError:
            pass

    system = compute_system(
        t,
        alpha_endo=alpha_endo,
        alpha_epi=alpha_epi,
    )

    if outdir is not None:
        utils.save_microstructure(mesh, system, outdir)

    return system
