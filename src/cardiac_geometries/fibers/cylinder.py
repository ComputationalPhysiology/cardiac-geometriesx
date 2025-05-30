from pathlib import Path

import dolfinx
import numpy as np

from ..utils import space_from_string
from . import utils


def compute_system(
    mesh: dolfinx.mesh.Mesh,
    alpha_endo: float = -60,
    alpha_epi: float = 60,
    r_inner: float = 10.0,
    r_outer: float = 20.0,
    function_space: str = "P_1",
    **kwargs,
) -> utils.Microstructure:
    """Compute ldrb system for cylinder, assuming linear
    angle between endo and epi

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        A cylinder mesh
    alpha_endo : float, optional
        Angle on endocardium, by default -60
    alpha_epi : float, optional
        Angle on epicardium, by default 60
    r_inner : float, optional
        Inner radius, by default 10.0
    r_outer : float, optional
        Outer radius, by default 20.0
    function_space : str, optional
        Function space to interpolate the fibers, by default "P_1"
    Returns
    -------
    Microstructure
        Tuple with fiber, sheet and sheet normal
    """

    Vv = space_from_string(function_space, mesh, dim=3)

    x, y, z = Vv.tabulate_dof_coordinates().T
    r = np.sqrt(x**2 + y**2)

    # Circumferential direction
    e_r = np.array([x / r, y / r, np.zeros_like(r)])
    e_theta = np.array([-y / r, x / r, np.zeros_like(r)])
    e_z = np.array([np.zeros_like(r), np.zeros_like(r), np.ones_like(r)])

    n0 = e_r
    alpha = (alpha_endo + (alpha_epi - alpha_endo) * (r - r_inner) / (r_outer - r_inner)) * (
        np.pi / 180
    )

    f0 = e_theta * np.cos(alpha) - e_z * np.sin(alpha)
    s0 = e_theta * np.sin(alpha) + e_z * np.cos(alpha)

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
    alpha_endo: float,
    alpha_epi: float,
    r_inner: float,
    r_outer: float,
    function_space: str = "P_1",
    outdir: str | Path | None = None,
    **kwargs,
) -> utils.Microstructure:
    """Generate microstructure for cylinder

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        A cylinder mesh
    alpha_endo : float
        Angle on the endocardium
    alpha_epi : float
        Angle on the epicardium
    r_inner : float
        Inner radius
    r_outer : float
        Outer radius
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

    system = compute_system(
        mesh=mesh,
        function_space=function_space,
        r_inner=r_inner,
        r_outer=r_outer,
        alpha_endo=alpha_endo,
        alpha_epi=alpha_epi,
    )

    if outdir is not None:
        utils.save_microstructure(mesh, system, outdir)

    return system
