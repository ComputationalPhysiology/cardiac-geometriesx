from pathlib import Path

import dolfinx
import numpy as np

from ..utils import space_from_string
from . import utils


def compute_system(
    mesh: dolfinx.mesh.Mesh,
    alpha_endo: float = -60,
    alpha_epi: float = 60,
    r_inner_x: float = 10.0,
    r_inner_y: float = 10.0,
    r_outer_x: float = 20.0,
    r_outer_y: float = 20.0,
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
    r_inner_x : float, optional
        Inner radius along the x-axis, by default 10.0
    r_inner_y : float, optional
        Inner radius along the y-axis, by default 10.0
    r_outer_x : float, optional
        Outer radius along the x-axis, by default 20.0
    r_outer_y : float, optional
        Outer radius along the y-axis, by default 20.0
    function_space : str, optional
        Function space to interpolate the fibers, by default "P_1"
    Returns
    -------
    Microstructure
        Tuple with fiber, sheet and sheet normal
    """

    Vv = space_from_string(function_space, mesh, dim=3)

    x, y, z = Vv.tabulate_dof_coordinates().T

    # We need the angle to know "where" we are on the ellipse ring
    theta = np.arctan2(y, x)

    # B. Compute "Local" Inner/Outer Radii at this angle
    # Formula for radius of ellipse r(theta) = (a*b) / sqrt((b*cos)^2 + (a*sin)^2)
    # This tells us: "At this angle theta, what is the distance to the inner/outer wall?"

    denom_in = np.sqrt((r_inner_y * np.cos(theta)) ** 2 + (r_inner_x * np.sin(theta)) ** 2)
    rad_limit_inner = (r_inner_x * r_inner_y) / denom_in

    denom_out = np.sqrt((r_outer_y * np.cos(theta)) ** 2 + (r_outer_x * np.sin(theta)) ** 2)
    rad_limit_outer = (r_outer_x * r_outer_y) / denom_out

    # C. Compute Transmural Depth (u)
    # u goes from 0 (endo) to 1 (epi)
    r_current = np.sqrt(x**2 + y**2)
    u = (r_current - rad_limit_inner) / (rad_limit_outer - rad_limit_inner)
    # Clip to avoid numerical noise outside [0,1]
    u = np.clip(u, 0.0, 1.0)

    # D. Compute Local Basis Vectors
    # For a circle, e_r is just (x, y). For an ellipse, it is (x/a^2, y/b^2).
    # We interpolate the semi-axes 'a' and 'b' based on depth 'u'.
    a_local = r_inner_x + u * (r_outer_x - r_inner_x)
    b_local = r_inner_y + u * (r_outer_y - r_inner_y)

    # Normal Vector components (Gradient of ellipse)
    nx = x / (a_local**2)
    ny = y / (b_local**2)

    # Normalize e_r (this replaces 'x/r, y/r' from original)
    n_mag = np.sqrt(nx**2 + ny**2)
    e_r_x = nx / n_mag
    e_r_y = ny / n_mag

    # e_r vector (The sheet normal direction)
    e_r = np.array([e_r_x, e_r_y, np.zeros_like(x)])

    # e_theta vector (The circumferential direction)
    # Perpendicular to e_r: (-y, x)
    e_theta = np.array([-e_r_y, e_r_x, np.zeros_like(x)])

    # e_z vector (The longitudinal direction)
    e_z = np.array([np.zeros_like(x), np.zeros_like(x), np.ones_like(x)])

    # E. Rotation (Identical to original)
    n0 = e_r

    # Calculate Helix Angle
    alpha = (alpha_endo + (alpha_epi - alpha_endo) * u) * (np.pi / 180)

    # Apply Rotation
    f0 = e_theta * np.cos(alpha) - e_z * np.sin(alpha)
    s0 = e_theta * np.sin(alpha) + e_z * np.cos(alpha)

    # F. Assignment (Identical to original)
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
    r_inner_x: float,
    r_inner_y: float,
    r_outer_x: float,
    r_outer_y: float,
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
    r_inner_x : float
        Inner radius along the x-axis
    r_inner_y : float
        Inner radius along the y-axis
    r_outer_x : float
        Outer radius along the x-axis
    r_outer_y : float
        Outer radius along the y-axis
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
        r_inner_x=r_inner_x,
        r_inner_y=r_inner_y,
        r_outer_x=r_outer_x,
        r_outer_y=r_outer_y,
        alpha_endo=alpha_endo,
        alpha_epi=alpha_epi,
    )

    if outdir is not None:
        utils.save_microstructure(mesh, system, path=Path(outdir) / "geometry.bp")

    return system
