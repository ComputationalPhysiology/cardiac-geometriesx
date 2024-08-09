from pathlib import Path

import dolfinx
import numpy as np

from ..utils import space_from_string
from . import utils


def mu_theta(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, long_axis: int = 0
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Get the angles mu and theta from the coordinates x, y, z
    given the long axis.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates
    y : np.ndarray
        The y-coordinates
    z : np.ndarray
        The z-coordinates
    long_axis : int, optional
        The long axis, by default 0 (x-axis)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[int]]
        The angles mu and theta and the permutation of the axes

    Raises
    ------
    ValueError
        If the long axis is not 0, 1 or 2
    """
    if long_axis == 0:
        a = np.sqrt(y**2 + z**2)
        b = x
        theta = np.pi - np.arctan2(z, -y)
        perm = [0, 1, 2]
    elif long_axis == 1:
        a = np.sqrt(x**2 + z**2)
        b = y
        theta = np.pi - np.arctan2(z, -x)
        perm = [1, 0, 2]
    elif long_axis == 2:
        a = np.sqrt(x**2 + y**2)
        b = z
        theta = np.pi - np.arctan2(x, -y)
        perm = [2, 1, 0]
    else:
        raise ValueError("Invalid long_axis")

    mu = np.arctan2(a, b)

    theta[mu < 1e-7] = 0.0

    return mu, theta, perm


def compute_system(
    t_func: dolfinx.fem.Function,
    r_short_endo=0.025,
    r_short_epi=0.035,
    r_long_endo=0.09,
    r_long_epi=0.097,
    alpha_endo: float = -60,
    alpha_epi: float = 60,
    long_axis: int = 0,
    **kwargs,
) -> utils.Microstructure:
    """Compute the microstructure for the given time function.

    Parameters
    ----------
    t_func : dolfinx.fem.Function
        Solution of the Laplace equation
    r_short_endo : float, optional
        Short radius at the endocardium, by default 0.025
    r_short_epi : float, optional
        Short radius at the epicardium, by default 0.035
    r_long_endo : float, optional
        Long radius at the endocardium, by default 0.09
    r_long_epi : float, optional
        Long radius at the epicardium, by default 0.097
    alpha_endo : float, optional
        Angle at the endocardium, by default -60
    alpha_epi : float, optional
        Angle at the epicardium, by default 60
    long_axis : int, optional
        Long axis, by default 0 (x-axis)

    Returns
    -------
    utils.Microstructure
        The microstructure
    """

    V = t_func.function_space
    element = V.ufl_element()
    mesh = V.mesh

    dof_coordinates = V.tabulate_dof_coordinates()

    alpha = lambda x: (alpha_endo + (alpha_epi - alpha_endo) * x) * (np.pi / 180)
    r_long = lambda x: r_long_endo + (r_long_epi - r_long_endo) * x
    r_short = lambda x: r_short_endo + (r_short_epi - r_short_endo) * x

    drl_dt = r_long_epi - r_long_endo
    drs_dt = r_short_epi - r_short_endo

    t = t_func.x.array

    rl = r_long(t)
    rs = r_short(t)
    al = alpha(t)

    x = dof_coordinates[:, 0]
    y = dof_coordinates[:, 1]
    z = dof_coordinates[:, 2]

    mu, theta, perm = mu_theta(x, y, z, long_axis=long_axis)

    e_t = np.array(
        [
            drl_dt * np.cos(mu),
            drs_dt * np.sin(mu) * np.cos(theta),
            drs_dt * np.sin(mu) * np.sin(theta),
        ],
    )[perm]
    e_t = utils.normalize(e_t)

    e_mu = np.array(
        [
            -rl * np.sin(mu),
            rs * np.cos(mu) * np.cos(theta),
            rs * np.cos(mu) * np.sin(theta),
        ],
    )[perm]
    e_mu = utils.normalize(e_mu)

    e_theta = np.array(
        [
            np.zeros_like(t),
            -rs * np.sin(mu) * np.sin(theta),
            rs * np.sin(mu) * np.cos(theta),
        ],
    )[perm]
    e_theta = utils.normalize(e_theta)

    f0 = np.sin(al) * e_mu + np.cos(al) * e_theta
    f0 = utils.normalize(f0)

    n0 = np.cross(e_mu, e_theta, axis=0)
    n0 = utils.normalize(n0)

    s0 = np.cross(f0, n0, axis=0)
    s0 = utils.normalize(s0)

    Vv = space_from_string(
        space_string=f"{element.family_name}_{element.degree}", mesh=mesh, dim=mesh.geometry.dim
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
    mesh,
    ffun,
    markers,
    function_space="P_1",
    r_short_endo=0.025,
    r_short_epi=0.035,
    r_long_endo=0.09,
    r_long_epi=0.097,
    alpha_endo: float = -60,
    alpha_epi: float = 60,
    long_axis: int = 0,
    outdir: str | Path | None = None,
):
    endo_marker = markers["ENDO"][0]
    epi_marker = markers["EPI"][0]
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
        function_space=function_space,
        r_short_endo=r_short_endo,
        r_short_epi=r_short_epi,
        r_long_endo=r_long_endo,
        r_long_epi=r_long_epi,
        alpha_endo=alpha_endo,
        alpha_epi=alpha_epi,
        long_axis=long_axis,
    )
    if outdir is not None:
        utils.save_microstructure(mesh, system, outdir)

    return system
