from pathlib import Path

import basix
import dolfinx
import numpy as np

from . import utils


def compute_system(
    t_func: dolfinx.fem.Function,
    r_short_endo=0.025,
    r_short_epi=0.035,
    r_long_endo=0.09,
    r_long_epi=0.097,
    alpha_endo: float = -60,
    alpha_epi: float = 60,
    **kwargs,
) -> utils.Microstructure:
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

    a = np.sqrt(y**2 + z**2) / rs
    b = x / rl
    mu = np.arctan2(a, b)
    theta = np.pi - np.arctan2(z, -y)
    theta[mu < 1e-7] = 0.0

    e_t = np.array(
        [
            drl_dt * np.cos(mu),
            drs_dt * np.sin(mu) * np.cos(theta),
            drs_dt * np.sin(mu) * np.sin(theta),
        ],
    )
    e_t = utils.normalize(e_t)

    e_mu = np.array(
        [
            -rl * np.sin(mu),
            rs * np.cos(mu) * np.cos(theta),
            rs * np.cos(mu) * np.sin(theta),
        ],
    )
    e_mu = utils.normalize(e_mu)

    e_theta = np.array(
        [
            np.zeros_like(t),
            -rs * np.sin(mu) * np.sin(theta),
            rs * np.sin(mu) * np.cos(theta),
        ],
    )
    e_theta = utils.normalize(e_theta)

    f0 = np.sin(al) * e_mu + np.cos(al) * e_theta
    f0 = utils.normalize(f0)

    n0 = np.cross(e_mu, e_theta, axis=0)
    n0 = utils.normalize(n0)

    s0 = np.cross(f0, n0, axis=0)
    s0 = utils.normalize(s0)

    el = basix.ufl.element(
        element.family_name,
        mesh.ufl_cell().cellname(),
        degree=element.degree,
        discontinuous=element.discontinuous,
        shape=(mesh.geometry.dim,),
    )
    Vv = dolfinx.fem.functionspace(mesh, el)

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
        with dolfinx.io.XDMFFile(mesh.comm, Path(outdir) / "laplace.xdmf", "w") as file:
            file.write_mesh(mesh)
            file.write_function(t)
    system = compute_system(
        t,
        function_space=function_space,
        r_short_endo=r_short_endo,
        r_short_epi=r_short_epi,
        r_long_endo=r_long_endo,
        r_long_epi=r_long_epi,
        alpha_endo=alpha_endo,
        alpha_epi=alpha_epi,
    )
    if outdir is not None:
        utils.save_microstructure(mesh, system, outdir)

    return system
