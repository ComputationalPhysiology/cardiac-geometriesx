import dolfinx
from dolfinx.fem.petsc import LinearProblem
import numpy as np
from petsc4py import PETSc
import ufl
import basix

from typing import Dict
from typing import NamedTuple
from typing import Tuple
from typing import Union


class Microstructure(NamedTuple):
    f0: dolfinx.fem.Function
    s0: dolfinx.fem.Function
    n0: dolfinx.fem.Function


def check_mesh_params(mesh_params: Dict[str, float]):
    for key in ["r_short_endo", "r_short_epi", "r_long_endo", "r_long_epi"]:
        assert key in mesh_params, f"Missing key '{key}' in mesh parameters"


def check_fiber_params(fiber_params: Dict[str, Union[str, float]]) -> None:
    for key in ["alpha_endo", "alpha_epi", "function_space"]:
        assert key in fiber_params, f"Missing key '{key}' in fiber parameters"


def laplace(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    markers: Dict[str, Tuple[int, int]],
    function_space: str,
):
    endo_marker = markers["ENDO"][0]
    epi_marker = markers["EPI"][0]

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = v * dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) * ufl.dx

    endo_facets = ffun.find(endo_marker)
    endo_dofs = dolfinx.fem.locate_dofs_topological(V, 2, endo_facets)
    epi_facets = ffun.find(epi_marker)
    epi_dofs = dolfinx.fem.locate_dofs_topological(V, 2, epi_facets)
    zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

    endo_bc = dolfinx.fem.dirichletbc(zero, endo_dofs, V)
    epi_bc = dolfinx.fem.dirichletbc(one, epi_dofs, V)

    bcs = [endo_bc, epi_bc]

    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    with dolfinx.io.XDMFFile(mesh.comm, "solution.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)

    if function_space != "P_1":
        family, degree = function_space.split("_")
        W = dolfinx.fem.functionspace(mesh, (family, int(degree)))
        t = dolfinx.fem.Function(W)
        t.interpolate(uh)
    else:
        t = uh

    return t


def normalize(u):
    return u / np.linalg.norm(u, axis=0)


def compute_system(
    t_func: dolfinx.fem.Function,
    r_short_endo=0.025,
    r_short_epi=0.035,
    r_long_endo=0.09,
    r_long_epi=0.097,
    alpha_endo: float = -60,
    alpha_epi: float = 60,
    **kwargs,
):
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
    e_t = normalize(e_t)

    e_mu = np.array(
        [
            -rl * np.sin(mu),
            rs * np.cos(mu) * np.cos(theta),
            rs * np.cos(mu) * np.sin(theta),
        ],
    )
    e_mu = normalize(e_mu)

    e_theta = np.array(
        [
            np.zeros_like(t),
            -rs * np.sin(mu) * np.sin(theta),
            rs * np.sin(mu) * np.cos(theta),
        ],
    )
    e_theta = normalize(e_theta)

    f0 = np.sin(al) * e_mu + np.cos(al) * e_theta
    f0 = normalize(f0)

    n0 = np.cross(e_mu, e_theta, axis=0)
    n0 = normalize(n0)

    s0 = np.cross(f0, n0, axis=0)
    s0 = normalize(s0)

    el = basix.ufl.element(
        element.family_name,
        mesh.ufl_cell().cellname(),
        degree=element.degree,
        discontinuous=element.discontinuous,
        shape=(mesh.geometry.dim,),
    )
    Vv = dolfinx.fem.functionspace(mesh, el)

    # FIXME: Need to make it work for parallel
    bs = Vv.dofmap.bs
    # breakpoint()
    start, end = Vv.sub(0).dofmap.index_map.local_range
    x_dofs = np.arange(0, bs * (end - start), bs)
    y_dofs = np.arange(1, bs * (end - start), bs)
    z_dofs = np.arange(2, bs * (end - start), bs)

    start, end = V.dofmap.index_map.local_range
    # scalar_dofs = V.dofmap.index_map.local_to_global(
    #     np.arange(0, end - start, dtype=np.int32)
    # )
    scalar_dofs = np.arange(0, end - start, dtype=np.int32)
    # scalar_dofs = [
    #     dof
    #     for dof in range(end - start)
    #     if V.dofmap().local_to_global_index(dof)
    #     not in V.dofmap().local_to_global_unowned()
    # ]

    fiber = dolfinx.fem.Function(Vv)
    f = np.zeros_like(fiber.x.array)

    f[x_dofs] = f0[0, scalar_dofs] / np.linalg.norm(f0, axis=0)
    f[y_dofs] = f0[1, scalar_dofs] / np.linalg.norm(f0, axis=0)
    f[z_dofs] = f0[2, scalar_dofs] / np.linalg.norm(f0, axis=0)
    fiber.vector.setArray(f)
    fiber.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    fiber.name = "fiber"

    sheet = dolfinx.fem.Function(Vv)
    s = np.zeros_like(f)
    s[x_dofs] = s0[0, scalar_dofs]
    s[y_dofs] = s0[1, scalar_dofs]
    s[z_dofs] = s0[2, scalar_dofs]
    sheet.vector.setArray(s)
    sheet.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    sheet.name = "sheet"

    sheet_normal = dolfinx.fem.Function(Vv)
    n = np.zeros_like(f)
    n[x_dofs] = n0[0, scalar_dofs]
    n[y_dofs] = n0[1, scalar_dofs]
    n[z_dofs] = n0[2, scalar_dofs]
    sheet_normal.vector.setArray(n)
    sheet_normal.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    sheet_normal.name = "sheet_normal"

    return Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)


def create_microstructure(mesh, ffun, markers):
    # check_mesh_params(mesh_params)
    # check_fiber_params(fiber_params)
    # function_space = fiber_params.get("function_space", "P_1")
    function_space = "DG_0"
    t = laplace(mesh, ffun, markers, function_space=function_space)
    return compute_system(t)  # , **mesh_params, **fiber_params)
