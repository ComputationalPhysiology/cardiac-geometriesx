from typing import NamedTuple
from pathlib import Path

import dolfinx
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
import adios4dolfinx


class Microstructure(NamedTuple):
    f0: dolfinx.fem.Function
    s0: dolfinx.fem.Function
    n0: dolfinx.fem.Function


def save_microstructure(
    mesh: dolfinx.mesh.Mesh, system: Microstructure, outdir: str | Path
) -> None:
    from ..utils import element2array

    # Save for paraview visualization
    with dolfinx.io.XDMFFile(mesh.comm, Path(outdir) / "microstructure.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(system.f0)
        file.write_function(system.s0)
        file.write_function(system.n0)

    # Save with proper function space
    filename = Path(outdir) / "microstructure.bp"
    adios4dolfinx.write_function(u=system.f0, filename=filename)
    adios4dolfinx.write_function(u=system.s0, filename=filename)
    adios4dolfinx.write_function(u=system.n0, filename=filename)
    arr = element2array(system.f0.ufl_element().basix_element)
    adios4dolfinx.write_attributes(
        comm=mesh.comm,
        filename=filename,
        name="function_space",
        attributes={k: arr for k in ("f0", "s0", "n0")},
    )


def normalize(u):
    return u / np.linalg.norm(u, axis=0)


def laplace(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    endo_marker: int,
    epi_marker: int,
    function_space: str,
):
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

    if function_space != "P_1":
        family, degree = function_space.split("_")
        W = dolfinx.fem.functionspace(mesh, (family, int(degree)))
        t = dolfinx.fem.Function(W)
        t.interpolate(uh)
    else:
        t = uh

    return t
