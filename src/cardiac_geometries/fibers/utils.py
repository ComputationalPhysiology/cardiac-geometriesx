from pathlib import Path
from typing import NamedTuple, Sequence

import adios4dolfinx
import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem


class Microstructure(NamedTuple):
    f0: dolfinx.fem.Function
    s0: dolfinx.fem.Function
    n0: dolfinx.fem.Function


def save_microstructure(
    mesh: dolfinx.mesh.Mesh, functions: Sequence[dolfinx.fem.Function], outdir: str | Path
) -> None:
    from ..utils import element2array

    # Save for paraview visualization
    with dolfinx.io.XDMFFile(mesh.comm, Path(outdir) / "microstructure.xdmf", "w") as file:
        file.write_mesh(mesh)
        for f in functions:
            file.write_function(f)

    # Save with proper function space
    filename = Path(outdir) / "microstructure.bp"
    for function in functions:
        adios4dolfinx.write_function(u=function, filename=filename)

    attributes = {f.name: element2array(f.ufl_element().basix_element) for f in functions}
    adios4dolfinx.write_attributes(
        comm=mesh.comm,
        filename=filename,
        name="function_space",
        attributes=attributes,
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
