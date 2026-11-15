import json
import shutil
from pathlib import Path
from typing import NamedTuple, Sequence

import dolfinx
import io4dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem

from ..utils import element2array, json_serial, space_from_string


class Microstructure(NamedTuple):
    f0: dolfinx.fem.Function
    s0: dolfinx.fem.Function
    n0: dolfinx.fem.Function


def save_microstructure(
    mesh: dolfinx.mesh.Mesh,
    functions: Sequence[dolfinx.fem.Function],
    path: Path,
    viz_path: Path | None = None,
    viz: bool = True,
    checkpoint: bool = True,
    backend: str | None = None,
) -> None:
    if len(functions) == 0:
        return
    # Save for paraview visualization

    if viz:
        if functions[0].function_space.ufl_element().family_name == "quadrature":
            from scifem.xdmf import XDMFFile

            if viz_path is None:
                viz_path = path.parent / "microstructure-viz.xdmf"

            viz_path = viz_path.with_suffix(".xdmf")
            viz_path.unlink(missing_ok=True)
            viz_path.with_suffix(".h5").unlink(missing_ok=True)

            with XDMFFile(viz_path, functions) as xdmf:
                xdmf.write(0.0)

        else:
            if viz_path is None:
                viz_path = path.parent / "microstructure-viz.bp"
            viz_path = viz_path.with_suffix(".bp")
            shutil.rmtree(viz_path, ignore_errors=True)
            try:
                with dolfinx.io.VTXWriter(
                    mesh.comm, viz_path, list(functions), engine="BP4"
                ) as file:
                    file.write(0.0)
            except (RuntimeError, ValueError) as ex:
                print(f"Failed to write microstructure: {ex}")

    # Save with proper function space

    if checkpoint:
        from ..geometry import infer_io_backend, microstructure_path

        backend = backend or infer_io_backend(path)
        attributes = {f.name: element2array(f.ufl_element()) for f in functions}
        if mesh.comm.rank == 0:
            microstructure_path(path).write_text(
                json.dumps(attributes, indent=4, default=json_serial)
            )
        for name, u in zip(("f0", "s0", "n0"), functions):
            io4dolfinx.write_function(u=u, filename=path, name=name, backend=backend)


def normalize(u):
    return u / np.linalg.norm(u, axis=0)


# Conjugate gradients with algebraic multigrid: faster than a direct solver in serial and in
# parallel; a solve that does not converge raises an error
LAPLACE_PETSC_OPTIONS = {
    "ksp_type": "cg",
    "pc_type": "gamg",
    "ksp_rtol": 1e-10,
    "ksp_error_if_not_converged": True,
}


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

    fdim = mesh.topology.dim - 1
    endo_facets = ffun.find(endo_marker)
    endo_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, endo_facets)
    epi_facets = ffun.find(epi_marker)
    epi_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, epi_facets)
    zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    one = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

    endo_bc = dolfinx.fem.dirichletbc(zero, endo_dofs, V)
    epi_bc = dolfinx.fem.dirichletbc(one, epi_dofs, V)

    bcs = [endo_bc, epi_bc]

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options=LAPLACE_PETSC_OPTIONS,
        petsc_options_prefix="cardiac_geometriesx_laplace",
    )
    uh = problem.solve()
    assert isinstance(uh, dolfinx.fem.Function)
    # An iterative solver meets the boundary values only to its tolerance
    for bc in bcs:
        bc.set(uh.x.array)

    if function_space != "P_1":
        W = space_from_string(function_space, mesh, dim=1)
        t = dolfinx.fem.Function(W)
        points = W.element.interpolation_points

        expr = dolfinx.fem.Expression(uh, points)
        t.interpolate(expr)
    else:
        t = uh

    return t
