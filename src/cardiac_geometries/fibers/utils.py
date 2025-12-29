import json
import shutil
from pathlib import Path
from typing import NamedTuple, Sequence

import adios4dolfinx
import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from packaging.version import Version

from ..utils import space_from_string

_dolfinx_version = Version(dolfinx.__version__)


class Microstructure(NamedTuple):
    f0: dolfinx.fem.Function
    s0: dolfinx.fem.Function
    n0: dolfinx.fem.Function


def save_microstructure(
    mesh: dolfinx.mesh.Mesh, functions: Sequence[dolfinx.fem.Function], outdir: str | Path
) -> None:
    from ..utils import element2array

    if len(functions) == 0:
        return
    # Save for paraview visualization

    if functions[0].function_space.ufl_element().family_name == "quadrature":
        from scifem.xdmf import XDMFFile

        fname = Path(outdir) / "microstructure-viz.xdmf"
        fname.unlink(missing_ok=True)
        fname.with_suffix(".h5").unlink(missing_ok=True)

        with XDMFFile(fname, functions) as xdmf:
            xdmf.write(0.0)

    else:
        fname = Path(outdir) / "microstructure-viz.bp"
        shutil.rmtree(fname, ignore_errors=True)
        try:
            with dolfinx.io.VTXWriter(mesh.comm, fname, functions, engine="BP4") as file:
                file.write(0.0)
        except RuntimeError as ex:
            print(f"Failed to write microstructure: {ex}")

    # Save with proper function space

    filename = Path(outdir) / "microstructure.bp"
    shutil.rmtree(filename, ignore_errors=True)
    adios4dolfinx.write_mesh(mesh=mesh, filename=filename)

    for function in functions:
        # adios4dolfinx.write_function_on_input_mesh(u=function, filename=filename)
        adios4dolfinx.write_function(u=function, filename=filename)

    attributes = {f.name: element2array(f.ufl_element()) for f in functions}
    adios4dolfinx.write_attributes(
        comm=mesh.comm,
        filename=filename,
        name="function_space",
        attributes=attributes,
    )

    def json_serial(obj):
        if isinstance(obj, (np.ndarray)):
            return obj.tolist()
        raise TypeError("Type %s not serializable" % type(obj))

    if mesh.comm.rank == 0:
        (Path(outdir) / "microstructure.json").write_text(
            json.dumps(attributes, indent=4, default=json_serial)
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

    kwargs = {}
    if _dolfinx_version >= Version("0.10"):
        kwargs["petsc_options_prefix"] = "cardiac_geometriesx_laplace"

    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, **kwargs
    )
    uh = problem.solve()

    if function_space != "P_1":
        W = space_from_string(function_space, mesh, dim=1)
        t = dolfinx.fem.Function(W)
        if _dolfinx_version >= Version("0.10"):
            points = W.element.interpolation_points
        else:
            points = W.element.interpolation_points()

        expr = dolfinx.fem.Expression(uh, points)
        t.interpolate(expr)
    else:
        t = uh

    return t
