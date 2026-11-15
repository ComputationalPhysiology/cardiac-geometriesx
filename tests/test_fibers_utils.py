from mpi4py import MPI

import dolfinx
import numpy as np

from cardiac_geometries.fibers import utils


def test_laplace_is_iterative_with_exact_boundary_values():
    assert utils.LAPLACE_PETSC_OPTIONS["ksp_type"] == "cg"
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    fdim = mesh.topology.dim - 1
    left = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 1.0))
    facets = np.concatenate([left, right])
    values = np.concatenate([np.full(len(left), 1), np.full(len(right), 2)]).astype(np.int32)
    order = np.argsort(facets)
    ffun = dolfinx.mesh.meshtags(mesh, fdim, facets[order], values[order])
    t = utils.laplace(mesh, ffun, endo_marker=1, epi_marker=2, function_space="P_1")
    x = t.function_space.tabulate_dof_coordinates()[:, 0]
    n = t.function_space.dofmap.index_map.size_local
    assert np.all(t.x.array[:n][np.isclose(x[:n], 0.0)] == 0.0)
    assert np.all(t.x.array[:n][np.isclose(x[:n], 1.0)] == 1.0)
    assert np.allclose(t.x.array[:n], x[:n], atol=1e-8)  # linear in x
