from mpi4py import MPI

import basix
import dolfinx
import numpy as np

import cardiac_geometries
import cardiac_geometries.geometry


def test_save_load_mesh_only(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)

    geo = cardiac_geometries.geometry.Geometry(mesh=mesh)

    folder = comm.bcast(tmp_path, root=0)
    path = folder / "geo.bp"
    geo.save(path)
    geo2 = cardiac_geometries.geometry.Geometry.from_file(comm, path)

    # Just assert that they have the same number of cells
    assert geo2.mesh.topology.index_map(3).size_global == geo.mesh.topology.index_map(3).size_global


def test_save_load_mesh_and_tags(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    # Create some artificial tags
    facet_tags = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        np.array([0, 1, 2, 3], dtype=np.int32),
        1,
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    geo = cardiac_geometries.geometry.Geometry(mesh=mesh, ffun=facet_tags)

    folder = comm.bcast(tmp_path, root=0)
    path = folder / "geo.bp"
    geo.save(path)
    geo2 = cardiac_geometries.geometry.Geometry.from_file(comm, path)

    # Just assert that they have the same number of cells
    assert geo2.mesh.topology.index_map(3).size_global == geo.mesh.topology.index_map(3).size_global
    # A bit hard to compare in parallel, so just check that the ffun is not None
    assert geo.ffun is not None
    assert geo2.ffun is not None


def test_save_load_mesh_and_function(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    # Create and arbitrary function

    V = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element(
            family="Lagrange",
            cell=mesh.ufl_cell().cellname(),
            degree=1,
            discontinuous=False,
            shape=(3,),
        ),
    )
    f = dolfinx.fem.Function(V)
    f.interpolate(lambda x: x)
    geo = cardiac_geometries.geometry.Geometry(mesh=mesh, f0=f)

    folder = comm.bcast(tmp_path, root=0)
    path = folder / "geo.bp"
    geo.save(path)
    geo2 = cardiac_geometries.geometry.Geometry.from_file(comm, path)
    assert geo2.mesh.topology.index_map(3).size_global == geo.mesh.topology.index_map(3).size_global
    assert geo.f0 is not None
    assert geo2.f0 is not None


def test_load_from_folder_mesh_only(tmp_path):
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    mesh.name = "Mesh"
    with dolfinx.io.XDMFFile(comm, folder / "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, folder)
    assert geo.mesh.topology.index_map(3).size_global == mesh.topology.index_map(3).size_global


def test_load_from_folder_mesh_and_tags(tmp_path):
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    mesh.name = "Mesh"
    # Create some artificial tags
    facet_tags = dolfinx.mesh.meshtags(
        mesh,
        mesh.topology.dim - 1,
        np.array([0, 1, 2, 3], dtype=np.int32),
        1,
    )
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facet_tags.name = "Facet tags"

    with dolfinx.io.XDMFFile(comm, folder / "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_meshtags(
            facet_tags,
            mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
        )

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, folder)
    assert geo.mesh.topology.index_map(3).size_global == mesh.topology.index_map(3).size_global
    assert geo.ffun is not None


def test_load_from_folder_mesh_and_function(tmp_path):
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    mesh.name = "Mesh"

    with dolfinx.io.XDMFFile(comm, folder / "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)

    # Create and arbitrary function
    V = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element(
            family="Lagrange",
            cell=mesh.ufl_cell().cellname(),
            degree=1,
            discontinuous=False,
            shape=(3,),
        ),
    )
    f = dolfinx.fem.Function(V)
    f.interpolate(lambda x: x)
    f.name = "f0"
    cardiac_geometries.fibers.utils.save_microstructure(mesh=mesh, functions=(f,), outdir=folder)

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, folder)
    assert geo.mesh.topology.index_map(3).size_global == mesh.topology.index_map(3).size_global
    assert geo.f0 is not None
