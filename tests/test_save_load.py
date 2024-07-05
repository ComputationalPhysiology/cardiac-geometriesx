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

    path = tmp_path / "geo.bp"
    geo.save(path)
    geo2 = cardiac_geometries.geometry.Geometry.from_file(comm, path)
    assert (geo2.mesh.geometry.x == geo.mesh.geometry.x).all()


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

    path = tmp_path / "geo.bp"
    geo.save(path)
    geo2 = cardiac_geometries.geometry.Geometry.from_file(comm, path)
    assert (geo2.mesh.geometry.x == geo.mesh.geometry.x).all()
    assert (geo2.ffun.values == geo.ffun.values).all()


def test_save_load_mesh_and_function(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    # Create and arbitrary function
    # breakpoint()
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

    path = tmp_path / "geo.bp"
    geo.save(path)
    geo2 = cardiac_geometries.geometry.Geometry.from_file(comm, path)
    assert (geo2.mesh.geometry.x == geo.mesh.geometry.x).all()
    assert (geo2.f0.x.array == geo.f0.x.array).all()


def test_load_from_folder_mesh_only(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    mesh.name = "Mesh"
    with dolfinx.io.XDMFFile(comm, tmp_path / "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, tmp_path)
    assert (geo.mesh.geometry.x == mesh.geometry.x).all()


def test_load_from_folder_mesh_and_tags(tmp_path):
    comm = MPI.COMM_WORLD
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

    with dolfinx.io.XDMFFile(comm, tmp_path / "mesh.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_meshtags(
            facet_tags,
            mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
        )

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, tmp_path)
    assert (geo.mesh.geometry.x == mesh.geometry.x).all()
    assert (geo.ffun.values == facet_tags.values).all()


def test_load_from_folder_mesh_and_function(tmp_path):
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    mesh.name = "Mesh"

    with dolfinx.io.XDMFFile(comm, tmp_path / "mesh.xdmf", "w") as file:
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
    cardiac_geometries.fibers.utils.save_microstructure(mesh=mesh, functions=(f,), outdir=tmp_path)

    geo = cardiac_geometries.geometry.Geometry.from_folder(comm, tmp_path)
    assert (geo.mesh.geometry.x == mesh.geometry.x).all()
    assert (geo.f0.x.array == f.x.array).all()
