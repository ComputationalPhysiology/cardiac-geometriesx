from pathlib import Path

from mpi4py import MPI

import gmsh
import pytest

import cardiac_geometries as cg

try:
    import ldrb  # noqa: F401

    HAS_LDRB = True
except ImportError:
    HAS_LDRB = False

try:
    import ukb.cli  # noqa: F401

    HAS_UKB = True
except ImportError:
    HAS_UKB = False

MPI_SIZE = MPI.COMM_WORLD.size


@pytest.mark.parametrize(
    "script",
    [
        cg.mesh.slab,
        cg.mesh.lv_ellipsoid,
        cg.mesh.cylinder,
    ],
    ids=["slab", "lv_ellipsoid", "cylinder"],
)
def test_refine_analytic_fibers(script, tmp_path: Path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)
    outdir = path / "mesh"
    script(outdir=outdir, create_fibers=True, comm=comm)
    geo = cg.Geometry.from_folder(comm=comm, folder=outdir)
    assert (outdir / "mesh.xdmf").is_file()
    assert geo.f0 is not None
    # assert geo.mesh.geometry.dim == 3
    refined_outdir = path / "refined"
    refined = geo.refine(outdir=refined_outdir, n=1)
    assert refined.f0 is not None
    assert (refined_outdir / "mesh.xdmf").is_file()
    assert (
        refined.f0.function_space.dofmap.index_map.size_global
        > geo.f0.function_space.dofmap.index_map.size_global
    )
    assert refined.mesh.geometry.index_map().size_global > geo.mesh.geometry.index_map().size_global


@pytest.mark.skipif(gmsh.__version__ == "4.14.0", reason="GMSH 4.14.0 has a bug with fuse")
@pytest.mark.skipif(not HAS_LDRB, reason="LDRB atlas is not installed")
def test_refine_biv(tmp_path: Path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)
    outdir = path / "mesh"
    cg.mesh.biv_ellipsoid(outdir=outdir, create_fibers=True, comm=comm)
    geo = cg.Geometry.from_folder(comm=comm, folder=outdir)
    assert geo.f0 is not None
    assert (outdir / "mesh.xdmf").is_file()
    # assert geo.mesh.geometry.dim == 3
    refined_outdir = path / "refined"
    refined = geo.refine(outdir=refined_outdir, n=1)
    assert refined.f0 is not None
    assert (refined_outdir / "mesh.xdmf").is_file()
    assert (
        refined.f0.function_space.dofmap.index_map.size_global
        > geo.f0.function_space.dofmap.index_map.size_global
    )
    assert refined.mesh.geometry.index_map().size_global > geo.mesh.geometry.index_map().size_global


@pytest.mark.skipif(not HAS_LDRB, reason="LDRB atlas is not installed")
@pytest.mark.skipif(not HAS_UKB, reason="UKB atlas is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Pyvista operations is not parallelized yet")
def test_refine_ukb(tmp_path: Path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)
    outdir = path / "mesh"
    cg.mesh.ukb(outdir=outdir, create_fibers=True, comm=comm)
    geo = cg.Geometry.from_folder(comm=comm, folder=outdir)
    assert geo.f0 is not None
    assert (outdir / "mesh.xdmf").is_file()
    # assert geo.mesh.geometry.dim == 3
    refined_outdir = path / "refined"
    refined = geo.refine(outdir=refined_outdir, n=1)
    assert refined.f0 is not None
    assert (refined_outdir / "mesh.xdmf").is_file()
    assert (
        refined.f0.function_space.dofmap.index_map.size_global
        > geo.f0.function_space.dofmap.index_map.size_global
    )
    assert refined.mesh.geometry.index_map().size_global > geo.mesh.geometry.index_map().size_global
