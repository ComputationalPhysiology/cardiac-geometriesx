from pathlib import Path

from mpi4py import MPI

import gmsh
import pytest
from click.testing import CliRunner

from cardiac_geometries import Geometry, cli

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
        cli.slab,
        cli.lv_ellipsoid,
        cli.cylinder,
        cli.cylinder_D_shaped,
        cli.cylinder_racetrack,
    ],
    ids=["slab", "lv_ellipsoid", "cylinder", "cylinder_D_shaped", "cylinder_racetrack"],
)
@pytest.mark.parametrize("fiber_space", [None, "P_1", "P_2", "Quadrature_2", "DG_1"])
def test_script(fiber_space, script, tmp_path: Path):
    runner = CliRunner()

    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)

    args = [path.as_posix()]
    if fiber_space is not None:
        args.extend(["--create-fibers", "--fiber-space", fiber_space])

    res = runner.invoke(script, args)
    assert res.exit_code == 0
    assert path.is_dir()
    geo = Geometry.from_folder(comm=comm, folder=path)
    assert geo.mesh.geometry.dim == 3

    if fiber_space is not None:
        assert geo.f0 is not None


@pytest.mark.skipif(gmsh.__version__ == "4.14.0", reason="GMSH 4.14.0 has a bug with fuse")
@pytest.mark.skipif(not HAS_LDRB, reason="LDRB atlas is not installed")
def test_biv_fibers(tmp_path: Path):
    runner = CliRunner()

    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)

    res = runner.invoke(
        cli.biv_ellipsoid,
        [path.as_posix(), "--create-fibers", "--fiber-space", "P_1"],
    )
    assert res.exit_code == 0
    assert path.is_dir()

    geo = Geometry.from_folder(comm=comm, folder=path)
    assert geo.mesh.geometry.dim == 3


@pytest.mark.parametrize(
    "script",
    [
        cli.slab_in_bath,
        cli.biv_ellipsoid,
        cli.biv_ellipsoid_torso,
    ],
    ids=["slab_in_bath", "biv_ellipsoid", "biv_ellipsoid_torso"],
)
@pytest.mark.skipif(gmsh.__version__ == "4.14.0", reason="GMSH 4.14.0 has a bug with fuse")
def test_script_no_fibers(script, tmp_path: Path):
    runner = CliRunner()

    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)

    res = runner.invoke(script, [path.as_posix()])
    assert res.exit_code == 0
    assert path.is_dir()

    geo = Geometry.from_folder(comm=comm, folder=path)
    assert geo.mesh.geometry.dim == 3


@pytest.mark.parametrize("clipped", [True, False])
@pytest.mark.parametrize("case", ["ED", "ES"])
@pytest.mark.skipif(not HAS_UKB, reason="UKB atlas is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Pyvista operations is not parallelized yet")
def test_ukb(tmp_path: Path, case: str, clipped: bool):
    runner = CliRunner()

    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)

    args = [
        path.as_posix(),
        "--case",
        case,
        "--char-length-max",
        "10.0",
        "--char-length-min",
        "10.0",
    ]
    if clipped:
        if case == "ES":
            pytest.skip("Clipped case with default values are for ED")
        args.append("--clipped")

    res = runner.invoke(cli.ukb, args)
    assert res.exit_code == 0
    assert path.is_dir()

    assert (path / "mesh.xdmf").exists()
    if clipped:
        assert (path / f"{case}_clipped.msh").exists()
    else:
        assert (path / f"{case}.msh").exists()
    geo = Geometry.from_folder(comm=comm, folder=path)
    assert geo.mesh.geometry.dim == 3
