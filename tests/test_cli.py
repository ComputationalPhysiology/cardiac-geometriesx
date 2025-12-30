from pathlib import Path

from mpi4py import MPI

import numpy as np
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
    ],
    ids=["slab_in_bath", "biv_ellipsoid"],
)
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


def test_lv_aha(tmp_path: Path):
    runner = CliRunner()

    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)

    res = runner.invoke(
        cli.lv_ellipsoid,
        [
            path.as_posix(),
            "--aha",
            "--dmu-factor",
            "0.2",
        ],
    )
    assert res.exit_code == 0
    assert path.is_dir()

    geo = Geometry.from_folder(comm=comm, folder=path)

    assert geo.cfun is not None
    values = np.hstack(comm.allgather(geo.cfun.values))
    assert len(np.setdiff1d(np.unique(values), np.arange(1, 18))) == 0


@pytest.mark.parametrize(
    "script",
    [
        cli.lv_ellipsoid,
        cli.biv_ellipsoid,
        cli.ukb,
    ],
    ids=["lv_ellipsoid", "biv_ellipsoid", "ukb"],
)
@pytest.mark.parametrize("fiber_space", [None, "P_1"])
def test_rotate(script, fiber_space, tmp_path: Path):
    comm = MPI.COMM_WORLD
    if comm.size > 1:
        return pytest.skip("rotate works in serial.")
    path = comm.bcast(tmp_path, root=0)
    path_orig = path / "original"
    args = [path_orig.as_posix()]
    if fiber_space is not None:
        args.extend(["--create-fibers", "--fiber-space", fiber_space])

    if "ukb" in str(script):
        if not HAS_UKB:
            return pytest.skip("UKB atlas package is not installed")
        if not HAS_LDRB:
            return pytest.skip("ldrb package is not installed")
        args.extend(["--clipped"])

    if "biv-ellipsoid" in str(script):
        if not HAS_LDRB:
            return pytest.skip("ldrb package is not installed")

    runner = CliRunner()
    res = runner.invoke(script, args)
    assert res.exit_code == 0
    path_rotated = path / "rotated"
    args_rot = [
        path_orig.as_posix(),
        "-o",
        path_rotated.as_posix(),
        "--target-normal",
        "0.0",
        "1.0",
        "0.0",
    ]
    res = runner.invoke(cli.rotate, args_rot)

    assert res.exit_code == 0
    geo_rotated = Geometry.from_folder(comm=comm, folder=path_rotated)
    assert geo_rotated.mesh.geometry.dim == 3
    if fiber_space is not None:
        assert geo_rotated.f0 is not None
