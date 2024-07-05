from pathlib import Path

from mpi4py import MPI

import pytest
from click.testing import CliRunner

from cardiac_geometries import Geometry, cli


@pytest.mark.parametrize(
    "script",
    [
        cli.slab,
        cli.lv_ellipsoid,
    ],
    ids=["slab", "lv_ellipsoid"],
)
@pytest.mark.parametrize("fiber_space", [None, "P_1", "P_2", "Quadrature_2"])
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


@pytest.mark.parametrize(
    "script",
    [
        cli.slab_in_bath,
        cli.biv_ellipsoid,
        cli.biv_ellipsoid_torso,
    ],
    ids=["slab_in_bath", "biv_ellipsoid", "biv_ellipsoid_torso"],
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
