from pathlib import Path

from mpi4py import MPI

import dolfinx
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
        cli.cylinder_elliptical,
        cli.cylinder_D_shaped,
        cli.cylinder_racetrack,
    ],
    ids=[
        "slab",
        "lv_ellipsoid",
        "cylinder",
        "cylinder_elliptical",
        "cylinder_D_shaped",
        "cylinder_racetrack",
    ],
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


UKB_MARKERS = {
    "biv": {"LV_ENDO", "RV_ENDO", "EPI", "MV", "AV", "TV", "PV", "Wall"},
    "biv-closed": {"LV_ENDO", "RV_ENDO", "EPI", "MV", "AV", "TV", "PV"}
    | {"MV_CAP", "AV_CAP", "TV_CAP", "PV_CAP", "Wall", "LV_CAVITY", "RV_CAVITY"},
    "lv": {"LV_ENDO", "EPI", "RV_SEPTUM", "SEPTAL_BRIDGE", "MV", "AV", "Wall"},
    "lv-closed": {"LV_ENDO", "EPI", "RV_SEPTUM", "SEPTAL_BRIDGE", "MV", "AV"}
    | {"MV_CAP", "AV_CAP", "Wall", "LV_CAVITY"},
}


@pytest.mark.parametrize("geometry", ["biv", "biv-closed", "lv", "lv-closed"])
@pytest.mark.parametrize("clipped", [True, False])
@pytest.mark.parametrize("case", ["ED", "ES"])
@pytest.mark.skipif(not HAS_UKB, reason="UKB atlas is not installed")
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Pyvista operations is not parallelized yet")
def test_ukb(tmp_path: Path, case: str, clipped: bool, geometry: str):
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
        "--geometry",
        geometry,
    ]
    if clipped:
        if case == "ES":
            pytest.skip("Clipped case with default values are for ED")
        if geometry != "biv":
            pytest.skip("Clipping is only supported for the biv geometry")
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
    if clipped:
        assert set(geo.markers) == {"LV_ENDO", "RV_ENDO", "EPI", "BASE", "Wall"}
    else:
        assert set(geo.markers) == UKB_MARKERS[geometry]


@pytest.mark.parametrize("geometry", ["biv-closed", "lv-closed"])
@pytest.mark.skipif(not HAS_UKB, reason="UKB atlas is not installed")
@pytest.mark.skipif(not HAS_LDRB, reason="ldrb package is not installed")
def test_ukb_closed_fibers_only_in_wall(tmp_path: Path, geometry: str):
    runner = CliRunner()

    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)

    args = [
        path.as_posix(),
        "--char-length-max",
        "10.0",
        "--char-length-min",
        "10.0",
        "--geometry",
        geometry,
        "--create-fibers",
    ]
    res = runner.invoke(cli.ukb, args)
    assert res.exit_code == 0, res.output

    geo = Geometry.from_folder(comm=comm, folder=path)
    assert geo.f0 is not None
    assert geo.cfun is not None
    V = geo.f0.function_space
    bs = V.dofmap.bs
    num_owned = V.dofmap.index_map.size_local

    # Mark the dofs of the wall cells, also those owned by another process
    wall = dolfinx.fem.Function(V)
    wall_cells = geo.cfun.find(geo.markers["Wall"][0])
    wall.x.array[V.dofmap.list[wall_cells] * bs] = 1.0
    wall.x.scatter_reverse(dolfinx.la.InsertMode.add)
    in_wall = wall.x.array[: num_owned * bs : bs] > 0

    norms = np.linalg.norm(geo.f0.x.array[: num_owned * bs].reshape(-1, bs), axis=1)
    assert np.allclose(norms[in_wall], 1.0)
    assert np.allclose(norms[~in_wall], 0.0)
    assert comm.allreduce(np.sum(~in_wall)) > 0


def test_transform_markers_ukb():
    from cardiac_geometries.mesh import transform_markers

    biv = {"LV_ENDO": (1, 2), "RV_ENDO": (2, 2), "EPI": (3, 2)}
    biv |= {"MV": (4, 2), "AV": (5, 2), "TV": (6, 2), "PV": (7, 2), "Wall": (8, 3)}
    assert transform_markers(biv) == {"lv": [1], "rv": [2], "epi": [3], "base": [4, 5, 6, 7]}

    clipped = {"LV_ENDO": (1, 2), "RV_ENDO": (2, 2), "EPI": (3, 2), "BASE": (4, 2)}
    assert transform_markers(clipped) == {"lv": [1], "rv": [2], "epi": [3], "base": [4]}

    lv = {"LV_ENDO": (1, 2), "EPI": (2, 2), "RV_SEPTUM": (3, 2), "SEPTAL_BRIDGE": (4, 2)}
    lv |= {"MV": (5, 2), "AV": (6, 2), "MV_CAP": (7, 2), "AV_CAP": (8, 2), "Wall": (9, 3)}
    assert transform_markers(lv) == {"lv": [1], "epi": [2, 3], "base": [5, 6]}


@pytest.mark.skipif(not HAS_UKB, reason="UKB atlas is not installed")
def test_ukb_clipped_requires_biv(tmp_path: Path):
    from cardiac_geometries import mesh

    with pytest.raises(ValueError, match="only supported for the 'biv' geometry"):
        mesh.ukb(tmp_path, geometry="lv", clipped=True)


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
