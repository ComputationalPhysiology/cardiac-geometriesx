"""Stage timers of :func:`cardiac_geometries.mesh.ukb`, and reusing the mesh of an earlier call."""

from mpi4py import MPI

import pytest

pytest.importorskip("ukb")

import dolfinx  # noqa: E402

from cardiac_geometries import mesh  # noqa: E402

CHAR_LENGTH = {"char_length_max": 5.0, "char_length_min": 5.0}


def timer_count(name):
    try:
        return dolfinx.common.timing(name)[0]
    except RuntimeError:  # never started
        return 0


def counts():
    return {name: timer_count(name) for name in mesh.UKB_TIMERS}


def shared_path(tmp_path):
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


def test_ukb_records_stage_timers(tmp_path):
    before = counts()
    mesh.ukb(shared_path(tmp_path), create_fibers=True, aha=True, **CHAR_LENGTH)
    after = counts()
    assert {name: after[name] - before[name] for name in mesh.UKB_TIMERS} == dict.fromkeys(
        mesh.UKB_TIMERS, 1
    )


def test_ukb_regions_timer_only_with_aha(tmp_path):
    before = counts()
    mesh.ukb(shared_path(tmp_path), create_fibers=False, aha=False, **CHAR_LENGTH)
    after = counts()
    assert {name: after[name] - before[name] for name in mesh.UKB_TIMERS} == {
        mesh.TIMER_MESH: 1,
        mesh.TIMER_READ_MESH: 1,
        mesh.TIMER_REGIONS: 0,
        mesh.TIMER_FIBERS: 0,
        mesh.TIMER_SAVE: 1,
    }


def test_ukb_save_writes_json_after_fibers(tmp_path):
    path = shared_path(tmp_path)
    mesh.ukb(path, create_fibers=True, **CHAR_LENGTH)
    viz = (path / "f0-viz.bp").stat().st_mtime
    assert (path / "markers.json").stat().st_mtime >= viz
    assert (path / "info.json").stat().st_mtime >= viz


def test_ukb_reuse_mesh_same_mesh_without_meshing(tmp_path):
    path = shared_path(tmp_path)
    first = mesh.ukb(path, create_fibers=False, aha=True, **CHAR_LENGTH)
    n = timer_count(mesh.TIMER_MESH)
    again = mesh.ukb(path, create_fibers=False, aha=True, reuse_mesh=True, **CHAR_LENGTH)
    assert timer_count(mesh.TIMER_MESH) == n
    cells = again.mesh.topology.index_map(3).size_global
    assert cells == first.mesh.topology.index_map(3).size_global
    assert again.markers == first.markers


def test_ukb_reuse_mesh_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="ED.msh"):
        mesh.ukb(shared_path(tmp_path), reuse_mesh=True)


def test_ukb_reuse_mesh_rejects_other_parameters(tmp_path):
    path = shared_path(tmp_path)
    mesh.ukb(path, create_fibers=False, **CHAR_LENGTH)
    with pytest.raises(ValueError, match="char_length_max") as error:
        mesh.ukb(path, char_length_max=4.0, char_length_min=5.0, reuse_mesh=True)
    assert "4.0" in str(error.value) and "5.0" in str(error.value)
