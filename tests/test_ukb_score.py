"""PCA scores in :func:`cardiac_geometries.mesh.ukb` (passed on to ukb-atlas)."""

import json

from mpi4py import MPI

import numpy as np
import pytest

pytest.importorskip("ukb")

from cardiac_geometries import mesh  # noqa: E402

CHAR_LENGTH = {"char_length_max": 5.0, "char_length_min": 5.0}
SCORE = [3.0] + [0.0] * 24


def shared_path(tmp_path):
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


def bounds(geo):
    x = geo.mesh.geometry.x
    comm = geo.mesh.comm
    lo = [comm.allreduce(float(v), op=MPI.MIN) for v in x.min(axis=0)]
    hi = [comm.allreduce(float(v), op=MPI.MAX) for v in x.max(axis=0)]
    return np.array([lo, hi])


def test_ukb_score_changes_shape(tmp_path):
    path = shared_path(tmp_path)
    mean = mesh.ukb(path / "a", mode=-1, create_fibers=False, **CHAR_LENGTH)
    scored = mesh.ukb(path / "b", score=SCORE, create_fibers=False, **CHAR_LENGTH)
    assert np.abs(bounds(mean) - bounds(scored)).max() > 1.0


def test_ukb_score_recorded(tmp_path):
    path = shared_path(tmp_path)
    mesh.ukb(path, score=SCORE, create_fibers=False, **CHAR_LENGTH)
    assert json.loads((path / "info.json").read_text())["score"] == SCORE


def test_ukb_reuse_mesh_rejects_other_score(tmp_path):
    path = shared_path(tmp_path)
    mesh.ukb(path, score=SCORE, create_fibers=False, **CHAR_LENGTH)
    with pytest.raises(ValueError, match="score"):
        mesh.ukb(
            path, score=[-3.0] + [0.0] * 24, create_fibers=False, reuse_mesh=True, **CHAR_LENGTH
        )


def test_cli_ukb_score(tmp_path):
    from click.testing import CliRunner

    from cardiac_geometries import cli

    path = shared_path(tmp_path)
    args = [str(path), "--char-length-max", "5", "--char-length-min", "5"]
    res = CliRunner().invoke(cli.ukb, [*args, "--score", "3", "--score", "0"])
    assert res.exit_code == 0, res.output
    assert json.loads((path / "info.json").read_text())["score"] == [3.0, 0.0]
