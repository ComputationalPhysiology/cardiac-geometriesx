from pathlib import Path
import pytest
from cardiac_geometries import cli

from cardiac_geometries import Geometry
from click.testing import CliRunner


@pytest.mark.parametrize(
    "script",
    [
        cli.slab,
        cli.lv_ellipsoid,
    ],
)
def test_script(script, tmp_path: Path):
    runner = CliRunner()
    path = tmp_path

    res = runner.invoke(script, [path.as_posix(), "--create-fibers"])
    assert res.exit_code == 0
    assert path.is_dir()
    geo = Geometry.from_folder(path)
    assert geo.mesh.geometry.dim == 3
    assert geo.f0 is not None


@pytest.mark.parametrize(
    "script",
    [
        cli.slab_in_bath,
        cli.biv_ellipsoid,
        cli.biv_ellipsoid_torso,
    ],
)
def test_script_no_fibers(script, tmp_path: Path):
    runner = CliRunner()
    path = tmp_path
    res = runner.invoke(script, [path.as_posix()])
    assert res.exit_code == 0
    assert path.is_dir()
    geo = Geometry.from_folder(path)
    assert geo.mesh.geometry.dim == 3
