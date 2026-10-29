from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from click.testing import CliRunner

import cardiac_geometries as cg
from cardiac_geometries import Geometry, cli
from cardiac_geometries.fibers.lv_ellipsoid import create_microstructure, transmural_coordinate
from cardiac_geometries.utils import space_from_string

RADII: dict[str, Any] = dict(r_short_endo=7.0, r_short_epi=10.0, r_long_endo=17.0, r_long_epi=20.0)


def ellipsoid_residual(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    rs = RADII["r_short_endo"] + (RADII["r_short_epi"] - RADII["r_short_endo"]) * t
    rl = RADII["r_long_endo"] + (RADII["r_long_epi"] - RADII["r_long_endo"]) * t
    return (x[:, 1] ** 2 + x[:, 2] ** 2) / rs**2 + x[:, 0] ** 2 / rl**2 - 1.0


@pytest.fixture(scope="module")
def geo_2d(tmp_path_factory) -> Geometry:
    outdir = tmp_path_factory.mktemp("lv_ellipsoid_2D")
    return cg.mesh.lv_ellipsoid_2D(
        outdir=MPI.COMM_WORLD.bcast(outdir, root=0),
        order=2,
        create_fibers=True,
        fiber_angle_endo=90,
        fiber_angle_epi=-90,
        fiber_space="Quadrature_4",
    )


def test_lv_ellipsoid_2D_mesh(geo_2d: Geometry):
    mesh = geo_2d.mesh
    assert mesh.geometry.dim == 2
    assert mesh.topology.dim == 2
    assert mesh.geometry.cmaps[0].degree == 2
    assert geo_2d.efun is None
    for name, dim in [("ENDO", 1), ("EPI", 1), ("BASE", 1), ("APEX", 1), ("ENDOPT", 0)]:
        assert geo_2d.markers[name][1] == dim
    assert geo_2d.vfun is not None
    assert geo_2d.vfun.dim == 0
    # The meridian is in the upper half plane, with the epicardial apex at x = -20 and the base
    # at x = 5
    x = mesh.geometry.x
    assert x[:, 1].min() > -1e-12
    assert np.isclose(mesh.comm.allreduce(x[:, 0].min(), op=MPI.MIN), -20.0)
    assert np.isclose(mesh.comm.allreduce(x[:, 0].max(), op=MPI.MAX), 5.0)


def test_lv_ellipsoid_2D_lower_half_plane_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="mu_apex_endo"):
        cg.mesh.lv_ellipsoid_2D(outdir=tmp_path, mu_apex_endo=-np.pi)


def test_transmural_coordinate(geo_2d: Geometry):
    mesh = geo_2d.mesh
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    t = transmural_coordinate(V, **RADII)
    assert t.x.array.min() >= 0.0
    assert t.x.array.max() <= 1.0

    # The nodes of the curved mesh are on the ellipsoids
    assert geo_2d.ffun is not None
    for marker, value in [("ENDO", 0.0), ("EPI", 1.0)]:
        facets = geo_2d.ffun.find(geo_2d.markers[marker][0])
        dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
        assert np.allclose(t.x.array[dofs], value, atol=1e-8)

    # and t solves the equation of the ellipsoid everywhere inside
    x = V.tabulate_dof_coordinates()
    inside = (t.x.array > 1e-8) & (t.x.array < 1 - 1e-8)
    assert mesh.comm.allreduce(inside.any(), op=MPI.LOR)
    assert np.allclose(ellipsoid_residual(x[inside], t.x.array[inside]), 0.0, atol=1e-12)


def test_lv_ellipsoid_2D_fibers(geo_2d: Geometry):
    assert geo_2d.f0 is not None and geo_2d.s0 is not None and geo_2d.n0 is not None
    assert geo_2d.f0.ufl_shape == (3,)
    f0 = geo_2d.f0.x.array.reshape(-1, 3)
    s0 = geo_2d.s0.x.array.reshape(-1, 3)
    n0 = geo_2d.n0.x.array.reshape(-1, 3)
    for v in (f0, s0, n0):
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose((f0 * s0).sum(axis=1), 0.0)
    assert np.allclose((f0 * n0).sum(axis=1), 0.0)
    assert np.allclose((s0 * n0).sum(axis=1), 0.0)

    # The third component is the circumferential one, cos(alpha) of the fiber
    # angle alpha = 90 - 180 t
    V = space_from_string("Quadrature_4", geo_2d.mesh, dim=1)
    t = transmural_coordinate(V, **RADII).x.array
    alpha = np.deg2rad(90.0 - 180.0 * t)
    assert np.allclose(np.abs(f0[:, 2]), np.abs(np.cos(alpha)))


def test_create_microstructure_unknown_transmural(geo_2d: Geometry):
    with pytest.raises(ValueError, match="Unknown transmural coordinate"):
        create_microstructure(
            mesh=geo_2d.mesh,
            ffun=geo_2d.ffun,
            markers=geo_2d.markers,
            transmural="unknown",
            **RADII,
        )


def test_lv_ellipsoid_2D_cli(tmp_path: Path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(tmp_path, root=0)
    runner = CliRunner()
    res = runner.invoke(
        cli.lv_ellipsoid_2D,
        [path.as_posix(), "--order", "2", "--create-fibers", "--fiber-space", "Quadrature_4"],
    )
    assert res.exit_code == 0, res.output

    geo = Geometry.from_folder(comm=comm, folder=path)
    assert geo.mesh.geometry.dim == 2
    assert geo.mesh.geometry.cmaps[0].degree == 2
    assert geo.f0 is not None
    assert geo.f0.ufl_shape == (3,)
    assert geo.vfun is not None
    # The tags are local, so the apex is on one of the processes only
    assert comm.allreduce(geo.markers["ENDOPT"][0] in geo.vfun.values, op=MPI.LOR)
