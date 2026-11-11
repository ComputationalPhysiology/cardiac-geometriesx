"""RV regions of Zhong et al. (2012), with the septum split by a transventricular Laplace solution.

The pure-numpy core (:func:`aha.rv_segments`) is checked against the analytic cylinder of
``test_aha_biv.py``, where every tag is known by hand. The mesh-level function
(:func:`aha.biv_regions`) is checked on the UKB mean shape.
"""

import numpy as np
import pytest

from cardiac_geometries import aha

# The analytic cylinder of test_aha_biv.py: r in [10, 15], z in [0, 30], insertions at 80 and -60
# degrees (septal arc 140 degrees), s = z / 30
APEX = np.zeros(3)
BASE = np.array([0.0, 0.0, 30.0])


def polar(r, theta_deg, z):
    t = np.radians(theta_deg)
    return np.column_stack(np.broadcast_arrays(r * np.cos(t), r * np.sin(t), z)).astype(float)


INSERTION = np.vstack(
    [polar(12.5, 80.0, np.array([-1.0, 31.0])), polar(12.5, -60.0, np.array([31.0, -1.0]))]
)
ANTERIOR_REF = polar(20.0, 85.0, 30.0)[0]
SEPTUM_REF = polar(20.0, 10.0, 15.0)[0]
# The RV body runs from s = 0 to s = 0.75 (z = 22.5); above it, outlet or inlet by nearest valve
S_MIN, S_TOP = 0.0, 0.75
PV = polar(20.0, 85.0, 30.0)[0]
TV = polar(20.0, -55.0, 30.0)[0]


def rv_at(theta, z, septal):
    x = polar(12.5, np.atleast_1d(theta), np.atleast_1d(z))
    return aha.rv_segments(
        x,
        np.full(len(x), septal),
        APEX,
        BASE,
        INSERTION,
        ANTERIOR_REF,
        SEPTUM_REF,
        S_MIN,
        S_TOP,
        PV,
        TV,
    )


@pytest.mark.parametrize(
    "theta, z, septal, expected",
    [
        # Basal level (z in [15, 22.5]); psi = 80 - theta, width 140
        (70.0, 20.0, False, 18),  # frac 10/140: basal anterior
        (10.0, 20.0, False, 19),  # frac 70/140: basal lateral
        (-50.0, 20.0, False, 20),  # frac 130/140: basal inferior
        (-40.0, 20.0, True, 21),  # septal, frac 120/140: basal inferoseptal
        (60.0, 20.0, True, 22),  # septal, frac 20/140: basal anteroseptal
        (10.0, 10.0, False, 24),  # mid lateral
        (60.0, 10.0, True, 27),  # mid anteroseptal
        (40.0, 3.0, False, 28),  # apical anterior (frac 40/140 < 1/2)
        (-20.0, 3.0, False, 29),  # apical inferior
        (10.0, 3.0, True, 30),  # apical septal
    ],
)
def test_rv_segments_analytic(theta, z, septal, expected):
    assert rv_at(theta, z, septal)[0] == expected


def test_rv_segments_outlet_inlet():
    # Above s_top (z > 22.5): nearer the PV (85 degrees) is the outlet, nearer the TV the inlet
    assert rv_at(84.0, 27.0, False)[0] == 31
    assert rv_at(-54.0, 27.0, False)[0] == 32
    assert rv_at(-54.0, 27.0, True)[0] == 32


def test_rv_segments_outside_arc_nearest_end():
    # 5 degrees past the inferior insertion: inferior, not anterior
    assert rv_at(-65.0, 20.0, False)[0] == 20
    # 5 degrees before the anterior insertion (psi = 355): anterior
    assert rv_at(85.0, 20.0, False)[0] == 18
    # The same for the septal sectors
    assert rv_at(-65.0, 20.0, True)[0] == 21
    assert rv_at(85.0, 20.0, True)[0] == 22


def test_rv_segments_below_s_min_is_apical():
    assert rv_at(40.0, -0.5, False)[0] == 28


def test_rv_segments_equal_angle_sectors():
    # On a dense ring in the basal level the three free-wall sectors take equal angles, and the
    # two septal sectors too
    theta = np.linspace(80.0, -60.0, 1401)[1:-1]
    free = rv_at(theta, 20.0, False)
    assert [int(np.sum(free == t)) for t in (18, 19, 20)] == pytest.approx([466, 466, 466], abs=2)
    sept = rv_at(theta, 20.0, True)
    assert set(np.unique(sept)) == {21, 22}
    assert np.sum(sept == 21) == pytest.approx(np.sum(sept == 22), abs=2)


def test_rv_region_names():
    assert len(aha.RV_REGION_NAMES) == 15
    assert aha.RV_REGION_NAMES[0] == "RV-BASAL-ANTERIOR"
    assert aha.RV_REGION_NAMES[12] == "RV-APICAL-SEPTAL"
    assert aha.RV_REGION_NAMES[13:] == ["RV-OUTLET", "RV-INLET"]
    assert aha.RV_REGION_OFFSET == 17
    assert aha.LVOT_TAG_RV_REGIONS == 33


# aha_segments on a 9 x 7 grid (theta every 40 degrees from -160, z from -2 to 34), recorded from
# 0.19.1 before the short-axis frame was factored out
AHA_GRID = [
    [17, 16, 11, 11, 5, 5, 19],
    [17, 15, 11, 11, 5, 5, 19],
    [17, 15, 10, 10, 4, 4, 19],
    [17, 15, 9, 9, 3, 3, 19],
    [17, 14, 9, 9, 3, 3, 19],
    [17, 14, 8, 8, 2, 2, 19],
    [17, 13, 8, 8, 2, 2, 19],
    [17, 13, 7, 7, 1, 1, 19],
    [17, 16, 12, 12, 6, 6, 19],
]


def test_aha_segments_unchanged_by_refactor():
    theta, z = np.arange(-160, 200, 40.0), np.linspace(-2, 34, 7)
    T, Z = np.meshgrid(theta, z, indexing="ij")
    x = polar(12.5, T.ravel(), Z.ravel())
    labels = aha.aha_segments(x, APEX, BASE, INSERTION, ANTERIOR_REF, septum_ref=SEPTUM_REF)
    assert labels.reshape(len(theta), len(z)).tolist() == AHA_GRID


# ---------------------------------------------------------------------------
# The UKB mean shape: AHA segments plus the RV regions, the septum split by u
# ---------------------------------------------------------------------------
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import ufl  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from cardiac_geometries import cli, mesh  # noqa: E402
from cardiac_geometries.geometry import Geometry  # noqa: E402

RV_SEPTAL = (21, 22, 26, 27, 30)
LV_SEPTAL = (2, 3, 8, 9)  # basal and mid; the apical septal segment is 90 degrees, not the arc


def build(path, **kwargs):
    return mesh.ukb(
        path, char_length_max=5.0, char_length_min=5.0, create_fibers=False, aha=True, **kwargs
    )


@pytest.fixture(scope="module")
def ukb_regions(tmp_path_factory):
    path = MPI.COMM_WORLD.bcast(tmp_path_factory.mktemp("ukb-rv"), root=0)
    geo = build(path, rv_regions=True)
    landmarks = json.loads((path / "landmarks_ED.json").read_text())
    return path, geo, landmarks


def all_values(tags):
    return np.unique(np.hstack(MPI.COMM_WORLD.allgather(tags.values)))


def volumes(m, tags, ids):
    dx = ufl.Measure("dx", domain=m, subdomain_data=tags)
    out = {}
    for i in ids:
        local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * dx(int(i), domain=m)))
        out[int(i)] = m.comm.allreduce(local)
    return out


def regions(geo, landmarks, fraction):
    return aha.biv_regions(geo.mesh, geo.ffun, geo.markers, landmarks, rv_septum_fraction=fraction)


def test_biv_regions_tags_every_cell(ukb_regions):
    _, geo, _ = ukb_regions
    ncells = geo.mesh.topology.index_map(3).size_local
    owned = geo.cfun.indices[geo.cfun.indices < ncells]
    assert len(np.unique(owned)) == ncells
    assert all_values(geo.cfun).tolist() == list(range(1, 34))


def test_biv_regions_rv_volumes(ukb_regions):
    _, geo, _ = ukb_regions
    vol = volumes(geo.mesh, geo.cfun, range(18, 33))
    rv = sum(vol.values())
    shares = {k: v / rv for k, v in vol.items()}
    assert all(0.02 <= shares[k] <= 0.20 for k in range(18, 31)), shares
    # Measured on the mean shape: outlet + inlet hold 12-14% of the RV (with the RV half of the
    # septum), about two regions' share; the body regions 3-11% each
    assert 0.08 <= shares[31] + shares[32] <= 0.30, shares


def test_biv_regions_septum_halves(ukb_regions):
    _, geo, landmarks = ukb_regions
    half = volumes(geo.mesh, geo.cfun, RV_SEPTAL)
    tags_all, _ = regions(geo, landmarks, 1.0)
    whole = volumes(geo.mesh, tags_all, RV_SEPTAL)
    ratio = sum(half.values()) / sum(whole.values())
    assert 0.3 <= ratio <= 0.7, ratio


def test_biv_regions_fraction_extremes(ukb_regions):
    _, geo, landmarks = ukb_regions
    ncells = geo.mesh.topology.index_map(3).size_local
    tags0, _ = regions(geo, landmarks, 0.0)
    assert not set(all_values(tags0)) & set(RV_SEPTAL)
    # With no RV septum the LV cells are biv_aha's (with the LVOT as 33), except the RV wall at the
    # junction (outside the septal arc), which biv_regions moves to the RV free wall
    ref, _ = aha.biv_aha(geo.mesh, geo.ffun, geo.markers, landmarks)
    ref_lv = ref.values[:ncells] != aha.RV_TAG
    expected = np.where(ref.values[:ncells] == aha.LVOT_TAG, 33, ref.values[:ncells])
    new = tags0.values[:ncells]
    still_lv = ref_lv & ((new <= 17) | (new == 33))
    assert np.array_equal(new[still_lv], expected[still_lv])
    moved = ref_lv & ~still_lv
    assert set(np.unique(new[moved])) <= {18, 19, 20, 23, 24, 25, 28, 29, 31, 32}
    # ... and that band is thin: under 15% of the cells biv_aha calls LV
    assert MPI.COMM_WORLD.allreduce(int(moved.sum())) < 0.15 * MPI.COMM_WORLD.allreduce(
        int(ref_lv.sum())
    )
    tags1, _ = regions(geo, landmarks, 1.0)
    assert not set(all_values(tags1)) & set(LV_SEPTAL)
    with pytest.raises(ValueError, match="rv_septum_fraction"):
        regions(geo, landmarks, 1.5)


def test_biv_regions_markers(ukb_regions):
    _, geo, _ = ukb_regions
    markers = aha.biv_region_markers()
    assert markers["LVOT"] == (33, 3) and "RV" not in markers
    assert [markers[n][0] for n in aha.RV_REGION_NAMES] == list(range(18, 33))
    assert markers["BASAL-ANTERIOR"] == (1, 3) and markers["APEX"] == (17, 3)
    assert aha.aha_markers()["LVOT"] == (19, 3)  # unchanged without RV regions
    assert tuple(geo.markers["LVOT"]) == (33, 3) and "RV" not in geo.markers


def test_transventricular_boundary_values(ukb_regions):
    _, geo, landmarks = ukb_regions
    u = aha.transventricular(geo.mesh, geo.ffun, geo.markers)
    V = u.function_space
    for name, value in (("RV_ENDO", 0.0), ("LV_ENDO", 1.0)):
        dofs = dolfinx.fem.locate_dofs_topological(V, 2, geo.ffun.find(geo.markers[name][0]))
        assert np.allclose(u.x.array[dofs], value, atol=1e-12)
    # On the septal candidates (all RV septal with fraction 1) u is about one half on average
    tags1, _ = regions(geo, landmarks, 1.0)
    dx = ufl.Measure("dx", domain=geo.mesh, subdomain_data=tags1)
    num = sum(
        geo.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(u * dx(i))))
        for i in RV_SEPTAL
    )
    den = sum(volumes(geo.mesh, tags1, RV_SEPTAL).values())
    assert 0.4 < num / den < 0.6


def test_aha_alone_unchanged(ukb_regions):
    path, geo, landmarks = ukb_regions
    from cardiac_geometries import utils

    plain = utils.gmsh2dolfin(comm=MPI.COMM_WORLD, msh_file=path / "ED.msh")
    ref, ref_markers = aha.biv_aha(plain.mesh, plain.ffun, plain.markers, landmarks)
    assert all_values(ref).tolist() == list(range(1, 20))
    assert ref_markers == aha.aha_markers()


def test_biv_regions_mpi(ukb_regions, tmp_path):
    if MPI.COMM_WORLD.size > 1:
        pytest.skip("launches its own MPI runs")
    path, _, _ = ukb_regions
    script = tmp_path / "volumes.py"
    script.write_text(
        f"""
import json
from pathlib import Path
from mpi4py import MPI
import dolfinx, ufl
from cardiac_geometries import aha, utils
geo = utils.gmsh2dolfin(comm=MPI.COMM_WORLD, msh_file=Path({str(path / "ED.msh")!r}))
landmarks = json.loads(Path({str(path / "landmarks_ED.json")!r}).read_text())
tags, _ = aha.biv_regions(geo.mesh, geo.ffun, geo.markers, landmarks)
dx = ufl.Measure("dx", domain=geo.mesh, subdomain_data=tags)
def volume(i):
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * dx(i, domain=geo.mesh)))
    return geo.mesh.comm.allreduce(local)
vols = [volume(i) for i in range(1, 34)]
if MPI.COMM_WORLD.rank == 0:
    Path({str(tmp_path)!r}, f"vols{{MPI.COMM_WORLD.size}}.json").write_text(json.dumps(vols))
"""
    )
    for n in (1, 2):
        r = subprocess.run(
            ["mpirun", "-n", str(n), sys.executable, str(script)], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-2000:]
    v1 = json.loads((tmp_path / "vols1.json").read_text())
    v2 = json.loads((tmp_path / "vols2.json").read_text())
    assert np.allclose(v1, v2, rtol=1e-10)
    assert min(v1) > 0


def test_ukb_rv_regions_roundtrip(ukb_regions):
    path, geo, _ = ukb_regions
    loaded = Geometry.from_folder(comm=MPI.COMM_WORLD, folder=path)
    assert all_values(loaded.cfun).tolist() == list(range(1, 34))
    vol = volumes(geo.mesh, geo.cfun, range(1, 34))
    vol_loaded = volumes(loaded.mesh, loaded.cfun, range(1, 34))
    assert np.allclose(list(vol.values()), list(vol_loaded.values()), rtol=1e-12)
    assert {k: tuple(v) for k, v in loaded.markers.items()} == {
        k: tuple(v) for k, v in geo.markers.items()
    }
    info = json.loads((path / "info.json").read_text())
    assert info["rv_regions"] is True and info["rv_septum_fraction"] == 0.5


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(rv_regions=True), "aha"),
        (dict(aha=True, rv_septum_fraction=0.3), "rv_regions"),
        (dict(aha=True, rv_regions=True, rv_septum_fraction=1.5), "rv_septum_fraction"),
        (dict(aha=True, rv_regions=True, geometry="lv"), "biv"),
    ],
)
def test_ukb_rv_regions_errors(tmp_path, kwargs, message):
    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    kwargs = {"aha": False, **kwargs}
    with pytest.raises(ValueError, match=message):
        mesh.ukb(path, char_length_max=5.0, char_length_min=5.0, create_fibers=False, **kwargs)


def test_ukb_cli_rv_regions(tmp_path):
    runner = CliRunner()
    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    args = [str(path / "biv"), "--char-length-max", "5", "--char-length-min", "5", "--aha"]
    res = runner.invoke(cli.ukb, args + ["--rv-regions", "--rv-septum-fraction", "0.4"])
    assert res.exit_code == 0, res.output
    info = json.loads((path / "biv" / "info.json").read_text())
    assert info["rv_regions"] is True and info["rv_septum_fraction"] == 0.4


def test_biv_regions_insertion_band_is_rv(ukb_regions):
    # Cells just outside the septal arc whose nearest RV landmark node is an insertion node are not
    # free wall by biv_aha's test, but where they are nearer the RV endocardium than the LV
    # endocardium they are RV wall at the junction and must get an RV region, not an AHA segment
    _, geo, landmarks = ukb_regions
    m = geo.mesh
    ncells = m.topology.index_map(3).size_local
    x = dolfinx.mesh.compute_midpoints(m, 3, np.arange(ncells, dtype=np.int32))
    apex, mv = aha.lv_long_axis(m, geo.ffun, geo.markers)
    septum = np.asarray(landmarks["rv_septum"], dtype=float)
    s, psi, width = aha._short_axis_frame(
        x, apex, mv, landmarks["insertion"], landmarks["valves"]["PV"], septum.mean(axis=0)
    )
    rv_nodes = np.vstack([np.asarray(landmarks["rv_freewall"], dtype=float), septum])
    d_rv, _ = aha._nearest(x, rv_nodes)
    lv_endo = aha._gather_facet_midpoints(m, geo.ffun, geo.markers["LV_ENDO"][0])
    d_lv, _ = aha._nearest(x, lv_endo)
    band = (psi >= width) & (s >= 0.0) & (s <= 1.0) & (d_rv < d_lv)
    values = geo.cfun.values[:ncells]
    lv_tagged = band & (values <= 17)
    assert MPI.COMM_WORLD.allreduce(int(lv_tagged.sum())) == 0
