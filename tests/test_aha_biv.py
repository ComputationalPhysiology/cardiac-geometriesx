"""AHA segmentation of biventricular geometries (Cerqueira et al. 2002).

The segmentation is checked in three independent ways: against an analytic
geometry with a known answer, against the anatomical landmarks of the UKB
template (from ukb-atlas), and by its stability across atlas shapes, which share
the template's node numbering.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
from click.testing import CliRunner

from cardiac_geometries import aha, cli, mesh
from cardiac_geometries.geometry import Geometry

# Apex cap, apical, mid and basal levels and the region above the base
MIRROR = {1: 4, 2: 3, 3: 2, 4: 1, 5: 6, 6: 5, 13: 15, 14: 14, 15: 13, 16: 16, 17: 17, 19: 19}
MIRROR |= {k + 6: v + 6 for k, v in MIRROR.items() if k <= 6}

# ---------------------------------------------------------------------------
# Analytic geometry: a thick cylinder about z, r in [10, 15], z in [0, 30], with
# vertical insertion lines at 80 and -60 degrees (septum 140 degrees, so the
# septal segments are 70 degrees and the free-wall segments 55 degrees).
# ---------------------------------------------------------------------------
APEX = np.zeros(3)
BASE = np.array([0.0, 0.0, 30.0])
THETA_A, THETA_I = 80.0, -60.0


def polar(r, theta_deg, z):
    t = np.radians(theta_deg)
    return np.column_stack(np.broadcast_arrays(r * np.cos(t), r * np.sin(t), z)).astype(float)


INSERTION = np.vstack(
    [polar(12.5, THETA_A, np.array([-1.0, 31.0])), polar(12.5, THETA_I, np.array([31.0, -1.0]))]
)
ANTERIOR_REF = polar(20.0, 85.0, 30.0)[0]
SEPTUM_REF = polar(20.0, 10.0, 15.0)[0]


def labels_at(theta, z, anterior_ref=ANTERIOR_REF):
    x = polar(12.5, np.atleast_1d(theta), np.atleast_1d(z))
    return aha.aha_segments(x, APEX, BASE, INSERTION, anterior_ref, septum_ref=SEPTUM_REF)


@pytest.mark.parametrize(
    "theta, z, expected",
    [
        # Mid level: septum from -60 to 80, split at its bisector (10 degrees)
        (40.0, 15.0, 8),  # anteroseptal
        (-20.0, 15.0, 9),  # inferoseptal
        # Free wall from 80 on round to 300 (= -60) in four 55 degree sectors
        (100.0, 15.0, 7),  # anterior (80-135)
        (160.0, 15.0, 12),  # anterolateral (135-190)
        (220.0, 15.0, 11),  # inferolateral (190-245)
        (280.0, 15.0, 10),  # inferior (245-300)
        # Basal level, same sectors
        (40.0, 25.0, 2),
        (-20.0, 25.0, 3),
        (100.0, 25.0, 1),
        (280.0, 25.0, 4),
        # Apical level: septal 90 degrees centred on the bisector (-35 to 55)
        (10.0, 5.0, 14),
        (100.0, 5.0, 13),  # anterior (55-145)
        (190.0, 5.0, 16),  # lateral (145-235)
        (280.0, 5.0, 15),  # inferior (235-325)
        # Beyond the apex and above the base
        (100.0, -0.5, 17),
        (100.0, 30.5, 19),
    ],
)
def test_segments_analytic_points(theta, z, expected):
    assert labels_at(theta, z)[0] == expected


def test_segments_analytic_mirror():
    # With the other insertion as the anterior one, every segment becomes its mirror
    rng = np.random.default_rng(0)
    theta, z = rng.uniform(-180, 180, 500), rng.uniform(-0.9, 30.9, 500)
    labels = labels_at(theta, z)
    swapped = labels_at(theta, z, anterior_ref=polar(20.0, -65.0, 30.0)[0])
    assert np.array_equal(swapped, [MIRROR[v] for v in labels])


# Like the UKB template: the insertion loop has two near-vertical branches and closes with runs
# across the top of the septum (rising from z = 23 at the inferior branch to z = 26 at the anterior
# one, below the base at z = 30) and near the apex (z = 3)
_RUN = np.linspace(0.0, 1.0, 15)[1:-1]
CLOSED_LOOP = np.vstack(
    [
        polar(12.5, THETA_A, np.linspace(3.0, 26.0, 12)),
        polar(12.5, THETA_A + (THETA_I - THETA_A) * _RUN, 26.0 - 3.0 * _RUN),
        polar(12.5, THETA_I, np.linspace(23.0, 3.0, 12)),
        polar(12.5, THETA_I + (THETA_A - THETA_I) * _RUN, np.full(len(_RUN), 3.0)),
    ]
)


@pytest.mark.parametrize("z", [24.5, 25.5, 27.0, 29.5])
def test_segments_above_the_insertion_branches(z):
    # Where short-axis planes cut the run across the top of the septum, or nothing, the insertions
    # keep the azimuths of the branches below
    x = polar(12.5, np.array([40.0, -20.0, 100.0, 280.0]), np.full(4, z))
    labels = aha.aha_segments(x, APEX, BASE, CLOSED_LOOP, ANTERIOR_REF, septum_ref=SEPTUM_REF)
    assert labels.tolist() == [2, 3, 1, 4]


@pytest.mark.parametrize("z", [1.0, 2.5, 4.0])
def test_segments_near_the_apical_run(z):
    x = polar(12.5, np.array([10.0, 100.0, 280.0]), np.full(3, z))
    labels = aha.aha_segments(x, APEX, BASE, CLOSED_LOOP, ANTERIOR_REF, septum_ref=SEPTUM_REF)
    assert labels.tolist() == [14, 13, 15]


def test_segments_shorter_arc_without_septum_ref():
    x = polar(12.5, np.array([40.0, 100.0]), np.array([15.0, 15.0]))
    labels = aha.aha_segments(x, APEX, BASE, INSERTION, ANTERIOR_REF)
    assert labels.tolist() == [8, 7]  # the 140 degree arc is the septum


def test_segments_analytic_volumes():
    rng = np.random.default_rng(1)
    n = 200_000
    r = np.sqrt(rng.uniform(10.0**2, 15.0**2, n))
    theta = rng.uniform(-180.0, 180.0, n)
    z = rng.uniform(0.0, 30.0, n)
    labels = aha.aha_segments(
        polar(r, theta, z), APEX, BASE, INSERTION, ANTERIOR_REF, septum_ref=SEPTUM_REF
    )
    widths = {2: 70, 3: 70, 1: 55, 6: 55, 5: 55, 4: 55}
    for level, offset in (("basal", 0), ("mid", 6)):
        in_level = np.isin(labels, np.arange(1, 7) + offset)
        for segment, width in widths.items():
            fraction = np.mean(labels[in_level] == segment + offset)
            assert fraction == pytest.approx(width / 360, abs=0.01), (level, segment)
    in_apical = np.isin(labels, [13, 14, 15, 16])
    for segment in (13, 14, 15, 16):
        assert np.mean(labels[in_apical] == segment) == pytest.approx(0.25, abs=0.01)


# ---------------------------------------------------------------------------
# UKB BiV, with the anatomical landmarks written by ukb-atlas
# ---------------------------------------------------------------------------
pytest.importorskip("ukb.template")


def nearest(points, targets):
    d = np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=2)
    return d.min(axis=1), d.argmin(axis=1)


def cell_data(geo):
    m = geo.mesh
    ncells = m.topology.index_map(3).size_local
    cells = np.arange(ncells, dtype=np.int32)
    x = dolfinx.mesh.compute_midpoints(m, 3, cells)
    tags = np.zeros(ncells, dtype=np.int32)
    tags[geo.cfun.indices[geo.cfun.indices < ncells]] = geo.cfun.values[geo.cfun.indices < ncells]
    return x, tags


def build(path, **kwargs):
    return mesh.ukb(
        path, char_length_max=5.0, char_length_min=5.0, create_fibers=False, aha=True, **kwargs
    )


@pytest.fixture(scope="module")
def ukb_geo(tmp_path_factory):
    path = MPI.COMM_WORLD.bcast(tmp_path_factory.mktemp("ukb-aha"), root=0)
    geo = build(path)
    landmarks = json.loads((path / "landmarks_ED.json").read_text())
    return path, geo, landmarks


def test_biv_aha_tags_every_cell(ukb_geo):
    _, geo, _ = ukb_geo
    ncells = geo.mesh.topology.index_map(3).size_local
    owned = geo.cfun.indices[geo.cfun.indices < ncells]
    assert len(np.unique(owned)) == ncells
    values = np.unique(np.hstack(MPI.COMM_WORLD.allgather(geo.cfun.values)))
    assert values.tolist() == list(range(1, 20))
    for name, marker in [
        ("BASAL-ANTERIOR", (1, 3)),
        ("APEX", (17, 3)),
        ("RV", (18, 3)),
        ("LVOT", (19, 3)),
    ]:
        assert tuple(geo.markers[name]) == marker


def lv_axis_s(geo, x):
    apex, mv_centre = aha.lv_long_axis(geo.mesh, geo.ffun, geo.markers)
    e_l = mv_centre - apex
    length = np.linalg.norm(e_l)
    return (x - apex) @ (e_l / length) / length


def test_biv_aha_insertions_on_boundaries(ukb_geo):
    # The template's insertion line is independent of the segmentation's cut: at mid
    # level the LV cells next to the anterior insertion are anterior or anteroseptal,
    # and next to the inferior insertion inferoseptal or inferior.
    _, geo, landmarks = ukb_geo
    x, tags = cell_data(geo)
    lv = (tags != 18) & (tags != 0)
    x, tags = (
        np.vstack(MPI.COMM_WORLD.allgather(x[lv])),
        np.hstack(MPI.COMM_WORLD.allgather(tags[lv])),
    )
    loop = np.array(landmarks["insertion"])
    s = lv_axis_s(geo, loop)
    mid = (s > 0.4) & (s < 0.6)
    # Contiguous runs of mid-level loop points: one anterior and one inferior branch
    runs = np.split(np.arange(len(loop)), np.flatnonzero(np.diff(mid.astype(int))) + 1)
    branches = [r for r in runs if mid[r[0]]]
    if len(branches) == 3:  # the loop's start split one branch in two
        branches = [np.concatenate([branches[0], branches[2]]), branches[1]]
    assert len(branches) == 2
    pv = np.array(landmarks["valves"]["PV"])
    branches.sort(key=lambda b: np.linalg.norm(loop[b] - pv, axis=1).mean())
    for branch, allowed in zip(branches, ({7, 8}, {9, 10})):
        _, idx = nearest(loop[branch], x)
        assert np.mean(np.isin(tags[idx], list(allowed))) >= 0.9, (allowed, tags[idx])


def test_biv_aha_rv_free_wall(ukb_geo):
    _, geo, landmarks = ukb_geo
    x, tags = cell_data(geo)
    septum = np.array(landmarks["rv_septum"])
    freewall = np.array(landmarks["rv_freewall"])
    d_sep, _ = nearest(x, septum)
    d_fw, _ = nearest(x, freewall)
    # The septum belongs to the LV: cells clearly nearer the RV septum than the free wall are LV
    assert not np.any((tags == 18) & (d_sep < 0.5 * d_fw))
    # The RV free wall is thin: its cells lie close to the RV free-wall nodes
    assert d_fw[tags == 18].max() < 10.0


def test_biv_aha_basal_septum_does_not_collapse(ukb_geo):
    # The insertion loop ends below the base: the basal septal segments (2, 3) must keep their share
    # of the basal level up to the base, not shrink where the short-axis planes no longer cut the
    # loop's branches
    _, geo, _ = ukb_geo
    x, tags = cell_data(geo)
    x, tags = np.vstack(MPI.COMM_WORLD.allgather(x)), np.hstack(MPI.COMM_WORLD.allgather(tags))
    s = lv_axis_s(geo, x)
    basal = np.isin(tags, range(1, 7))
    lower, upper = basal & (s < 5 / 6), basal & (s >= 5 / 6)
    for seg in (2, 3):
        share_lower = np.mean(tags[lower] == seg)
        share_upper = np.mean(tags[upper] == seg)
        assert share_upper > 0.5 * share_lower, (seg, share_lower, share_upper)


def test_biv_aha_septum_is_lv(ukb_geo):
    # Cells whose nearest RV endocardial node (insertion nodes excluded, as they belong to both
    # surfaces) is on the septum are septal myocardium, and so LV, not RV
    _, geo, landmarks = ukb_geo
    x, tags = cell_data(geo)
    septum = np.array(landmarks["rv_septum"])
    freewall = np.array(landmarks["rv_freewall"])
    insertion = np.array(landmarks["insertion"])
    d_ins_fw, _ = nearest(freewall, insertion)
    d_ins_sep, _ = nearest(septum, insertion)
    fw_only, sep_only = freewall[d_ins_fw > 1e-9], septum[d_ins_sep > 1e-9]
    d_fw, _ = nearest(x, fw_only)
    d_sep, _ = nearest(x, sep_only)
    septal = d_sep < d_fw
    assert MPI.COMM_WORLD.allreduce(int(np.sum((tags == 18) & septal))) == 0


def test_biv_aha_anterior_faces_pv(ukb_geo):
    _, geo, landmarks = ukb_geo
    x, tags = cell_data(geo)
    x, tags = np.vstack(MPI.COMM_WORLD.allgather(x)), np.hstack(MPI.COMM_WORLD.allgather(tags))
    pv = np.array(landmarks["valves"]["PV"])
    d = np.linalg.norm(x - pv, axis=1)
    assert d[np.isin(tags, [1, 7, 13])].mean() < d[np.isin(tags, [4, 10, 15])].mean()


def test_biv_aha_stable_across_shapes(tmp_path):
    # The atlas shapes share the template's node numbering, so a node should keep
    # its segment from one shape to the next, unless it lies on a segment boundary.
    from ukb import atlas, template

    if MPI.COMM_WORLD.size > 1:
        pytest.skip("serial only")
    t = template.load_template()
    nodes = np.unique(t.faces[t.regions["LV_ENDO"]])
    filename = atlas.download_atlas(Path.home() / ".ukb")

    def node_labels(std):
        path = tmp_path / f"std{std}"
        geo = build(path, mode=-1 if std == 0 else 0, std=std)
        points = atlas.generate_points(filename=filename, mode=-1 if std == 0 else 0, std=std).ED[
            nodes
        ]
        x, tags = cell_data(geo)
        lv = tags != 18
        d, idx = nearest(points, x[lv])
        return tags[lv][idx], x[lv], tags[lv]

    labels, x, tags = node_labels(0)
    # Nodes away from boundaries on the mean shape: all cells within 5 mm share the label
    points = atlas.generate_points(filename=filename, mode=-1, std=0).ED[nodes]
    d = np.linalg.norm(points[:, None, :] - x[None, :, :], axis=2)
    interior = np.array([len(np.unique(tags[row < 5.0])) == 1 for row in d])
    assert interior.mean() > 0.3
    for std in (-2.0, 2.0):
        other, _, _ = node_labels(std)
        agreement = np.mean(other[interior] == labels[interior])
        assert agreement >= 0.9, (std, agreement)


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available")
def test_biv_aha_mpi(ukb_geo, tmp_path):
    if MPI.COMM_WORLD.size > 1:
        pytest.skip("launches its own MPI runs")
    path, _, _ = ukb_geo
    script = tmp_path / "volumes.py"
    script.write_text(
        f"""
import json
from pathlib import Path
from mpi4py import MPI
import dolfinx, numpy as np, ufl
from cardiac_geometries import aha, utils
geo = utils.gmsh2dolfin(comm=MPI.COMM_WORLD, msh_file=Path({str(path / "ED.msh")!r}))
landmarks = json.loads(Path({str(path / "landmarks_ED.json")!r}).read_text())
tags, _ = aha.biv_aha(geo.mesh, geo.ffun, geo.markers, landmarks)
dx = ufl.Measure("dx", domain=geo.mesh, subdomain_data=tags)
def volume(i):
    local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * dx(i, domain=geo.mesh)))
    return geo.mesh.comm.allreduce(local)
vols = [volume(i) for i in range(1, 20)]
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
    assert np.allclose(v1, v2, rtol=1e-12)
    assert min(v1) > 0


def test_ukb_cli_aha(tmp_path):
    runner = CliRunner()
    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    args = [str(path / "biv"), "--char-length-max", "5", "--char-length-min", "5", "--aha"]
    res = runner.invoke(cli.ukb, args)
    assert res.exit_code == 0, res.output
    geo = Geometry.from_folder(comm=MPI.COMM_WORLD, folder=path / "biv")
    values = np.unique(np.hstack(MPI.COMM_WORLD.allgather(geo.cfun.values)))
    assert values.tolist() == list(range(1, 20))

    res = runner.invoke(cli.ukb, [str(path / "lv"), "--aha", "--geometry", "lv"])
    assert res.exit_code != 0
    assert "biv" in str(res.exception)


def test_biv_aha_missing_landmarks(ukb_geo):
    _, geo, landmarks = ukb_geo
    with pytest.raises(ValueError, match="insertion"):
        aha.biv_aha(
            geo.mesh,
            geo.ffun,
            geo.markers,
            {k: v for k, v in landmarks.items() if k != "insertion"},
        )
    markers = {k: v for k, v in geo.markers.items() if k != "MV"}
    with pytest.raises(ValueError, match="MV"):
        aha.biv_aha(geo.mesh, geo.ffun, markers, landmarks)


def test_ukb_aha_needs_landmarks_file(tmp_path, monkeypatch):
    # A folder from ukb-atlas < 2.2 has no landmarks file
    import ukb.surface

    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    orig = ukb.surface.main

    def surf_without_landmarks(**kwargs):
        orig(**kwargs)
        if MPI.COMM_WORLD.rank == 0:
            (Path(kwargs["folder"]) / f"landmarks_{kwargs['case']}.json").unlink()

    monkeypatch.setattr(ukb.surface, "main", surf_without_landmarks)
    with pytest.raises(ValueError, match="landmarks_ED.json"):
        build(path)
