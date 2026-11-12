"""Patient meshes from a tetrahedral volume mesh and labelled boundary surfaces.

The labelling core (:func:`labelled.label_boundary`) is checked on a structured box whose faces are
known; the mesh-level function (:func:`mesh.from_volume`) on a thick spherical shell made with gmsh.
"""

import itertools

import numpy as np
import pytest

from cardiac_geometries import labelled
from cardiac_geometries import mesh as cgmesh

# Kuhn split of the unit cube into 6 tetrahedra along the main diagonal (conforming across cubes)
KUHN = [
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
]


def box_mesh(n=(4, 2, 2), size=(2.0, 1.0, 1.0)):
    """Structured tetrahedral mesh of [0, size] with n cells per direction."""
    nx, ny, nz = n
    xs = [np.linspace(0, s, k + 1) for s, k in zip(size, n)]
    points = np.array(list(itertools.product(*xs)))

    def node(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    tets = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        corners = [node(i + a, j + b, k + c) for a, b, c in itertools.product((0, 1), repeat=3)]
        tets += [[corners[v] for v in t] for t in KUHN]
    return points, np.array(tets)


def boundary_triangles(points, tets):
    faces = np.sort(np.vstack([tets[:, list(f)] for f in itertools.combinations(range(4), 3)]), 1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return unique[counts == 1]


@pytest.fixture
def box():
    points, tets = box_mesh()
    tri = boundary_triangles(points, tets)
    x = points[tri][:, :, 0]
    base = np.all(np.isclose(x, 0.0), axis=1)
    epi = np.all(np.isclose(x, 2.0), axis=1)
    surfaces = {
        "BASE": (points, tri[base]),
        "EPI": (points, tri[epi]),
        "LV_ENDO": (points, tri[~base & ~epi]),
    }
    return points, tets, surfaces


def facet_coords(points, tets):
    return points[labelled.boundary_faces(tets)]


def label_counts(names):
    values, n = np.unique(names, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, n)}


def test_label_facets_surfaces_exact(box):
    points, tets, surfaces = box
    names, matched = labelled.label_facets(facet_coords(points, tets), surfaces=surfaces)
    assert label_counts(names) == {k: len(t) for k, (_, t) in surfaces.items()}
    assert matched == {k: len(t) for k, (_, t) in surfaces.items()}


def test_label_facets_rule_with_data(box):
    points, tets, surfaces = box
    coords = facet_coords(points, tets)
    # The rule sees each facet's vertex coordinates and the values of named point data there
    data = {"x": coords[:, :, 0]}

    def rule(c, d):
        x = d["x"]
        return np.where(
            np.all(np.isclose(x, 0.0), axis=1),
            "BASE",
            np.where(np.all(np.isclose(x, 2.0), axis=1), "EPI", "LV_ENDO"),
        )

    by_rule, _ = labelled.label_facets(coords, rule=rule, data=data)
    by_surfaces, _ = labelled.label_facets(coords, surfaces=surfaces)
    assert np.array_equal(by_rule, by_surfaces)


def test_marker_ids():
    markers = labelled.marker_ids({"EPI", "BASE", "LV_ENDO"})
    # Fixed ids: the position in MARKER_ORDER, plus one (other names after, sorted)
    assert markers["LV_ENDO"] == (1, 2) and markers["EPI"] == (3, 2) and markers["BASE"] == (4, 2)
    assert "RV_ENDO" not in markers and markers["Wall"] == (1, 3)
    assert labelled.marker_ids({"EPI", "ZZ", "AA"})["AA"] == (9, 2)
    assert labelled.MARKER_ORDER == ["LV_ENDO", "RV_ENDO", "EPI", "BASE", "MV", "AV", "TV", "PV"]


def test_check_labels_unlabelled(box):
    points, tets, surfaces = box
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (p, t[1:])
    names, matched = labelled.label_facets(facet_coords(points, tets), surfaces=surfaces)
    sizes = {k: len(t) for k, (_, t) in surfaces.items()}
    with pytest.raises(ValueError, match="unlabelled: 1"):
        labelled.check_labels(names, matched, sizes)


def test_label_facets_double_label_errors(box):
    points, tets, surfaces = box
    p, t = surfaces["EPI"]
    surfaces["BASE"] = (p, np.vstack([surfaces["BASE"][1], t[:1]]))
    with pytest.raises(ValueError, match="2 names"):
        labelled.label_facets(facet_coords(points, tets), surfaces=surfaces)


def test_check_labels_rejects_interior_face(box):
    points, tets, surfaces = box
    faces = np.sort(np.vstack([tets[:, list(f)] for f in itertools.combinations(range(4), 3)]), 1)
    unique, n = np.unique(faces, axis=0, return_counts=True)
    interior = unique[n == 2][:1]
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (p, np.vstack([t, interior]))
    names, matched = labelled.label_facets(facet_coords(points, tets), surfaces=surfaces)
    # The interior triangle is never tagged: no boundary facet has its vertices
    assert label_counts(names)["EPI"] == len(t)
    sizes = {k: len(t) for k, (_, t) in surfaces.items()}
    with pytest.raises(ValueError, match="not boundary faces: 1"):
        labelled.check_labels(names, matched, sizes)


def test_label_facets_inexact_needs_tol(box):
    points, tets, surfaces = box
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (
        p + 1e-3,
        t,
    )  # far beyond round-off (the default matches to 1e-6 of the size)
    names, matched = labelled.label_facets(facet_coords(points, tets), surfaces=surfaces)
    sizes = {k: len(t) for k, (_, t) in surfaces.items()}
    with pytest.raises(ValueError, match="tol"):
        labelled.check_labels(names, matched, sizes)


def test_label_facets_tol(box):
    points, tets, surfaces = box
    exact, _ = labelled.label_facets(facet_coords(points, tets), surfaces=surfaces)
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (p + 1e-3, t)
    names, _ = labelled.label_facets(facet_coords(points, tets), surfaces=surfaces, tol=1e-2)
    assert np.array_equal(names, exact)


def test_label_facets_one_of_surfaces_or_rule(box):
    points, tets, surfaces = box
    coords = facet_coords(points, tets)
    with pytest.raises(ValueError, match="one of"):
        labelled.label_facets(coords)
    with pytest.raises(ValueError, match="one of"):
        labelled.label_facets(coords, surfaces=surfaces, rule=lambda c, d: None)


def test_transform_markers_base_and_valves():
    markers = {
        "LV_ENDO": (1, 2),
        "RV_ENDO": (2, 2),
        "EPI": (3, 2),
        "BASE": (4, 2),
        "MV": (5, 2),
        "AV": (6, 2),
        "TV": (7, 2),
        "PV": (8, 2),
    }
    assert cgmesh.transform_markers(markers)["base"] == [4, 5, 6, 7, 8]


@pytest.mark.parametrize(
    "markers, base",
    [
        ({"LV_ENDO": (1, 2), "EPI": (3, 2), "MV": (5, 2), "AV": (6, 2)}, [5, 6]),
        ({"LV_ENDO": (1, 2), "EPI": (3, 2), "BASE": (4, 2)}, [4]),
    ],
)
def test_transform_markers_ukb_unchanged(markers, base):
    assert cgmesh.transform_markers(markers)["base"] == base


# ---------------------------------------------------------------------------
# mesh.from_volume on a thick spherical shell (radii 1 and 2) cut at z = 0.5 (open at the base)
# ---------------------------------------------------------------------------
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from mpi4py import MPI  # noqa: E402

from click.testing import CliRunner  # noqa: E402

from cardiac_geometries import cli  # noqa: E402
from cardiac_geometries.geometry import Geometry  # noqa: E402


def make_shell(path):
    """A tetrahedral shell written as volume.vtu and one .ply per surface (rank 0), with pyvista."""
    import gmsh
    import pyvista

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("shell")
    outer = gmsh.model.occ.addSphere(0, 0, 0, 2.0)
    inner = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    shell, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    top = gmsh.model.occ.addBox(-3, -3, 0.5, 6, 6, 3)
    gmsh.model.occ.cut(shell, [(3, top)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.35)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    points = coords.reshape(-1, 3)
    index = np.empty(int(tags.max()) + 1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    _, nodes = gmsh.model.mesh.getElementsByType(4)
    gmsh.finalize()
    tets = index[nodes.astype(np.int64)].reshape(-1, 4)
    grid = pyvista.UnstructuredGrid({pyvista.CellType.TETRA: tets}, points)
    grid.point_data["height"] = points[:, 2]
    grid.save(path / "volume.vtu")
    faces = labelled.boundary_faces(tets)
    centre = points[faces].mean(axis=1)
    radius = np.linalg.norm(centre, axis=1)
    names = np.where(
        np.isclose(centre[:, 2], 0.5, atol=1e-8),
        "BASE",
        np.where(radius < 1.5, "LV_ENDO", "EPI"),
    )
    files = {}
    for name in ("LV_ENDO", "EPI", "BASE"):
        files[name] = path / f"{name.lower()}.ply"
        pyvista.PolyData.from_regular_faces(points, faces[names == name]).save(files[name])
    return path / "volume.vtu", files, {n: int(np.sum(names == n)) for n in files}


@pytest.fixture(scope="module")
def shell(tmp_path_factory):
    path = MPI.COMM_WORLD.bcast(tmp_path_factory.mktemp("shell"), root=0)
    result = make_shell(path) if MPI.COMM_WORLD.rank == 0 else None
    return MPI.COMM_WORLD.bcast(result, root=0)


def facet_counts(geo):
    out = {}
    for name, (marker, dim) in geo.markers.items():
        if dim != 2:
            continue
        owned = (
            geo.ffun.indices[geo.ffun.values == marker] < geo.mesh.topology.index_map(2).size_local
        )
        local = int(np.sum(owned))
        out[name] = geo.mesh.comm.allreduce(local)
    return out


def test_from_volume_markers(shell, tmp_path):
    volume, surfaces, expected = shell
    geo = cgmesh.from_volume(MPI.COMM_WORLD.bcast(tmp_path, root=0), volume, surfaces=surfaces)
    assert facet_counts(geo) == expected
    assert geo.markers["LV_ENDO"] == (1, 2) and geo.markers["EPI"] == (3, 2)
    assert geo.markers["BASE"] == (4, 2)


def test_from_volume_rule_data(shell, tmp_path):
    # The rule reads the point data "height" (z) through io4dolfinx and labels by it, and by radius
    volume, _, expected = shell

    def rule(coords, data):
        z = data["height"]
        radius = np.linalg.norm(coords.mean(axis=1), axis=1)
        base = np.all(np.isclose(z, 0.5, atol=1e-8), axis=1)
        return np.where(base, "BASE", np.where(radius < 1.5, "LV_ENDO", "EPI"))

    geo = cgmesh.from_volume(
        MPI.COMM_WORLD.bcast(tmp_path, root=0), volume, rule=rule, rule_data=["height"]
    )
    assert facet_counts(geo) == expected


def test_from_volume_scale(shell, tmp_path):
    volume, surfaces, _ = shell
    geo = cgmesh.from_volume(
        MPI.COMM_WORLD.bcast(tmp_path, root=0), volume, surfaces=surfaces, scale=1e-3
    )
    x = geo.mesh.geometry.x
    top = geo.mesh.comm.allreduce(float(x[:, 2].max()) if len(x) else -np.inf, op=MPI.MAX)
    assert top == pytest.approx(0.5e-3)


def test_from_volume_fibers_bayer(shell, tmp_path):
    volume, surfaces, _ = shell
    geo = cgmesh.from_volume(
        MPI.COMM_WORLD.bcast(tmp_path, root=0), volume, surfaces=surfaces, create_fibers=True
    )
    f = geo.f0.x.array.reshape(-1, 3)
    assert np.all(np.isfinite(f))
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-8)


def test_from_volume_doste_needs_valves(shell, tmp_path):
    volume, surfaces, _ = shell
    with pytest.raises(ValueError, match="MV"):
        cgmesh.from_volume(
            MPI.COMM_WORLD.bcast(tmp_path, root=0),
            volume,
            surfaces=surfaces,
            create_fibers=True,
            fiber_method="doste",
        )


def test_from_volume_roundtrip(shell, tmp_path):
    volume, surfaces, expected = shell
    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    geo = cgmesh.from_volume(path, volume, surfaces=surfaces, create_fibers=True)
    loaded = Geometry.from_folder(comm=MPI.COMM_WORLD, folder=path)
    assert {k: tuple(v) for k, v in loaded.markers.items()} == {
        k: tuple(v) for k, v in geo.markers.items()
    }
    assert facet_counts(loaded) == expected
    assert loaded.f0 is not None
    info = json.loads((path / "info.json").read_text())
    assert info["mesh_type"] == "from_volume" and info["scale"] == 1.0


def test_from_volume_mpi(shell, tmp_path):
    if MPI.COMM_WORLD.size > 1:
        pytest.skip("launches its own MPI runs")
    volume, surfaces, expected = shell
    script = tmp_path / "counts.py"
    script.write_text(
        f"""
import json, sys
from pathlib import Path
from mpi4py import MPI
import numpy as np
from cardiac_geometries import mesh
out = Path({str(tmp_path)!r}) / f"run{{MPI.COMM_WORLD.size}}"
surfaces = {{k: Path(v) for k, v in {json.dumps({k: str(v) for k, v in surfaces.items()})}.items()}}
geo = mesh.from_volume(out, Path({str(volume)!r}), surfaces=surfaces)
n = geo.mesh.topology.index_map(2).size_local
counts = {{name: geo.mesh.comm.allreduce(int(np.sum(geo.ffun.indices[geo.ffun.values == m] < n)))
          for name, (m, d) in geo.markers.items() if d == 2}}
if MPI.COMM_WORLD.rank == 0:
    (Path({str(tmp_path)!r}) / f"counts{{MPI.COMM_WORLD.size}}.json").write_text(json.dumps(counts))
"""
    )
    for n in (1, 2):
        r = subprocess.run(
            ["mpirun", "-n", str(n), sys.executable, str(script)], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-2000:]
    c1 = json.loads((tmp_path / "counts1.json").read_text())
    c2 = json.loads((tmp_path / "counts2.json").read_text())
    assert c1 == c2 == expected


def test_cli_from_volume(shell, tmp_path):
    volume, surfaces, expected = shell
    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    args = [str(path / "out"), str(volume)]
    for name, file in surfaces.items():
        args += ["--surface", f"{name}={file}"]
    res = CliRunner().invoke(cli.app, ["from-volume", *args])
    assert res.exit_code == 0, res.output
    geo = Geometry.from_folder(comm=MPI.COMM_WORLD, folder=path / "out")
    assert facet_counts(geo) == expected


def test_from_volume_needs_pyvista_message(shell, tmp_path, monkeypatch):
    volume, surfaces, _ = shell
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(ImportError, match="cardiac-geometriesx\\[pyvista\\]"):
        cgmesh.from_volume(MPI.COMM_WORLD.bcast(tmp_path, root=0), volume, surfaces=surfaces)


def test_from_volume_single_precision_volume(shell, tmp_path):
    # A volume stored in single precision (e.g. legacy VTK in float32) still gives a
    # double-precision mesh, which the fiber methods need
    import pyvista

    volume, surfaces, expected = shell
    path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    single = path / "volume32.vtu"
    if MPI.COMM_WORLD.rank == 0:
        grid = pyvista.read(volume)
        grid.points = grid.points.astype(np.float32)
        grid.save(single)
    MPI.COMM_WORLD.barrier()
    geo = cgmesh.from_volume(path / "out", single, surfaces=surfaces, create_fibers=True)
    assert geo.mesh.geometry.x.dtype == np.float64
    assert facet_counts(geo) == expected
    assert np.allclose(np.linalg.norm(geo.f0.x.array.reshape(-1, 3), axis=1), 1.0)
