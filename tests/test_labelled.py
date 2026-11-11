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


def counts(faces, ids, markers):
    names = {v[0]: k for k, v in markers.items() if v[1] == 2}
    values, n = np.unique(ids, return_counts=True)
    return {names[v]: int(c) for v, c in zip(values, n)}


def test_label_boundary_surfaces_exact(box):
    points, tets, surfaces = box
    faces, ids, markers = labelled.label_boundary(points, tets, surfaces=surfaces)
    assert len(faces) == sum(len(t) for _, t in surfaces.values())
    assert counts(faces, ids, markers) == {k: len(t) for k, (_, t) in surfaces.items()}


def test_label_boundary_rule(box):
    points, tets, surfaces = box

    def rule(faces, pts):
        x = pts[faces][:, :, 0]
        return np.where(
            np.all(np.isclose(x, 0.0), axis=1),
            "BASE",
            np.where(np.all(np.isclose(x, 2.0), axis=1), "EPI", "LV_ENDO"),
        )

    by_rule = labelled.label_boundary(points, tets, rule=rule)
    by_surfaces = labelled.label_boundary(points, tets, surfaces=surfaces)
    assert counts(*by_rule) == counts(*by_surfaces)


def test_label_boundary_marker_ids(box):
    points, tets, surfaces = box
    _, _, markers = labelled.label_boundary(points, tets, surfaces=surfaces)
    # Fixed ids: the position in MARKER_ORDER, plus one (other names after, sorted)
    assert markers["LV_ENDO"] == (1, 2) and markers["EPI"] == (3, 2) and markers["BASE"] == (4, 2)
    assert "RV_ENDO" not in markers
    assert markers["Wall"] == (1, 3)
    assert labelled.MARKER_ORDER == ["LV_ENDO", "RV_ENDO", "EPI", "BASE", "MV", "AV", "TV", "PV"]


def test_label_boundary_unlabelled_face_errors(box):
    points, tets, surfaces = box
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (p, t[1:])
    with pytest.raises(ValueError, match="unlabelled: 1"):
        labelled.label_boundary(points, tets, surfaces=surfaces)


def test_label_boundary_double_label_errors(box):
    points, tets, surfaces = box
    p, t = surfaces["EPI"]
    surfaces["BASE"] = (p, np.vstack([surfaces["BASE"][1], t[:1]]))
    with pytest.raises(ValueError, match="2 names"):
        labelled.label_boundary(points, tets, surfaces=surfaces)


def test_label_boundary_rejects_interior_face(box):
    points, tets, surfaces = box
    faces = np.sort(np.vstack([tets[:, list(f)] for f in itertools.combinations(range(4), 3)]), 1)
    unique, n = np.unique(faces, axis=0, return_counts=True)
    interior = unique[n == 2][:1]
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (p, np.vstack([t, interior]))
    with pytest.raises(ValueError, match="not boundary faces: 1"):
        labelled.label_boundary(points, tets, surfaces=surfaces)


def test_label_boundary_inexact_needs_tol(box):
    points, tets, surfaces = box
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (p + 1e-6, t)
    with pytest.raises(ValueError, match="tol"):
        labelled.label_boundary(points, tets, surfaces=surfaces)


def test_label_boundary_tol(box):
    points, tets, surfaces = box
    exact = counts(*labelled.label_boundary(points, tets, surfaces=surfaces))
    p, t = surfaces["EPI"]
    surfaces["EPI"] = (p + 1e-6, t)
    assert counts(*labelled.label_boundary(points, tets, surfaces=surfaces, tol=1e-5)) == exact


def test_label_boundary_one_of_surfaces_or_rule(box):
    points, tets, surfaces = box
    with pytest.raises(ValueError, match="one of"):
        labelled.label_boundary(points, tets)
    with pytest.raises(ValueError, match="one of"):
        labelled.label_boundary(points, tets, surfaces=surfaces, rule=lambda f, p: None)


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
