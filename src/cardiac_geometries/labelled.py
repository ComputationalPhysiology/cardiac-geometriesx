"""Facet labels for a tetrahedral volume mesh, from labelled boundary surfaces or a rule.

This is how a patient mesh from another tool gets the markers the rest of the package uses: each
boundary facet of the (distributed) volume mesh is matched, by its vertex coordinates, to the
triangles of surface meshes (one per marker name) or labelled by a rule, and every boundary facet
must get exactly one label.
"""

from collections.abc import Callable, Mapping

import numpy as np

# Marker names as in the UKB geometries; their facet ids are their position here plus one
MARKER_ORDER = ["LV_ENDO", "RV_ENDO", "EPI", "BASE", "MV", "AV", "TV", "PV"]
# Without a tolerance, coordinates match to this fraction of the size of the surfaces (round-off,
# e.g. single-precision PLY files against a double-precision volume)
ROUND_OFF = 1e-6

Rule = Callable[[np.ndarray, dict[str, np.ndarray]], np.ndarray]


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    """The faces of a tetrahedral mesh that belong to one tetrahedron, as sorted node triples."""
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.sort(
        np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]),
        axis=1,
    )
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return unique[counts == 1]


def marker_ids(names) -> dict[str, tuple[int, int]]:
    """Facet markers for the given names: those in :data:`MARKER_ORDER` keep their position there
    plus one as id, other names follow sorted; ``"Wall"`` is the cell marker."""
    markers = {name: (MARKER_ORDER.index(name) + 1, 2) for name in MARKER_ORDER if name in names}
    others = sorted(set(names) - set(MARKER_ORDER))
    markers |= {name: (len(MARKER_ORDER) + 1 + i, 2) for i, name in enumerate(others)}
    markers["Wall"] = (1, 3)
    return markers


def label_facets(
    coords: np.ndarray,
    surfaces: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    rule: Rule | None = None,
    data: Mapping[str, np.ndarray] | None = None,
    tol: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Label boundary facets given by their vertex coordinates.

    Parameters
    ----------
    coords : np.ndarray
        The vertex coordinates of the facets, shape (F, 3, 3)
    surfaces : Mapping[str, tuple[np.ndarray, np.ndarray]] | None
        Marker name to a surface mesh (points (K, 3), triangles (T, 3)). A facet gets the name of
        the surface that has a triangle with the same three vertices
    rule : Callable | None
        ``rule(coords, data)`` returns the marker name of each facet. Exactly one of ``surfaces``
        and ``rule`` must be given
    data : Mapping[str, np.ndarray] | None
        For ``rule``: named values at the facet vertices, each of shape (F, 3)
    tol : float | None
        The distance within which a facet vertex matches a surface point. By default, round-off:
        ``ROUND_OFF`` times the diagonal of the surfaces' bounding box

    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        The name of each facet ("" where no surface matched) and, per surface, the number of its
        triangles that matched a facet

    Raises
    ------
    ValueError
        If a facet matches triangles of two surfaces, or the rule returns the wrong number of names
    """
    if (surfaces is None) == (rule is None):
        raise ValueError("Give exactly one of surfaces and rule")
    coords = np.asarray(coords, dtype=float).reshape(-1, 3, 3)
    if rule is not None:
        names = np.asarray(rule(coords, dict(data or {})), dtype=object)
        if names.shape != (len(coords),):
            raise ValueError(f"The rule must return one name per facet ({len(coords)})")
        return np.where(names == None, "", names), {}  # noqa: E711

    from scipy.spatial import cKDTree

    assert surfaces is not None
    if tol is None:
        allpoints = np.vstack([np.asarray(p, dtype=float) for p, _ in surfaces.values()])
        size = (
            float(np.linalg.norm(allpoints.max(axis=0) - allpoints.min(axis=0)))
            if len(allpoints)
            else 1.0
        )
        tol = ROUND_OFF * size
    names = np.full(len(coords), "", dtype=object)
    matched: dict[str, int] = {}
    vertices = coords.reshape(-1, 3)
    for name, (points, triangles) in surfaces.items():
        points = np.asarray(points, dtype=float)
        triangles = np.sort(np.asarray(triangles, dtype=np.int64), axis=1)
        known = {tuple(t) for t in triangles.tolist()}
        if len(points) == 0 or len(coords) == 0:
            matched[name] = 0
            continue
        dist, nodes = cKDTree(points).query(vertices, distance_upper_bound=max(tol, 1e-300))
        close = (dist <= tol).reshape(-1, 3).all(axis=1)
        keys = np.sort(nodes.reshape(-1, 3), axis=1)
        hit = np.array([c and tuple(k) in known for c, k in zip(close, keys.tolist())], dtype=bool)
        double = int(np.sum(hit & (names != "")))
        if double:
            raise ValueError(f"Boundary facets labelled by 2 names or more: {double} ({name!r})")
        names[hit] = name
        matched[name] = int(np.sum(hit))
    return names, matched


def check_labels(
    names: np.ndarray, matched: Mapping[str, int], sizes: Mapping[str, int], comm=None
) -> None:
    """Every boundary facet labelled, and every surface triangle a boundary facet (globally, with
    ``comm``).

    Raises
    ------
    ValueError
        With the counts, if facets are unlabelled or surface triangles matched no boundary facet
    """

    def total(n):
        return comm.allreduce(int(n)) if comm is not None else int(n)

    unlabelled = total(np.sum(names == ""))
    facets = total(len(names))
    missing = {name: sizes[name] - total(matched.get(name, 0)) for name in sizes}
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        raise ValueError(
            f"Surface triangles that are not boundary faces: {sum(missing.values())} {missing} "
            "(interior faces, or points that do not match the volume: pass tol, or scale the "
            "surfaces to the units of the volume)"
        )
    if unlabelled:
        raise ValueError(f"Boundary facets unlabelled: {unlabelled} of {facets}")
