"""Facet labels for a tetrahedral volume mesh, from labelled boundary surfaces or a rule.

This is how a patient mesh from another tool gets the markers the rest of the package uses: the
boundary faces of the volume are matched to surface meshes (one per marker name) or labelled by a
rule, and every boundary face must get exactly one label.
"""

from collections.abc import Callable, Mapping

import numpy as np

# Marker names as in the UKB geometries; their facet ids are their position here plus one
MARKER_ORDER = ["LV_ENDO", "RV_ENDO", "EPI", "BASE", "MV", "AV", "TV", "PV"]

Rule = Callable[[np.ndarray, np.ndarray], np.ndarray]


def boundary_faces(tets: np.ndarray) -> np.ndarray:
    """The faces of a tetrahedral mesh that belong to one tetrahedron, as sorted node triples."""
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.sort(np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]],
                               tets[:, [1, 2, 3]]]), axis=1)  # fmt: skip
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    return unique[counts == 1]


def _marker_ids(names) -> dict[str, tuple[int, int]]:
    markers = {name: (MARKER_ORDER.index(name) + 1, 2) for name in MARKER_ORDER if name in names}
    others = sorted(set(names) - set(MARKER_ORDER))
    markers |= {name: (len(MARKER_ORDER) + 1 + i, 2) for i, name in enumerate(others)}
    markers["Wall"] = (1, 3)
    return markers


def _surface_labels(
    points: np.ndarray,
    faces: np.ndarray,
    surfaces: Mapping[str, tuple[np.ndarray, np.ndarray]],
    tol: float | None,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    index = {tuple(f): i for i, f in enumerate(faces.tolist())}
    labels = np.full(len(faces), "", dtype=object)
    hits = np.zeros(len(faces), dtype=int)
    limit = 0.0 if tol is None else tol
    for name, (spoints, triangles) in surfaces.items():
        dist, nodes = tree.query(np.asarray(spoints, dtype=float))
        far = int(np.sum(dist > limit))
        if far:
            hint = "exactly" if tol is None else f"within tol={tol}"
            raise ValueError(
                f"Surface {name!r}: {far} points do not coincide {hint} with a node of the volume "
                "mesh; pass tol (or scale the surfaces to the units of the volume)"
            )
        tri = np.sort(nodes[np.asarray(triangles, dtype=np.int64)], axis=1)
        found = [index.get(tuple(t)) for t in tri.tolist()]
        missing = sum(i is None for i in found)
        if missing:
            raise ValueError(
                f"Surface {name!r}: triangles that are not boundary faces: {missing} "
                "(interior faces of the volume, or no face at all)"
            )
        for i in set(found):
            hits[i] += 1
            labels[i] = name
    double = int(np.sum(hits > 1))
    if double:
        raise ValueError(f"Boundary faces labelled by 2 names or more: {double}")
    return labels


def label_boundary(
    points: np.ndarray,
    tets: np.ndarray,
    surfaces: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    rule: Rule | None = None,
    tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    """Label every boundary face of a tetrahedral mesh.

    Parameters
    ----------
    points : np.ndarray
        The nodes, shape (N, 3)
    tets : np.ndarray
        The tetrahedra, shape (M, 4)
    surfaces : Mapping[str, tuple[np.ndarray, np.ndarray]] | None
        Marker name to a surface mesh (points (K, 3), triangles (T, 3)). Each surface node must
        coincide with a node of the volume (exactly, or within ``tol``), and each triangle must be a
        boundary face of the volume
    rule : Callable | None
        ``rule(faces, points)`` returns the marker name of each boundary face (faces as (F, 3) node
        indices). Exactly one of ``surfaces`` and ``rule`` must be given
    tol : float | None
        The distance within which a surface node matches a volume node. By default they must
        coincide exactly

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]
        The boundary faces (F, 3, sorted node indices), their marker ids (F,) and the markers: the
        names in :data:`MARKER_ORDER` keep their position there plus one as id, other names follow
        sorted; ``"Wall"`` is the cell marker

    Raises
    ------
    ValueError
        If a boundary face is left unlabelled or labelled twice, if a surface triangle is not a
        boundary face, or if a surface node matches no volume node
    """
    if (surfaces is None) == (rule is None):
        raise ValueError("Give exactly one of surfaces and rule")
    points = np.asarray(points, dtype=float)
    faces = boundary_faces(tets)
    if surfaces is not None:
        labels = _surface_labels(points, faces, surfaces, tol)
    else:
        assert rule is not None
        labels = np.asarray(rule(faces, points), dtype=object)
        if labels.shape != (len(faces),):
            raise ValueError(f"The rule must return one name per face ({len(faces)})")
        labels = np.where(labels == None, "", labels)  # noqa: E711
    unlabelled = int(np.sum(labels == ""))
    if unlabelled:
        raise ValueError(f"Boundary faces unlabelled: {unlabelled} of {len(faces)}")
    markers = _marker_ids(set(labels.tolist()))
    ids = np.array([markers[name][0] for name in labels], dtype=np.int32)
    return faces, ids, markers
