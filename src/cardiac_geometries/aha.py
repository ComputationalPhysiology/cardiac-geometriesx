from typing import Any

import dolfinx
import numpy as np


def focal(r_long_endo: float | np.ndarray, r_short_endo: float | np.ndarray) -> float | np.ndarray:
    """Calculate the focal distance for a prolate ellipsoid

    Parameters
    ----------
    r_long_endo : float | np.ndarray
        Radius of the long axis of the endocardium
    r_short_endo : float | np.ndarray
        Radius of the short axis of the endocardium"

    Returns
    -------
    float | np.ndarray
        The focal distance for the prolate ellipsoid
    """
    return np.sqrt(r_long_endo**2 - r_short_endo**2)


def full_arctangent(x, y) -> float | np.ndarray:
    """Compute the full arctangent in [0, 2pi]

    Parameters
    ----------
    x : float | np.ndarray
    y : float | np.ndarray

    Returns
    -------
    float | np.ndarray
        The angle in [0, 2pi]
    """
    t = np.arctan2(x, y)
    return np.where(t < 0, t + 2 * np.pi, t)


def cartesian_to_prolate_ellipsoidal(
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
    a: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
    """Convert Cartesian coordinates to prolate ellipsoidal coordinates

    Parameters
    ----------
    x : float | np.ndarray
    y : float | np.ndarray
    z : float | np.ndarray
    a : float | np.ndarray

    Returns
    -------
    tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]
        The prolate ellipsoidal coordinates (nu, mu, phi)
    """
    b1 = np.sqrt((x + a) ** 2 + y**2 + z**2)
    b2 = np.sqrt((x - a) ** 2 + y**2 + z**2)

    sigma = 1 / (2.0 * a) * (b1 + b2)
    tau = 1 / (2.0 * a) * (b1 - b2)
    phi = full_arctangent(z, y)
    nu = np.arccosh(sigma)
    mu = np.arccos(tau)
    return nu, mu, phi


def get_level(region: int, mu: np.ndarray | float, mu_base: float, dmu: float) -> np.ndarray | bool:
    """Get the level set for a given AHA region.

    Notes
    -----
    The regions are defined as follows:
    - Regions 1-6: Basal (mu_base <= mu <= mu_base + dmu)
    - Regions 7-12: Midventricular (mu_base + dmu < mu <= mu_base + 2 * dmu)
    - Regions 13-16: Apical (mu_base + 2 * dmu < mu <= mu_base + 3 * dmu)
    - Region 17: Apex (mu > mu_base + 3 * dmu)
    This function returns a boolean array indicating whether
    each mu value belongs to the specified region.

    Parameters
    ----------
    region : int
        The AHA region (1-17)
    mu : np.ndarray | float
        The mu coordinate(s)
    mu_base : float
        The base value of mu for segmentation
    dmu : float
        The segmentation width
    Returns
    -------
    np.ndarray | bool
        The level set for the given AHA region
    """
    if 1 <= region <= 6:
        return np.logical_and(mu_base <= mu, mu <= mu_base + dmu)
    elif 7 <= region <= 12:
        return np.logical_and(mu_base + dmu < mu, mu <= mu_base + 2 * dmu)
    elif 13 <= region <= 16:
        return np.logical_and(mu_base + 2 * dmu < mu, mu <= mu_base + 3 * dmu)
    else:
        return mu > mu_base + 3 * dmu


def get_sector(region: int, phi: np.ndarray | float) -> np.ndarray | bool:
    if region in (1, 7):
        return np.logical_and(0 < phi, phi <= np.pi / 3)
    elif region in (2, 8):
        return np.logical_and(np.pi / 3 < phi, phi <= 2 * np.pi / 3)
    elif region in (3, 9):
        return np.logical_and(2 * np.pi / 3 < phi, phi <= np.pi)
    elif region in (4, 10):
        return np.logical_and(np.pi < phi, phi <= 4 * np.pi / 3)
    elif region in (5, 11):
        return np.logical_and(4 * np.pi / 3 < phi, phi <= 5 * np.pi / 3)
    elif region in (6, 12):
        return np.logical_and(5 * np.pi / 3 < phi, phi <= 2 * np.pi)
    elif region == 13:
        return np.logical_and(0 < phi, phi <= np.pi / 2)
    elif region == 14:
        return np.logical_and(np.pi / 2 < phi, phi <= np.pi)
    elif region == 15:
        return np.logical_and(np.pi < phi, phi <= 3 * np.pi / 2)
    elif region == 16:
        return np.logical_and(3 * np.pi / 2 < phi, phi <= 2 * np.pi)
    else:
        # 17 APEX
        return np.ones_like(phi, dtype=bool)


def find_region_entities(
    mu: np.ndarray, phi: np.ndarray, mu_base: float, region: int, dmu: float
) -> np.ndarray:
    """Find the entities belonging to a given AHA region.

    Parameters
    ----------
    mu : np.ndarray
        The mu coordinate(s)
    phi : np.ndarray
        The phi coordinate(s)
    mu_base : float
        The base value of mu for segmentation
    region : int
        The AHA region (1-17)
    dmu : float
        The segmentation width

    Returns
    -------
    np.ndarray
        The indices of entities belonging to the specified AHA region
    """
    level = get_level(region=region, mu=mu, mu_base=mu_base, dmu=dmu)
    sector = get_sector(region=region, phi=phi)
    return np.where(np.logical_and(level, sector))[0]


def lv_aha(
    mesh: dolfinx.mesh.Mesh,
    r_long_endo: float,
    r_short_endo: float,
    mu_base: float,
    dmu_factor: float = 1 / 4,
) -> tuple[dolfinx.mesh.MeshTags, dict[str, tuple[int, int]]]:
    """Generate AHA segments for idealized LV ellipsoid

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The LV ellipsoidal mesh
    r_long_endo : float
        Radius of the long axis of the endocardium
    r_short_endo : float
        Radius of the short axis of the endocardium
    mu_base : float
        Base value of mu for segmentation
    dmu_factor : float, optional
        Factor to determine the segmentation width, by default 1/4

    Returns
    -------
    tuple[dolfinx.mesh.MeshTags, dict[str, tuple[int, int]]]
        A tuple with the MeshTags object containing the cell tags
        for the AHA segments and a dictionary with the marker
        names and corresponding (tag, dim) values
    """
    assert mesh.topology.dim == 3, "AHA segmentation only implemented for 3D geometries"
    foc = focal(r_long_endo=r_long_endo, r_short_endo=r_short_endo)
    mu_base = abs(mu_base)
    x, y, z = dolfinx.mesh.compute_midpoints(
        mesh,
        3,
        entities=np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32),
    ).T

    dmu = (np.pi - mu_base) * dmu_factor
    _, mu, phi = cartesian_to_prolate_ellipsoidal(x, y, z, a=foc)
    mu = np.asarray(mu)
    phi = np.asarray(phi)
    entities = [
        find_region_entities(mu=mu, phi=phi, mu_base=mu_base, region=region, dmu=dmu)
        for region in range(1, 18)
    ]

    values = [np.full(len(e), i + 1, dtype=np.int32) for i, e in enumerate(entities)]

    cell_tags = dolfinx.mesh.meshtags(
        mesh,
        3,
        np.hstack(entities),
        np.hstack(values),
    )

    markers = {}
    i = 1

    for level in ["BASAL", "MID", "APICAL"]:
        for sector in [
            "ANTERIOR",
            "ANTEROSEPTAL",
            "SEPTAL",
            "INFERIOR",
            "POSTERIOR",
            "LATERAL",
        ]:
            if level == "APICAL" and sector in ("ANTEROSEPTAL", "POSTERIOR"):
                continue

            markers["-".join((level, sector))] = (i, 3)
            i += 1

    markers["APEX"] = (17, 3)

    return cell_tags, markers


SECTOR_NAMES = ["ANTERIOR", "ANTEROSEPTAL", "SEPTAL", "INFERIOR", "POSTERIOR", "LATERAL"]
LEVEL_NAMES = ["BASAL", "MID", "APICAL"]
RV_TAG = 18
LVOT_TAG = 19
NUM_LEVELS = 200  # resolution of the insertion cuts along the long axis


def aha_markers() -> dict[str, tuple[int, int]]:
    """The names of the 17 AHA segments (as in :func:`lv_aha`), the RV free wall
    and the LV above the base."""
    markers = {}
    i = 1
    for level in LEVEL_NAMES:
        for sector in SECTOR_NAMES:
            if level == "APICAL" and sector in ("ANTEROSEPTAL", "POSTERIOR"):
                continue
            markers["-".join((level, sector))] = (i, 3)
            i += 1
    markers["APEX"] = (17, 3)
    markers["RV"] = (RV_TAG, 3)
    markers["LVOT"] = (LVOT_TAG, 3)
    return markers


def _nearest(
    points: np.ndarray, targets: np.ndarray, chunk: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to, and index of, the nearest of ``targets`` for each of ``points``."""
    dist = np.empty(len(points))
    index = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk):
        p = points[start : start + chunk]
        d2 = (
            np.sum(p**2, axis=1)[:, None]
            - 2.0 * p @ targets.T
            + np.sum(targets**2, axis=1)[None, :]
        )
        index[start : start + chunk] = np.argmin(d2, axis=1)
        dist[start : start + chunk] = np.sqrt(
            np.maximum(d2[np.arange(len(p)), index[start : start + chunk]], 0.0)
        )
    return dist, index


def _insertion_azimuths(
    insertion: np.ndarray,
    apex: np.ndarray,
    e_l: np.ndarray,
    length: float,
    azimuth: Any,
    anterior_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuths of the anterior and inferior insertion at ``NUM_LEVELS`` levels
    along the long axis, where the short-axis planes cut the insertion loop."""
    levels = (np.arange(NUM_LEVELS) + 0.5) / NUM_LEVELS
    s = (insertion - apex) @ e_l / length
    a, b = insertion, np.roll(insertion, -1, axis=0)
    sa, sb = s, np.roll(s, -1)
    anterior = np.full(NUM_LEVELS, np.nan)
    inferior = np.full(NUM_LEVELS, np.nan)
    for i, level in enumerate(levels):
        crossing = np.flatnonzero((sa - level) * (sb - level) <= 0.0)
        crossing = crossing[sa[crossing] != sb[crossing]]
        if len(crossing) != 2:
            continue
        t = (level - sa[crossing]) / (sb[crossing] - sa[crossing])
        cuts = a[crossing] + t[:, None] * (b[crossing] - a[crossing])
        first = np.argmin(np.linalg.norm(cuts - anterior_ref, axis=1))
        anterior[i] = azimuth(cuts[first])
        inferior[i] = azimuth(cuts[1 - first])
    valid = np.flatnonzero(~np.isnan(anterior))
    if len(valid) == 0:
        raise ValueError("No short-axis plane cuts the insertion loop twice")
    # Levels without two cuts (e.g. near the apex) take those of the nearest level with two
    fill = valid[np.argmin(np.abs(np.arange(NUM_LEVELS)[:, None] - valid[None, :]), axis=1)]
    return levels, anterior[fill], inferior[fill]


def aha_segments(
    x: np.ndarray,
    apex: np.ndarray,
    base_centre: np.ndarray,
    insertion: np.ndarray,
    anterior_ref: np.ndarray,
    septum_ref: np.ndarray | None = None,
) -> np.ndarray:
    """Label points of the LV wall with the 17 AHA segments of Cerqueira et al. (2002).

    The long axis runs from ``apex`` to ``base_centre``. With s the normalized
    position along it, points with s < 0 (beyond the apex) are the apex (17),
    s in [0, 1/3) apical, [1/3, 2/3) mid and [2/3, 1] basal; points with s > 1
    are above the base (19). In each short-axis plane the RV insertion loop is cut
    twice, at the anterior insertion (the cut nearer ``anterior_ref``) and the
    inferior insertion. In the basal and mid levels the septum between the
    insertions is split into two equal segments (anteroseptal and inferoseptal)
    and the rest of the wall into four (anterior, anterolateral, inferolateral and
    inferior). In the apical level the septal segment is the 90 degrees centred on
    the septum, followed by anterior, lateral and inferior segments of 90 degrees.

    Parameters
    ----------
    x : np.ndarray
        Points to label, shape (N, 3)
    apex : np.ndarray
        The (endocardial) apex
    base_centre : np.ndarray
        The centre of the base, e.g. of the mitral valve
    insertion : np.ndarray
        The RV insertion line as a closed polyline, shape (M, 3)
    anterior_ref : np.ndarray
        A point on the anterior side, e.g. the centre of the pulmonary valve
    septum_ref : np.ndarray | None
        A point on the septal side of the insertions, e.g. a point of the RV
        septum. If not given, the septum is the shorter arc between the insertions.

    Returns
    -------
    np.ndarray
        The segment of each point (1-17, or 19 above the base)
    """
    x = np.asarray(x, dtype=float).reshape(-1, 3)
    apex = np.asarray(apex, dtype=float)
    e_l = np.asarray(base_centre, dtype=float) - apex
    length = float(np.linalg.norm(e_l))
    e_l /= length
    u = np.asarray(anterior_ref, dtype=float) - apex
    u -= (u @ e_l) * e_l
    u /= np.linalg.norm(u)
    v = np.cross(e_l, u)

    def azimuth(p: np.ndarray) -> np.ndarray:
        q = np.asarray(p) - apex
        return np.degrees(np.arctan2(q @ v, q @ u))

    levels, anterior, inferior = _insertion_azimuths(
        np.asarray(insertion, dtype=float), apex, e_l, length, azimuth, np.asarray(anterior_ref)
    )
    s = (x - apex) @ e_l / length
    level = np.clip(np.floor(s * NUM_LEVELS).astype(int), 0, NUM_LEVELS - 1)
    a, i = anterior[level], inferior[level]

    # psi: angle from the anterior insertion, increasing towards the septum
    arc = np.mod(i - a, 360.0)  # from the anterior to the inferior insertion, counter-clockwise
    if septum_ref is not None:
        ref_level = np.clip(
            int(np.floor(((septum_ref - apex) @ e_l / length) * NUM_LEVELS)), 0, NUM_LEVELS - 1
        )
        ccw_contains = np.mod(azimuth(septum_ref) - anterior[ref_level], 360.0) < np.mod(
            inferior[ref_level] - anterior[ref_level], 360.0
        )
        sign = np.where(ccw_contains, 1.0, -1.0) * np.ones_like(arc)
    else:
        sign = np.where(arc <= 180.0, 1.0, -1.0)
    width = np.where(sign > 0, arc, 360.0 - arc)  # of the septum
    psi = np.mod(sign * (azimuth(x) - a), 360.0)

    # Basal/mid sector index in the order anteroseptal, inferoseptal, inferior,
    # inferolateral, anterolateral, anterior
    free = (psi - width) / ((360.0 - width) / 4.0)
    sector = np.where(
        psi < width / 2, 0, np.where(psi < width, 1, 2 + np.clip(free.astype(int), 0, 3))
    )
    basal = np.array([2, 3, 4, 5, 6, 1])[sector]
    # Apical: septal (14), inferior (15), lateral (16) and anterior (13), in 90 degree
    # steps from the septum
    apical_sector = np.mod(np.floor((psi - width / 2 + 45.0) / 90.0).astype(int), 4)
    apical = np.array([14, 15, 16, 13])[apical_sector]

    labels = np.where(s < 1 / 3, apical, np.where(s < 2 / 3, basal + 6, basal))
    labels = np.where(s < 0, 17, labels)
    labels = np.where(s > 1, LVOT_TAG, labels)
    return labels.astype(np.int32)


def _gather_facet_midpoints(
    mesh: dolfinx.mesh.Mesh, ffun: dolfinx.mesh.MeshTags, marker: int
) -> np.ndarray:
    facets = ffun.find(marker)
    num_owned = mesh.topology.index_map(mesh.topology.dim - 1).size_local
    facets = facets[facets < num_owned]
    points = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim - 1, facets)
    return np.vstack(mesh.comm.allgather(points))


def _marker(markers: dict[str, Any], name: str) -> int:
    value = markers[name]
    return int(value[0]) if isinstance(value, (tuple, list)) else int(value)


def lv_long_axis(
    mesh: dolfinx.mesh.Mesh, ffun: dolfinx.mesh.MeshTags, markers: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """The LV long axis of a UKB geometry, the same on every process.

    Returns the apex, the point of the LV endocardium farthest from the centre of
    the mitral valve (as in :func:`ldrb.doste.find_apex`), and that centre, the
    mean of the midpoints of the mitral valve facets.
    """
    missing = [name for name in ("LV_ENDO", "MV") if name not in markers]
    if missing:
        raise ValueError(f"Missing markers {missing}")
    mv = _gather_facet_midpoints(mesh, ffun, _marker(markers, "MV"))
    endo = _gather_facet_midpoints(mesh, ffun, _marker(markers, "LV_ENDO"))
    if len(mv) == 0 or len(endo) == 0:
        raise ValueError("No facets marked as MV or LV_ENDO")
    mv_centre = mv.mean(axis=0)
    apex = endo[np.argmax(np.linalg.norm(endo - mv_centre, axis=1))]
    return apex, mv_centre


def biv_aha(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    markers: dict[str, Any],
    landmarks: dict[str, Any],
) -> tuple[dolfinx.mesh.MeshTags, dict[str, tuple[int, int]]]:
    """AHA segments of a UKB biventricular geometry (Cerqueira et al. 2002).

    Every cell gets one tag: the 17 AHA segments of the LV (see
    :func:`aha_segments`, with the names of :func:`lv_aha`), 18 for the RV free
    wall and 19 for the LV above the mitral plane (around the LV outflow tract).
    The septum belongs to the LV. A cell is in the RV free wall if its nearest RV
    endocardial landmark node is on the free wall and it is nearer to that node
    than to the LV endocardium.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    ffun : dolfinx.mesh.MeshTags
        The facet tags, with the markers ``LV_ENDO`` and ``MV``
    markers : dict[str, Any]
        The markers of the geometry
    landmarks : dict[str, Any]
        The landmarks written by ``ukb-atlas surf`` (``landmarks_<case>.json``,
        ukb-atlas >= 2.2): ``insertion``, ``rv_freewall``, ``rv_septum`` and
        ``valves``

    Returns
    -------
    tuple[dolfinx.mesh.MeshTags, dict[str, tuple[int, int]]]
        The cell tags and their markers
    """
    missing = [k for k in ("insertion", "rv_freewall", "rv_septum", "valves") if k not in landmarks]
    if missing:
        raise ValueError(f"Missing landmarks {missing}")
    apex, mv_centre = lv_long_axis(mesh, ffun, markers)
    ncells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(ncells, dtype=np.int32)
    x = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, cells)

    freewall = np.asarray(landmarks["rv_freewall"], dtype=float)
    septum = np.asarray(landmarks["rv_septum"], dtype=float)
    rv_nodes = np.vstack([freewall, septum])
    d_rv, index = _nearest(x, rv_nodes)
    d_lv, _ = _nearest(x, _gather_facet_midpoints(mesh, ffun, _marker(markers, "LV_ENDO")))
    rv = (index < len(freewall)) & (d_rv < d_lv)

    values = np.full(ncells, RV_TAG, dtype=np.int32)
    values[~rv] = aha_segments(
        x[~rv],
        apex,
        mv_centre,
        np.asarray(landmarks["insertion"], dtype=float),
        np.asarray(landmarks["valves"]["PV"], dtype=float),
        septum_ref=septum.mean(axis=0),
    )
    tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, values)
    return tags, aha_markers()
