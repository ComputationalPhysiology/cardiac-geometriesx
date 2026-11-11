from typing import Any

import dolfinx
import numpy as np
import ufl


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
        # The expanded form loses digits (identical points come out ~1e-5 apart): recompute the
        # distance
        dist[start : start + chunk] = np.linalg.norm(
            p - targets[index[start : start + chunk]], axis=1
        )
    return dist, index


def _crossings(s: np.ndarray, level: float) -> list[tuple[int, float]]:
    """Segments (i, i + 1) of the closed polyline with heights `s` that the plane s = level cuts,
    with the position t in [0, 1] along each.
    """
    sa, sb = s, np.roll(s, -1)
    out = []
    for i in np.flatnonzero((sa - level) * (sb - level) <= 0.0):
        if sa[i] != sb[i]:
            out.append((int(i), float((level - sa[i]) / (sb[i] - sa[i]))))
    return out


def _follow_branch(
    insertion: np.ndarray, s: np.ndarray, e_l: np.ndarray, segment: int, t: float, steepness: float
) -> tuple[np.ndarray, np.ndarray]:
    """Follow the insertion loop from the cut (segment, t) upwards and downwards for as long as it
    rises (or falls) more steeply than `steepness` (the cosine of the largest angle to the long
    axis).

    Returns the heights and points of this section of the branch, ordered by height.
    """
    n = len(insertion)
    cut = insertion[segment] + t * (insertion[(segment + 1) % n] - insertion[segment])
    s_cut = s[segment] + t * (s[(segment + 1) % n] - s[segment])
    sections = []
    for sign in (1.0, -1.0):  # upwards, then downwards
        # Along the loop, the direction of the segment's end that lies higher (or lower)
        forward = (s[(segment + 1) % n] - s[segment]) * sign > 0
        node = (segment + 1) % n if forward else segment
        step = 1 if forward else -1
        prev, prev_s, points = cut, s_cut, []
        for _ in range(n):
            d = insertion[node] - prev
            rise = (s[node] - prev_s) * sign
            if rise <= 0 or abs(d @ e_l) < steepness * np.linalg.norm(d):
                break
            points.append((s[node], insertion[node]))
            prev, prev_s = insertion[node], s[node]
            node = (node + step) % n
        sections.append(points)
    ordered = sections[1][::-1] + [(s_cut, cut)] + sections[0]
    return np.array([h for h, _ in ordered]), np.array([x for _, x in ordered])


def _insertion_azimuths(
    insertion: np.ndarray,
    apex: np.ndarray,
    e_l: np.ndarray,
    length: float,
    azimuth: Any,
    anterior_ref: np.ndarray,
    steepness: float = np.cos(np.pi / 4),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuths of the anterior and inferior insertion at ``NUM_LEVELS`` levels along the long axis.

    The insertion loop of the template has two near-vertical branches, the anterior and the inferior
    insertion, joined by runs across the top of the septum (below the base) and near the apex. The
    branches are found from the two cuts of the level nearest mid-ventricle that cuts the loop
    twice, and followed along the loop while they run within 45 degrees of the long axis. Above and
    below that section each insertion keeps the azimuth of its end, so levels that cut the runs, or
    nothing, are not misread.
    """
    levels = (np.arange(NUM_LEVELS) + 0.5) / NUM_LEVELS
    s = (insertion - apex) @ e_l / length
    candidates = sorted(levels, key=lambda level: abs(level - 0.5))
    cuts = next((c for c in (_crossings(s, level) for level in candidates) if len(c) == 2), None)
    if cuts is None:
        raise ValueError("No short-axis plane cuts the insertion loop twice")
    branches = []
    for segment, t in cuts:
        heights, points = _follow_branch(insertion, s, e_l, segment, t, steepness)
        angles = np.degrees(np.unwrap(np.radians([azimuth(p) for p in points])))
        branches.append((heights, angles, points))
    # The anterior insertion is the branch nearer the anterior reference (e.g. the pulmonary valve)
    distance = [np.min(np.linalg.norm(points - anterior_ref, axis=1)) for _, _, points in branches]
    first = int(np.argmin(distance))
    out = []
    for heights, angles, _ in (branches[first], branches[1 - first]):
        # np.interp holds the end values outside the followed section
        out.append(np.mod(np.interp(levels, heights, angles) + 180.0, 360.0) - 180.0)
    return levels, out[0], out[1]


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
    s, psi, width = _short_axis_frame(x, apex, base_centre, insertion, anterior_ref, septum_ref)

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


def _short_axis_frame(
    x: np.ndarray,
    apex: np.ndarray,
    base_centre: np.ndarray,
    insertion: np.ndarray,
    anterior_ref: np.ndarray,
    septum_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-axis coordinates of points: (s, psi, width), each of shape (N,).

    s is the normalized position along the long axis from ``apex`` to ``base_centre``; psi in
    [0, 360) is the angle about the long axis from the anterior insertion, increasing towards the
    septum; width is the angle of the septal arc (from the anterior to the inferior insertion) at
    each point's level. See :func:`aha_segments` for the parameters.
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
    return s, psi, width


# The 15 RV regions of Zhong et al. (2012), Appendix B, in their order; with RV regions, region k
# has the cell tag RV_REGION_OFFSET + k (18-32, the numbering of Doste et al. 2019, Table A1)
RV_REGION_NAMES = [
    "RV-BASAL-ANTERIOR",
    "RV-BASAL-LATERAL",
    "RV-BASAL-INFERIOR",
    "RV-BASAL-INFEROSEPTAL",
    "RV-BASAL-ANTEROSEPTAL",
    "RV-MID-ANTERIOR",
    "RV-MID-LATERAL",
    "RV-MID-INFERIOR",
    "RV-MID-INFEROSEPTAL",
    "RV-MID-ANTEROSEPTAL",
    "RV-APICAL-ANTERIOR",
    "RV-APICAL-INFERIOR",
    "RV-APICAL-SEPTAL",
    "RV-OUTLET",
    "RV-INLET",
]
RV_REGION_OFFSET = 17
LVOT_TAG_RV_REGIONS = 33


def rv_segments(
    x: np.ndarray,
    septal: np.ndarray,
    apex: np.ndarray,
    base_centre: np.ndarray,
    insertion: np.ndarray,
    anterior_ref: np.ndarray,
    septum_ref: np.ndarray | None,
    s_min: float,
    s_top: float,
    pv_centre: np.ndarray,
    tv_centre: np.ndarray,
) -> np.ndarray:
    """Label points of the RV wall with the 15 regions of Zhong et al. (2012), Appendix B.

    The levels lie in short-axis planes of the LV long axis (``apex`` to ``base_centre``), as Zhong
    et al. drew them on short-axis MRI slices. With s the normalized position along that axis and
    r = (s - s_min) / (s_top - s_min), the RV body is apical for r < 1/3 (and below s_min), mid for
    r < 2/3 and basal for r <= 1. Above it (r > 1) a point is in the RV outlet (14) if it is closer
    to ``pv_centre`` than to ``tv_centre``, and in the RV inlet (15) otherwise.

    The sectors start at the anterior insertion. With frac the position along the septal arc
    between the insertions (0 at the anterior, 1 at the inferior insertion; points beyond the arc
    take its nearer end), the free wall is split into anterior, lateral and inferior thirds in the
    basal and mid levels, and anterior and inferior halves in the apical level; the septum
    (``septal``) into anteroseptal and inferoseptal halves, and one apical septal region. Zhong et
    al. give the ratio of free wall to septum (3:2, and 2:1 at the apex) but not where the levels
    fall or how the outlet meets the inlet: s_top, the nearest-valve rule and the equal-angle
    sectors are this implementation's choices.

    Parameters
    ----------
    x : np.ndarray
        Points to label, shape (N, 3)
    septal : np.ndarray
        True for points of the RV septum, shape (N,)
    apex, base_centre, insertion, anterior_ref, septum_ref
        As in :func:`aha_segments`
    s_min : float
        The lowest point of the RV, in units of s
    s_top : float
        The top of the basal level, in units of s (e.g. the centre of the tricuspid valve)
    pv_centre : np.ndarray
        The centre of the pulmonary valve
    tv_centre : np.ndarray
        The centre of the tricuspid valve

    Returns
    -------
    np.ndarray
        The tag of each point, RV_REGION_OFFSET + the region number (18-32)
    """
    x = np.asarray(x, dtype=float).reshape(-1, 3)
    septal = np.asarray(septal, dtype=bool).reshape(-1)
    s, psi, width = _short_axis_frame(x, apex, base_centre, insertion, anterior_ref, septum_ref)
    r = (s - s_min) / (s_top - s_min)
    frac = np.where(psi <= width, psi / width, np.where(psi - width < 360.0 - psi, 1.0, 0.0))

    free_body = np.where(
        frac < 1 / 3, 1, np.where(frac < 2 / 3, 2, 3)
    )  # anterior, lateral, inferior
    septal_body = np.where(frac < 1 / 2, 5, 4)  # anteroseptal, inferoseptal
    body = np.where(septal, septal_body, free_body)  # basal numbering (1-5)
    apical = np.where(septal, 13, np.where(frac < 1 / 2, 11, 12))
    above = np.where(
        np.linalg.norm(x - np.asarray(pv_centre, dtype=float), axis=1)
        < np.linalg.norm(x - np.asarray(tv_centre, dtype=float), axis=1),
        14,
        15,
    )
    region = np.where(r < 1 / 3, apical, np.where(r < 2 / 3, body + 5, body))
    region = np.where(r > 1, above, region)
    return (RV_REGION_OFFSET + region).astype(np.int32)


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
    endocardial landmark node is on the free wall (the insertion nodes, shared by
    both surfaces, count as septum) and it is nearer to that node than to the LV
    endocardium.

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
    c = _biv_cells(mesh, ffun, markers, landmarks)
    rv = c["free_wall"]
    values = np.full(len(c["cells"]), RV_TAG, dtype=np.int32)
    values[~rv] = aha_segments(c["x"][~rv], *c["frame"])
    tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, c["cells"], values)
    return tags, aha_markers()


def _biv_cells(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    markers: dict[str, Any],
    landmarks: dict[str, Any],
) -> dict[str, Any]:
    """The owned cells of a UKB BiV, their midpoints, the RV free-wall mask and the short-axis
    frame arguments (apex, base centre, insertion, anterior and septal references)."""
    missing = [k for k in ("insertion", "rv_freewall", "rv_septum", "valves") if k not in landmarks]
    if missing:
        raise ValueError(f"Missing landmarks {missing}")
    apex, mv_centre = lv_long_axis(mesh, ffun, markers)
    ncells = mesh.topology.index_map(mesh.topology.dim).size_local
    cells = np.arange(ncells, dtype=np.int32)
    x = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, cells)

    freewall = np.asarray(landmarks["rv_freewall"], dtype=float)
    septum = np.asarray(landmarks["rv_septum"], dtype=float)
    # The insertion nodes belong to both surfaces; count them as septum, so a cell is in the free
    # wall only if its nearest RV node lies on the free wall proper
    on_septum, _ = _nearest(freewall, septum)
    freewall = freewall[on_septum > 1e-9]
    rv_nodes = np.vstack([freewall, septum])
    d_rv, index = _nearest(x, rv_nodes)
    d_lv, _ = _nearest(x, _gather_facet_midpoints(mesh, ffun, _marker(markers, "LV_ENDO")))
    frame = (
        apex,
        mv_centre,
        np.asarray(landmarks["insertion"], dtype=float),
        np.asarray(landmarks["valves"]["PV"], dtype=float),
        septum.mean(axis=0),
    )
    return {
        "cells": cells,
        "x": x,
        "free_wall": (index < len(freewall)) & (d_rv < d_lv),
        "nearer_rv": d_rv < d_lv,
        "frame": frame,
        "rv_nodes": np.vstack([np.asarray(landmarks["rv_freewall"], dtype=float), septum]),
    }


def transventricular(
    mesh: dolfinx.mesh.Mesh, ffun: dolfinx.mesh.MeshTags, markers: dict[str, Any]
) -> dolfinx.fem.Function:
    """The transventricular Laplace solution (P1): 0 on the RV endocardium (``RV_ENDO``), 1 on
    the LV endocardium (``LV_ENDO``) and zero flux elsewhere. This is the transventricular
    coordinate of Cobiveco (Schuler et al. 2021), whose 0.5 level runs through the middle of the
    septum. Doste et al. (2019) solve a related problem (-2 on the LV and +1 on the RV
    endocardium, 0 on the epicardium) whose zero level gives one third of the septum to the RV;
    ``rv_septum_fraction=1/3`` in :func:`biv_regions` approximates that split."""
    from dolfinx.fem.petsc import LinearProblem

    missing = [name for name in ("RV_ENDO", "LV_ENDO") if name not in markers]
    if missing:
        raise ValueError(f"Missing markers {missing}")
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    bcs = [
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(value)),
            dolfinx.fem.locate_dofs_topological(V, fdim, ffun.find(_marker(markers, name))),
            V,
        )
        for name, value in (("RV_ENDO", 0.0), ("LV_ENDO", 1.0))
    ]
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    zero = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    problem = LinearProblem(
        ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        zero * v * ufl.dx,
        bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="cardiac_geometriesx_transventricular",
    )
    uh = problem.solve()
    assert isinstance(uh, dolfinx.fem.Function)
    uh.x.scatter_forward()
    return uh


def biv_region_markers() -> dict[str, tuple[int, int]]:
    """The names of the 17 AHA segments (1-17), the 15 RV regions of Zhong et al. (2012) (18-32)
    and the LV above the base (33), as tagged by :func:`biv_regions`."""
    markers = {k: v for k, v in aha_markers().items() if v[0] <= 17}
    for k, name in enumerate(RV_REGION_NAMES, start=1):
        markers[name] = (RV_REGION_OFFSET + k, 3)
    markers["LVOT"] = (LVOT_TAG_RV_REGIONS, 3)
    return markers


def biv_regions(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    markers: dict[str, Any],
    landmarks: dict[str, Any],
    rv_septum_fraction: float = 0.5,
) -> tuple[dolfinx.mesh.MeshTags, dict[str, tuple[int, int]]]:
    """AHA segments of the LV and the RV regions of Zhong et al. (2012) on a UKB BiV.

    The RV free wall is found as in :func:`biv_aha`, and the RV wall at the junction with the LV
    is added to it: cells outside the septal arc that are nearer the RV endocardium than the LV
    endocardium (their nearest RV landmark node is an insertion node, so :func:`biv_aha` keeps
    them in the LV). The septum (the cells between the two insertions, with 0 <= s <= 1, that are
    not RV free wall) is split by the transventricular Laplace solution u
    (:func:`transventricular`): a septal cell belongs to the RV if u at its midpoint is below
    ``rv_septum_fraction``. So 0 gives the whole septum to the LV (as :func:`biv_aha`), 1 gives it
    all to the RV, the default 0.5 splits it at the middle of u (the transventricular boundary of
    Cobiveco, Schuler et al. 2021), and 1/3 approximates the one third of Doste et al. (2019).
    Zhong et al. segment the RV endocardial surface and do not split the septum through its
    thickness, so the split is a choice. RV cells get :func:`rv_segments` (tags 18-32), with the
    RV body from its lowest landmark node to the centre of the tricuspid valve; LV cells get
    :func:`aha_segments` (1-17), and the LV above the mitral plane is 33.

    Parameters
    ----------
    mesh, ffun, markers, landmarks
        As in :func:`biv_aha`; the markers also need ``RV_ENDO``
    rv_septum_fraction : float
        The share of the septum, in u from the RV side, that belongs to the RV, in [0, 1]

    Returns
    -------
    tuple[dolfinx.mesh.MeshTags, dict[str, tuple[int, int]]]
        The cell tags and their markers (:func:`biv_region_markers`)
    """
    if not 0.0 <= rv_septum_fraction <= 1.0:
        raise ValueError(f"rv_septum_fraction must be in [0, 1], got {rv_septum_fraction}")
    c = _biv_cells(mesh, ffun, markers, landmarks)
    x, free_wall, frame = c["x"], c["free_wall"], c["frame"]
    s, psi, width = _short_axis_frame(x, *frame)
    # RV wall at the junction: outside the septal arc, nearer the RV than the LV endocardium
    free_wall = free_wall | (~free_wall & (psi >= width) & c["nearer_rv"])
    candidates = ~free_wall & (s >= 0.0) & (s <= 1.0) & (psi < width)
    if rv_septum_fraction == 0.0:
        rv_septum = np.zeros_like(candidates)
    elif rv_septum_fraction == 1.0:
        rv_septum = candidates
    else:
        u = transventricular(mesh, ffun, markers)
        u0 = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
        u0.interpolate(u)
        rv_septum = candidates & (u0.x.array[: len(c["cells"])] < rv_septum_fraction)

    apex, mv_centre = frame[0], frame[1]
    e_l = mv_centre - apex
    length = float(np.linalg.norm(e_l))
    e_l = e_l / length
    s_min = float(np.min((c["rv_nodes"] - apex) @ e_l) / length)
    tv = np.asarray(landmarks["valves"]["TV"], dtype=float)
    s_top = float((tv - apex) @ e_l / length)

    rv = free_wall | rv_septum
    values = np.empty(len(c["cells"]), dtype=np.int32)
    _, _, insertion, anterior_ref, septum_ref = frame
    values[rv] = rv_segments(
        x[rv],
        rv_septum[rv],
        apex,
        mv_centre,
        insertion,
        anterior_ref,
        septum_ref,
        s_min,
        s_top,
        np.asarray(landmarks["valves"]["PV"], dtype=float),
        tv,
    )
    lv_values = aha_segments(x[~rv], *frame)
    values[~rv] = np.where(lv_values == LVOT_TAG, LVOT_TAG_RV_REGIONS, lv_values)
    tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, c["cells"], values)
    return tags, biv_region_markers()
