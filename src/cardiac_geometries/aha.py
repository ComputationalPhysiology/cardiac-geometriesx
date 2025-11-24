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


def biv_aha(): ...
