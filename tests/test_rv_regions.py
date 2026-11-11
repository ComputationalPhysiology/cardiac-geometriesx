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
