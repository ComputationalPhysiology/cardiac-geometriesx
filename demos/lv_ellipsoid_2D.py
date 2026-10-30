# # LV ellipsoid for axisymmetric problems
# In this example we will create the meridian of the ellipsoid model of the left ventricle (LV),
# i.e. the cross section of the wall with a plane through the long axis. If the problem you
# want to solve is axisymmetric, such as the inflation of the LV in the benchmark of
# [Land et al. (2015)](https://doi.org/10.1098/rspa.2015.0641), you only need to solve it on
# this two dimensional mesh.
#
# The long axis is the $x$-axis and the radial direction is the $y$-axis, and the meridian lies in
# the upper half plane $y \geq 0$. The default radii and angles give the ellipsoid of the
# benchmark (in mm), which is the same ellipsoid as the default one in
# [the LV ellipsoid demo](lv_ellipsoid.py). Note, however, that the angles $\mu$ here must be
# in $[0, \pi]$, so that the meridian is above the axis.

from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pyvista

import cardiac_geometries
from cardiac_geometries.fibers.lv_ellipsoid import transmural_coordinate

geodir = Path("lv_ellipsoid_2D")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid_2D(outdir=geodir, create_fibers=True, fiber_space="P_1")

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# The mesh has geometric dimension 2

print(f"Geometric dimension: {geo.mesh.geometry.dim}, topological dimension {geo.mesh.topology.dim}")

vtk_mesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("lv_2D_mesh.png")

# ## Markers
#
# The markers are the same as in three dimensions, but one dimension lower: `ENDO`, `EPI` and
# `BASE` are now curves, and `APEX` is the segment of the symmetry axis between the two apices,
# which is where an axisymmetric problem needs its symmetry condition. The points `ENDOPT` and
# `EPIPT` (the apices) and `ENDORING` and `EPIRING` (the base of the endocardium and the
# epicardium) are marked in the vertex tags `geo.vfun`.

print(geo.markers)

# Next lets plot the facets with each of the markers with pyvista.

assert geo.ffun is not None
p = pyvista.Plotter(window_size=[800, 800])
for name, color in [("ENDO", "red"), ("EPI", "blue"), ("BASE", "green"), ("APEX", "orange")]:
    facets = geo.ffun.find(geo.markers[name][0])
    bgrid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(geo.mesh, geo.ffun.dim, facets))
    p.add_mesh(bgrid, color=color, line_width=5, label=name)
p.add_legend()
p.view_xy()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("lv_2D_facet_tags.png")

# ## Fibers
#
# The fiber angle varies linearly in the transmural coordinate $t$ of the ellipsoid itself:
# the point is on the ellipsoid with radii interpolated linearly between the endocardium
# ($t = 0$) and the epicardium ($t = 1$), as in the benchmark. A Laplace problem on the
# meridian would not give this coordinate, as the Laplace problem in the plane is not the
# axisymmetric one. Let us plot $t$

V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))
t = transmural_coordinate(
    V,
    r_short_endo=geo.info["r_short_endo"],
    r_short_epi=geo.info["r_short_epi"],
    r_long_endo=geo.info["r_long_endo"],
    r_long_epi=geo.info["r_long_epi"],
)
grid.point_data["t"] = t.x.array
plotter = pyvista.Plotter()
plotter.add_mesh(grid, scalars="t", cmap="viridis", clim=[0.0, 1.0])
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("lv_2D_transmural.png")

# The fibers have three components also on the meridian: the axial ($x$), the radial ($y$) and
# the circumferential one. The circumferential direction is $e_\theta = (0, 0, 1)$, and since the
# sheet normal `n0` is perpendicular to the wall, the direction along the meridian is
# $e_\mu = e_\theta \times n_0$. The helix angle of the fibers is therefore

assert geo.f0 is not None and geo.n0 is not None
f0 = geo.f0.x.array.reshape(-1, 3)
n0 = geo.n0.x.array.reshape(-1, 3)
e_mu = np.stack([-n0[:, 1], n0[:, 0], np.zeros(len(n0))], axis=1)
helix_angle = np.degrees(np.arctan2((f0 * e_mu).sum(axis=1), f0[:, 2]))

# which varies linearly from the angle at the endocardium to the angle at the epicardium

print(
    f"Fiber angle {geo.info['fiber_angle_endo']} at the endocardium and "
    f"{geo.info['fiber_angle_epi']} at the epicardium",
)
grid.point_data["helix angle"] = helix_angle
plotter = pyvista.Plotter()
plotter.add_mesh(grid, scalars="helix angle", cmap="coolwarm")
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("lv_2D_helix_angle.png")

# To see the fibers in the more familiar way, we can revolve the meridian around the long axis.
# We revolve it three quarters of a turn, so that we can see the wall, and rotate the fibers
# with it. A rotation by the angle $\theta$ around the $x$-axis takes the point $(x, y, 0)$ to
# $(x, y \cos \theta, y \sin \theta)$, and the circumferential direction $(0, 0, 1)$ to
# $(0, -\sin \theta, \cos \theta)$. We color the fibers by the helix angle, and make the wall
# transparent so that we can see the fibers inside it.

solid = grid.extract_surface().extrude_rotate(
    resolution=60,
    angle=270.0,
    rotation_axis=(1.0, 0.0, 0.0),
    capping=True,
)

x = geo.f0.function_space.tabulate_dof_coordinates()
points = []
vectors = []
thetas = np.radians(np.linspace(0, 270, 19))
for theta in thetas:
    c, s = np.cos(theta), np.sin(theta)
    points.append(np.stack([x[:, 0], x[:, 1] * c, x[:, 1] * s], axis=1))
    vectors.append(
        np.stack([f0[:, 0], f0[:, 1] * c - f0[:, 2] * s, f0[:, 1] * s + f0[:, 2] * c], axis=1),
    )
point_cloud = pyvista.PolyData(np.concatenate(points))
point_cloud["fibers"] = np.concatenate(vectors)
point_cloud["helix angle"] = np.tile(helix_angle, len(thetas))
glyphs = point_cloud.glyph(orient="fibers", scale=False, factor=1.5)

plotter = pyvista.Plotter()
plotter.add_mesh(solid, color="lightgrey", opacity=0.2)
plotter.add_mesh(glyphs, scalars="helix angle", cmap="coolwarm")
plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("lv_2D_fibers.png")
