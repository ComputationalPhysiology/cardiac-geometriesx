# # Higher order geometries
#
# By default the meshes are made of linear (straight sided) tetrahedra, so curved
# surfaces such as the endocardium and epicardium are approximated by flat triangles.
# All geometries can also be created with elements of higher order by passing the `order`
# argument (or `--order` on the command line). The nodes of the higher order elements are
# placed on the geometry, so the elements are curved to follow the boundaries.
#
# This is useful when you want an accurate representation of the geometry without refining
# the mesh, e.g. to compute cavity volumes, or when you solve for the displacement with
# quadratic elements and want the geometry to be represented with the same order
# (isoparametric elements).

import math
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pyvista
import ufl

import cardiac_geometries

# Let us start by creating the same coarse LV ellipsoid with linear, quadratic and cubic
# elements

geos = {}
for order in [1, 2, 3]:
    geodir = Path(f"lv_ellipsoid_order{order}")
    if not geodir.exists():
        cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, psize_ref=8.0, order=order)
    geos[order] = cardiac_geometries.geometry.Geometry.from_folder(
        comm=MPI.COMM_WORLD,
        folder=geodir,
    )

# The three meshes have the same cells, but the higher order meshes have additional nodes on
# the edges, faces and inside the cells

for order, geo in geos.items():
    num_cells = geo.mesh.topology.index_map(3).size_global
    num_nodes = geo.mesh.geometry.index_map().size_global
    print(f"Order {order}: {num_cells} cells and {num_nodes} nodes")

# Let us plot the linear and the quadratic mesh next to each other. Since the cells of the
# quadratic mesh are curved, we subdivide each of them when we extract the surface to plot

plotter = pyvista.Plotter(shape=(1, 2))
for i, order in enumerate([1, 2]):
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(geos[order].mesh))
    plotter.subplot(0, i)
    plotter.add_text(f"Order {order}")
    plotter.add_mesh(grid.extract_surface(nonlinear_subdivision=4))
    plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("lv_higher_order.png")

# The LV ellipsoid is the part of the region between two ellipsoids that lies below the base
# at $x = 5$. Since the volume of an ellipsoid with semi-axes $a$, $b$ and $b$ truncated at
# $x = h$ is
#
# $$
# V = \pi b^2 \left( (h + a) - \frac{h^3 + a^3}{3 a^2} \right)
# $$
#
# we know the exact volume of the wall and the cavity


def truncated_ellipsoid_volume(a, b, h):
    return math.pi * b**2 * ((h + a) - (h**3 + a**3) / (3 * a**2))


cavity_exact = truncated_ellipsoid_volume(a=17.0, b=7.0, h=5.0)
wall_exact = truncated_ellipsoid_volume(a=20.0, b=10.0, h=5.0) - cavity_exact

# The volume of the wall is the integral of one over the mesh, and the volume of the cavity can
# be computed from the endocardium by the divergence theorem
#
# $$
# V_{\text{cavity}} = -\frac{1}{3} \int_{\text{ENDO}} (x - x_0) \cdot n \, ds
# $$
#
# where $x_0 = (5, 0, 0)$ is a point on the base, so that the base, which closes the
# cavity, does not contribute to the integral. Let us compare the error for the different orders

for order, geo in geos.items():
    x = ufl.SpatialCoordinate(geo.mesh)
    n = ufl.FacetNormal(geo.mesh)
    x0 = ufl.as_vector([5.0, 0.0, 0.0])
    ds = ufl.Measure("ds", domain=geo.mesh, subdomain_data=geo.ffun)
    wall = geo.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.dx(domain=geo.mesh))),
    )
    cavity = geo.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(-1.0 / 3.0 * ufl.dot(x - x0, n) * ds(geo.markers["ENDO"][0])),
        ),
    )
    print(
        f"Order {order}: relative error in the volume of the wall "
        f"{abs(wall - wall_exact) / wall_exact:.1e} and of the cavity "
        f"{abs(cavity - cavity_exact) / cavity_exact:.1e}",
    )

# On this coarse mesh the cavity volume is off by about 10% with linear elements, while with
# quadratic elements the error is below 0.2%, without adding any cells.
#
# ## Fibers on a higher order mesh
#
# Fibers can be created on the higher order meshes just like on the linear meshes.
# With quadratic elements it is natural to also represent the fibers in a quadratic space

geodir = Path("lv_ellipsoid_order2_fibers")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir,
        order=2,
        create_fibers=True,
        fiber_space="P_2",
    )
geo = cardiac_geometries.geometry.Geometry.from_folder(comm=MPI.COMM_WORLD, folder=geodir)
assert geo.f0 is not None

topology, cell_types, geometry = dolfinx.plot.vtk_mesh(geo.f0.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["f0"] = geo.f0.x.array.real.reshape((geometry.shape[0], 3))
glyphs = function_grid.glyph(orient="f0", factor=1.0)
grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(geo.mesh))
plotter = pyvista.Plotter()
plotter.add_mesh(grid.extract_surface(nonlinear_subdivision=3), style="wireframe", color="r")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("lv_higher_order_fibers.png")

# ## Other geometries
#
# The `order` argument is available for all the geometries, e.g. for the BiV ellipsoid

geodir = Path("biv_ellipsoid_order2")
if not geodir.exists():
    cardiac_geometries.mesh.biv_ellipsoid(outdir=geodir, order=2, create_fibers=True)
biv = cardiac_geometries.geometry.Geometry.from_folder(comm=MPI.COMM_WORLD, folder=geodir)

grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(biv.mesh))
plotter = pyvista.Plotter()
plotter.add_mesh(grid.extract_surface(nonlinear_subdivision=4))
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("biv_higher_order.png")

# and for the meshes from the UK Biobank atlas (see the [UK Biobank demo](ukb_atlas.py)),
# i.e. `cardiac_geometries.mesh.ukb(outdir, order=2)`. Note that the surfaces of the atlas are
# triangulations, so here the higher order nodes are placed on flat triangles. These meshes
# therefore do not represent the geometry more accurately than the linear ones, but they are
# still useful when you want to use isoparametric elements. Placing the nodes on the
# triangulated surfaces can also invert some elements, which gmsh untangles with its high order
# optimizer. This is fast for order 2, but can take a few minutes for order 3.
#
# ```{note}
# XDMF does not support meshes of order higher than 2, so for these meshes the file `mesh.xdmf`
# is not written. The geometry is always saved to `geometry.bp`, which is what
# `Geometry.from_folder` reads.
# ```
#
# Finally, we can check that the fibers have unit length also on the quadratic mesh

print(np.linalg.norm(biv.f0.x.array.reshape(-1, 3), axis=1).min())
