# # Slab
# In this example we will create a tissue slab

from pathlib import Path
from mpi4py import MPI
import pyvista
import dolfinx
import numpy as np
import cardiac_geometries


geodir = Path("slab")
if not geodir.exists():
    cardiac_geometries.mesh.slab(
        lx=10.0,
        ly=4.0,
        lz=1.0,
        dx=0.5,
        outdir=geodir,
        create_fibers=True,
        fiber_space="P_1",
        fiber_angle_endo=60,
        fiber_angle_epi=-60,
    )

# If the folder already exist, then we just load the geometry

geo = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

# Next we will use `pyvista` to plot the mesh

vtk_mesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("slab.png")

# The facets of the mesh are also marked with a `MeshTags` object. We can access the facet tags with `geo.ffun.values` and the markers with `geo.markers`.

print(geo.markers)

# Next lets plot the facet tags with pyvista.

assert geo.ffun is not None
vtk_bmesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.ffun.dim, geo.ffun.indices)
bgrid = pyvista.UnstructuredGrid(*vtk_bmesh)
bgrid.cell_data["Facet tags"] = geo.ffun.values
bgrid.set_active_scalars("Facet tags")
p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(bgrid, show_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_tags.png")

# Now let us look at the fibers of the LV ellipsoid. The fibers are stored in the `f0` attribute of the `Geometry` object.
assert geo.f0 is not None
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(geo.f0.function_space)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, : len(geo.f0)] = geo.f0.x.array.real.reshape((geometry.shape[0], len(geo.f0)))
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=2.0)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="r")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("fiber.png")
