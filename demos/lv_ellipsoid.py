# # LV ellipsoid
# In this example we will create a simple ellipsoid model of the left ventricle (LV) of the heart.

from pathlib import Path
from mpi4py import MPI
import pyvista
import dolfinx
import numpy as np
import cardiac_geometries


geodir = Path("lv_ellipsoid")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True, fiber_space="P_1")

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
    figure = plotter.screenshot("lv_mesh.png")

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

# Now we could try to redo the exercise but with create a thicker ellipsoid with fibers
# oriented more longitudinally at the endocardium and more circumferentially at the epicardium.

geodir2 = Path("lv_ellipsoid2")
if not geodir2.exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir2,
        create_fibers=True,
        fiber_space="P_1",
        fiber_angle_endo=80,
        fiber_angle_epi=-30,
        r_short_endo=5.0,
        r_short_epi=10.0,
    )

# If the folder already exist, then we just load the geometry

geo2 = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir2,
)

# Next we will use `pyvista` to plot the mesh

vtk_mesh = dolfinx.plot.vtk_mesh(geo2.mesh, geo2.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("lv_mesh2.png")


assert geo2.ffun is not None
vtk_bmesh = dolfinx.plot.vtk_mesh(geo2.mesh, geo2.ffun.dim, geo2.ffun.indices)
bgrid = pyvista.UnstructuredGrid(*vtk_bmesh)
bgrid.cell_data["Facet tags"] = geo2.ffun.values
bgrid.set_active_scalars("Facet tags")
p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(bgrid, show_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_tags2.png")


assert geo2.f0 is not None
topology, cell_types, geometry = dolfinx.plot.vtk_mesh(geo2.f0.function_space)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, : len(geo2.f0)] = geo2.f0.x.array.real.reshape((geometry.shape[0], len(geo2.f0)))
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
    fig_as_array = plotter.screenshot("fiber2.png")


# ## Creating AHA segments for the LV ellipsoid
# Finally, we can also create AHA segments for the LV ellipsoid by setting the `
# aha=True` flag when creating the ellipsoid mesh. This will create a `MeshTags` object
# containing the AHA segments which can be accessed with the `cfun` attribute of the
# `Geometry` object. Here we also set `psize_ref=1` to get a finer mesh and we specify
# `dmu_factor=1/4` (which is also the default). The `dmu_factor` controls the
# thickness of the basal, midventricular, and apical segments in the long axis.
# A value of 1/4 means that 25% of the distance from base to apex is used for each segment.
# A value of 1/3 means that 33% of the distance from base to apex is used for each segment,
# which means that the 17th segment (the apex) will be taken out, while a value of 0.2 (20%)
# means that 20% of the distance from base to apex is used for each segment. In this case,
# We have 20% base, 20% midventricular, 20% apical, and 40% apex.


geodir_aha = Path("lv_ellipsoid_aha")
if not geodir_aha.exists():
    cardiac_geometries.mesh.lv_ellipsoid(
        outdir=geodir_aha,
        aha=True, psize_ref=1, dmu_factor=1/4
    )

geo_aha = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir_aha,
)
assert geo_aha.cfun is not None

# Now  you can also see that we have additional markers for the AHA segments

print(geo_aha.markers)

# We can also plot the AHA segments with pyvista

vtk_mesh = dolfinx.plot.vtk_mesh(geo_aha.mesh, geo_aha.mesh.topology.dim)
p = pyvista.Plotter(window_size=[800, 800])
grid = pyvista.UnstructuredGrid(*vtk_mesh)
grid.cell_data["AHA"] = geo_aha.cfun.values
grid.set_active_scalars("AHA")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("aha_segments.png")
p.show()
