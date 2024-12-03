# # UK-Biobank atlas
#
# The UK-Biobank provides an atlas for BiV meshes (see https://www.cardiacatlas.org/biventricular-modes/).
# We also provide a separate package to generate meshes from this atlas, see https://computationalphysiology.github.io/ukb-atlas.
#
# However, if you want to use these meshes for cardiac simulations you probably need to generate fiber orientations using [`fenicsx-ldrb](https://finsberg.github.io/fenicsx-ldrb/README.html) as well.
# Here we will show how to do this programmatically from python

from pathlib import Path
import numpy as np
import json
from mpi4py import MPI
import pyvista
import dolfinx
import ldrb
import cardiac_geometries as cg
import ukb.cli

# Select the mode you want to use (-1 will you the mean mode)

mode = -1

# Standard deviation of the mode
std = 1.5

# Choose output directory

outdir = Path("ukb_mesh")
outdir.mkdir(exist_ok=True)
subdir = outdir / f"mode_{int(mode)}"

# Characteristic length of the mesh (smaller values will give finer meshes)
char_length = 10.0

# Generate mesh

ukb.cli.main(
    [
        str(outdir),
        "--mode",
        str(mode),
        "--std",
        str(std),
        "--subdir",
        f"mode_{int(mode)}",
        "--mesh",
        "--char_length_max",
        str(char_length),
        "--char_length_min",
        str(char_length),
    ]
)
# Choose the ED mesh
mesh_name = subdir / "ED.msh"
# Convert mesh to dolfinx
comm = MPI.COMM_WORLD
geometry = cg.utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

# Now we can plot the mesh

pyvista.start_xvfb()
vtk_mesh = dolfinx.plot.vtk_mesh(geometry.mesh, geometry.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("ukb_mesh.png")

# We als save the markers to json

print(geometry.markers)
if comm.rank == 0:
    (subdir / "markers.json").write_text(json.dumps(geometry.markers, default=cg.utils.json_serial))
comm.barrier()

# And plot the facet tags

assert geometry.ffun is not None
vtk_bmesh = dolfinx.plot.vtk_mesh(geometry.mesh, geometry.ffun.dim, geometry.ffun.indices)
bgrid = pyvista.UnstructuredGrid(*vtk_bmesh)
bgrid.cell_data["Facet tags"] = geometry.ffun.values
bgrid.set_active_scalars("Facet tags")
p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(bgrid, show_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_tags_ukb.png")

# We could also combine all the outflow tracts into one entity, i.e the `BASE`

entities = [
    geometry.ffun.find(1),  # LV
    geometry.ffun.find(2),  # RV
    geometry.ffun.find(7),  # EPI
    np.hstack(
        [
            geometry.ffun.find(3),
            geometry.ffun.find(4),
            geometry.ffun.find(5),
            geometry.ffun.find(6),
        ]
    ),  # BASE
]
entity_values = [np.full(e.shape, i + 1, dtype=np.int32) for i, e in enumerate(entities)]

# and create new mesh tags

ffun = dolfinx.mesh.meshtags(geometry.mesh, 2, np.hstack(entities), np.hstack(entity_values))

# and plot the new tags

vtk_bmesh_new = dolfinx.plot.vtk_mesh(geometry.mesh, ffun.dim, ffun.indices)
bgrid_new = pyvista.UnstructuredGrid(*vtk_bmesh_new)
bgrid_new.cell_data["Facet tags"] = ffun.values
bgrid_new.set_active_scalars("Facet tags")
p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(bgrid_new, show_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("new_facet_tags_ukb.png")

# Let us also save the markers so that we can inspect them later

with dolfinx.io.XDMFFile(geometry.mesh.comm, subdir / "new_ffun.xdmf", "w") as xdmf:
    xdmf.write_mesh(geometry.mesh)
    xdmf.write_meshtags(ffun, geometry.mesh.geometry)

# Now let ut combine the markers as well so that they align with the markers needed in fenicsx-ldrb

markers = {}
keymap = {"LV": "lv", "RV": "rv", "EPI": "epi"}
for k, v in geometry.markers.items():
    markers[keymap.get(k, k)] = [v[0]]
markers["base"] = [geometry.markers[m][0] for m in ["MV", "PV", "TV", "AV"]]

# And finally generate the fibers

fiber_space = "P_1"
fiber_angle_endo = 60
fiber_angle_epi = -60
system = ldrb.dolfinx_ldrb(
    mesh=geometry.mesh,
    ffun=geometry.ffun,
    markers=markers,
    alpha_endo_lv=fiber_angle_endo,
    alpha_epi_lv=fiber_angle_epi,
    beta_endo_lv=0,
    beta_epi_lv=0,
    fiber_space=fiber_space,
)

# and save them

cg.fibers.utils.save_microstructure(
    mesh=geometry.mesh, functions=(system.f0, system.s0, system.n0), outdir=subdir
)

# Let us also plot the fibers
topology_f0, cell_types_f0, geometry_f0 = dolfinx.plot.vtk_mesh(system.f0.function_space)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, : len(system.f0)] = system.f0.x.array.real.reshape((geometry_f0.shape[0], len(system.f0)))
function_grid = pyvista.UnstructuredGrid(topology_f0, cell_types_f0, geometry_f0)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=1.0)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="r")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("fiber_ukb.png")

# You should now be able to load the geometry and the fibers using the following code

geo = cg.geometry.Geometry.from_folder(comm=comm, folder=subdir)
