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

# Characteristic length of the mesh (smaller values will give finer meshes). Let us also pick the case from end-diastole (ED)
char_length = 10.0
case = "ED"

# Generate surfaces

ukb.cli.main(["surf", str(outdir), "--mode", str(mode), "--std", str(std), "--case", case])

# Generate mesh

ukb.cli.main([
    "mesh",
    str(outdir),
    "--case",
    case,
    "--char_length_max",
    str(char_length),
    "--char_length_min",
    str(char_length),
])

# Choose the ED mesh
mesh_name = outdir / "ED.msh"
# Convert mesh to dolfinx
comm = MPI.COMM_WORLD
geometry = cg.utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

# Now we can plot the mesh

vtk_mesh = dolfinx.plot.vtk_mesh(geometry.mesh, geometry.mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("ukb_mesh.png")

# We also print the markers. The surfaces are the endocardium of each ventricle (`LV_ENDO` and
# `RV_ENDO`), the epicardium (`EPI`) and the rims of the four valves (`MV`, `AV`, `TV` and `PV`),
# and the volume is the myocardium (`Wall`)

print(geometry.markers)


# And plot the facet tags

assert geometry.ffun is not None
vtk_bmesh = dolfinx.plot.vtk_mesh(
    geometry.mesh, geometry.ffun.dim, geometry.ffun.indices
)
bgrid = pyvista.UnstructuredGrid(*vtk_bmesh)
bgrid.cell_data["Facet tags"] = geometry.ffun.values
bgrid.set_active_scalars("Facet tags")
p = pyvista.Plotter(window_size=[800, 800])
p.add_mesh(bgrid, show_edges=True)
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure = p.screenshot("facet_tags_ukb.png")

# We could also combine all the valves into one entity, i.e the `BASE`

m = {name: value[0] for name, value in geometry.markers.items()}
entities = [
    geometry.ffun.find(m["LV_ENDO"]),
    geometry.ffun.find(m["RV_ENDO"]),
    geometry.ffun.find(m["EPI"]),
    np.hstack([geometry.ffun.find(m[valve]) for valve in ["MV", "AV", "TV", "PV"]]),  # BASE
]
entity_values = [
    np.full(e.shape, i + 1, dtype=np.int32) for i, e in enumerate(entities)
]

# and create new mesh tags

ffun = dolfinx.mesh.meshtags(
    geometry.mesh, 2, np.hstack(entities), np.hstack(entity_values)
)

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

# Let us also save the mesh and  markers so that we can inspect them later. Let us save them in a adios2 file called `geometry.bp`

cg.geometry.save_geometry(
    path=outdir / "geometry.bp",
    mesh=geometry.mesh,
    markers=geometry.markers,
    ffun=ffun,
)

# Now let us combine the markers as well so that they align with the markers needed in
# fenicsx-ldrb, where the valves together make up the base

markers = {
    "lv": [m["LV_ENDO"]],
    "rv": [m["RV_ENDO"]],
    "epi": [m["EPI"]],
    "base": [m[valve] for valve in ["MV", "AV", "TV", "PV"]],
}

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

# and save them in the same file

cg.fibers.utils.save_microstructure(
    mesh=geometry.mesh, functions=(system.f0, system.s0, system.n0), path=outdir / "geometry.bp"
)

# Let us also plot the fibers
topology_f0, cell_types_f0, geometry_f0 = dolfinx.plot.vtk_mesh(
    system.f0.function_space
)
values = np.zeros((geometry_f0.shape[0], 3), dtype=np.float64)
values[:, : len(system.f0)] = system.f0.x.array.real.reshape(
    (geometry_f0.shape[0], len(system.f0))
)
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

geo = cg.geometry.Geometry.from_folder(comm=comm, folder=outdir)


# We have also implemented a wrapper around the `ukb-atlas` package that allows you to generate the
# mesh and the fibers directly

outdir = Path("ukb_mesh2")
geo = cg.mesh.ukb(
    outdir=outdir,
    comm=comm,
    char_length_max=2.0,
    char_length_min=2.0,
    fiber_angle_endo=60,
    fiber_angle_epi=-60,
)

# Since the UKB geometries include the outflow tracts, the fibers are by default generated with the
# rule-based method of [Doste et al. (2019)](https://doi.org/10.1002/cnm.3185) (see
# `ldrb.dolfinx_doste`), which defines the longitudinal direction in each ventricle from the apex
# to both the inflow and the outflow valve, and uses separate fiber angles in the RV and the
# outflow tracts. Here `fiber_angle_endo` and `fiber_angle_epi` are the angles in the LV. To use
# the method above instead, pass `fiber_method="ldrb"`, which is also the default for clipped
# meshes (`clipped=True`), as these have no outflow tracts

print(geo.info["fiber_method"])

# The wrapper can also create an LV-only geometry (`geometry="lv"`), and geometries where the
# valves are closed by caps (`geometry="biv-closed"` and `geometry="lv-closed"`). In the closed
# geometries the cavities are meshed as well, and are marked as `LV_CAVITY` and `RV_CAVITY` in the
# cell tags, while the fibers are only created in the wall (`Wall`)

geo_closed = cg.mesh.ukb(
    outdir=Path("ukb_mesh_lv_closed"),
    comm=comm,
    geometry="lv-closed",
    char_length_max=5.0,
    char_length_min=5.0,
)
print(geo_closed.markers)

# ## AHA segments for the UKB BiV
# With `aha=True` the cells are tagged with the 17 AHA segments of the LV (Cerqueira et al. 2002,
# https://doi.org/10.1161/hc0402.102975), the RV free wall (`RV`) and the part of the LV above the
# mitral plane (`LVOT`). The segments use anatomical landmarks that `ukb-atlas` writes next to the
# surfaces (`landmarks_ED.json`): the RV insertion line, where the RV septum meets the RV free wall,
# which every short-axis plane cuts at the anterior and the inferior insertion. The septum between
# the insertions is split into two segments and the rest of the wall into four (basal and mid
# levels), and the levels are thirds of the long axis from the apex to the mitral valve. This is
# only supported for the open BiV geometry (`geometry="biv"`, not clipped)

outdir_aha = Path("ukb_mesh_aha")
geo_aha = cg.mesh.ukb(
    outdir=outdir_aha,
    comm=comm,
    char_length_max=3.0,
    char_length_min=3.0,
    create_fibers=False,
    aha=True,
)
print(geo_aha.markers)

# Let us plot the LV segments (hiding the RV free wall, tag 18) from the RV side and from the
# opposite side, together with the RV insertion line (in black)

landmarks = json.loads((outdir_aha / "landmarks_ED.json").read_text())
insertion = np.array(landmarks["insertion"])
insertion_line = pyvista.lines_from_points(np.vstack([insertion, insertion[:1]]))

vtk_mesh_aha = dolfinx.plot.vtk_mesh(geo_aha.mesh, geo_aha.mesh.topology.dim)
grid_aha = pyvista.UnstructuredGrid(*vtk_mesh_aha)
grid_aha.cell_data["AHA"] = geo_aha.cfun.values
lv_aha = grid_aha.threshold([0.5, 17.5], scalars="AHA")
# Number each segment at the cell centre nearest to the mean of its cell centres
centres = lv_aha.cell_centers().points
tags = np.asarray(lv_aha.cell_data["AHA"])
segment_numbers = [str(i) for i in range(1, 18) if np.any(tags == i)]
segment_centres = []
for i in map(int, segment_numbers):
    c = centres[tags == i]
    segment_centres.append(c[np.argmin(np.linalg.norm(c - c.mean(axis=0), axis=1))])
segment_centres = np.array(segment_centres)
rv_centre = np.mean(landmarks["rv_freewall"], axis=0)
lv_centre = np.array(lv_aha.center)
plotter = pyvista.Plotter(shape=(1, 2), window_size=[1600, 800])
for col, side in enumerate((1.0, -1.0)):
    plotter.subplot(0, col)
    plotter.add_mesh(
        lv_aha, scalars="AHA", cmap="tab20", n_colors=17, clim=[0.5, 17.5], show_scalar_bar=False
    )
    plotter.add_mesh(insertion_line, color="black", line_width=6)
    # Label the segments on the side facing the camera
    facing = (segment_centres - lv_centre) @ (side * (rv_centre - lv_centre)) > 0
    plotter.add_point_labels(
        segment_centres[facing],
        [n for n, f in zip(segment_numbers, facing) if f],
        font_size=20,
        shape_opacity=0.6,
        always_visible=True,
    )
    plotter.camera_position = [
        lv_centre + side * (rv_centre - lv_centre),
        lv_centre,
        np.array(landmarks["valves"]["MV"]) - lv_centre,
    ]
    plotter.reset_camera()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("aha_segments_ukb.png")

# The same segments are also available from Python for any UKB BiV mesh, given the landmarks, with
# `cardiac_geometries.aha.biv_aha`, and from the command line with `geo ukb --aha`
