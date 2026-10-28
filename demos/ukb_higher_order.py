# # Higher order meshes from the UK Biobank atlas
#
# In this demo we compare linear and quadratic meshes created from the UK Biobank atlas (see the
# [UK Biobank demo](ukb_atlas.py) for an introduction to the atlas, and the
# [higher order demo](higher_order.py) for higher order meshes in general).
#
# The surfaces of the atlas are triangulations with sides of 3-4 mm. A linear mesh can never be
# more accurate than these triangles, and to resolve the curvature of the heart well you need a
# fine mesh. With quadratic elements the cells are curved, but the nodes on the boundary can only
# be as accurate as the surfaces they are placed on. Therefore we can also refine the surfaces
# with `refine`, which splits each triangle of the atlas into four `refine` times, and places the
# new nodes on a smooth surface through the nodes of the atlas.
#
# We will create the biventricular geometry with closed valves, so that we can also measure the
# volumes of the cavities, using three different meshes
#
# - a coarse linear mesh with a characteristic length of 10 mm
# - a fine linear mesh with a characteristic length of 2.5 mm
# - a coarse quadratic mesh with a characteristic length of 10 mm, of the surfaces refined twice
#
# We also create fibers with the rule-based method of
# [Doste et al. (2019)](https://doi.org/10.1002/cnm.3185), which handles the outflow tracts of the
# atlas. The fibers are represented in a space of the same degree as the mesh.

from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np
import pyvista
import ufl
import ukb.atlas
import ukb.smooth
import ukb.template

import cardiac_geometries

comm = MPI.COMM_WORLD
meshes = {
    "linear, 10 mm": dict(order=1, char_length=10.0, refine=0, fiber_space="P_1"),
    "linear, 2.5 mm": dict(order=1, char_length=2.5, refine=0, fiber_space="P_1"),
    "quadratic, 10 mm": dict(order=2, char_length=10.0, refine=2, fiber_space="P_2"),
}

geos = {}
for name, kwargs in meshes.items():
    geodir = Path("ukb_biv_closed_" + name.replace(", ", "_").replace(" ", ""))
    if not geodir.exists():
        cardiac_geometries.mesh.ukb(
            outdir=geodir,
            comm=comm,
            geometry="biv-closed",
            order=kwargs["order"],
            char_length_max=kwargs["char_length"],
            char_length_min=kwargs["char_length"],
            refine=kwargs["refine"],
            fiber_method="doste",
            fiber_space=kwargs["fiber_space"],
        )
    geos[name] = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)

# Let us first look at the size of the meshes. Since you would typically use quadratic elements
# for the displacement in a mechanics simulation, we also print the number of degrees of freedom
# of a vector valued quadratic function space on the wall


def wall_cells(geo):
    assert geo.cfun is not None
    return geo.cfun.find(geo.markers["Wall"][0])


for name, geo in geos.items():
    num_cells = comm.allreduce(
        (wall_cells(geo) < geo.mesh.topology.index_map(3).size_local).sum(),
    )
    wall = dolfinx.mesh.create_submesh(geo.mesh, 3, wall_cells(geo))[0]
    V = dolfinx.fem.functionspace(wall, ("Lagrange", 2, (3,)))
    num_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"{name}: {num_cells} cells in the wall and {num_dofs} P2 degrees of freedom")

# The two coarse meshes have about the same size, while the fine linear mesh has six to seven
# times as many cells and degrees of freedom.
#
# Let us plot the coarse linear and the coarse quadratic mesh next to each other. The cells of the
# quadratic mesh are curved, so we subdivide them when we extract the surface to plot.


def wall_surface(geo, subdivision=0):
    wall = dolfinx.mesh.create_submesh(geo.mesh, 3, wall_cells(geo))[0]
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(wall))
    return grid.extract_surface(nonlinear_subdivision=subdivision)


plotter = pyvista.Plotter(shape=(1, 2))
for i, name in enumerate(["linear, 10 mm", "quadratic, 10 mm"]):
    plotter.subplot(0, i)
    plotter.add_text(name)
    plotter.add_mesh(wall_surface(geos[name], subdivision=4), color="lightcoral")
    plotter.view_xy()
plotter.link_views()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("ukb_higher_order.png")

# ## Comparing with the smooth surfaces
#
# As reference we use the surfaces of the atlas refined three times, which are within 0.01 mm of
# the smooth surfaces. The surfaces are generated from the same atlas shape as the meshes above,
# i.e. the mean shape (`mode=-1`) at end-diastole

points = ukb.atlas.generate_points(
    ukb.atlas.download_atlas(Path.home() / ".ukb"), mode=-1, std=1.5
).ED
refined_points, refined_template = ukb.smooth.refine(points, 3)


def reference_surface(regions):
    faces = refined_template.faces[np.concatenate([refined_template.regions[r] for r in regions])]
    return pyvista.PolyData.from_regular_faces(refined_points, faces)


# The triangles of the template are not oriented consistently, so before we compute the volume
# enclosed by a surface we let pyvista orient the normals


def enclosed_volume(regions):
    surface = reference_surface(regions).clean()
    return surface.compute_normals(auto_orient_normals=True).volume


caps = ["MV_CAP", "AV_CAP", "TV_CAP", "PV_CAP"]
rims = ["MV", "AV", "TV", "PV"]
reference = {
    "LV_CAVITY": enclosed_volume(["LV_ENDO", "MV_CAP", "AV_CAP"]),
    "RV_CAVITY": enclosed_volume(["RV_SEPTUM", "RV_FREEWALL", "TV_CAP", "PV_CAP"]),
}
reference["Wall"] = (
    enclosed_volume(["LV_EPI", "RV_EPI"] + rims + caps)
    - reference["LV_CAVITY"]
    - reference["RV_CAVITY"]
)

# Now we compute the volume of the wall and of the two cavities by integrating over the cells
# with the corresponding tags, and compare with the reference

for name, geo in geos.items():
    dx = ufl.Measure("dx", domain=geo.mesh, subdomain_data=geo.cfun)
    errors = []
    for region, volume in reference.items():
        value = comm.allreduce(
            dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(1.0 * dx(geo.markers[region][0])),
            ),
        )
        errors.append(f"{region} {abs(value - volume) / volume:.1e}")
    print(f"{name}: relative error in the volume of " + ", ".join(errors))

# The coarse quadratic mesh is more accurate than the fine linear mesh, for all three volumes.
#
# We can also look at where the meshes differ from the smooth surfaces, by computing the distance
# from the boundary of each mesh to the reference surfaces

reference_grid = reference_surface(
    ["LV_ENDO", "RV_SEPTUM", "RV_FREEWALL", "LV_EPI", "RV_EPI"] + rims,
)

plotter = pyvista.Plotter(shape=(1, 3))
for i, name in enumerate(geos):
    surface = wall_surface(geos[name], subdivision=3)
    surface.compute_implicit_distance(reference_grid, inplace=True)
    surface["distance (mm)"] = np.abs(surface["implicit_distance"])
    distance = surface["distance (mm)"]
    print(f"{name}: mean distance {distance.mean():.3f} mm, maximum {distance.max():.2f} mm")
    plotter.subplot(0, i)
    plotter.add_text(name, font_size=10)
    plotter.add_mesh(
        surface, scalars="distance (mm)", clim=[0, 0.5], cmap="viridis", copy_mesh=True
    )
    plotter.view_xy()
    plotter.camera.zoom(1.5)
plotter.link_views()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("ukb_higher_order_distance.png")

# The linear meshes cut through the curved surfaces between the nodes. For the coarse linear mesh
# this gives errors of up to almost 2 mm where the surfaces are most curved, and even the fine
# linear mesh is typically off by about 0.06 mm. The quadratic mesh follows the smooth surfaces
# about ten times closer than the fine linear mesh, and the largest errors are found where the
# valves meet.
#
# ## Fibers
#
# The fibers are computed from the gradients of solutions of Laplace problems, which
# `fenicsx-ldrb` solves with Lagrange elements of the same degree as the mesh. On the quadratic
# mesh both the geometry and the Laplace solutions are therefore more accurate than on the coarse
# linear mesh, and so are the fibers.
#
# To see this, we compare the fibers on the two coarse meshes with the fibers on the fine linear
# mesh. We evaluate all the fibers at a subset of the nodes of the fine mesh, and compute the angle
# between them. Locating points in the curved cells of the quadratic mesh needs a few more Newton
# iterations than the default.


def evaluate(f, points):
    """Evaluate `f` at `points`, with NaN for the points outside the (local) mesh"""
    mesh = f.function_space.mesh
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, points)
    values = np.full((len(points), 3), np.nan)
    found = np.flatnonzero(np.diff(colliding.offsets) > 0)
    cells = colliding.array[colliding.offsets[found]]
    values[found] = f.eval(points[found], cells, tol=1e-8, maxit=50)
    return values


reference = geos["linear, 2.5 mm"].f0
assert reference is not None
points = reference.function_space.tabulate_dof_coordinates()[::5]
fibers = {name: evaluate(geo.f0, points) for name, geo in geos.items()}
# Use the points in the wall of all the meshes (the fibers are zero in the cavities)
inside = np.all([np.linalg.norm(f, axis=1) > 0.5 for f in fibers.values()], axis=0)
f_ref = fibers["linear, 2.5 mm"][inside]
for name in ["linear, 10 mm", "quadratic, 10 mm"]:
    f = fibers[name][inside]
    cos = (
        np.abs(np.sum(f * f_ref, axis=1))
        / np.linalg.norm(f, axis=1)
        / np.linalg.norm(f_ref, axis=1)
    )
    angle = np.degrees(np.arccos(np.clip(cos, 0, 1)))
    print(
        f"{name}: angle to the fibers on the fine linear mesh, median {np.median(angle):.1f} "
        f"degrees and 90th percentile {np.percentile(angle, 90):.1f} degrees",
    )

# The fibers on the coarse quadratic mesh are much closer to the fibers on the fine mesh.
#
# Finally, let us plot the fibers on the quadratic mesh

geo = geos["quadratic, 10 mm"]
assert geo.f0 is not None
topology, cell_types, x = dolfinx.plot.vtk_mesh(geo.f0.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cell_types, x)
function_grid["f0"] = geo.f0.x.array.real.reshape((x.shape[0], 3))
# Plot only some of the fibers, so that the plot is not too cluttered
glyphs = function_grid.glyph(orient="f0", factor=4.0, tolerance=0.02)
plotter = pyvista.Plotter()
plotter.add_mesh(wall_surface(geo, subdivision=3), color="lightgrey", opacity=0.3)
plotter.add_mesh(glyphs, show_scalar_bar=False)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("ukb_higher_order_fibers.png")
