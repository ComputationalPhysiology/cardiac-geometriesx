# # Storing Additional Data in the Geometry
#
# This demo illustrates how to extend the standard `Geometry` storage format to include
# custom fields. We use a Bi-Ventricular (BiV) mesh from the UK Biobank Atlas as an example.
#
# We often need more than just the standard microstructure ($\mathbf{f}_0, \mathbf{s}_0, \mathbf{n}_0$).
# Common requirements include:
# 1.  **AHA Segments**: The 17-segment model for regional analysis.
# 2.  **Auxiliary Vectors**: Purely circumferential ($\mathbf{c}_0$) or longitudinal ($\mathbf{l}_0$) directions for strain analysis.
# 3.  **Custom Markers**: Patient-specific regions of interest (e.g., scars).
#
# We will generate these fields using `cardiac-geometries` and `fenicsx-ldrb`, save them
# alongside the mesh, and demonstrate how to reload them.
#
# ---

# ## Imports

import shutil
from pathlib import Path
import numpy as np
from mpi4py import MPI
import dolfinx
import ldrb
import scifem
import cardiac_geometries as cg

# ## 1. Geometry Generation
#
# We generate a mesh from the UK Biobank (UKB) statistical atlas.
# See [UKB Atlas Documentation](https://computationalphysiology.github.io/ukb-atlas) for details.

comm = MPI.COMM_WORLD
mode = -1  # Use the mean mode
std = 0
char_length = 5.0
outdir = Path("ukb_atlas_additional_data")
geodir = outdir / "geometry"

# Clean up previous geometry if it exists
if geodir.exists():
    shutil.rmtree(geodir)

geo = cg.mesh.ukb(
    outdir=geodir,
    comm=comm,
    mode=mode,
    std=std,
    case="ED",
    create_fibers=True,
    char_length_max=char_length,
    char_length_min=char_length,
    clipped=True,
    fiber_angle_endo=60,
    fiber_angle_epi=-60,
    fiber_space="Quadrature_6",
)

# Visualize the standard fiber field
try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    plotter = pyvista.Plotter()
    N = 5  # Only show every 5th point to avoid clutter
    assert geo.f0 is not None
    points = geo.f0.function_space.tabulate_dof_coordinates()
    point_cloud = pyvista.PolyData(points[::N, :])
    f0_arr = geo.f0.x.array.reshape((points.shape[0], 3))
    point_cloud["fibers"] = f0_arr[::N, :]
    fibers = point_cloud.glyph(
        orient="fibers",
        scale=False,
        factor=5.0,
    )
    plotter.add_mesh(fibers, color="red")
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(outdir / "f0.png")

# ## 2. Generating AHA Segments
#
# We use `fenicsx-ldrb` to generate the 17-segment American Heart Association (AHA) model.
# This assigns an integer ID to each cell in the mesh corresponding to its region (Basal Anterior, Septal, etc.).

# Generate AHA function (DG0 space)
aha = ldrb.aha.gernerate_aha_biv(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=cg.mesh.transform_markers(geo.markers, clipped=True),
    function_space="DG_0",
    base_max=0.75,
    mid_base=0.70,
    apex_mid=0.65,
)

# Convert the DG0 function to dolfinx.mesh.MeshTags
# This is often more convenient for post-processing and sub-domain integration.
entities = np.hstack(comm.allgather(np.arange(*geo.mesh.topology.index_map(3).local_range, dtype=np.int32)))
values = np.hstack(comm.allgather(aha.x.array.astype(np.int32)))

aha_mt = dolfinx.mesh.meshtags(
    geo.mesh, 3, entities, values,
)

# Visualize AHA segments
try:
    import pyvista
except ImportError:
    pass
else:
    vtk_mesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(*vtk_mesh)
    grid.cell_data["AHA"] = aha_mt.values
    grid.set_active_scalars("AHA")

    p = pyvista.Plotter(window_size=[800, 800])
    p.add_mesh(grid, show_edges=True, cmap="tab20")
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "aha_segments.png")

# Note however that the markers near the base is a little bit off in this case.

# ## 3. Generating Custom Fiber Vectors
#
# For strain analysis, it is useful to have vector fields representing the purely
# **Circumferential** ($\mathbf{c}_0$) and **Longitudinal** ($\mathbf{l}_0$) directions,
# separate from the helical myofibers.
#
# We can generate these by running `ldrb` with specific angles ($0^\circ$ and $90^\circ$).

fiber_space = "DG_1"

# ### Circumferential Vectors (0 degrees)

system_c = ldrb.dolfinx_ldrb(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=cg.mesh.transform_markers(geo.markers, clipped=True),
    alpha_endo_lv=0,
    alpha_epi_lv=0,
    fiber_space=fiber_space,
)
c0 = system_c.f0
c0.name = "c0"

# Visualize Circumferential Vectors
try:
    import pyvista
except ImportError:
    pass
else:
    plotter = pyvista.Plotter()
    points = c0.function_space.tabulate_dof_coordinates()
    point_cloud = pyvista.PolyData(points[::N, :])
    c0_arr = c0.x.array.reshape((points.shape[0], 3))
    point_cloud["fibers"] = c0_arr[::N, :]
    fibers = point_cloud.glyph(orient="fibers", scale=False, factor=5.0)
    plotter.add_mesh(fibers, color="blue")
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(outdir / "c0.png")

# ### Longitudinal Vectors (90 degrees)

system_l = ldrb.dolfinx_ldrb(
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=cg.mesh.transform_markers(geo.markers, clipped=True),
    alpha_endo_lv=90,
    alpha_epi_lv=90,
    epi_only=True, # Optimization: Longitudinal direction is roughly constant transmurally
    fiber_space=fiber_space,
)
l0 = system_l.f0
l0.name = "l0"

# Visualize Longitudinal Vectors
try:
    import pyvista
except ImportError:
    pass
else:
    plotter = pyvista.Plotter()
    points = l0.function_space.tabulate_dof_coordinates()
    point_cloud = pyvista.PolyData(points[::N, :])
    l0_arr = l0.x.array.reshape((points.shape[0], 3))
    point_cloud["fibers"] = l0_arr[::N, :]
    fibers = point_cloud.glyph(orient="fibers", scale=False, factor=5.0)
    plotter.add_mesh(fibers, color="blue")
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(outdir / "l0.png")

# ## 4. Saving Geometry with Additional Data
#
# We now bundle our custom fields into a dictionary and pass it to `save_geometry`.
# This function serializes the mesh, standard markers, standard fibers, *and* our additional data
# into a consistent folder structure (using HDF5/XDMF for mesh tags and ADIOS2 for functions).

additional_data = {
    "aha_segments": aha_mt,  # MeshTags
    "c0": c0,                # Function
    "l0": l0,                # Function
}

# Overwrite the existing geometry.bp folder to include the new data
# Note: We use the existing mesh/markers to ensure consistency and delete the old data first
shutil.rmtree(geodir / "geometry.bp")
cg.geometry.save_geometry(
    path=geodir / "geometry.bp",
    mesh=geo.mesh,
    ffun=geo.ffun,
    markers=geo.markers,
    info=geo.info,
    f0=geo.f0,
    s0=geo.s0,
    n0=geo.n0,
    additional_data=additional_data,
)

# ## 5. Reloading and Verification
#
# We load the geometry from the folder. The `Geometry` object will now populate its
# `additional_data` attribute with our custom fields.

geo_loaded = cg.geometry.Geometry.from_folder(
    comm=comm,
    folder=geodir,
)

print("Loaded keys:", geo_loaded.additional_data.keys())

# Verify AHA Segments
try:
    import pyvista
except ImportError:
    pass
else:
    # Access the loaded MeshTags
    loaded_aha = geo_loaded.additional_data["aha_segments"]

    vtk_mesh = dolfinx.plot.vtk_mesh(geo_loaded.mesh, geo_loaded.mesh.topology.dim)
    grid = pyvista.UnstructuredGrid(*vtk_mesh)
    grid.cell_data["AHA"] = loaded_aha.values
    grid.set_active_scalars("AHA")

    p = pyvista.Plotter(window_size=[800, 800])
    p.add_mesh(grid, show_edges=True, cmap="tab20")
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        p.screenshot(outdir / "loaded_aha_segments.png")

# Verify Circumferential Vectors
try:
    import pyvista
except ImportError:
    pass
else:
    # Access the loaded Function
    loaded_c0 = geo_loaded.additional_data["c0"]
    assert isinstance(loaded_c0, dolfinx.fem.Function)

    plotter = pyvista.Plotter()
    points = loaded_c0.function_space.tabulate_dof_coordinates()
    point_cloud = pyvista.PolyData(points[::N, :])
    c0_arr = loaded_c0.x.array.reshape((points.shape[0], 3))
    point_cloud["fibers"] = c0_arr[::N, :]
    fibers = point_cloud.glyph(orient="fibers", scale=False, factor=5.0)
    plotter.add_mesh(fibers, color="green")

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(outdir / "loaded_c0.png")

# Verify Longitudinal Vectors
try:
    import pyvista
except ImportError:
    pass
else:
    # Access the loaded Function
    loaded_l0 = geo_loaded.additional_data["l0"]
    assert isinstance(loaded_l0, dolfinx.fem.Function)

    plotter = pyvista.Plotter()
    points = loaded_l0.function_space.tabulate_dof_coordinates()
    point_cloud = pyvista.PolyData(points[::N, :])
    f0_arr = loaded_l0.x.array.reshape((points.shape[0], 3))
    point_cloud["fibers"] = f0_arr[::N, :]
    fibers = point_cloud.glyph(orient="fibers", scale=False, factor=5.0)
    plotter.add_mesh(fibers, color="green")

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(outdir / "loaded_l0.png")

# Also make sure the fiber fields still looks good

# Visualize the standard fiber field
try:
    import pyvista
except ImportError:
    print("Pyvista is not installed")
else:
    plotter = pyvista.Plotter()
    N = 5  # Only show every 5th point to avoid clutter
    assert geo_loaded.f0 is not None
    points = geo_loaded.f0.function_space.tabulate_dof_coordinates()
    point_cloud = pyvista.PolyData(points[::N, :])
    f0_arr = geo_loaded.f0.x.array.reshape((points.shape[0], 3))
    point_cloud["fibers"] = f0_arr[::N, :]
    fibers = point_cloud.glyph(
        orient="fibers",
        scale=False,
        factor=5.0,
    )
    plotter.add_mesh(fibers, color="red")
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(outdir / "f0.png")
