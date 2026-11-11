# # Patient meshes from another tool
#
# Meshes of patient hearts are often made with other tools and shared as a tetrahedral volume mesh
# together with the boundary surfaces (endocardium, epicardium, base, valves) as separate files. With
# `cardiac_geometries.mesh.from_volume` such a mesh becomes a geometry with facet markers, and
# optionally fibers, like the other geometries in this package. Every boundary face of the volume must
# lie on exactly one of the surfaces; the surfaces are matched to the volume by their nodes.
#
# Use the marker names of the UK Biobank geometries (`LV_ENDO`, `RV_ENDO`, `EPI`, `BASE`, `MV`, `AV`,
# `TV`, `PV`), so that the fiber methods find their boundaries. Here we make a small example: a thick
# spherical shell, cut at the top, standing in for an LV.

from pathlib import Path

from mpi4py import MPI

import dolfinx
import gmsh
import meshio
import numpy as np
import pyvista

import cardiac_geometries as cg
from cardiac_geometries.labelled import boundary_faces

comm = MPI.COMM_WORLD
outdir = Path("patient_mesh")
outdir.mkdir(exist_ok=True)

# First we make the "patient" files: a volume mesh and three surfaces, as another tool would give them

if comm.rank == 0:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    outer = gmsh.model.occ.addSphere(0, 0, 0, 20.0)
    inner = gmsh.model.occ.addSphere(0, 0, 0, 10.0)
    shell, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)])
    top = gmsh.model.occ.addBox(-30, -30, 5.0, 60, 60, 30)
    gmsh.model.occ.cut(shell, [(3, top)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.5)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    index = np.empty(int(tags.max()) + 1, dtype=np.int64)
    index[tags.astype(np.int64)] = np.arange(len(tags))
    points = coords.reshape(-1, 3)
    tets = index[gmsh.model.mesh.getElementsByType(4)[1].astype(np.int64)].reshape(-1, 4)
    gmsh.finalize()
    meshio.write(outdir / "volume.vtu", meshio.Mesh(points, [("tetra", tets)]))
    faces = boundary_faces(tets)
    centre = points[faces].mean(axis=1)
    names = np.where(
        np.isclose(centre[:, 2], 5.0),
        "BASE",
        np.where(np.linalg.norm(centre, axis=1) < 15.0, "LV_ENDO", "EPI"),
    )
    for name in ("LV_ENDO", "EPI", "BASE"):
        meshio.write(
            outdir / f"{name.lower()}.ply", meshio.Mesh(points, [("triangle", faces[names == name])])
        )
comm.barrier()

# Now we import it, with fibers by the method of Bayer et al.

geo = cg.mesh.from_volume(
    outdir / "geometry",
    outdir / "volume.vtu",
    surfaces={name: outdir / f"{name.lower()}.ply" for name in ("LV_ENDO", "EPI", "BASE")},
    create_fibers=True,
    fiber_space="P_1",
)
print(geo.markers)

# The same from the command line:
#
# ```
# geo from-volume patient_mesh/geometry patient_mesh/volume.vtu \
#     --surface LV_ENDO=patient_mesh/lv_endo.ply --surface EPI=patient_mesh/epi.ply \
#     --surface BASE=patient_mesh/base.ply --create-fibers
# ```
#
# A mesh in other units can be scaled with `scale` (`--scale`), e.g. `scale=1e-3` from micrometres to
# millimetres, and surfaces whose nodes do not coincide exactly with the volume's can be matched within
# a distance `tol` (`--tol`). Instead of surface files, a rule on the boundary faces can label them
# (`rule=...`, from Python only).
#
# Let us plot the facet markers and the fibers

vtk_mesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.mesh.topology.dim - 1, geo.ffun.indices)
grid = pyvista.UnstructuredGrid(*vtk_mesh)
grid.cell_data["Marker"] = geo.ffun.values
V = geo.f0.function_space
topology, cell_types, x = dolfinx.plot.vtk_mesh(V)
fibers = pyvista.UnstructuredGrid(topology, cell_types, x)
fibers["f0"] = geo.f0.x.array.reshape(-1, 3)
glyphs = fibers.glyph(orient="f0", factor=1.5, tolerance=0.02)
plotter = pyvista.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(grid, scalars="Marker", cmap="tab10", show_edges=False)
plotter.subplot(0, 1)
plotter.add_mesh(glyphs, color="firebrick")
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    plotter.screenshot("patient_mesh.png")
