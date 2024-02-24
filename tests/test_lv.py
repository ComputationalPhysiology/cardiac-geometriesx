import cardiac_geometries.lv as lv
import cardiac_geometries.fibers as fibers
import dolfinx


def test_geo():
    msh_name = "lv.msh"
    # lv.create_benchmark_ellipsoid_mesh_gmsh(msh_name)
    geo = lv.gmsh2dolfin(msh_name)
    micro = fibers.create_microstructure(geo.mesh, geo.ffun, geo.markers)

    with dolfinx.io.XDMFFile(geo.mesh.comm, "dolfinx_micro.xdmf", "w") as xdmf:
        xdmf.write_mesh(geo.mesh)
        xdmf.write_function(micro.f0)
        xdmf.write_function(micro.s0)
        xdmf.write_function(micro.n0)
