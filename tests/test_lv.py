import cardiac_geometries.lv as lv
import cardiac_geometries.fibers as fibers
import dolfinx
from pathlib import Path
import json
import adios4dolfinx


def test_geo():
    msh_name = "lv.msh"
    lv.lv_ellipsoid_flat_base(msh_name)
    geo = lv.gmsh2dolfin(msh_name)
    micro = fibers.create_microstructure(geo.mesh, geo.ffun, geo.markers)

    with dolfinx.io.XDMFFile(geo.mesh.comm, "dolfinx_micro.xdmf", "w") as xdmf:
        xdmf.write_mesh(geo.mesh)
        xdmf.write_function(micro.f0)
        xdmf.write_function(micro.s0)
        xdmf.write_function(micro.n0)

    folder = Path("test_data")
    folder.mkdir(exist_ok=True)
    adios4dolfinx.write_mesh(mesh=geo.mesh, filename=folder / "mesh.bp")
    adios4dolfinx.write_meshtags(meshtags=geo.ffun, mesh=geo.mesh, filename=folder / "ffun.bp")
    adios4dolfinx.write_function(u=micro.f0, filename=folder / "f0.bp")
    adios4dolfinx.write_function(u=micro.s0, filename=folder / "s0.bp")
    adios4dolfinx.write_function(u=micro.n0, filename=folder / "n0.bp")
    if geo.mesh.comm.rank == 0:
        Path(folder / "markers.json").write_text(
            json.dumps({k: [int(vi) for vi in v] for k, v in geo.markers.items()})
        )
