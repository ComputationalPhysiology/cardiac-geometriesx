from pathlib import Path
import tempfile
from typing import NamedTuple, Iterable
from enum import Enum

import numpy as np
from mpi4py import MPI
from structlog import get_logger
import basix
import dolfinx


logger = get_logger()


def element2array(el: basix.finite_element.FiniteElement) -> np.ndarray:
    return np.array(
        [int(el.family), int(el.cell_type), int(el.degree), int(el.discontinuous)],
        dtype=np.uint8,
    )


def number2Enum(num: int, enum: Iterable) -> Enum:
    for e in enum:
        if int(e) == num:
            return e
    raise ValueError(f"Invalid value {num} for enum {enum}")


def array2element(arr: np.ndarray) -> basix.finite_element.FiniteElement:
    family = number2Enum(arr[0], basix.ElementFamily)
    cell_type = number2Enum(arr[1], basix.CellType)
    degree = int(arr[2])
    discontinuous = bool(arr[3])
    # TODO: Shape is hardcoded to (3,) for now, but this should also be stored
    return basix.ufl.element(
        family=family,
        cell=cell_type,
        degree=degree,
        discontinuous=discontinuous,
        shape=(3,),
    )


def handle_mesh_name(mesh_name: str = "") -> Path:
    if mesh_name == "":
        fd, mesh_name = tempfile.mkstemp(suffix=".msh")
    return Path(mesh_name).with_suffix(".msh")


def json_serial(obj):
    if isinstance(obj, (np.ndarray)):
        return obj.tolist()
    else:
        try:
            return str(obj)
        except Exception:
            raise TypeError("Type %s not serializable" % type(obj))


class GMshGeometry(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    cfun: dolfinx.mesh.MeshTags | None
    ffun: dolfinx.mesh.MeshTags | None
    efun: dolfinx.mesh.MeshTags | None
    vfun: dolfinx.mesh.MeshTags | None
    markers: dict[str, tuple[int, int]]


def read_ffun(mesh, filename: str | Path) -> dolfinx.mesh.MeshTags | None:
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with dolfinx.io.XDMFFile(mesh.comm, filename, "r") as xdmf:
        ffun = xdmf.read_meshtags(mesh, name="Facet tags")

    return ffun


def read_mesh(comm, filename: str | Path) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags | None]:
    with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Mesh")
        cfun = xdmf.read_meshtags(mesh, name="Cell tags")
    return mesh, cfun


def gmsh2dolfin(comm: MPI.Intracomm, msh_file, rank: int = 0) -> GMshGeometry:
    logger.debug(f"Convert file {msh_file} to dolfin")
    outdir = Path(msh_file).parent

    import gmsh

    # We could make this work in parallel in the future

    if comm.rank == rank:
        gmsh.initialize()
        gmsh.model.add("Mesh from file")
        gmsh.merge(str(msh_file))
        mesh, ct, ft = dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm, 0)
        markers = {gmsh.model.getPhysicalName(*v): v for v in gmsh.model.getPhysicalGroups()}
        gmsh.finalize()
    else:
        mesh, ct, ft = dolfinx.io.gmshio.model_to_mesh(gmsh.model, comm, 0)
        markers = {}
    mesh.name = "Mesh"
    ct.name = "Cell tags"
    ft.name = "Facet tags"

    markers = comm.bcast(markers, root=rank)

    # Save tags to xdmf
    with dolfinx.io.XDMFFile(comm, outdir / "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        mesh.topology.create_connectivity(2, 3)
        xdmf.write_meshtags(
            ct, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
        )
        xdmf.write_meshtags(
            ft, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
        )

    vfun = None
    efun = None

    return GMshGeometry(mesh, ct, ft, efun, vfun, markers)
