from pathlib import Path
import tempfile
from typing import NamedTuple, Iterable
from enum import Enum

import numpy as np
from mpi4py import MPI
from structlog import get_logger
import basix
import dolfinx
import meshio


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


def create_mesh(mesh, cell_type):
    # From http://jsdokken.com/converted_files/tutorial_pygmsh.html
    cells = mesh.get_cells_type(cell_type)
    if cells.size == 0:
        return None

    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    return out_mesh


def read_ffun(mesh, filename: str | Path) -> dolfinx.mesh.MeshTags | None:
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with dolfinx.io.XDMFFile(mesh.comm, filename, "r") as xdmf:
        ffun = xdmf.read_meshtags(mesh, name="Grid")
    ffun.name = "Facet tags"
    return ffun


def read_mesh(comm, filename: str | Path) -> tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags | None]:
    with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cfun = xdmf.read_meshtags(mesh, name="Grid")
    return mesh, cfun


def gmsh2dolfin(comm: MPI.Intracomm, msh_file, unlink: bool = False) -> GMshGeometry:
    logger.debug(f"Convert file {msh_file} to dolfin")
    outdir = Path(msh_file).parent

    vertex_mesh_name = outdir / "vertex_mesh.xdmf"
    line_mesh_name = outdir / "line_mesh.xdmf"
    triangle_mesh_name = outdir / "triangle_mesh.xdmf"
    tetra_mesh_name = outdir / "mesh.xdmf"

    if comm.rank == 0:
        msh = meshio.gmsh.read(msh_file)
        vertex_mesh = create_mesh(msh, "vertex")
        line_mesh = create_mesh(msh, "line")
        triangle_mesh = create_mesh(msh, "triangle")
        tetra_mesh = create_mesh(msh, "tetra")

        if vertex_mesh is not None:
            meshio.write(vertex_mesh_name, vertex_mesh)

        if line_mesh is not None:
            meshio.write(line_mesh_name, line_mesh)

        if triangle_mesh is not None:
            meshio.write(triangle_mesh_name, triangle_mesh)

        if tetra_mesh is not None:
            meshio.write(
                tetra_mesh_name,
                tetra_mesh,
            )
        markers = msh.field_data
    else:
        markers = {}
    # Broadcast markers
    markers = comm.bcast(markers, root=0)
    comm.barrier()

    mesh, cfun = read_mesh(comm, tetra_mesh_name)

    if triangle_mesh_name.is_file():
        ffun = read_ffun(mesh, triangle_mesh_name)
    else:
        ffun = None

    if line_mesh_name.is_file():
        mesh.topology.create_connectivity(mesh.topology.dim - 2, mesh.topology.dim)
        with dolfinx.io.XDMFFile(comm, line_mesh_name, "r") as xdmf:
            efun = xdmf.read_meshtags(mesh, name="Grid")
    else:
        efun = None

    if vertex_mesh_name.is_file():
        mesh.topology.create_connectivity(mesh.topology.dim - 3, mesh.topology.dim)
        with dolfinx.io.XDMFFile(comm, vertex_mesh_name, "r") as xdmf:
            vfun = xdmf.read_meshtags(mesh, name="Grid")
    else:
        vfun = None

    if unlink:
        # Wait for all processes to finish reading
        comm.barrier()
        if comm.rank == 0:
            vertex_mesh_name.unlink(missing_ok=True)
            line_mesh_name.unlink(missing_ok=True)
            triangle_mesh_name.unlink(missing_ok=True)
            tetra_mesh_name.unlink(missing_ok=True)
            vertex_mesh_name.with_suffix(".h5").unlink(missing_ok=True)
            line_mesh_name.with_suffix(".h5").unlink(missing_ok=True)
            triangle_mesh_name.with_suffix(".h5").unlink(missing_ok=True)
            tetra_mesh_name.with_suffix(".h5").unlink(missing_ok=True)

    return GMshGeometry(mesh, cfun, ffun, efun, vfun, markers)
