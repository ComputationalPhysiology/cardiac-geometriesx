import tempfile
import typing
from enum import Enum
from pathlib import Path
from typing import Iterable, NamedTuple

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
from structlog import get_logger

logger = get_logger()

quads = ("Quadrature", "Q", "Quad", "quadrature", "q", "quad")
QUADNR = 42


class GMshModel(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    cell_tags: dolfinx.mesh.MeshTags
    facet_tags: dolfinx.mesh.MeshTags
    edge_tags: dolfinx.mesh.MeshTags
    vertex_tags: dolfinx.mesh.MeshTags


# copied from https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/io/gmshio.py
def model_to_mesh(
    model,
    comm: MPI.Comm,
    rank: int,
    gdim: int = 3,
    partitioner: typing.Optional[
        typing.Callable[
            [MPI.Comm, int, int, dolfinx.cpp.graph.AdjacencyList_int32],
            dolfinx.cpp.graph.AdjacencyList_int32,
        ]
    ] = None,
    dtype=dolfinx.default_real_type,
) -> GMshModel:
    """Create a Mesh from a Gmsh model.

    Creates a :class:`dolfinx.mesh.Mesh` from the physical entities of
    the highest topological dimension in the Gmsh model. In parallel,
    the gmsh model is processed on one MPI rank, and the
    :class:`dolfinx.mesh.Mesh` is distributed across ranks.

    Args:
        model: Gmsh model.
        comm: MPI communicator to use for mesh creation.
        rank: MPI rank that the Gmsh model is initialized on.
        gdim: Geometrical dimension of the mesh.
        partitioner: Function that computes the parallel
            distribution of cells across MPI ranks.

    Returns:
        A tuple containing the :class:`dolfinx.mesh.Mesh` and
        :class:`dolfinx.mesh.MeshTags` for cells, facets, edges and
        vertices.


    Note:
        For performance, this function should only be called once for
        large problems. For re-use, it is recommended to save the mesh
        and corresponding tags using :class:`dolfinxio.XDMFFile` after
        creation for efficient access.

    """
    if comm.rank == rank:
        assert model is not None, "Gmsh model is None on rank responsible for mesh creation."
        # Get mesh geometry and mesh topology for each element
        x = dolfinx.io.gmshio.extract_geometry(model)
        topologies = dolfinx.io.gmshio.extract_topology_and_markers(model)

        # Extract Gmsh cell id, dimension of cell and number of nodes to
        # cell for each
        num_cell_types = len(topologies.keys())
        cell_information = dict()
        cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)
        for i, element in enumerate(topologies.keys()):
            _, dim, _, num_nodes, _, _ = model.mesh.getElementProperties(element)
            cell_information[i] = {"id": element, "dim": dim, "num_nodes": num_nodes}
            cell_dimensions[i] = dim

        # Sort elements by ascending dimension
        perm_sort = np.argsort(cell_dimensions)

        # Broadcast cell type data and geometric dimension
        cell_id = cell_information[perm_sort[-1]]["id"]
        tdim = cell_information[perm_sort[-1]]["dim"]
        num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
        cell_id, num_nodes = comm.bcast([cell_id, num_nodes], root=rank)

        # Check for facet data and broadcast relevant info if True
        has_facet_data = False
        if tdim - 1 in cell_dimensions:
            has_facet_data = True
        has_edge_data = False
        if tdim - 2 in cell_dimensions:
            has_edge_data = True
        has_vertex_data = False
        if tdim - 3 in cell_dimensions:
            has_vertex_data = True

        has_facet_data = comm.bcast(has_facet_data, root=rank)
        if has_facet_data:
            num_facet_nodes = comm.bcast(cell_information[perm_sort[-2]]["num_nodes"], root=rank)
            gmsh_facet_id = cell_information[perm_sort[-2]]["id"]
            marked_facets = np.asarray(topologies[gmsh_facet_id]["topology"], dtype=np.int64)
            facet_values = np.asarray(topologies[gmsh_facet_id]["cell_data"], dtype=np.int32)

        has_edge_data = comm.bcast(has_edge_data, root=rank)
        if has_edge_data:
            num_edge_nodes = comm.bcast(cell_information[perm_sort[-3]]["num_nodes"], root=rank)
            gmsh_edge_id = cell_information[perm_sort[-3]]["id"]
            marked_edges = np.asarray(topologies[gmsh_edge_id]["topology"], dtype=np.int64)
            edge_values = np.asarray(topologies[gmsh_edge_id]["cell_data"], dtype=np.int32)

        has_vertex_data = comm.bcast(has_vertex_data, root=rank)
        if has_vertex_data:
            num_vertex_nodes = comm.bcast(cell_information[perm_sort[-4]]["num_nodes"], root=rank)
            gmsh_vertex_id = cell_information[perm_sort[-4]]["id"]
            marked_vertices = np.asarray(topologies[gmsh_vertex_id]["topology"], dtype=np.int64)
            vertex_values = np.asarray(topologies[gmsh_vertex_id]["cell_data"], dtype=np.int32)

        cells = np.asarray(topologies[cell_id]["topology"], dtype=np.int64)
        cell_values = np.asarray(topologies[cell_id]["cell_data"], dtype=np.int32)
    else:
        cell_id, num_nodes = comm.bcast([None, None], root=rank)
        cells, x = np.empty([0, num_nodes], dtype=np.int32), np.empty([0, gdim], dtype=dtype)
        cell_values = np.empty((0,), dtype=np.int32)

        has_facet_data = comm.bcast(None, root=rank)
        if has_facet_data:
            num_facet_nodes = comm.bcast(None, root=rank)
            marked_facets = np.empty((0, num_facet_nodes), dtype=np.int32)
            facet_values = np.empty((0,), dtype=np.int32)

        has_edge_data = comm.bcast(None, root=rank)
        if has_edge_data:
            num_edge_nodes = comm.bcast(None, root=rank)
            marked_edges = np.empty((0, num_edge_nodes), dtype=np.int32)
            edge_values = np.empty((0,), dtype=np.int32)

        has_vertex_data = comm.bcast(None, root=rank)
        if has_vertex_data:
            num_vertex_nodes = comm.bcast(None, root=rank)
            marked_vertices = np.empty((0, num_vertex_nodes), dtype=np.int32)
            vertex_values = np.empty((0,), dtype=np.int32)

    # Create distributed mesh
    ufl_domain = dolfinx.io.gmshio.ufl_mesh(cell_id, gdim, dtype=dtype)
    gmsh_cell_perm = dolfinx.io.gmshio.cell_perm_array(
        dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes
    )
    cells = cells[:, gmsh_cell_perm].copy()
    mesh = dolfinx.mesh.create_mesh(
        comm, cells, x[:, :gdim].astype(dtype, copy=False), ufl_domain, partitioner
    )

    # Create MeshTags for cells
    local_entities, local_values = dolfinx.io.utils.distribute_entity_data(
        mesh._cpp_object, mesh.topology.dim, cells, cell_values
    )
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)
    ct = dolfinx.mesh.meshtags_from_entities(
        mesh, mesh.topology.dim, adj, local_values.astype(np.int32, copy=False)
    )
    ct.name = "Cell tags"

    # Create MeshTags for facets
    topology = mesh.topology
    tdim = topology.dim
    if has_facet_data:
        # Permute facets from MSH to DOLFINx ordering
        # FIXME: This does not work for prism meshes
        if (
            topology.cell_type == dolfinx.mesh.CellType.prism
            or topology.cell_type == dolfinx.mesh.CellType.pyramid
        ):
            raise RuntimeError(f"Unsupported cell type {topology.cell_type}")

        facet_type = dolfinx.cpp.mesh.cell_entity_type(
            dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), tdim - 1, 0
        )
        gmsh_facet_perm = dolfinx.io.gmshio.cell_perm_array(facet_type, num_facet_nodes)
        marked_facets = marked_facets[:, gmsh_facet_perm]

        local_entities, local_values = dolfinx.io.utils.distribute_entity_data(
            mesh._cpp_object, tdim - 1, marked_facets, facet_values
        )
        mesh.topology.create_connectivity(topology.dim - 1, tdim)
        adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)
        ft = dolfinx.io.gmshio.meshtags_from_entities(
            mesh, tdim - 1, adj, local_values.astype(np.int32, copy=False)
        )
        ft.name = "Facet tags"
    else:
        ft = dolfinx.mesh.meshtags(
            mesh, tdim - 1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        )

    if has_edge_data:
        # Permute edges from MSH to DOLFINx ordering
        edge_type = dolfinx.cpp.mesh.cell_entity_type(
            dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), tdim - 2, 0
        )
        gmsh_edge_perm = dolfinx.io.gmshio.cell_perm_array(edge_type, num_edge_nodes)
        marked_edges = marked_edges[:, gmsh_edge_perm]

        local_entities, local_values = dolfinx.io.utils.distribute_entity_data(
            mesh._cpp_object, tdim - 2, marked_edges, edge_values
        )
        mesh.topology.create_connectivity(topology.dim - 2, tdim)
        adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)
        et = dolfinx.io.gmshio.meshtags_from_entities(
            mesh, tdim - 2, adj, local_values.astype(np.int32, copy=False)
        )
        et.name = "Edge tags"
    else:
        et = dolfinx.mesh.meshtags(
            mesh, tdim - 2, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        )

    if has_vertex_data:
        # Permute vertices from MSH to DOLFINx ordering
        vertex_type = dolfinx.cpp.mesh.cell_entity_type(
            dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), tdim - 3, 0
        )
        gmsh_vertex_perm = dolfinx.io.gmshio.cell_perm_array(vertex_type, num_vertex_nodes)
        marked_vertices = marked_vertices[:, gmsh_vertex_perm]

        local_entities, local_values = dolfinx.io.utils.distribute_entity_data(
            mesh._cpp_object, tdim - 3, marked_vertices, vertex_values
        )
        mesh.topology.create_connectivity(topology.dim - 3, tdim)
        adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)
        vt = dolfinx.io.gmshio.meshtags_from_entities(
            mesh, tdim - 3, adj, local_values.astype(np.int32, copy=False)
        )
        vt.name = "Vertex tags"
    else:
        vt = dolfinx.mesh.meshtags(
            mesh, tdim - 3, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        )

    return GMshModel(mesh, ct, ft, et, vt)


def parse_element(space_string: str, mesh: dolfinx.mesh.Mesh, dim: int) -> basix.ufl._ElementBase:
    """
    Parse a string representation of a basix element family
    """

    family_str, degree_str = space_string.split("_")
    kwargs = {"degree": int(degree_str), "cell": mesh.ufl_cell().cellname()}
    if dim > 1:
        if family_str in quads:
            kwargs["value_shape"] = (dim,)
        else:
            kwargs["shape"] = (dim,)

    if family_str in ["Lagrange", "P", "CG"]:
        el = basix.ufl.element(family=basix.ElementFamily.P, discontinuous=False, **kwargs)
    elif family_str in ["Discontinuous Lagrange", "DG", "dP"]:
        el = basix.ufl.element(family=basix.ElementFamily.P, discontinuous=True, **kwargs)

    elif family_str in quads:
        el = basix.ufl.quadrature_element(scheme="default", **kwargs)
    else:
        families = list(basix.ElementFamily.__members__.keys())
        msg = f"Unknown element family: {family_str!r}, available families: {families}"
        raise ValueError(msg)
    return el


def space_from_string(
    space_string: str, mesh: dolfinx.mesh.Mesh, dim: int
) -> dolfinx.fem.functionspace:
    """
    Constructed a finite elements space from a string
    representation of the space

    Arguments
    ---------
    space_string : str
        A string on the form {family}_{degree} which
        determines the space. Example 'Lagrange_1'.
    mesh : df.Mesh
        The mesh
    dim : int
        1 for scalar space, 3 for vector space.
    """
    el = parse_element(space_string, mesh, dim)
    return dolfinx.fem.functionspace(mesh, el)


def element2array(el: basix.ufl._BlockedElement) -> np.ndarray:
    if el.family_name in quads:
        return np.array(
            [QUADNR, int(el.cell_type), int(el.degree), 0],
            dtype=np.uint8,
        )
    else:
        return np.array(
            [
                int(el.basix_element.family),
                int(el.cell_type),
                int(el.degree),
                int(el.basix_element.discontinuous),
            ],
            dtype=np.uint8,
        )


def number2Enum(num: int, enum: Iterable) -> Enum:
    for e in enum:
        if int(e) == num:
            return e
    raise ValueError(f"Invalid value {num} for enum {enum}")


def array2element(arr: np.ndarray) -> basix.finite_element.FiniteElement:
    cell_type = number2Enum(arr[1], basix.CellType)
    degree = int(arr[2])
    discontinuous = bool(arr[3])
    if arr[0] == QUADNR:
        return basix.ufl.quadrature_element(
            scheme="default", degree=degree, value_shape=(3,), cell=cell_type
        )
    else:
        family = number2Enum(arr[0], basix.ElementFamily)

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


def read_mesh(
    comm, filename: str | Path
) -> tuple[dolfinx.mesh.Mesh, dict[str, dolfinx.mesh.MeshTags]]:
    tags = {}
    with dolfinx.io.XDMFFile(comm, filename, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Mesh")
        for var, name, dim in [
            ("cfun", "Cell tags", mesh.topology.dim),
            ("ffun", "Facet tags", mesh.topology.dim - 1),
            ("efun", "Edge tags", mesh.topology.dim - 2),
            ("vfun", "Vertex tags", mesh.topology.dim - 3),
        ]:
            mesh.topology.create_connectivity(dim, mesh.topology.dim)
            try:
                tags[var] = xdmf.read_meshtags(mesh, name=name)
            except RuntimeError:
                continue

    return mesh, tags


def gmsh2dolfin(comm: MPI.Intracomm, msh_file, rank: int = 0) -> GMshGeometry:
    logger.debug(f"Convert file {msh_file} to dolfin")
    outdir = Path(msh_file).parent

    import gmsh

    # We could make this work in parallel in the future

    if comm.rank == rank:
        gmsh.initialize()
        gmsh.model.add("Mesh from file")
        gmsh.merge(str(msh_file))
        mesh, ct, ft, et, vt = model_to_mesh(gmsh.model, comm, 0)
        markers = {
            gmsh.model.getPhysicalName(*v): tuple(reversed(v))
            for v in gmsh.model.getPhysicalGroups()
        }
        gmsh.finalize()
    else:
        mesh, ct, ft, et, vt = model_to_mesh(gmsh.model, comm, 0)
        markers = {}
    mesh.name = "Mesh"
    ct.name = "Cell tags"
    ft.name = "Facet tags"
    et.name = "Edge tags"
    vt.name = "Vertex tags"

    markers = comm.bcast(markers, root=rank)

    # Save tags to xdmf
    with dolfinx.io.XDMFFile(comm, outdir / "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(
            ct, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
        )
        mesh.topology.create_connectivity(2, 3)
        xdmf.write_meshtags(
            ft, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
        )
        mesh.topology.create_connectivity(1, 3)
        xdmf.write_meshtags(
            et, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
        )
        mesh.topology.create_connectivity(0, 3)
        xdmf.write_meshtags(
            vt, mesh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry"
        )

    return GMshGeometry(mesh, ct, ft, et, vt, markers)
