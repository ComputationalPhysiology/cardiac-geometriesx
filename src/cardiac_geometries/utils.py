import tempfile
import typing
import xml.etree.ElementTree as ET
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
from dolfinx.graph import adjacencylist
from packaging.version import Version
from structlog import get_logger

try:
    import dolfinx.io.gmsh as gmshio
except ImportError:
    import dolfinx.io.gmshio as gmshio  # type: ignore[import]

logger = get_logger()

quads = ("Quadrature", "Q", "Quad", "quadrature", "q", "quad")
QUADNR = 42


class GMshModel(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    cell_tags: dolfinx.mesh.MeshTags
    facet_tags: dolfinx.mesh.MeshTags
    edge_tags: dolfinx.mesh.MeshTags
    vertex_tags: dolfinx.mesh.MeshTags


def distribute_entity_data(
    mesh: dolfinx.mesh.Mesh,
    tdim: int,
    marked_entities: np.ndarray,
    entity_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if Version(dolfinx.__version__) >= Version("0.9.0"):
        local_entities, local_values = dolfinx.io.utils.distribute_entity_data(
            mesh, tdim, marked_entities, entity_values
        )
    else:
        local_entities, local_values = dolfinx.io.utils.distribute_entity_data(
            mesh._cpp_object, tdim, marked_entities, entity_values
        )
    return local_entities, local_values


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
        x = gmshio.extract_geometry(model)
        topologies = gmshio.extract_topology_and_markers(model)

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
    ufl_domain = gmshio.ufl_mesh(cell_id, gdim, dtype=dtype)
    gmsh_cell_perm = gmshio.cell_perm_array(
        dolfinx.cpp.mesh.to_type(str(ufl_domain.ufl_cell())), num_nodes
    )
    cells = cells[:, gmsh_cell_perm].copy()
    mesh = dolfinx.mesh.create_mesh(
        comm, cells, x[:, :gdim].astype(dtype, copy=False), ufl_domain, partitioner
    )

    # Create MeshTags for cells
    local_entities, local_values = distribute_entity_data(
        mesh, mesh.topology.dim, cells, cell_values
    )

    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    # adj = dolfinx.cpp.graph.AdjacencyList_int32(local_entities)
    adj = adjacencylist(local_entities)
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
        gmsh_facet_perm = gmshio.cell_perm_array(facet_type, num_facet_nodes)
        marked_facets = marked_facets[:, gmsh_facet_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, mesh.topology.dim - 1, marked_facets, facet_values
        )

        mesh.topology.create_connectivity(topology.dim - 1, tdim)
        adj = adjacencylist(local_entities)

        ft = gmshio.meshtags_from_entities(
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
        gmsh_edge_perm = gmshio.cell_perm_array(edge_type, num_edge_nodes)
        marked_edges = marked_edges[:, gmsh_edge_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, tdim - 2, marked_edges, edge_values
        )
        mesh.topology.create_connectivity(topology.dim - 2, tdim)
        adj = adjacencylist(local_entities)
        et = gmshio.meshtags_from_entities(
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
        gmsh_vertex_perm = gmshio.cell_perm_array(vertex_type, num_vertex_nodes)
        marked_vertices = marked_vertices[:, gmsh_vertex_perm]

        local_entities, local_values = distribute_entity_data(
            mesh, tdim - 3, marked_vertices, vertex_values
        )
        mesh.topology.create_connectivity(topology.dim - 3, tdim)
        adj = adjacencylist(local_entities)
        vt = gmshio.meshtags_from_entities(
            mesh, tdim - 3, adj, local_values.astype(np.int32, copy=False)
        )
        vt.name = "Vertex tags"
    else:
        vt = dolfinx.mesh.meshtags(
            mesh, tdim - 3, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        )

    return GMshModel(mesh, ct, ft, et, vt)


def parse_element(
    space_string: str, mesh: dolfinx.mesh.Mesh, dim: int, discontinuous: bool = False
) -> basix.ufl._ElementBase:
    """
    Parse a string representation of a basix element family
    """

    family_str, degree_str = space_string.split("_")
    kwargs = {"degree": int(degree_str), "cell": mesh.basix_cell()}
    if dim > 1:
        if family_str in quads:
            kwargs["value_shape"] = (dim,)
        else:
            kwargs["shape"] = (dim,)

    # breakpoint()
    if family_str in ["Lagrange", "P", "CG"]:
        el = basix.ufl.element(family=basix.ElementFamily.P, discontinuous=discontinuous, **kwargs)
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
    space_string: str, mesh: dolfinx.mesh.Mesh, dim: int, discontinuous: bool = False
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
    discontinuous: bool
        If true force element to be discontinuous, by default False
    """
    el = parse_element(space_string, mesh, dim, discontinuous=discontinuous)
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
            lagrange_variant=basix.LagrangeVariant.unset,
        )


@lru_cache
def array2functionspace(mesh: dolfinx.mesh.Mesh, arr: np.ndarray) -> dolfinx.fem.functionspace:
    el = array2element(arr)
    return dolfinx.fem.functionspace(mesh, el)


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
        mesh = xdmf.read_mesh(name="Mesh", ghost_mode=dolfinx.mesh.GhostMode.none)
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
    outdir.mkdir(parents=True, exist_ok=True)

    if Version(dolfinx.__version__) >= Version("0.10.0"):
        mesh_data = gmshio.read_from_msh(comm=comm, filename=msh_file)
        mesh = mesh_data.mesh
        markers_ = mesh_data.physical_groups
        ct = mesh_data.cell_tags
        tdim = mesh.topology.dim
        if ct is None:
            ct = dolfinx.mesh.meshtags(
                mesh, tdim, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
            )

        ft = mesh_data.facet_tags
        if ft is None:
            ft = dolfinx.mesh.meshtags(
                mesh, tdim - 1, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
            )

        if hasattr(mesh_data, "edge_tags"):
            et = mesh_data.edge_tags
        else:
            et = mesh_data.ridge_tags
        if et is None:
            et = dolfinx.mesh.meshtags(
                mesh, tdim - 2, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
            )
        if hasattr(mesh_data, "vertex_tags"):
            vt = mesh_data.vertex_tags
        else:
            vt = mesh_data.peak_tags
        if vt is None:
            vt = dolfinx.mesh.meshtags(
                mesh, tdim - 3, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
            )

        markers = {k: tuple(reversed(v)) for k, v in markers_.items()}

    else:
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

        markers = comm.bcast(markers, root=rank)

    mesh.name = "Mesh"
    ct.name = "Cell tags"
    ft.name = "Facet tags"
    et.name = "Edge tags"
    vt.name = "Vertex tags"

    save_mesh_to_xdmf(comm, outdir / "mesh.xdmf", mesh, ct, ft, et, vt)

    return GMshGeometry(mesh, ct, ft, et, vt, markers)


def save_mesh_to_xdmf(
    comm: MPI.Intracomm,
    fname: Path,
    mesh: dolfinx.mesh.Mesh,
    ct: dolfinx.mesh.MeshTags,
    ft: dolfinx.mesh.MeshTags,
    et: dolfinx.mesh.MeshTags,
    vt: dolfinx.mesh.MeshTags,
):
    # Save tags to xdmf
    with dolfinx.io.XDMFFile(comm, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
        if ct is not None:
            xdmf.write_meshtags(
                ct,
                mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
            )
        if ft is not None:
            mesh.topology.create_connectivity(2, 3)
            xdmf.write_meshtags(
                ft,
                mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
            )
        if et is not None:
            mesh.topology.create_connectivity(1, 3)
            xdmf.write_meshtags(
                et,
                mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
            )
        if vt is not None:
            mesh.topology.create_connectivity(0, 3)
            xdmf.write_meshtags(
                vt,
                mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
            )


def create_xdmf_pointcloud(filename: Path, us: typing.Sequence[dolfinx.fem.Function]) -> None:
    # Adopted from https://gist.github.com/jorgensd/8bae61ad7a0c211570dff0116a68a356
    if len(us) == 0:
        return

    import adios2

    u = us[0]
    points = u.function_space.tabulate_dof_coordinates()
    h5name = filename.with_suffix(".h5")

    bs = u.function_space.dofmap.index_map_bs
    comm = u.function_space.mesh.comm
    num_dofs_global = u.function_space.dofmap.index_map.size_global
    num_dofs_local = u.function_space.dofmap.index_map.size_local
    local_range = np.array(u.function_space.dofmap.index_map.local_range, dtype=np.int64)

    # Write XDMF on rank 0
    if comm.rank == 0:
        xdmf = ET.Element("XDMF")
        xdmf.attrib["Version"] = "3.0"
        xdmf.attrib["xmlns:xi"] = "http://www.w3.org/2001/XInclude"
        domain = ET.SubElement(xdmf, "Domain")
        grid = ET.SubElement(domain, "Grid")
        grid.attrib["GridType"] = "Uniform"
        grid.attrib["Name"] = "Point Cloud"
        topology = ET.SubElement(grid, "Topology")
        topology.attrib["NumberOfElements"] = str(num_dofs_global)
        topology.attrib["TopologyType"] = "PolyVertex"
        topology.attrib["NodesPerElement"] = "1"
        geometry = ET.SubElement(grid, "Geometry")
        geometry.attrib["GeometryType"] = "XY" if points.shape[1] == 2 else "XYZ"
        for u in us:
            it0 = ET.SubElement(geometry, "DataItem")
            it0.attrib["Dimensions"] = f"{num_dofs_global} {points.shape[1]}"
            it0.attrib["Format"] = "HDF"
            it0.text = f"{h5name.name}:/Step0/Points"
            attrib = ET.SubElement(grid, "Attribute")
            attrib.attrib["Name"] = u.name
            if bs == 1:
                attrib.attrib["AttributeType"] = "Scalar"
            else:
                attrib.attrib["AttributeType"] = "Vector"
            attrib.attrib["Center"] = "Node"
            it1 = ET.SubElement(attrib, "DataItem")
            it1.attrib["Dimensions"] = f"{num_dofs_global} {bs}"
            it1.attrib["Format"] = "HDF"
            it1.text = f"{h5name.name}:/Step0/Values_{u.name}"
        text = [
            '<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n',
            ET.tostring(xdmf, encoding="unicode"),
        ]
        filename.write_text("".join(text))
    # Create ADIOS2 reader
    # start = time.perf_counter()
    adios = adios2.ADIOS(comm)
    io = adios.DeclareIO("Point cloud writer")
    io.SetEngine("HDF5")
    outfile = io.Open(h5name.as_posix(), adios2.Mode.Write)
    points_out = points[:num_dofs_local, :]

    pointvar = io.DefineVariable(
        "Points",
        points_out,
        shape=[num_dofs_global, points.shape[1]],
        start=[local_range[0], 0],
        count=[num_dofs_local, points.shape[1]],
    )
    outfile.Put(pointvar, points_out)
    for u in us:
        data = u.x.array[: num_dofs_local * bs].reshape(-1, bs)

        valuevar = io.DefineVariable(
            f"Values_{u.name}",
            data,
            shape=[num_dofs_global, bs],
            start=[local_range[0], 0],
            count=[num_dofs_local, bs],
        )
        outfile.Put(valuevar, data)
    outfile.PerformPuts()
    outfile.Close()
    assert adios.RemoveIO("Point cloud writer")


class BaseData(NamedTuple):
    centroid: npt.NDArray[np.float64]
    vector: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float64]


def compute_base_data(
    mesh: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    marker,
) -> BaseData:
    """Compute the centroid, vector and normal of the base

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    facet_tags : dolfinx.mesh.MeshTags
        The facet tags
    marker : _type_
        Marker for the base

    Returns
    -------
    BaseData
        NamedTuple containing the centroid, vector and normal of the base
    """
    base_facets = facet_tags.find(marker)
    base_midpoints = mesh.comm.gather(
        dolfinx.mesh.compute_midpoints(mesh, 2, base_facets),
        root=0,
    )
    base_vector = np.zeros(3)
    base_centroid = np.zeros(3)
    base_normal = np.zeros(3)
    if mesh.comm.rank == 0:
        bm = np.concatenate(base_midpoints)
        base_centroid = bm.mean(axis=0)
        # print("Base centroid", len(base_midpoints))
        base_points_centered = bm - base_centroid
        u, s, vh = np.linalg.svd(base_points_centered)
        base_normal = vh[-1, :]
        # Initialize vector to be used for cross product
        vector_init = np.array([0, 1, 0])

        # If the normal is parallel to the initial vector, change the initial vector
        if np.allclose(np.abs(base_normal), np.abs(vector_init)):
            vector_init = np.array([0, 0, 1])

        # Find two vectors in the plane, orthogonal to the normal
        vector = np.cross(base_normal, vector_init)
        base_vector = np.cross(base_normal, vector)

    base_centroid = mesh.comm.bcast(base_centroid, root=0)
    base_vector = mesh.comm.bcast(base_vector, root=0)
    base_normal = mesh.comm.bcast(base_normal, root=0)
    return BaseData(centroid=base_centroid, vector=base_vector, normal=base_normal)


def rotate_geometry_and_fields(
    mesh: dolfinx.mesh.Mesh,
    ffun: dolfinx.mesh.MeshTags,
    base_marker: int,
    target_normal: typing.Sequence[float] = tuple((1.0, 0.0, 0.0)),
    fields: typing.Sequence[dolfinx.fem.Function] | None = None,
) -> tuple[dolfinx.mesh.Mesh, np.ndarray, list[dolfinx.fem.Function]]:
    """
    Rotates the mesh and vector fields so that the base normal points in the +x direction.
    """
    comm = mesh.comm
    if comm.size > 1:
        raise NotImplementedError("rotate_geometry_and_fields only works in serial.")

    if fields is None:
        fs: list[dolfinx.fem.Function] = []
    else:
        fs = list(fields)
    # Compute current base normal
    # We use pulse's built-in helper to find the normal of the surface marked as base
    base_data = compute_base_data(mesh, ffun, base_marker)
    current_normal = base_data.normal

    # Create rotation matrix
    R_matrix = np.eye(3)
    target_normal_normalized: np.ndarray = np.array(target_normal) / np.linalg.norm(target_normal)

    # Compute Rotation Matrix
    # We find the rotation that aligns current_normal to target_normal
    if np.allclose(current_normal, target_normal_normalized):
        logger.info("Geometry already aligned.")
        return mesh, R_matrix, fs

    # Simple check to avoid singular rotation if vectors are opposite
    if np.allclose(current_normal, -target_normal_normalized):
        R_matrix = np.diag([-1.0, -1.0, 1.0])  # 180 flip
    else:
        # Calculate cross product and angle
        v = np.cross(current_normal, target_normal_normalized)
        s = np.linalg.norm(v)
        c = np.dot(current_normal, target_normal_normalized)

        if s > 1e-8:
            # Skew-symmetric cross-product matrix of v
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    new_fields = []
    for f in fs:
        # Ensure it is a vector function
        bs = f.function_space.dofmap.bs
        if bs != 3:
            continue

        # Get underlying array
        values = f.x.array
        # Reshape to (Num_DOFs, 3)
        num_dofs = len(values) // 3
        vectors = values.reshape((num_dofs, 3))

        # Rotate vectors: v_new = R * v_old
        vectors_rotated = vectors @ R_matrix.T

        # Assign back
        f.x.array[:] = vectors_rotated.flatten()
        new_fields.append(f)

    # Rotate Mesh Coordinates
    # geometry.x is N x 3. We apply x_new = R * x_old^T -> x_new^T = x_old * R^T
    coords = mesh.geometry.x[:, :3]

    mesh.geometry.x[:, :3] = coords @ R_matrix.T

    logger.info(f"Rotated geometry. Base normal {current_normal} aligned to {target_normal}")
    logger.debug(f"Rotation matrix:\n{R_matrix}")
    return mesh, R_matrix, new_fields
