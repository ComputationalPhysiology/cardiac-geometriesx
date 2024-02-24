from pathlib import Path
import math
from mpi4py import MPI
from typing import Dict, NamedTuple, Tuple
from structlog import get_logger
import dolfinx
import gmsh
import meshio

import tempfile


def handle_mesh_name(mesh_name: str = "") -> Path:
    if mesh_name == "":
        fd, mesh_name = tempfile.mkstemp(suffix=".msh")
    return Path(mesh_name).with_suffix(".msh")


logger = get_logger()


class GMshGeometry(NamedTuple):
    mesh: dolfinx.mesh.Mesh
    cfun: dolfinx.mesh.MeshTags
    ffun: dolfinx.mesh.MeshTags
    efun: dolfinx.mesh.MeshTags
    vfun: dolfinx.mesh.MeshTags
    markers: Dict[str, Tuple[int, int]]


def create_mesh(mesh, cell_type):
    # From http://jsdokken.com/converted_files/tutorial_pygmsh.html
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells={cell_type: cells},
        cell_data={"name_to_read": [cell_data]},
    )
    return out_mesh


def gmsh2dolfin(msh_file):
    logger.debug(f"Convert file {msh_file} to dolfin")
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        msh = meshio.gmsh.read(msh_file)
        vertex_mesh = create_mesh(msh, "vertex")
        line_mesh = create_mesh(msh, "line")
        triangle_mesh = create_mesh(msh, "triangle")
        tetra_mesh = create_mesh(msh, "tetra")
        vertex_mesh_name = Path("vertex_mesh.xdmf")
        meshio.write(vertex_mesh_name, vertex_mesh)

        line_mesh_name = Path("line_mesh.xdmf")
        meshio.write(line_mesh_name, line_mesh)

        triangle_mesh_name = Path("triangle_mesh.xdmf")
        meshio.write(triangle_mesh_name, triangle_mesh)

        tetra_mesh_name = Path("mesh.xdmf")
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

    with dolfinx.io.XDMFFile(comm, "mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cfun = xdmf.read_meshtags(mesh, name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    with dolfinx.io.XDMFFile(comm, "triangle_mesh.xdmf", "r") as xdmf:
        ffun = xdmf.read_meshtags(mesh, name="Grid")
    ffun.name = "Facet tags"

    mesh.topology.create_connectivity(mesh.topology.dim - 2, mesh.topology.dim)
    with dolfinx.io.XDMFFile(comm, "line_mesh.xdmf", "r") as xdmf:
        efun = xdmf.read_meshtags(mesh, name="Grid")

    mesh.topology.create_connectivity(mesh.topology.dim - 3, mesh.topology.dim)
    with dolfinx.io.XDMFFile(comm, "vertex_mesh.xdmf", "r") as xdmf:
        vfun = xdmf.read_meshtags(mesh, name="Grid")

    return GMshGeometry(mesh, cfun, ffun, efun, vfun, markers)


def lv_ellipsoid_flat_base(
    mesh_name: str = "",
    r_short_endo: float = 7.0,
    r_short_epi: float = 10.0,
    r_long_endo: float = 17.0,
    r_long_epi: float = 20.0,
    quota_base: float = -5.0,
    psize: float = 3.0,
    ndiv: float = 1.0,
) -> Path:
    """Create an LV ellipsoids with a flat base

    Parameters
    ----------
    mesh_name : str, optional
        Name of the mesh by default ""
    r_short_endo : float, optional
        Shortest radius on the endocardium layer, by default 7.0
    r_short_epi : float, optional
       Shortest radius on the epicardium layer, by default 10.0
    r_long_endo : float, optional
        Longest radius on the endocardium layer, by default 17.0
    r_long_epi : float, optional
        Longest radius on the epicardium layer, by default 20.0
    quota_base : float, optional
        Position of the base relative to the x=0 plane, by default -5.0
    psize : float, optional
        Point size, by default 3.0
    ndiv : float, optional
        Number of divisions, by default 1.0

    Returns
    -------
    Path
        Path to file
    """
    mu_base_endo = math.acos(quota_base / r_long_endo)
    mu_base_epi = math.acos(quota_base / r_long_epi)
    mu_apex_endo = mu_apex_epi = 0
    psize_ref = psize / ndiv
    return lv_ellipsoid(
        mesh_name=mesh_name,
        r_short_endo=r_short_endo,
        r_short_epi=r_short_epi,
        r_long_endo=r_long_endo,
        r_long_epi=r_long_epi,
        psize_ref=psize_ref,
        mu_base_endo=mu_base_endo,
        mu_base_epi=mu_base_epi,
        mu_apex_endo=mu_apex_endo,
        mu_apex_epi=mu_apex_epi,
    )


def lv_ellipsoid(
    mesh_name: str = "",
    r_short_endo=0.025,
    r_short_epi=0.035,
    r_long_endo=0.09,
    r_long_epi=0.097,
    psize_ref=0.005,
    mu_apex_endo=-math.pi,
    mu_base_endo=-math.acos(5 / 17),
    mu_apex_epi=-math.pi,
    mu_base_epi=-math.acos(5 / 20),
) -> Path:
    """Create general LV ellipsoid

    Parameters
    ----------
    mesh_name : str, optional
        Name of the mesh, by default ""
    r_short_endo : float, optional
        Shortest radius on the endocardium layer, by default 0.025
    r_short_epi : float, optional
       Shortest radius on the epicardium layer, by default 0.035
    r_long_endo : float, optional
        Longest radius on the endocardium layer, by default 0.09
    r_long_epi : float, optional
        Longest radius on the epicardium layer, by default 0.097
    psize_ref : float, optional
        The reference point size (smaller values yield as finer mesh, by default 0.005
    mu_apex_endo : float, optional
        Angle for the endocardial apex, by default -math.pi
    mu_base_endo : float, optional
        Angle for the endocardial base, by default -math.acos(5 / 17)
    mu_apex_epi : float, optional
        Angle for the epicardial apex, by default -math.pi
    mu_base_epi : float, optional
        Angle for the epicardial apex, by default -math.acos(5 / 20)

    Returns
    -------
    Path
        Path to the generated gmsh file
    """

    path = handle_mesh_name(mesh_name=mesh_name)

    gmsh.initialize()
    gmsh.option.setNumber("Geometry.CopyMeshingMethod", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    def ellipsoid_point(mu, theta, r_long, r_short, psize):
        return gmsh.model.geo.addPoint(
            r_long * math.cos(mu),
            r_short * math.sin(mu) * math.cos(theta),
            r_short * math.sin(mu) * math.sin(theta),
            psize,
        )

    center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)

    apex_endo = ellipsoid_point(
        mu=mu_apex_endo,
        theta=0.0,
        r_short=r_short_endo,
        r_long=r_long_endo,
        psize=psize_ref / 2.0,
    )

    base_endo = ellipsoid_point(
        mu=mu_base_endo,
        theta=0.0,
        r_short=r_short_endo,
        r_long=r_long_endo,
        psize=psize_ref,
    )

    apex_epi = ellipsoid_point(
        mu=mu_apex_epi,
        theta=0.0,
        r_short=r_short_epi,
        r_long=r_long_epi,
        psize=psize_ref / 2.0,
    )

    base_epi = ellipsoid_point(
        mu=mu_base_epi,
        theta=0.0,
        r_short=r_short_epi,
        r_long=r_long_epi,
        psize=psize_ref,
    )

    apex = gmsh.model.geo.addLine(apex_endo, apex_epi)
    base = gmsh.model.geo.addLine(base_endo, base_epi)
    endo = gmsh.model.geo.add_ellipse_arc(apex_endo, center, apex_endo, base_endo)
    epi = gmsh.model.geo.add_ellipse_arc(apex_epi, center, apex_epi, base_epi)

    ll1 = gmsh.model.geo.addCurveLoop([apex, epi, -base, -endo])

    s1 = gmsh.model.geo.addPlaneSurface([ll1])

    sendoringlist = []
    sepiringlist = []
    sendolist = []
    sepilist = []
    sbaselist = []
    vlist = []

    out = [(2, s1)]
    for _ in range(4):
        out = gmsh.model.geo.revolve(
            [out[0]],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            math.pi / 2,
        )

        sendolist.append(out[4][1])
        sepilist.append(out[2][1])
        sbaselist.append(out[3][1])
        vlist.append(out[1][1])

        gmsh.model.geo.synchronize()
        bnd = gmsh.model.getBoundary([out[0]])

        sendoringlist.append(bnd[1][1])
        sepiringlist.append(bnd[3][1])

    phys_apex_endo = gmsh.model.addPhysicalGroup(0, [apex_endo])
    gmsh.model.setPhysicalName(0, phys_apex_endo, "ENDOPT")

    phys_apex_epi = gmsh.model.addPhysicalGroup(0, [apex_epi])
    gmsh.model.setPhysicalName(0, phys_apex_epi, "EPIPT")

    phys_epiring = gmsh.model.addPhysicalGroup(1, sepiringlist)
    gmsh.model.setPhysicalName(1, phys_epiring, "EPIRING")

    phys_endoring = gmsh.model.addPhysicalGroup(1, sendoringlist)
    gmsh.model.setPhysicalName(1, phys_endoring, "ENDORING")

    phys_base = gmsh.model.addPhysicalGroup(2, sbaselist)
    gmsh.model.setPhysicalName(2, phys_base, "BASE")

    phys_endo = gmsh.model.addPhysicalGroup(2, sendolist)
    gmsh.model.setPhysicalName(2, phys_endo, "ENDO")

    phys_epi = gmsh.model.addPhysicalGroup(2, sepilist)
    gmsh.model.setPhysicalName(2, phys_epi, "EPI")

    phys_myo = gmsh.model.addPhysicalGroup(3, vlist)
    gmsh.model.setPhysicalName(3, phys_myo, "MYOCARDIUM")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    gmsh.write(path.as_posix())

    gmsh.finalize()
    return path
