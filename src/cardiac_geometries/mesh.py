from __future__ import annotations

import datetime
import json
import math
from importlib.metadata import metadata
from pathlib import Path

from mpi4py import MPI

import cardiac_geometries_core as cgc
import dolfinx
import numpy as np
from structlog import get_logger

from . import utils
from .fibers.utils import save_microstructure
from .geometry import Geometry, save_geometry

meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]

logger = get_logger()


def transform_biv_markers(markers: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    import warnings

    warnings.warn(
        "transform_biv_markers is deprecated and will be removed in a future release. "
        "Use transform_markers instead.",
        DeprecationWarning,
        stacklevel=3,
    )

    return {
        "base": [markers["BASE"][0]],
        "lv": [markers["LV_ENDO_FW"][0], markers["LV_SEPTUM"][0]],
        "rv": [markers["RV_ENDO_FW"][0], markers["RV_SEPTUM"][0]],
        "epi": [markers["LV_EPI_FW"][0], markers["RV_EPI_FW"][0]],
    }


def transform_markers(
    markers: dict[str, tuple[int, int]], clipped: bool = False
) -> dict[str, list[int]]:
    if "ENDO_RV" in markers:
        return {
            "lv": [markers["ENDO_LV"][0]],
            "rv": [markers["ENDO_RV"][0]],
            "epi": [markers["EPI"][0]],
            "base": [markers["BASE"][0]],
        }
    elif "PV" in markers:
        return {
            "lv": [markers["LV"][0]],
            "rv": [markers["RV"][0]],
            "epi": [markers["EPI"][0]],
            "base": [
                markers["PV"][0],
                markers["TV"][0],
                markers["AV"][0],
                markers["MV"][0],
            ],
        }
    elif "ENDO_LV" in markers:
        return {
            "lv": [markers["ENDO_LV"][0]],
            "epi": [markers["EPI"][0]],
            "base": [markers["BASE"][0]],
        }
    elif "LV_ENDO_FW" in markers:
        return {
            "base": [markers["BASE"][0]],
            "lv": [markers["LV_ENDO_FW"][0], markers["LV_SEPTUM"][0]],
            "rv": [markers["RV_ENDO_FW"][0], markers["RV_SEPTUM"][0]],
            "epi": [markers["LV_EPI_FW"][0], markers["RV_EPI_FW"][0]],
        }
    elif "ENDO" in markers:
        return {
            "lv": [markers["ENDO"][0]],
            "epi": [markers["EPI"][0]],
            "base": [markers["BASE"][0]],
        }
    elif "BASE" in markers:
        return {
            "lv": [markers["LV"][0]],
            "rv": [markers["RV"][0]],
            "epi": [markers["EPI"][0]],
            "base": [markers["BASE"][0]],
        }

    else:
        # Assume they are already transformed
        return {k: [v[0]] for k, v in markers.items()}


def ukb(
    outdir: str | Path,
    mode: int = -1,
    std: float = 1.5,
    case: str = "ED",
    char_length_max: float = 5.0,
    char_length_min: float = 5.0,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    create_fibers: bool = True,
    fiber_space: str = "P_1",
    clipped: bool = False,
    use_burns: bool = False,
    burns_path: Path | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create a mesh from the UK-Biobank atlas using
    the ukb-atlas package.

    Parameters
    ----------
    outdir : str | Path
        Directory where to save the results.
    mode : int, optional
        Mode for the UKB mesh, by default -1
    std : float, optional
        Standard deviation for the UKB mesh, by default 1.5
    case : str, optional
        Case for the UKB mesh (either "ED" or "ES"), by default "ED"
    char_length_max : float, optional
        Maximum characteristic length of the mesh, by default 2.0
    char_length_min : float, optional
        Minimum characteristic length of the mesh, by default 2.0
    fiber_angle_endo : float, optional
        Fiber angle for the endocardium, by default 60
    fiber_angle_epi : float, optional
        Fiber angle for the epicardium, by default -60
    create_fibers : bool, optional
        If True create rule-based fibers, by default True
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    clipped : bool, optional
        If True create a clipped mesh, by default False
    use_burns : bool
        If true, use the atlas from Richard Burns to generate the surfaces.
        This will override the `all` parameter and use the burns atlas instead.
    burns_path : Path | None
        Path to the burns atlas file. This will be a .mat file which will be loaded
        using scipy.io.loadmat. This needs to be specified if `use_burns`.
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """
    try:
        import ukb.cli
    except ImportError as e:
        msg = (
            "To create the UKB mesh you need to install the ukb package "
            "which you can install with pip install ukb-atlas"
        )
        raise ImportError(msg) from e

    if comm.rank == 0:
        surf_args = ["surf", str(outdir), "--mode", str(mode), "--std", str(std), "--case", case]
        if use_burns:
            surf_args.extend(["--use-burns", "--burns-path", str(burns_path)])

        ukb.cli.main(surf_args)
        mesh_args = [
            "mesh",
            str(outdir),
            "--case",
            case,
            "--char_length_max",
            str(char_length_max),
            "--char_length_min",
            str(char_length_min),
        ]
        if clipped:
            ukb.cli.main(["clip", str(outdir), "--case", case, "--smooth"])
            mesh_args.append("--clipped")
        print(comm.rank)

        ukb.cli.main(mesh_args)
    comm.barrier()
    outdir = Path(outdir)
    if clipped:
        mesh_name = outdir / f"{case}_clipped.msh"
    else:
        mesh_name = outdir / f"{case}.msh"

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)
    info = {
        "mode": mode,
        "std": std,
        "case": case,
        "char_length_max": char_length_max,
        "char_length_min": char_length_min,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "cardiac_geometry_version": __version__,
        "mesh_type": "ukb",
        "clipped": clipped,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        (outdir / "markers.json").write_text(
            json.dumps(geometry.markers, default=utils.json_serial)
        )
        (outdir / "info.json").write_text(json.dumps(info, default=utils.json_serial))

    fibers = {}
    if create_fibers:
        try:
            import ldrb
        except ImportError as ex:
            msg = (
                "To create fibers you need to install the ldrb package "
                "which you can install with pip install fenicsx-ldrb"
            )
            raise ImportError(msg) from ex

        markers = transform_markers(geometry.markers, clipped=clipped)
        system = ldrb.dolfinx_ldrb(
            mesh=geometry.mesh,
            ffun=geometry.ffun,
            markers=markers,
            alpha_endo_lv=fiber_angle_endo,
            alpha_epi_lv=fiber_angle_epi,
            beta_endo_lv=0,
            beta_epi_lv=0,
            fiber_space=fiber_space,
        )

        save_microstructure(
            mesh=geometry.mesh,
            functions=(system.f0, system.s0, system.n0),
            outdir=outdir,
        )
        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

        for k, v in system._asdict().items():
            if v is None:
                continue
            if fiber_space.startswith("Q"):
                # Cannot visualize Quadrature spaces yet
                continue

            logger.debug(f"Write {k}: {v}")
            with dolfinx.io.VTXWriter(comm, outdir / f"{k}-viz.bp", [v], engine="BP4") as vtx:
                vtx.write(0.0)

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        **fibers,
    )
    geo = Geometry.from_folder(comm=comm, folder=outdir)
    return geo


def biv_ellipsoid(
    outdir: str | Path,
    char_length: float = 0.5,
    base_cut_z: float = 2.5,
    box_size: float = 15.0,  # Size of the cutting box
    rv_wall_thickness: float = 0.4,  # cm
    lv_wall_thickness: float = 0.5,  # cm
    rv_offset_x: float = 2.5,
    lv_radius_x: float = 2.0,
    lv_radius_y: float = 1.8,
    lv_radius_z: float = 3.25,
    rv_radius_x: float = 1.9,
    rv_radius_y: float = 2.5,
    rv_radius_z: float = 3.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    verbose: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create BiV ellipsoidal geometry

    Parameters
    ----------
    outdir : str | Path
        Directory where to save the results.
    char_length : float, optional
        Characteristic length of mesh, by default 0.5
    box_size : float, optional
        Size of the cutting box, by default 15.0
    lv_radius_x : float, optional
        Radius of the left ventricle in the x-direction, by default 2.0
    lv_radius_y : float, optional
        Radius of the left ventricle in the y-direction, by default 1.8
    lv_radius_z : float, optional
        Radius of the left ventricle in the z-direction, by default 3.25
    rv_radius_x : float, optional
        Radius of the right ventricle in the x-direction, by default 1.9
    rv_radius_y : float, optional
        Radius of the right ventricle in the y-direction, by default 2.5
    rv_radius_z : float, optional
        Radius of the right ventricle in the z-direction, by default 3.0
    create_fibers : bool, optional
        If True create analytic fibers, by default False
    fiber_angle_endo : float, optional
        Angle for the endocardium, by default 60
    fiber_angle_epi : float, optional
        Angle for the epicardium, by default -60
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    verbose : bool, optional
        If True print information from gmsh, by default False
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """
    outdir = Path(outdir)
    mesh_name = outdir / "biv_ellipsoid.msh"
    info = {
        "char_length": char_length,
        "base_cut_z": base_cut_z,
        "box_size": box_size,
        "rv_wall_thickness": rv_wall_thickness,
        "lv_wall_thickness": lv_wall_thickness,
        "rv_offset_x": rv_offset_x,
        "lv_radius_x": lv_radius_x,
        "lv_radius_y": lv_radius_y,
        "lv_radius_z": lv_radius_z,
        "rv_radius_x": rv_radius_x,
        "rv_radius_y": rv_radius_y,
        "rv_radius_z": rv_radius_z,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "mesh_type": "biv_ellipsoid",
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                info,
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.biv_ellipsoid(
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            base_cut_z=base_cut_z,
            box_size=box_size,
            rv_wall_thickness=rv_wall_thickness,
            lv_wall_thickness=lv_wall_thickness,
            rv_offset_x=rv_offset_x,
            lv_radius_x=lv_radius_x,
            lv_radius_y=lv_radius_y,
            lv_radius_z=lv_radius_z,
            rv_radius_x=rv_radius_x,
            rv_radius_y=rv_radius_y,
            rv_radius_z=rv_radius_z,
            verbose=verbose,
        )
    comm.barrier()
    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)
    comm.barrier()
    fibers = {}
    if create_fibers:
        try:
            import ldrb
        except ImportError:
            msg = (
                "To create fibers you need to install the ldrb package "
                "which you can install with pip install fenicsx-ldrb"
            )
            raise ImportError(msg)

        ldrb_markers = transform_markers(geometry.markers)

        system = ldrb.dolfinx_ldrb(
            mesh=geometry.mesh,
            ffun=geometry.ffun,
            markers=ldrb_markers,
            alpha_endo_lv=fiber_angle_endo,
            alpha_epi_lv=fiber_angle_epi,
            beta_endo_lv=0,
            beta_epi_lv=0,
            fiber_space=fiber_space,
        )

        save_microstructure(
            mesh=geometry.mesh,
            functions=(system.f0, system.s0, system.n0),
            outdir=outdir,
        )
        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        info=info,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        **fibers,
    )
    geo = Geometry.from_folder(comm=comm, folder=outdir)
    return geo


def lv_ellipsoid(
    outdir: Path | str,
    r_short_endo: float = 7.0,
    r_short_epi: float = 10.0,
    r_long_endo: float = 17.0,
    r_long_epi: float = 20.0,
    psize_ref: float = 3,
    mu_apex_endo: float = -math.pi,
    mu_base_endo: float = -math.acos(5 / 17),
    mu_apex_epi: float = -math.pi,
    mu_base_epi: float = -math.acos(5 / 20),
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    aha: bool = False,
    dmu_factor: float = 1 / 4,
    verbose: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create an LV ellipsoidal geometry

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results.
    r_short_endo : float, optional
        Shortest radius on the endocardium layer, by default 7.0
    r_short_epi : float, optional
       Shortest radius on the epicardium layer, by default 10.0
    r_long_endo : float, optional
        Longest radius on the endocardium layer, by default 17.0
    r_long_epi : float, optional
        Longest radius on the epicardium layer, by default 20.0
    psize_ref : float, optional
        The reference point size (smaller values yield as finer mesh, by default 3
    mu_apex_endo : float, optional
        Angle for the endocardial apex, by default -math.pi
    mu_base_endo : float, optional
        Angle for the endocardial base, by default -math.acos(5 / 17)
    mu_apex_epi : float, optional
        Angle for the epicardial apex, by default -math.pi
    mu_base_epi : float, optional
        Angle for the epicardial apex, by default -math.acos(5 / 20)
    create_fibers : bool, optional
        If True create analytic fibers, by default False
    fiber_angle_endo : float, optional
        Angle for the endocardium, by default 60
    fiber_angle_epi : float, optional
        Angle for the epicardium, by default -60
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    aha : bool, optional
        If True create 17-segment AHA regions
    dmu_factor : float, optional
        Factor to determine the aha segmentation width, by default 1/4
    verbose : bool, optional
        If True print information from gmsh, by default False
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """

    outdir = Path(outdir)
    mesh_name = outdir / "lv_ellipsoid.msh"
    info = {
        "r_short_endo": r_short_endo,
        "r_short_epi": r_short_epi,
        "r_long_endo": r_long_endo,
        "r_long_epi": r_long_epi,
        "psize_ref": psize_ref,
        "mu_apex_endo": mu_apex_endo,
        "mu_base_endo": mu_base_endo,
        "mu_apex_epi": mu_apex_epi,
        "mu_base_epi": mu_base_epi,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "aha": aha,
        "dmu_factor": dmu_factor,
        "mesh_type": "lv_ellipsoid",
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                info,
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.lv_ellipsoid(
            mesh_name=mesh_name.as_posix(),
            r_short_endo=r_short_endo,
            r_short_epi=r_short_epi,
            r_long_endo=r_long_endo,
            r_long_epi=r_long_epi,
            mu_base_endo=mu_base_endo,
            mu_base_epi=mu_base_epi,
            mu_apex_endo=mu_apex_endo,
            mu_apex_epi=mu_apex_epi,
            psize_ref=psize_ref,
            verbose=verbose,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    kwargs = {"cfun": geometry.cfun}
    if aha:
        from .aha import lv_aha

        aha_func, markers = lv_aha(
            mesh=geometry.mesh,
            r_long_endo=r_long_endo,
            r_short_endo=r_short_endo,
            mu_base=mu_base_endo,
            dmu_factor=dmu_factor,
        )
        aha_func.name = "Cell tags"

        fname = outdir / "mesh.xdmf"
        fname.unlink(missing_ok=True)
        fname.with_suffix(".h5").unlink(missing_ok=True)
        comm.barrier()
        from .utils import save_mesh_to_xdmf

        save_mesh_to_xdmf(
            comm=comm,
            fname=fname,
            mesh=geometry.mesh,
            ct=aha_func,
            ft=geometry.ffun,
            et=geometry.efun,
            vt=geometry.vfun,
        )

        kwargs["cfun"] = aha_func

        for k, v in geometry.markers.items():
            # Add all markers except the volume markers
            if v[0] != 3:
                markers[k] = v

    else:
        markers = geometry.markers

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(markers, f, default=utils.json_serial)

    if create_fibers:
        from .fibers.lv_ellipsoid import create_microstructure

        system = create_microstructure(
            mesh=geometry.mesh,
            ffun=geometry.ffun,
            markers=geometry.markers,
            function_space=fiber_space,
            r_short_endo=r_short_endo,
            r_short_epi=r_short_epi,
            r_long_endo=r_long_endo,
            r_long_epi=r_long_epi,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
        )
        kwargs["f0"] = system.f0
        kwargs["s0"] = system.s0
        kwargs["n0"] = system.n0

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        markers=markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        **kwargs,
    )
    geo = Geometry.from_folder(comm=comm, folder=outdir)

    return geo


def slab_dolfinx(
    comm, outdir: Path, lx: float = 20.0, ly: float = 7.0, lz: float = 3.0, dx: float = 1.0
) -> utils.GMshGeometry:
    mesh = dolfinx.mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [lx, ly, lz]],
        [int(lx / dx), int(ly / dx), int(lz / dx)],
        dolfinx.mesh.CellType.tetrahedron,
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )
    mesh.name = "Mesh"
    fdim = mesh.topology.dim - 1
    x0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], 0))
    x1_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[0], lx))
    y0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0))
    y1_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], ly))
    z0_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0))
    z1_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], lz))

    # Concatenate and sort the arrays based on facet indices.
    # Left facets marked with 1, right facets with two
    marked_facets = np.hstack([x0_facets, x1_facets, y0_facets, y1_facets, z0_facets, z1_facets])

    marked_values = np.hstack(
        [
            np.full_like(x0_facets, 1),
            np.full_like(x1_facets, 2),
            np.full_like(y0_facets, 3),
            np.full_like(y1_facets, 4),
            np.full_like(z0_facets, 5),
            np.full_like(z1_facets, 6),
        ],
    )
    sorted_facets = np.argsort(marked_facets)
    ft = dolfinx.mesh.meshtags(
        mesh,
        fdim,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )
    ft.name = "Facet tags"
    markers = {
        "X0": (1, 2),
        "X1": (2, 2),
        "Y0": (3, 2),
        "Y1": (4, 2),
        "Z0": (5, 2),
        "Z1": (6, 2),
    }

    with dolfinx.io.XDMFFile(comm, outdir / "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft, mesh.geometry)

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=mesh,
        markers=markers,
        cfun=None,
        ffun=ft.indices,
        efun=None,
        vfun=None,
    )

    return utils.GMshGeometry(
        mesh=mesh,
        markers=markers,
        cfun=None,
        ffun=ft.indices,
        efun=None,
        vfun=None,
    )


def slab(
    outdir: Path | str,
    lx: float = 20.0,
    ly: float = 7.0,
    lz: float = 3.0,
    dx: float = 1.0,
    create_fibers: bool = True,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    verbose: bool = False,
    use_dolfinx: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create slab geometry

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results.
    lx : float, optional
        Length of slab the x-direction, by default 20.0
    ly : float, optional
        Length of slab the x-direction, by default 7.0
    lz : float, optional
        Length of slab the z-direction, by default 3.0
    dx : float, optional
        Element size, by default 1.0
    create_fibers : bool, optional
        If True create analytic fibers, by default True
    fiber_angle_endo : float, optional
        Angle for the endocardium, by default 60
    fiber_angle_epi : float, optional
        Angle for the epicardium, by default -60
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    verbose : bool, optional
        If True print information from gmsh, by default False
    use_dolfinx : bool, optional
        If True use dolfinx to create the mesh, by default False (gmsh)
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """
    outdir = Path(outdir)
    mesh_name = outdir / "slab.msh"
    info = {
        "Lx": lx,
        "Ly": ly,
        "Lz": lz,
        "dx": dx,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "mesh_type": "slab",
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                info,
                f,
                indent=2,
                default=utils.json_serial,
            )

        if not use_dolfinx:
            cgc.slab(
                mesh_name=mesh_name.as_posix(),
                lx=lx,
                ly=ly,
                lz=lz,
                dx=dx,
                verbose=verbose,
            )
    comm.barrier()

    if use_dolfinx:
        geometry = slab_dolfinx(
            comm=comm,
            outdir=outdir,
            lx=lx,
            ly=ly,
            lz=lz,
            dx=dx,
        )

    else:
        geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    fibers = {}
    if create_fibers:
        from .fibers.slab import create_microstructure

        system = create_microstructure(
            mesh=geometry.mesh,
            ffun=geometry.ffun,
            markers=geometry.markers,
            function_space=fiber_space,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
        )
        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        **fibers,
    )
    geo = Geometry.from_folder(comm=comm, folder=outdir)
    return geo


def slab_in_bath(
    outdir: Path | str,
    lx: float = 1.0,
    ly: float = 0.01,
    lz: float = 0.5,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.1,
    dx: float = 0.001,
    verbose: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create slab geometry

    Parameters
    ----------
    outdir : Path
        Directory where to save the results.
    lx : float, optional
        Length of slab the x-direction, by default 1.0
    ly : float, optional
        Length of slab the x-direction, by default 0.5
    lz : float, optional
        Length of slab the z-direction, by default 0.01
    bx : float, optional
        Thickness of bath the x-direction, by default 0.0
    by : float, optional
        Thickness of bath the x-direction, by default 0.0
    bz : float, optional
        Thickness of bath the z-direction, by default 0.1
    dx : float, optional
        Element size, by default 0.001
    verbose : bool, optional
        If True print information from gmsh, by default False
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """

    outdir = Path(outdir)
    mesh_name = outdir / "slab_in_bath.msh"
    info = {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "bx": bx,
        "by": by,
        "bz": bz,
        "dx": dx,
        "mesh_type": "slab-bath",
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                info,
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.slab_in_bath(
            mesh_name=mesh_name.as_posix(),
            lx=lx,
            ly=ly,
            lz=lz,
            bx=bx,
            by=by,
            bz=bz,
            dx=dx,
            verbose=verbose,
        )

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)
    comm.barrier()

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
    )
    geo = Geometry.from_folder(comm=comm, folder=outdir)

    return geo


def cylinder(
    outdir: Path | str,
    r_inner: float = 10.0,
    r_outer: float = 20.0,
    height: float = 40.0,
    char_length: float = 10.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    aha: bool = False,
    verbose: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create a cylindrical geometry

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results.
    r_inner : float, optional
        Radius on the endocardium layer, by default 10.0
    r_outer : float, optional
       Radius on the epicardium layer, by default 20.0
    height : float, optional
        Longest radius on the endocardium layer, by default 10.0
    char_length : float, optional
        Characteristic length of mesh, by default 10.0
    create_fibers : bool, optional
        If True create analytic fibers, by default False
    fiber_angle_endo : float, optional
        Angle for the endocardium, by default 60
    fiber_angle_epi : float, optional
        Angle for the epicardium, by default -60
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    aha : bool, optional
        If True create 17-segment AHA regions
    verbose : bool, optional
        If True print information from gmsh, by default False
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """

    outdir = Path(outdir)
    mesh_name = outdir / "cylinder.msh"
    info = {
        "r_inner": r_inner,
        "r_outer": r_outer,
        "height": height,
        "char_length": char_length,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "aha": aha,
        "mesh_type": "cylinder",
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                info,
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.cylinder(
            inner_radius=r_inner,
            outer_radius=r_outer,
            height=height,
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            verbose=verbose,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    fibers = {}
    if create_fibers:
        from .fibers.cylinder import create_microstructure

        system = create_microstructure(
            mesh=geometry.mesh,
            function_space=fiber_space,
            r_inner=r_inner,
            r_outer=r_outer,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
        )
        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        **fibers,
    )

    geo = Geometry.from_folder(comm=comm, folder=outdir)

    return geo


def cylinder_racetrack(
    outdir: Path | str,
    r_inner: float = 13.0,
    r_outer: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    char_length: float = 10.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    aha: bool = False,
    verbose: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create a racetrack-shaped thick cylindrical shell mesh using GMSH.

    Both the inner and outer surfaces have two flat faces on opposite sides.

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results.
    r_inner : float, optional
        Radius on the endocardium layer, by default 13.0
    r_outer : float, optional
       Radius on the epicardium layer, by default 20.0
    height : float, optional
        Longest radius on the endocardium layer, by default 10.0
    inner_flat_face_distance : float
        The distance of the inner flat face from the center (along the x-axis).
        This value must be less than inner_radius. Default is 10.0.
    outer_flat_face_distance : float
        The distance of the outer flat face from the center (along the x-axis).
        This value must be less than outer_radius. Default is 17.0.
    char_length : float, optional
        Characteristic length of mesh, by default 10.0
    create_fibers : bool, optional
        If True create analytic fibers, by default False
    fiber_angle_endo : float, optional
        Angle for the endocardium, by default 60
    fiber_angle_epi : float, optional
        Angle for the epicardium, by default -60
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    aha : bool, optional
        If True create 17-segment AHA regions
    verbose : bool, optional
        If True print information from gmsh, by default False
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """

    outdir = Path(outdir)
    mesh_name = outdir / "cylinder_racetrack.msh"
    info = {
        "r_inner": r_inner,
        "r_outer": r_outer,
        "height": height,
        "inner_flat_face_distance": inner_flat_face_distance,
        "outer_flat_face_distance": outer_flat_face_distance,
        "char_length": char_length,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "aha": aha,
        "mesh_type": "cylinder_racetrack",
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                info,
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.cylinder_racetrack(
            inner_radius=r_inner,
            outer_radius=r_outer,
            inner_flat_face_distance=inner_flat_face_distance,
            outer_flat_face_distance=outer_flat_face_distance,
            height=height,
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            verbose=verbose,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    fibers = {}
    if create_fibers:
        from .fibers.cylinder_flat import create_microstructure

        system = create_microstructure(
            mesh=geometry.mesh,
            function_space=fiber_space,
            r_inner=r_inner,
            r_outer=r_outer,
            inner_flat_face_distance=inner_flat_face_distance,
            ffun=geometry.ffun,
            markers=geometry.markers,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
        )

        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        **fibers,
    )

    geo = Geometry.from_folder(comm=comm, folder=outdir)

    return geo


def cylinder_D_shaped(
    outdir: Path | str,
    r_inner: float = 13.0,
    r_outer: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    char_length: float = 10.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    aha: bool = False,
    verbose: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create a D-shaped thick cylindrical shell mesh using GMSH.

    Both the inner and outer surfaces have two flat faces on opposite sides.

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results.
    r_inner : float, optional
        Radius on the endocardium layer, by default 13.0
    r_outer : float, optional
       Radius on the epicardium layer, by default 20.0
    height : float, optional
        Longest radius on the endocardium layer, by default 10.0
    inner_flat_face_distance : float
        The distance of the inner flat face from the center (along the x-axis).
        This value must be less than inner_radius. Default is 10.0.
    outer_flat_face_distance : float
        The distance of the outer flat face from the center (along the x-axis).
        This value must be less than outer_radius. Default is 17.0.
    char_length : float, optional
        Characteristic length of mesh, by default 10.0
    create_fibers : bool, optional
        If True create analytic fibers, by default False
    fiber_angle_endo : float, optional
        Angle for the endocardium, by default 60
    fiber_angle_epi : float, optional
        Angle for the epicardium, by default -60
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    aha : bool, optional
        If True create 17-segment AHA regions
    verbose : bool, optional
        If True print information from gmsh, by default False
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """

    outdir = Path(outdir)
    mesh_name = outdir / "cylinder_D_shaped.msh"
    info = {
        "r_inner": r_inner,
        "r_outer": r_outer,
        "height": height,
        "inner_flat_face_distance": inner_flat_face_distance,
        "outer_flat_face_distance": outer_flat_face_distance,
        "char_length": char_length,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "aha": aha,
        "mesh_type": "cylinder_D_shaped",
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                info,
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.cylinder_D_shaped(
            inner_radius=r_inner,
            outer_radius=r_outer,
            inner_flat_face_distance=inner_flat_face_distance,
            outer_flat_face_distance=outer_flat_face_distance,
            height=height,
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            verbose=verbose,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    fibers = {}
    if create_fibers:
        from .fibers.cylinder_flat import create_microstructure

        system = create_microstructure(
            mesh=geometry.mesh,
            function_space=fiber_space,
            r_inner=r_inner,
            r_outer=r_outer,
            inner_flat_face_distance=inner_flat_face_distance,
            ffun=geometry.ffun,
            markers=geometry.markers,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
        )

        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

    save_geometry(
        path=outdir / "geometry.bp",
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        **fibers,
    )

    geo = Geometry.from_folder(comm=comm, folder=outdir)

    return geo
