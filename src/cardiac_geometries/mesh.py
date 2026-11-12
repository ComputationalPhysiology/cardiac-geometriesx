from __future__ import annotations

import datetime
import json
import math
import typing
from importlib.metadata import metadata
from pathlib import Path

from mpi4py import MPI

import cardiac_geometries_core as cgc
import dolfinx
import numpy as np
from structlog import get_logger

from . import utils
from .geometry import DEFAULT_IO_BACKEND, Geometry, geometry_path, save_geometry

meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]

logger = get_logger()

UKB_GEOMETRIES = ("biv", "biv-closed", "lv", "lv-closed")


def transform_markers(
    markers: dict[str, tuple[int, int]], clipped: bool = False
) -> dict[str, list[int]]:
    if "LV_ENDO" in markers:
        # UKB geometries from ukb-atlas >= 2.0
        epi = [markers["EPI"][0]]
        if "RV_SEPTUM" in markers:
            # The LV-only geometries are closed on the RV side by the RV septum
            epi.append(markers["RV_SEPTUM"][0])
        # The base is the basal cap (BASE, e.g. a patient mesh) and the valves, whichever exist:
        # clipped UKB geometries have BASE only, the ones with outflow tracts the valves only
        valves = [markers[valve][0] for valve in ("MV", "AV", "TV", "PV") if valve in markers]
        base = ([markers["BASE"][0]] if "BASE" in markers else []) + valves
        transformed = {"lv": [markers["LV_ENDO"][0]], "epi": epi, "base": base}
        if "RV_ENDO" in markers:
            transformed["rv"] = [markers["RV_ENDO"][0]]
        return transformed
    elif "ENDO_RV" in markers:
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


def doste_markers(markers: dict[str, tuple[int, int]]) -> dict[str, list[int]]:
    """Markers for the method of Doste et al. (`ldrb.dolfinx_doste`) from the
    markers of the UKB geometries (ukb-atlas >= 2.0), which need the valves."""
    missing = {"LV_ENDO", "EPI", "MV", "AV"} - set(markers)
    if missing:
        raise ValueError(
            f"The method of Doste et al. needs the markers {sorted(missing)}, which are only "
            "in the UKB geometries with outflow tracts (not clipped)"
        )
    epi = [markers["EPI"][0]]
    if "RV_SEPTUM" in markers:
        # The LV-only geometries are closed on the RV side by the RV septum
        epi.append(markers["RV_SEPTUM"][0])
    transformed = {
        "lv": [markers["LV_ENDO"][0]],
        "epi": epi,
        "mv": [markers["MV"][0]],
        "av": [markers["AV"][0]],
    }
    if "RV_ENDO" in markers:
        transformed |= {
            "rv": [markers["RV_ENDO"][0]],
            "tv": [markers["TV"][0]],
            "pv": [markers["PV"][0]],
        }
    return transformed


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
    refine: int = 0,
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    geometry: str = "biv",
    fiber_method: str | None = None,
    io_backend: str = DEFAULT_IO_BACKEND,
    aha: bool = False,
    rv_regions: bool = False,
    rv_septum_fraction: float | None = None,
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
    refine : int, optional
        Number of times to refine the surfaces of the atlas before meshing,
        placing the new nodes on smooth surfaces through the nodes of the atlas.
        This makes meshes of higher order more accurate, by default 0
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    geometry : str, optional
        Which geometry to create: biventricular ("biv") or LV only ("lv"), with
        open valves, or with the valves closed by caps ("biv-closed", "lv-closed").
        The closed geometries also mesh the cavities, which are marked as
        "LV_CAVITY" and "RV_CAVITY" in the cell tags, while the myocardium is
        marked as "Wall". Fibers are only created in the wall. By default "biv".
    fiber_method : str | None, optional
        Rule-based method for the fibers: "doste" for the method of Doste et al.
        (2019), which handles the outflow tracts, or "ldrb" for the method of
        Bayer et al. (2012). With "doste", `fiber_angle_endo` and
        `fiber_angle_epi` are the angles in the LV, while the RV and outflow
        tracts get the angles from the paper. By default "doste", or "ldrb" for
        clipped meshes, which have no outflow tracts.
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"
    aha : bool, optional
        If True, the cell tags are the 17 AHA segments of the LV, the RV free wall
        ("RV") and the LV above the mitral plane ("LVOT"), see
        :func:`cardiac_geometries.aha.biv_aha`. Only for the "biv" geometry (not
        clipped), and it needs ukb-atlas >= 2.2 for the landmarks. By default False
    rv_regions : bool, optional
        If True (with `aha`), the RV is divided into the 15 regions of Zhong et al.
        (2012) (tags 18-32) and the septum is split between the LV and the RV by a
        transventricular Laplace solution; the LV above the mitral plane is 33. See
        :func:`cardiac_geometries.aha.biv_regions`. By default False
    rv_septum_fraction : float | None, optional
        With `rv_regions`, the share of the septum, in the transventricular coordinate
        from the RV side, that belongs to the RV, in [0, 1]: 0 gives the whole septum to
        the LV, 1 to the RV, 0.5 splits it at the middle (Cobiveco) and 1/3 approximates
        Doste et al. (2019). Only allowed with `rv_regions`. By default None, meaning 0.5

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """
    if geometry not in UKB_GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}, expected one of {UKB_GEOMETRIES}")
    if clipped and geometry != "biv":
        raise ValueError("Clipping is only supported for the 'biv' geometry")
    if fiber_method is None:
        fiber_method = "ldrb" if clipped else "doste"
    if fiber_method not in ("ldrb", "doste"):
        raise ValueError(f"Unknown fiber method {fiber_method!r}, expected 'ldrb' or 'doste'")
    if clipped and fiber_method == "doste":
        raise ValueError("The 'doste' fiber method needs the outflow tracts, use 'ldrb'")
    if aha and (geometry != "biv" or clipped):
        raise ValueError("AHA segments are only supported for the 'biv' geometry (not clipped)")
    if rv_regions and not aha:
        raise ValueError("rv_regions needs aha=True")
    if rv_septum_fraction is not None and not rv_regions:
        raise ValueError("rv_septum_fraction is only used with rv_regions=True")
    if rv_regions:
        rv_septum_fraction = 0.5 if rv_septum_fraction is None else float(rv_septum_fraction)
        if not 0.0 <= rv_septum_fraction <= 1.0:
            raise ValueError(f"rv_septum_fraction must be in [0, 1], got {rv_septum_fraction}")

    try:
        import ukb.cli
    except ImportError as e:
        msg = (
            "To create the UKB mesh you need to install the ukb package "
            "which you can install with pip install ukb-atlas"
        )
        raise ImportError(msg) from e

    if comm.rank == 0:
        surf_args = [
            "surf",
            str(outdir),
            "--mode",
            str(mode),
            "--std",
            str(std),
            "--case",
            case,
            "--geometry",
            geometry,
            "--refine",
            str(refine),
        ]
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
            "--order",
            str(order),
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

    gmsh_geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)
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
        "order": order,
        "geometry": geometry,
        "fiber_method": fiber_method,
        "refine": refine,
        "clipped": clipped,
        "timestamp": datetime.datetime.now().isoformat(),
        "use_burns": use_burns,
        "ghost_mode": ghost_mode.name,
        "aha": aha,
        "rv_regions": rv_regions,
        "rv_septum_fraction": rv_septum_fraction,
    }
    cfun, markers = gmsh_geometry.cfun, gmsh_geometry.markers
    if aha:
        from .aha import biv_aha

        if gmsh_geometry.ffun is None:
            raise ValueError("AHA segments need the facet tags of the mesh")
        landmarks_file = outdir / f"landmarks_{case}.json"
        if not landmarks_file.exists():
            raise ValueError(
                f"AHA segments need the landmarks in {landmarks_file}, "
                "written by ukb-atlas >= 2.2 (ukb-atlas surf)"
            )
        landmarks = json.loads(landmarks_file.read_text())
        if rv_regions:
            from .aha import biv_regions

            assert rv_septum_fraction is not None
            cfun, aha_markers = biv_regions(
                gmsh_geometry.mesh,
                gmsh_geometry.ffun,
                gmsh_geometry.markers,
                landmarks,
                rv_septum_fraction=rv_septum_fraction,
            )
        else:
            cfun, aha_markers = biv_aha(
                gmsh_geometry.mesh, gmsh_geometry.ffun, gmsh_geometry.markers, landmarks
            )
        # The AHA tags replace the cell tags
        markers = {k: v for k, v in markers.items() if v[1] != 3} | aha_markers
    if comm.rank == 0:
        (outdir / "markers.json").write_text(json.dumps(markers, default=utils.json_serial))
        (outdir / "info.json").write_text(json.dumps(info, default=utils.json_serial))

    fibers = {}
    if create_fibers:
        from .fibers import ldrb_fibers

        angles = {"alpha_endo_lv": fiber_angle_endo, "alpha_epi_lv": fiber_angle_epi}
        if fiber_method == "ldrb":
            angles |= {"beta_endo_lv": 0, "beta_epi_lv": 0}
        system, microstructure = ldrb_fibers(
            mesh=gmsh_geometry.mesh,
            ffun=gmsh_geometry.ffun,
            cfun=gmsh_geometry.cfun,
            markers=gmsh_geometry.markers,
            clipped=clipped,
            fiber_method=fiber_method,
            fiber_space=fiber_space,
            **angles,
        )

        fibers = microstructure._asdict()

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
        path=geometry_path(outdir, io_backend),
        mesh=gmsh_geometry.mesh,
        markers=markers,
        info=info,
        cfun=cfun,
        ffun=gmsh_geometry.ffun,
        efun=gmsh_geometry.efun,
        vfun=gmsh_geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
    )
    # geo = Geometry.from_folder(comm=comm, folder=outdir, ghost_mode=ghost_mode)
    # return geo
    return Geometry(
        mesh=gmsh_geometry.mesh,
        markers=markers,
        cfun=cfun,
        ffun=gmsh_geometry.ffun,
        efun=gmsh_geometry.efun,
        vfun=gmsh_geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
        info=info,
    )


def biv_ellipsoid(
    outdir: str | Path,
    char_length: float = 0.5,
    base_cut_z: float = 1.0,
    box_size: float = 20.0,  # Size of the cutting box
    rv_wall_thickness: float = 0.4,  # cm
    lv_wall_thickness: float = 0.5,  # cm
    rv_offset_x: float = 1.0,
    lv_radius_x: float = 2.2,
    lv_radius_y: float = 2.2,
    lv_radius_z: float = 4.5,
    rv_radius_x: float = 3.2,
    rv_radius_y: float = 2.3,
    rv_radius_z: float = 4.3,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    verbose: bool = False,
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
) -> Geometry:
    """Create BiV ellipsoidal geometry

    Parameters
    ----------
    outdir : str | Path
        Directory where to save the results.
    char_length : float, optional
        Characteristic length of mesh, by default 0.5
    box_size : float, optional
        Size of the cutting box, by default 20.0
    lv_radius_x : float, optional
        Radius of the left ventricle in the x-direction, by default 2.2
    lv_radius_y : float, optional
        Radius of the left ventricle in the y-direction, by default 2.2
    lv_radius_z : float, optional
        Radius of the left ventricle in the z-direction, by default 4.5
    rv_radius_x : float, optional
        Radius of the right ventricle in the x-direction, by default 3.2
    rv_radius_y : float, optional
        Radius of the right ventricle in the y-direction, by default 2.3
    rv_radius_z : float, optional
        Radius of the right ventricle in the z-direction, by default 4.3
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
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

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
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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
            order=order,
        )
    comm.barrier()
    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)

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

        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

    save_geometry(
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        info=info,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
    )

    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
        info=info,
    )


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
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
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
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

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
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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
            order=order,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)

    kwargs: dict[str, dolfinx.mesh.MeshTags | dolfinx.fem.Function | None] = {"cfun": geometry.cfun}
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

    cfun = kwargs.get("cfun", None)
    assert cfun is None or isinstance(cfun, dolfinx.mesh.MeshTags)
    f0 = kwargs.get("f0", None)
    assert f0 is None or isinstance(f0, dolfinx.fem.Function)
    s0 = kwargs.get("s0", None)
    assert s0 is None or isinstance(s0, dolfinx.fem.Function)
    n0 = kwargs.get("n0", None)
    assert n0 is None or isinstance(n0, dolfinx.fem.Function)

    save_geometry(
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=markers,
        info=info,
        cfun=cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=f0,
        s0=s0,
        n0=n0,
    )
    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=f0,
        s0=s0,
        n0=n0,
        info=info,
    )


def lv_ellipsoid_2D(
    outdir: Path | str,
    r_short_endo: float = 7.0,
    r_short_epi: float = 10.0,
    r_long_endo: float = 17.0,
    r_long_epi: float = 20.0,
    psize_ref: float = 3,
    mu_apex_endo: float = math.pi,
    mu_base_endo: float = math.acos(5 / 17),
    mu_apex_epi: float = math.pi,
    mu_base_epi: float = math.acos(5 / 20),
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    verbose: bool = False,
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
) -> Geometry:
    """Create the meridian of an LV ellipsoid, for axisymmetric problems

    The mesh lies in the plane z = 0, with the long axis along x and the radial
    direction along y, and has geometric dimension 2. The angles mu must be in
    [0, pi] so that y = r_short sin(mu) >= 0: unlike `lv_ellipsoid`, whose
    default angles are negative, the defaults here give the same ellipsoid
    with the meridian in the upper half plane. ``APEX`` marks
    the segment of the axis between the two apices, where an axisymmetric
    problem needs its symmetry condition.

    The fibers have three components: the axial, the radial and the
    circumferential one. The fiber angle varies linearly in the transmural
    coordinate of the ellipsoid itself (see
    `cardiac_geometries.fibers.lv_ellipsoid.transmural_coordinate`), as a
    planar Laplace problem on the meridian is not the axisymmetric one.

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
        Angle for the endocardial apex, by default math.pi
    mu_base_endo : float, optional
        Angle for the endocardial base, by default math.acos(5 / 17)
    mu_apex_epi : float, optional
        Angle for the epicardial apex, by default math.pi
    mu_base_epi : float, optional
        Angle for the epicardial base, by default math.acos(5 / 20)
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
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """

    for name, mu in [
        ("mu_apex_endo", mu_apex_endo),
        ("mu_base_endo", mu_base_endo),
        ("mu_apex_epi", mu_apex_epi),
        ("mu_base_epi", mu_base_epi),
    ]:
        if not 0.0 <= mu <= math.pi:
            raise ValueError(
                f"{name} = {mu} is not in [0, pi], so the meridian would not be at y >= 0"
            )

    outdir = Path(outdir)
    mesh_name = outdir / "lv_ellipsoid_2D.msh"
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
        "mesh_type": "lv_ellipsoid_2D",
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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

        cgc.lv_ellipsoid_2D(
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
            order=order,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode, gdim=2)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    f0 = s0 = n0 = None
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
            transmural="analytic",
        )
        f0, s0, n0 = system.f0, system.s0, system.n0

    save_geometry(
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        vfun=geometry.vfun,
        f0=f0,
        s0=s0,
        n0=n0,
    )
    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        vfun=geometry.vfun,
        f0=f0,
        s0=s0,
        n0=n0,
        info=info,
    )


def slab_dolfinx(
    comm,
    outdir: Path,
    lx: float = 20.0,
    ly: float = 7.0,
    lz: float = 3.0,
    dx: float = 1.0,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
) -> utils.GMshGeometry:
    mesh = dolfinx.mesh.create_box(
        comm,
        [[0.0, 0.0, 0.0], [lx, ly, lz]],
        [int(lx / dx), int(ly / dx), int(lz / dx)],
        dolfinx.mesh.CellType.tetrahedron,
        ghost_mode=ghost_mode,
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
        path=geometry_path(outdir, io_backend),
        mesh=mesh,
        markers=markers,
        cfun=None,
        ffun=ft,
        efun=None,
        vfun=None,
    )

    return utils.GMshGeometry(
        mesh=mesh,
        markers=markers,
        cfun=None,
        ffun=ft,
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
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
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
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """
    if use_dolfinx and order != 1:
        raise ValueError("Meshes of higher order are only supported with gmsh (use_dolfinx=False)")

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
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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
                order=order,
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
            ghost_mode=ghost_mode,
            io_backend=io_backend,
        )

    else:
        geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    fibers = {}
    if create_fibers:
        from .fibers.slab import create_microstructure

        assert geometry.ffun is not None
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
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
    )
    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
        info=info,
    )


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
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
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
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

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
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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
            order=order,
        )

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)
    comm.barrier()

    save_geometry(
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
    )
    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        info=info,
    )


def cylinder(
    outdir: Path | str,
    r_inner: float = 10.0,
    r_outer: float = 15.0,
    height: float = 40.0,
    floor_thickness: float = 0.0,
    roof_thickness: float = 0.0,
    char_length: float = 5.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    aha: bool = False,
    verbose: bool = False,
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
) -> Geometry:
    """Create a cylindrical geometry

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results.
    r_inner : float, optional
        Radius on the endocardium layer, by default 10.0
    r_outer : float, optional
       Radius on the epicardium layer, by default 15.0
    height : float, optional
        Longest radius on the endocardium layer, by default 40.0
    floor_thickness : float, optional
        Thickness of the floor, by default 0.0
    roof_thickness : float, optional
        Thickness of the roof, by default 0.0
    char_length : float, optional
        Characteristic length of mesh, by default 5.0
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
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

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
        "floor_thickness": floor_thickness,
        "roof_thickness": roof_thickness,
        "char_length": char_length,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "aha": aha,
        "mesh_type": "cylinder",
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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
            floor_thickness=floor_thickness,
            roof_thickness=roof_thickness,
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            verbose=verbose,
            order=order,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)

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
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
    )

    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
        info=info,
    )


def cylinder_elliptical(
    outdir: Path | str,
    r_inner_x: float = 10.0,
    r_inner_y: float = 10.0,
    r_outer_x: float = 20.0,
    r_outer_y: float = 20.0,
    height: float = 40.0,
    char_length: float = 10.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    aha: bool = False,
    verbose: bool = False,
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    io_backend: str = DEFAULT_IO_BACKEND,
) -> Geometry:
    """Create a cylindrical geometry

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results.
    r_inner_x : float, optional
        Radius on the endocardium layer along the x-axis, by default 10.0
    r_inner_y : float, optional
        Radius on the endocardium layer along the y-axis, by default 10.0
    r_outer_x : float, optional
       Radius on the epicardium layer along the x-axis, by default 20.0
    r_outer_y : float, optional
       Radius on the epicardium layer along the y-axis, by default 20.0
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
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

    Returns
    -------
    cardiac_geometries.geometry.Geometry
        A Geometry with the mesh, markers, markers functions and fibers.

    """

    outdir = Path(outdir)
    mesh_name = outdir / "cylinder.msh"
    info = {
        "r_inner_x": r_inner_x,
        "r_inner_y": r_inner_y,
        "r_outer_x": r_outer_x,
        "r_outer_y": r_outer_y,
        "height": height,
        "char_length": char_length,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "aha": aha,
        "mesh_type": "cylinder_elliptical",
        "order": order,
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

        cgc.cylinder_elliptical(
            inner_radius_x=r_inner_x,
            inner_radius_y=r_inner_y,
            outer_radius_x=r_outer_x,
            outer_radius_y=r_outer_y,
            height=height,
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            verbose=verbose,
            order=order,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    fibers = {}
    if create_fibers:
        from .fibers.cylinder_elliptical import create_microstructure

        system = create_microstructure(
            mesh=geometry.mesh,
            function_space=fiber_space,
            r_inner_x=r_inner_x,
            r_inner_y=r_inner_y,
            r_outer_x=r_outer_x,
            r_outer_y=r_outer_y,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
        )

        fibers = {"f0": system.f0, "s0": system.s0, "n0": system.n0}

    save_geometry(
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
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
    fillet_radius: float | None = None,
    verbose: bool = False,
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
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
    fillet_radius : float | None, optional
        Radius for the fillet on the flat faces. If None no fillet is created.
        By default None.
    verbose : bool, optional
        If True print information from gmsh, by default False
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

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
        "fillet_radius": fillet_radius,
        "mesh_type": "cylinder_racetrack",
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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

        cgc.cylinder_cut(
            inner_radius=r_inner,
            outer_radius=r_outer,
            inner_flat_face_distance=inner_flat_face_distance,
            outer_flat_face_distance=outer_flat_face_distance,
            height=height,
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            verbose=verbose,
            order=order,
            mode="racetrack",
            fillet_radius=fillet_radius,
        )

    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)

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
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
    )

    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
        info=info,
    )


def cylinder_D_shaped(
    outdir: Path | str,
    r_inner: float = 13.0,
    r_outer: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    floor_thickness: float = 0.0,
    roof_thickness: float = 0.0,
    char_length: float = 10.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    aha: bool = False,
    fillet_radius: float | None = None,
    verbose: bool = False,
    order: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
    ghost_mode: dolfinx.mesh.GhostMode = dolfinx.mesh.GhostMode.none,
    io_backend: str = DEFAULT_IO_BACKEND,
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
    floor_thickness : float, optional
        Thickness of the floor, by default 0.0
    roof_thickness : float, optional
        Thickness of the roof, by default 0.0
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
    fillet_radius : float | None, optional
        Radius for the fillet on the flat faces. If None no fillet is created.
        By default None.
    verbose : bool, optional
        If True print information from gmsh, by default False
    order : int, optional
        Order of the mesh elements. With order higher than one the elements
        are curved to follow the boundaries, by default 1
    comm : MPI.Comm, optional
        MPI communicator, by default MPI.COMM_WORLD
    ghost_mode : dolfinx.mesh.GhostMode, optional
        Ghost mode for the dolfinx mesh, by default dolfinx.mesh.GhostMode.none
    io_backend : str, optional
        io4dolfinx backend used to write the geometry, "adios2" (``geometry.bp``)
        or "h5py" (``geometry.h5``), by default "adios2"

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
        "floor_thickness": floor_thickness,
        "roof_thickness": roof_thickness,
        "char_length": char_length,
        "create_fibers": create_fibers,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "fiber_space": fiber_space,
        "aha": aha,
        "fillet_radius": fillet_radius,
        "mesh_type": "cylinder_D_shaped",
        "order": order,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
        "ghost_mode": ghost_mode.name,
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

        if fillet_radius is not None and np.isclose(fillet_radius, 0.0):
            cgc.cylinder_D_shaped(
                inner_radius=r_inner,
                outer_radius=r_outer,
                inner_flat_face_distance=inner_flat_face_distance,
                outer_flat_face_distance=outer_flat_face_distance,
                floor_thickness=floor_thickness,
                roof_thickness=roof_thickness,
                height=height,
                mesh_name=mesh_name.as_posix(),
                char_length=char_length,
                verbose=verbose,
                order=order,
            )
        else:
            cgc.cylinder_cut(
                inner_radius=r_inner,
                outer_radius=r_outer,
                inner_flat_face_distance=inner_flat_face_distance,
                outer_flat_face_distance=outer_flat_face_distance,
                floor_thickness=floor_thickness,
                roof_thickness=roof_thickness,
                height=height,
                mesh_name=mesh_name.as_posix(),
                char_length=char_length,
                verbose=verbose,
                order=order,
                mode="d_shaped",
                fillet_radius=fillet_radius,
            )

    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name, ghost_mode=ghost_mode)

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
        path=geometry_path(outdir, io_backend),
        mesh=geometry.mesh,
        markers=geometry.markers,
        info=info,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
    )

    return Geometry(
        mesh=geometry.mesh,
        markers=geometry.markers,
        cfun=geometry.cfun,
        ffun=geometry.ffun,
        efun=geometry.efun,
        vfun=geometry.vfun,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
        info=info,
    )


def _read_surfaces(surfaces: dict, scale: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read triangle surfaces (any format pyvista reads, e.g. PLY, VTP, STL) as (points,
    triangles)."""
    try:
        import pyvista
    except ImportError as e:
        raise ImportError(
            "Reading surfaces needs pyvista: pip install cardiac-geometriesx[pyvista]"
        ) from e
    out = {}
    for name, source in surfaces.items():
        surface = pyvista.read(source)
        if not isinstance(surface, pyvista.PolyData):
            surface = surface.extract_surface()
        if not surface.is_all_triangles:
            raise ValueError(f"Surface {name!r} ({source}) is not a triangle surface")
        out[name] = (
            np.asarray(surface.points, dtype=float) * scale,
            np.asarray(surface.regular_faces, dtype=np.int64),
        )
    return out


def from_volume(
    outdir: str | Path,
    volume,
    surfaces: dict | None = None,
    rule=None,
    rule_data: list[str] | None = None,
    scale: float = 1.0,
    tol: float | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
    create_fibers: bool = False,
    fiber_method: str = "ldrb",
    fiber_space: str = "P_1",
    fiber_angle_endo: float = 60.0,
    fiber_angle_epi: float = -60.0,
    io_backend: str = DEFAULT_IO_BACKEND,
) -> Geometry:
    """A geometry from a tetrahedral volume mesh and labelled boundary surfaces, e.g. a patient
    mesh made with another tool.

    The volume is read with io4dolfinx's pyvista backend, already distributed, and every boundary
    facet gets exactly one marker, either from the surface meshes in ``surfaces`` (marker name to a
    file pyvista reads, e.g. PLY; a facet gets the name of the surface with a triangle on its three
    vertices) or from ``rule`` (see :func:`cardiac_geometries.labelled.label_facets`). Use the
    marker
    names of the UKB geometries (``LV_ENDO``, ``RV_ENDO``, ``EPI``, ``BASE``, ``MV``, ``AV``,
    ``TV``, ``PV``) so the fiber methods find their boundaries: Bayer's base is ``BASE`` plus the
    valves, and the method of Doste et al. needs the valves.

    Parameters
    ----------
    outdir : str | Path
        Folder for the markers and the saved geometry
    volume : str | Path
        The volume mesh (linear tetrahedra), a file io4dolfinx's pyvista backend reads (e.g. VTU,
        VTK)
    surfaces : dict | None
        Marker name to a triangle surface file
    rule : Callable | None
        ``rule(coords, data)`` returns the marker name of each boundary facet, from its vertex
        coordinates (F, 3, 3, scaled) and ``data``: the point data named in ``rule_data`` at those
        vertices, each (F, 3)
    rule_data : list[str] | None
        Names of point data in the volume file that the rule needs
    scale : float
        Multiplies all coordinates (volume and surfaces), e.g. 1e-3 from micrometres to mm.
        Never guessed
    tol : float | None
        Distance within which a surface point matches a facet vertex (after scaling); by default
        round-off (1e-6 of the surfaces' size)
    comm : MPI.Comm
        The communicator
    create_fibers, fiber_method, fiber_space, fiber_angle_endo, fiber_angle_epi
        As in :func:`ukb`: fibers by the method of Bayer et al. ("ldrb") or Doste et al. ("doste")
    io_backend : str
        io4dolfinx backend for the saved geometry

    Returns
    -------
    Geometry
        The mesh, markers, facet tags and (optionally) fibers
    """
    import io4dolfinx

    from .labelled import check_labels, label_facets, marker_ids

    if (surfaces is None) == (rule is None):
        raise ValueError("Give exactly one of surfaces and rule")
    outdir = Path(outdir)
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    # Surfaces are read on rank 0 (so a reading error is raised on every rank) and sent to all
    surface_data: dict | None = None
    error: Exception | None = None
    if surfaces is not None and comm.rank == 0:
        try:
            surface_data = _read_surfaces(surfaces, scale)
        except Exception as e:
            error = e
    error = comm.bcast(error, root=0)
    if error is not None:
        raise error
    surface_data = comm.bcast(surface_data, root=0)

    try:
        import pyvista  # noqa: F401  (io4dolfinx's pyvista backend needs it)
    except ImportError as e:
        raise ImportError(
            "Reading the volume needs pyvista: pip install cardiac-geometriesx[pyvista]"
        ) from e
    mesh = io4dolfinx.read_mesh(Path(volume), comm, backend="pyvista")
    if mesh.topology.cell_name() != "tetrahedron":
        raise ValueError(f"The volume must be a tetrahedral mesh, got {mesh.topology.cell_name()}")
    mesh.geometry.x[:] *= scale

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    num_owned = mesh.topology.index_map(tdim - 1).size_local
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    facets = facets[facets < num_owned].astype(np.int32)
    geometry_dofs = dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, facets)
    coords = mesh.geometry.x[geometry_dofs]

    data: dict[str, np.ndarray] = {}
    for name in rule_data or []:
        f = io4dolfinx.read_point_data(Path(volume), name, mesh, backend="pyvista")
        from scipy.spatial import cKDTree

        x = f.function_space.tabulate_dof_coordinates()
        _, dof = cKDTree(x).query(coords.reshape(-1, 3))
        data[name] = np.asarray(f.x.array, dtype=float)[dof].reshape(-1, 3)

    names, matched = label_facets(coords, surfaces=surface_data, rule=rule, data=data, tol=tol)
    sizes = {k: len(t) for k, (_, t) in surface_data.items()} if surface_data else {}
    check_labels(names, matched, sizes, comm=comm)
    all_names = set().union(*comm.allgather(set(names.tolist())))
    markers = marker_ids(all_names)

    values = np.array([markers[n][0] for n in names], dtype=np.int32)
    order = np.argsort(facets)
    ffun = dolfinx.mesh.meshtags(mesh, tdim - 1, facets[order], values[order])
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    cfun = dolfinx.mesh.meshtags(
        mesh, tdim, cells, np.full(len(cells), markers["Wall"][0], np.int32)
    )
    ffun.name, cfun.name = "Facet tags", "Cell tags"
    info = {
        "mesh_type": "from_volume",
        "volume": str(volume),
        "rule_data": list(rule_data or []),
        "scale": scale,
        "tol": tol,
        "fiber_method": fiber_method if create_fibers else None,
        "fiber_space": fiber_space,
        "fiber_angle_endo": fiber_angle_endo,
        "fiber_angle_epi": fiber_angle_epi,
        "cardiac_geometry_version": __version__,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if comm.rank == 0:
        (outdir / "markers.json").write_text(json.dumps(markers, default=utils.json_serial))
        (outdir / "info.json").write_text(json.dumps(info, default=utils.json_serial))

    fibers: dict = {}
    if create_fibers:
        from .fibers import ldrb_fibers

        if fiber_method == "doste":
            doste_markers(markers)  # a clear error without the valves
        angles: dict[str, typing.Any] = {
            "alpha_endo_lv": fiber_angle_endo,
            "alpha_epi_lv": fiber_angle_epi,
        }
        if fiber_method == "ldrb":
            angles |= {"beta_endo_lv": 0, "beta_epi_lv": 0}
        _, microstructure = ldrb_fibers(
            mesh=mesh,
            ffun=ffun,
            cfun=cfun,
            markers=markers,
            fiber_method=fiber_method,
            fiber_space=fiber_space,
            **angles,
        )
        fibers = microstructure._asdict()

    save_geometry(
        path=geometry_path(outdir, io_backend),
        mesh=mesh,
        markers=markers,
        info=info,
        cfun=cfun,
        ffun=ffun,
        efun=None,
        vfun=None,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
    )
    return Geometry(
        mesh=mesh,
        markers=markers,
        cfun=cfun,
        ffun=ffun,
        efun=None,
        vfun=None,
        f0=fibers.get("f0", None),
        s0=fibers.get("s0", None),
        n0=fibers.get("n0", None),
        info=info,
    )
