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
from .geometry import Geometry

meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]

logger = get_logger()


def transform_markers(
    markers: dict[str, tuple[int, int]], clipped: bool = False
) -> dict[str, list[int]]:
    if clipped:
        return {
            "lv": [markers["LV"][0]],
            "rv": [markers["RV"][0]],
            "epi": [markers["EPI"][0]],
            "base": [markers["BASE"][0]],
        }
    else:
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

    if comm.rank == 0:
        (outdir / "markers.json").write_text(
            json.dumps(geometry.markers, default=utils.json_serial)
        )
        (outdir / "info.json").write_text(
            json.dumps(
                {
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
            )
        )

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

        for k, v in system._asdict().items():
            if v is None:
                continue
            if fiber_space.startswith("Q"):
                # Cannot visualize Quadrature spaces yet
                continue

            logger.debug(f"Write {k}: {v}")
            with dolfinx.io.VTXWriter(comm, outdir / f"{k}-viz.bp", [v], engine="BP4") as vtx:
                vtx.write(0.0)

    geo = Geometry.from_folder(comm=comm, folder=outdir)
    return geo


def biv_ellipsoid(
    outdir: str | Path,
    char_length: float = 0.5,
    center_lv_x: float = 0.0,
    center_lv_y: float = 0.0,
    center_lv_z: float = 0.0,
    a_endo_lv: float = 2.5,
    b_endo_lv: float = 1.0,
    c_endo_lv: float = 1.0,
    a_epi_lv: float = 3.0,
    b_epi_lv: float = 1.5,
    c_epi_lv: float = 1.5,
    center_rv_x: float = 0.0,
    center_rv_y: float = 0.5,
    center_rv_z: float = 0.0,
    a_endo_rv: float = 3.0,
    b_endo_rv: float = 1.5,
    c_endo_rv: float = 1.5,
    a_epi_rv: float = 4.0,
    b_epi_rv: float = 2.5,
    c_epi_rv: float = 2.0,
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
    center_lv_y : float, optional
        X-coordinate for the center of the lv, by default 0.0
    center_lv_y : float, optional
        Y-coordinate for the center of the lv, by default 0.0
    center_lv_z : float, optional
        Z-coordinate for the center of the lv, by default 0.0
    a_endo_lv : float, optional
        Dilation of lv endo ellipsoid in the x-direction, by default 2.5
    b_endo_lv : float, optional
       Dilation of lv endo ellipsoid in the y-direction, by default 1.0
    c_endo_lv : float, optional
       Dilation of lv endo ellipsoid in the z-direction, by default 1.0
    a_epi_lv : float, optional
        Dilation of lv epi ellipsoid in the x-direction, by default 3.0
    b_epi_lv : float, optional
        Dilation of lv epi ellipsoid in the y-direction, by default 1.5
    c_epi_lv : float, optional
        Dilation of lv epi ellipsoid in the z-direction, by default 1.5
    center_rv_x : float, optional
        X-coordinate for the center of the rv, by default 0.0
    center_rv_y : float, optional
        Y-coordinate for the center of the rv, by default 0.5
    center_rv_z : float, optional
        Z-coordinate for the center of the rv, by default 0.0
    a_endo_rv : float, optional
       Dilation of rv endo ellipsoid in the x-direction, by default 3.0
    b_endo_rv : float, optional
       Dilation of rv endo ellipsoid in the y-direction, by default 1.5
    c_endo_rv : float, optional
       Dilation of rv endo ellipsoid in the z-direction, by default 1.5
    a_epi_rv : float, optional
        Dilation of rv epi ellipsoid in the x-direction, by default 4.0
    b_epi_rv : float, optional
        Dilation of rv epi ellipsoid in the y-direction, by default 2.5
    c_epi_rv : float, optional
        Dilation of rv epi ellipsoid in the z-direction, by default 2.0
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
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                {
                    "char_length": char_length,
                    "center_lv_x": center_lv_x,
                    "center_lv_y": center_lv_y,
                    "center_lv_z": center_lv_z,
                    "a_endo_lv": a_endo_lv,
                    "b_endo_lv": b_endo_lv,
                    "c_endo_lv": c_endo_lv,
                    "a_epi_lv": a_epi_lv,
                    "b_epi_lv": b_epi_lv,
                    "c_epi_lv": c_epi_lv,
                    "center_rv_x": center_rv_x,
                    "center_rv_y": center_rv_y,
                    "center_rv_z": center_rv_z,
                    "a_endo_rv": a_endo_rv,
                    "b_endo_rv": b_endo_rv,
                    "c_endo_rv": c_endo_rv,
                    "a_epi_rv": a_epi_rv,
                    "b_epi_rv": b_epi_rv,
                    "c_epi_rv": c_epi_rv,
                    "create_fibers": create_fibers,
                    "fiber_angle_endo": fiber_angle_endo,
                    "fiber_angle_epi": fiber_angle_epi,
                    "fiber_space": fiber_space,
                    "mesh_type": "biv_ellipsoid",
                    "cardiac_geometry_version": __version__,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.biv_ellipsoid(
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            center_lv_x=center_lv_x,
            center_lv_y=center_lv_y,
            center_lv_z=center_lv_z,
            a_endo_lv=a_endo_lv,
            b_endo_lv=b_endo_lv,
            c_endo_lv=c_endo_lv,
            a_epi_lv=a_epi_lv,
            b_epi_lv=b_epi_lv,
            c_epi_lv=c_epi_lv,
            center_rv_x=center_rv_x,
            center_rv_y=center_rv_y,
            center_rv_z=center_rv_z,
            a_endo_rv=a_endo_rv,
            b_endo_rv=b_endo_rv,
            c_endo_rv=c_endo_rv,
            a_epi_rv=a_epi_rv,
            b_epi_rv=b_epi_rv,
            c_epi_rv=c_epi_rv,
            verbose=verbose,
        )
    comm.barrier()
    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)
    comm.barrier()
    if create_fibers:
        try:
            import ldrb
        except ImportError:
            msg = (
                "To create fibers you need to install the ldrb package "
                "which you can install with pip install fenicsx-ldrb"
            )
            raise ImportError(msg)

        system = ldrb.dolfinx_ldrb(
            mesh=geometry.mesh,
            ffun=geometry.ffun,
            markers=geometry.markers,
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

    geo = Geometry.from_folder(comm=comm, folder=outdir)
    return geo


def biv_ellipsoid_torso(
    outdir: str | Path,
    char_length: float = 0.5,
    heart_as_surface: bool = False,
    torso_length: float = 20.0,
    torso_width: float = 20.0,
    torso_height: float = 20.0,
    rotation_angle: float = math.pi / 6,
    center_lv_x: float = 0.0,
    center_lv_y: float = 0.0,
    center_lv_z: float = 0.0,
    a_endo_lv: float = 2.5,
    b_endo_lv: float = 1.0,
    c_endo_lv: float = 1.0,
    a_epi_lv: float = 3.0,
    b_epi_lv: float = 1.5,
    c_epi_lv: float = 1.5,
    center_rv_x: float = 0.0,
    center_rv_y: float = 0.5,
    center_rv_z: float = 0.0,
    a_endo_rv: float = 3.0,
    b_endo_rv: float = 1.5,
    c_endo_rv: float = 1.5,
    a_epi_rv: float = 4.0,
    b_epi_rv: float = 2.5,
    c_epi_rv: float = 2.0,
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
    heart_as_surface: bool
        If true, create the heart as a a surface inside the torso,
        otherwise let the heart be a volume, by default True.
    torso_length : float, optional
        Length of torso in the x-direction, by default 20.0
    torso_width : float, optional
        Length of torso in the y-direction, by default 20.0
    torso_height : float, optional
        Length of torso in the z-direction, by default 20.0
    rotation_angle: float, optional
        Angle to rotate the torso in order to object realistic position of
        the heart in a torso, by default pi / 6
    center_lv_x : float, optional
        X-coordinate for the center of the lv, by default 0.0
    center_lv_y : float, optional
        Y-coordinate for the center of the lv, by default 0.0
    center_lv_z : float, optional
        Z-coordinate for the center of the lv, by default 0.0
    a_endo_lv : float, optional
        Dilation of lv endo ellipsoid in the x-direction, by default 2.5
    b_endo_lv : float, optional
       Dilation of lv endo ellipsoid in the y-direction, by default 1.0
    c_endo_lv : float, optional
       Dilation of lv endo ellipsoid in the z-direction, by default 1.0
    a_epi_lv : float, optional
        Dilation of lv epi ellipsoid in the x-direction, by default 3.0
    b_epi_lv : float, optional
        Dilation of lv epi ellipsoid in the y-direction, by default 1.5
    c_epi_lv : float, optional
        Dilation of lv epi ellipsoid in the z-direction, by default 1.5
    center_rv_x : float, optional
        X-coordinate for the center of the rv, by default 0.0
    center_rv_y : float, optional
        Y-coordinate for the center of the rv, by default 0.5
    center_rv_z : float, optional
        Z-coordinate for the center of the rv, by default 0.0
    a_endo_rv : float, optional
       Dilation of rv endo ellipsoid in the x-direction, by default 3.0
    b_endo_rv : float, optional
       Dilation of rv endo ellipsoid in the y-direction, by default 1.5
    c_endo_rv : float, optional
       Dilation of rv endo ellipsoid in the z-direction, by default 1.5
    a_epi_rv : float, optional
        Dilation of rv epi ellipsoid in the x-direction, by default 4.0
    b_epi_rv : float, optional
        Dilation of rv epi ellipsoid in the y-direction, by default 2.5
    c_epi_rv : float, optional
        Dilation of rv epi ellipsoid in the z-direction, by default 2.0
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
    mesh_name = outdir / "biv_ellipsoid_torso.msh"
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                {
                    "char_length": char_length,
                    "heart_as_surface": heart_as_surface,
                    "torso_length": torso_length,
                    "torso_width": torso_width,
                    "torso_height": torso_height,
                    "rotation_angle": rotation_angle,
                    "center_lv_x": center_lv_x,
                    "center_lv_y": center_lv_y,
                    "center_lv_z": center_lv_z,
                    "a_endo_lv": a_endo_lv,
                    "b_endo_lv": b_endo_lv,
                    "c_endo_lv": c_endo_lv,
                    "a_epi_lv": a_epi_lv,
                    "b_epi_lv": b_epi_lv,
                    "c_epi_lv": c_epi_lv,
                    "center_rv_x": center_rv_x,
                    "center_rv_y": center_rv_y,
                    "center_rv_z": center_rv_z,
                    "a_endo_rv": a_endo_rv,
                    "b_endo_rv": b_endo_rv,
                    "c_endo_rv": c_endo_rv,
                    "a_epi_rv": a_epi_rv,
                    "b_epi_rv": b_epi_rv,
                    "c_epi_rv": c_epi_rv,
                    "create_fibers": create_fibers,
                    "fiber_angle_endo": fiber_angle_endo,
                    "fiber_angle_epi": fiber_angle_epi,
                    "fiber_space": fiber_space,
                    "mesh_type": "biv",
                    "cardiac_geometry_version": __version__,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                f,
                indent=2,
                default=utils.json_serial,
            )

        cgc.biv_ellipsoid_torso(
            mesh_name=mesh_name.as_posix(),
            char_length=char_length,
            heart_as_surface=heart_as_surface,
            torso_length=torso_length,
            torso_height=torso_height,
            torso_width=torso_width,
            rotation_angle=rotation_angle,
            center_lv_x=center_lv_x,
            center_lv_y=center_lv_y,
            center_lv_z=center_lv_z,
            a_endo_lv=a_endo_lv,
            b_endo_lv=b_endo_lv,
            c_endo_lv=c_endo_lv,
            a_epi_lv=a_epi_lv,
            b_epi_lv=b_epi_lv,
            c_epi_lv=c_epi_lv,
            center_rv_x=center_rv_x,
            center_rv_y=center_rv_y,
            center_rv_z=center_rv_z,
            a_endo_rv=a_endo_rv,
            b_endo_rv=b_endo_rv,
            c_endo_rv=c_endo_rv,
            a_epi_rv=a_epi_rv,
            b_epi_rv=b_epi_rv,
            c_epi_rv=c_epi_rv,
            verbose=verbose,
        )
    comm.barrier()

    geometry = utils.gmsh2dolfin(comm=comm, msh_file=mesh_name)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)
    comm.barrier()

    if create_fibers:
        if heart_as_surface:
            logger.warning("Can only create fibers when heart is a volume.")
        else:
            raise NotImplementedError("Fibers not implemented yet for biv ellipsoid.")
            # from .fibers._biv_ellipsoid import create_biv_in_torso_fibers

            # create_biv_in_torso_fibers(
            #     mesh=geometry.mesh,
            #     ffun=geometry.marker_functions.ffun,
            #     cfun=geometry.marker_functions.cfun,
            #     markers=geometry.markers,
            #     fiber_space=fiber_space,
            #     alpha_endo=fiber_angle_endo,
            #     alpha_epi=fiber_angle_epi,
            #     outdir=outdir,
            # )

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
    aha: bool = True,
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
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                {
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
                    "mesh_type": "lv_ellipsoid",
                    "cardiac_geometry_version": __version__,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
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

    # if aha:
    #     from .aha import lv_aha

    #     geometry = lv_aha(
    #         geometry=geometry,
    #         r_long_endo=r_long_endo,
    #         r_short_endo=r_short_endo,
    #         mu_base=mu_base_endo,
    #     )
    #     from dolfin import XDMFFile

    #     with XDMFFile((outdir / "cfun.xdmf").as_posix()) as xdmf:
    #         xdmf.write(geometry.marker_functions.cfun)

    if comm.rank == 0:
        with open(outdir / "markers.json", "w") as f:
            json.dump(geometry.markers, f, default=utils.json_serial)

    if create_fibers:
        from .fibers.lv_ellipsoid import create_microstructure

        create_microstructure(
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

    geo = Geometry.from_folder(comm=comm, folder=outdir)
    # if aha:
    #     # Update schema
    #     from .geometry import H5Path

    #     cfun = geo.schema["cfun"].to_dict()
    #     cfun["fname"] = "cfun.xdmf:f"
    #     geo.schema["cfun"] = H5Path(**cfun)

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
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                {
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
                },
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

    if create_fibers:
        from .fibers.slab import create_microstructure

        create_microstructure(
            mesh=geometry.mesh,
            ffun=geometry.ffun,
            markers=geometry.markers,
            function_space=fiber_space,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
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
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                {
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
                },
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
    aha: bool = True,
    verbose: bool = False,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Geometry:
    """Create an LV ellipsoidal geometry

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
    if comm.rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)

        with open(outdir / "info.json", "w") as f:
            json.dump(
                {
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
                },
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

    if create_fibers:
        from .fibers.cylinder import create_microstructure

        create_microstructure(
            mesh=geometry.mesh,
            function_space=fiber_space,
            r_inner=r_inner,
            r_outer=r_outer,
            alpha_endo=fiber_angle_endo,
            alpha_epi=fiber_angle_epi,
            outdir=outdir,
        )

    geo = Geometry.from_folder(comm=comm, folder=outdir)

    return geo
