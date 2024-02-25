from pathlib import Path
import math
import json
import tempfile
import datetime
from importlib.metadata import metadata

import cardiac_geometries_core as cgc

from . import utils

meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]


def lv_ellipsoid(
    outdir: Path | str | None = None,
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
    fiber_angle_endo: float = -60,
    fiber_angle_epi: float = +60,
    fiber_space: str = "P_1",
    aha: bool = True,
) -> None:
    """Create an LV ellipsoidal geometry

    Parameters
    ----------
    outdir : Optional[Path], optional
        Directory where to save the results. If not provided a temporary
        directory will be created, by default None
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
        Angle for the endocardium, by default -60
    fiber_angle_epi : float, optional
        Angle for the epicardium, by default +60
    fiber_space : str, optional
        Function space for fibers of the form family_degree, by default "P_1"
    aha : bool, optional
        If True create 17-segment AHA regions

    Returns
    -------
    Optional[Geometry]
        A Geometry with the mesh, markers, markers functions and fibers.
        Returns None if dolfin is not installed.

    Raises
    ------
    ImportError
        If gmsh is not installed
    """

    _tmpfile = None
    if outdir is None:
        _tmpfile = tempfile.TemporaryDirectory()
        outdir = _tmpfile.__enter__()

    outdir = Path(outdir)
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
                "fibers_angle_endo": fiber_angle_endo,
                "fibers_angle_epi": fiber_angle_epi,
                "fiber_space": fiber_space,
                "aha": aha,
                # "mesh_type": MeshTypes.lv_ellipsoid.value,
                "cardiac_geometry_version": __version__,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=utils.json_serial,
        )

    mesh_name = outdir / "lv_ellipsoid.msh"
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
    )

    geometry = utils.gmsh2dolfin(mesh_name, unlink=False)

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

    with open(outdir / "markers.json", "w") as f:
        json.dump(geometry.markers, f, default=utils.json_serial)

    if create_fibers:
        from .fibers._lv_ellipsoid import create_microstructure

        f0, s0, n0 = create_microstructure(
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

    # geo = Geometry.from_folder(outdir)
    # if aha:
    #     # Update schema
    #     from .geometry import H5Path

    #     cfun = geo.schema["cfun"].to_dict()
    #     cfun["fname"] = "cfun.xdmf:f"
    #     geo.schema["cfun"] = H5Path(**cfun)

    if _tmpfile is not None:
        _tmpfile.__exit__(None, None, None)

    # return geo
    return None
