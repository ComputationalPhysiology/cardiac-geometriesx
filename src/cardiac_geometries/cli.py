import logging
import math
from importlib.metadata import metadata
from pathlib import Path

from mpi4py import MPI

import rich_click as click

from . import mesh

logger = logging.getLogger(__name__)

meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


def init_logging(verbose: bool = False):
    loglevel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=loglevel)
    for name in ["numba"]:
        logging.getLogger(name).setLevel(logging.WARNING)


@click.group()
@click.version_option(__version__, prog_name="cardiac_geometriesx")
def app():
    """
    Cardiac Geometries - A library for creating meshes of
    cardiac geometries
    """
    pass


@click.command(help="Create UK Biobank geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--mode",
    default=-1,
    type=int,
    help="Mode of the geometry",
    show_default=True,
)
@click.option(
    "--std",
    default=1.5,
    type=float,
    help="Standard deviation of the geometry",
    show_default=True,
)
@click.option(
    "--case",
    default="ED",
    type=str,
    help="Case of the geometry",
    show_default=True,
)
@click.option(
    "--char-length-max",
    default=2.0,
    type=float,
    help="Maximum characteristic length of the mesh",
    show_default=True,
)
@click.option(
    "--char-length-min",
    default=2.0,
    type=float,
    help="Minimum characteristic length of the mesh",
    show_default=True,
)
@click.option(
    "--fiber-angle-endo",
    default=60,
    type=float,
    help="Angle for the endocardium",
    show_default=True,
)
@click.option(
    "--fiber-angle-epi",
    default=-60,
    type=float,
    help="Angle for the epicardium",
    show_default=True,
)
@click.option(
    "--fiber-space",
    default="P_1",
    type=str,
    help="Function space for fibers of the form family_degree",
    show_default=True,
)
@click.option(
    "--clipped/--no-clipped",
    default=False,
    is_flag=True,
    type=bool,
    help="If True create clip away the outflow tracts",
    show_default=True,
)
@click.option(
    "--create-fibers",
    default=False,
    is_flag=True,
    help="If True create rule-based fibers",
    show_default=True,
)
def ukb(
    outdir: Path | str,
    mode: int = -1,
    std: float = 1.5,
    case: str = "ED",
    char_length_max: float = 2.0,
    char_length_min: float = 2.0,
    fiber_angle_endo: float = 60,
    fiber_angle_epi: float = -60,
    fiber_space: str = "P_1",
    clipped: bool = False,
    create_fibers: bool = True,
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.ukb(
        outdir=outdir,
        mode=mode,
        std=std,
        case=case,
        char_length_max=char_length_max,
        char_length_min=char_length_min,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
        fiber_space=fiber_space,
        clipped=clipped,
        create_fibers=create_fibers,
    )
    geo.save(outdir / "ukb.bp")


@click.command(help="Create LV ellipsoidal geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--r-short-endo",
    default=7.0,
    type=float,
    help="Shortest radius on the endocardium layer",
    show_default=True,
)
@click.option(
    "--r-short-epi",
    default=10.0,
    type=float,
    help="Shortest radius on the epicardium layer",
    show_default=True,
)
@click.option(
    "--r-long-endo",
    default=17.0,
    type=float,
    help="Longest radius on the endocardium layer",
    show_default=True,
)
@click.option(
    "--r-long-epi",
    default=20.0,
    type=float,
    help="Longest radius on the epicardium layer",
    show_default=True,
)
@click.option(
    "--psize-ref",
    default=3.0,
    type=float,
    help="The reference point size (smaller values yield as finer mesh",
    show_default=True,
)
@click.option(
    "--mu-apex-endo",
    default=-math.pi,
    type=float,
    help="Angle for the endocardial apex",
    show_default=True,
)
@click.option(
    "--mu-base-endo",
    default=-math.acos(5 / 17),
    type=float,
    help="Angle for the endocardial base",
    show_default=True,
)
@click.option(
    "--mu-apex-epi",
    default=-math.pi,
    type=float,
    help="Angle for the epicardial apex",
    show_default=True,
)
@click.option(
    "--mu-base-epi",
    default=-math.acos(5 / 20),
    type=float,
    help="Angle for the epicardial base",
    show_default=True,
)
@click.option(
    "--create-fibers",
    default=False,
    is_flag=True,
    help="If True create analytic fibers",
    show_default=True,
)
@click.option(
    "--fiber-angle-endo",
    default=-60,
    type=float,
    help="Angle for the endocardium",
    show_default=True,
)
@click.option(
    "--fiber-angle-epi",
    default=+60,
    type=float,
    help="Angle for the epicardium",
    show_default=True,
)
@click.option(
    "--fiber-space",
    default="P_1",
    type=str,
    help="Function space for fibers of the form family_degree",
    show_default=True,
)
@click.option(
    "--aha/--no-aha",
    default=False,
    is_flag=True,
    type=bool,
    help="If True create 17-segment AHA regions",
    show_default=True,
)
@click.option(
    "--dmu-factor",
    default=1 / 4,
    type=float,
    help="Factor for adjusting the thickness of the AHA segments",
    show_default=True,
)
def lv_ellipsoid(
    outdir: Path,
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
    aha: bool = False,
    dmu_factor: float = 1 / 4,
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.lv_ellipsoid(
        outdir=outdir,
        r_short_endo=r_short_endo,
        r_short_epi=r_short_epi,
        r_long_endo=r_long_endo,
        r_long_epi=r_long_epi,
        mu_base_endo=mu_base_endo,
        mu_base_epi=mu_base_epi,
        mu_apex_endo=mu_apex_endo,
        mu_apex_epi=mu_apex_epi,
        psize_ref=psize_ref,
        create_fibers=create_fibers,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
        fiber_space=fiber_space,
        aha=aha,
        dmu_factor=dmu_factor,
    )
    geo.save(outdir / "lv_ellipsoid.bp")


@click.command(help="Create BiV ellipsoidal geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--char-length",
    default=0.5,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--base-cut-z",
    default=2.5,
    type=float,
    help="Z-coordinate of the base cut",
    show_default=True,
)
@click.option(
    "--box-size",
    default=15.0,
    type=float,
    help="Size of the cutting box",
    show_default=True,
)
@click.option(
    "--rv-wall-thickness",
    default=0.4,
    type=float,
    help="Thickness of the right ventricle wall",
    show_default=True,
)
@click.option(
    "--lv-wall-thickness",
    default=0.5,
    type=float,
    help="Thickness of the left ventricle wall",
    show_default=True,
)
@click.option(
    "--rv-offset-x",
    default=3.0,
    type=float,
    help="X-offset of the right ventricle",
    show_default=True,
)
@click.option(
    "--lv-radius-x",
    default=2.0,
    type=float,
    help="X-radius of the left ventricle",
    show_default=True,
)
@click.option(
    "--lv-radius-y",
    default=1.8,
    type=float,
    help="Y-radius of the left ventricle",
    show_default=True,
)
@click.option(
    "--lv-radius-z",
    default=3.25,
    type=float,
    help="Z-radius of the left ventricle",
    show_default=True,
)
@click.option(
    "--rv-radius-x",
    default=1.9,
    type=float,
    help="X-radius of the right ventricle",
    show_default=True,
)
@click.option(
    "--rv-radius-y",
    default=2.5,
    type=float,
    help="Y-radius of the right ventricle",
    show_default=True,
)
@click.option(
    "--rv-radius-z",
    default=3.0,
    type=float,
    help="Z-radius of the right ventricle",
    show_default=True,
)
@click.option(
    "--create-fibers",
    default=False,
    is_flag=True,
    help="If True create analytic fibers",
    show_default=True,
)
@click.option(
    "--fiber-angle-endo",
    default=-60,
    type=float,
    help="Angle for the endocardium",
    show_default=True,
)
@click.option(
    "--fiber-angle-epi",
    default=+60,
    type=float,
    help="Angle for the epicardium",
    show_default=True,
)
@click.option(
    "--fiber-space",
    default="P_1",
    type=str,
    help="Function space for fibers of the form family_degree",
    show_default=True,
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
    show_default=True,
)
def biv_ellipsoid(
    outdir: Path,
    char_length: float = 0.4,  # cm
    base_cut_z: float = 2.5,
    box_size: float = 15.0,  # Size of the cutting box
    rv_wall_thickness: float = 0.4,  # cm
    lv_wall_thickness: float = 0.5,  # cm
    rv_offset_x: float = 3.0,
    lv_radius_x: float = 2.0,
    lv_radius_y: float = 1.8,
    lv_radius_z: float = 3.25,
    rv_radius_x: float = 1.9,
    rv_radius_y: float = 2.5,
    rv_radius_z: float = 3.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = -60,
    fiber_angle_epi: float = +60,
    fiber_space: str = "P_1",
    verbose: bool = False,
):
    init_logging(verbose=verbose)
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.biv_ellipsoid(
        outdir=outdir,
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
        create_fibers=create_fibers,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
        fiber_space=fiber_space,
        verbose=verbose,
    )
    geo.save(outdir / "biv_ellipsoid.bp")


@click.command(help="Create slab geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--lx",
    default=20.0,
    type=float,
    help="Length of slab in the x-direction",
    show_default=True,
)
@click.option(
    "--ly",
    default=7.0,
    type=float,
    help="Length of slab in the y-direction",
    show_default=True,
)
@click.option(
    "--lz",
    default=1.0,
    type=float,
    help="Length of slab in the z-direction",
    show_default=True,
)
@click.option(
    "--dx",
    default=1.0,
    type=float,
    help="Element size",
    show_default=True,
)
@click.option(
    "--create-fibers",
    default=False,
    is_flag=True,
    help="If True create analytic fibers",
    show_default=True,
)
@click.option(
    "--fiber-angle-endo",
    default=-60,
    type=float,
    help="Angle for the endocardium",
    show_default=True,
)
@click.option(
    "--fiber-angle-epi",
    default=+60,
    type=float,
    help="Angle for the epicardium",
    show_default=True,
)
@click.option(
    "--fiber-space",
    default="P_1",
    type=str,
    help="Function space for fibers of the form family_degree",
    show_default=True,
)
def slab(
    outdir: Path,
    lx: float = 20.0,
    ly: float = 7.0,
    lz: float = 3.0,
    dx: float = 1.0,
    create_fibers: bool = True,
    fiber_angle_endo: float = -60,
    fiber_angle_epi: float = +60,
    fiber_space: str = "P_1",
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.slab(
        outdir=outdir,
        lx=lx,
        ly=ly,
        lz=lz,
        dx=dx,
        create_fibers=create_fibers,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
        fiber_space=fiber_space,
    )
    geo.save(outdir / "slab.bp")


@click.command(help="Create slab in bath geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--lx",
    default=1.0,
    type=float,
    help="Length of slab in the x-direction",
    show_default=True,
)
@click.option(
    "--ly",
    default=0.01,
    type=float,
    help="Length of slab in the y-direction",
    show_default=True,
)
@click.option(
    "--lz",
    default=0.5,
    type=float,
    help="Length of slab in the z-direction",
    show_default=True,
)
@click.option(
    "--bx",
    default=0.0,
    type=float,
    help="Thickness of bath in the x-direction",
    show_default=True,
)
@click.option(
    "--by",
    default=0.0,
    type=float,
    help="Thickness of bath in the y-direction",
    show_default=True,
)
@click.option(
    "--bz",
    default=0.1,
    type=float,
    help="Thickness of bath in the z-direction",
    show_default=True,
)
@click.option(
    "--dx",
    default=0.01,
    type=float,
    help="Element size",
    show_default=True,
)
def slab_in_bath(
    outdir: Path,
    lx: float = 1.0,
    ly: float = 0.01,
    lz: float = 0.5,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.1,
    dx: float = 0.01,
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.slab_in_bath(
        outdir=outdir,
        lx=lx,
        ly=ly,
        lz=lz,
        bx=bx,
        by=by,
        bz=bz,
        dx=dx,
    )
    geo.save(outdir / "slab_in_bath.bp")


@click.command(help="Create cylinder geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--char-length",
    default=10.0,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--r-inner",
    default=10.0,
    type=float,
    help="Inner radius of the cylinder",
    show_default=True,
)
@click.option(
    "--r-outer",
    default=20.0,
    type=float,
    help="Outer radius of the cylinder",
    show_default=True,
)
@click.option(
    "--height",
    default=40.0,
    type=float,
    help="Height of the cylinder",
    show_default=True,
)
@click.option(
    "--create-fibers",
    default=False,
    is_flag=True,
    help="If True create analytic fibers",
    show_default=True,
)
@click.option(
    "--fiber-angle-endo",
    default=-60,
    type=float,
    help="Angle for the endocardium",
    show_default=True,
)
@click.option(
    "--fiber-angle-epi",
    default=+60,
    type=float,
    help="Angle for the epicardium",
    show_default=True,
)
@click.option(
    "--fiber-space",
    default="P_1",
    type=str,
    help="Function space for fibers of the form family_degree",
    show_default=True,
)
def cylinder(
    outdir: Path,
    char_length: float = 10.0,
    r_inner: float = 10.0,
    r_outer: float = 20.0,
    height: float = 40.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = -60,
    fiber_angle_epi: float = +60,
    fiber_space: str = "P_1",
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.cylinder(
        outdir=outdir,
        r_inner=r_inner,
        r_outer=r_outer,
        height=height,
        char_length=char_length,
        create_fibers=create_fibers,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
        fiber_space=fiber_space,
    )
    geo.save(outdir / "cylinder.bp")


@click.command(help="Create racetrack cylinder geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--char-length",
    default=10.0,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--r-inner",
    default=13.0,
    type=float,
    help="Inner radius of the cylinder",
    show_default=True,
)
@click.option(
    "--r-outer",
    default=20.0,
    type=float,
    help="Outer radius of the cylinder",
    show_default=True,
)
@click.option(
    "--height",
    default=40.0,
    type=float,
    help="Height of the cylinder",
    show_default=True,
)
@click.option(
    "--inner-flat-face-distance",
    default=10.0,
    type=float,
    help="Distance from the inner flat face to the center",
    show_default=True,
)
@click.option(
    "--outer-flat-face-distance",
    default=17.0,
    type=float,
    help="Distance from the outer flat face to the center",
    show_default=True,
)
@click.option(
    "--create-fibers",
    default=False,
    is_flag=True,
    help="If True create analytic fibers",
    show_default=True,
)
@click.option(
    "--fiber-angle-endo",
    default=-60,
    type=float,
    help="Angle for the endocardium",
    show_default=True,
)
@click.option(
    "--fiber-angle-epi",
    default=+60,
    type=float,
    help="Angle for the epicardium",
    show_default=True,
)
@click.option(
    "--fiber-space",
    default="P_1",
    type=str,
    help="Function space for fibers of the form family_degree",
    show_default=True,
)
def cylinder_racetrack(
    outdir: Path,
    char_length: float = 10.0,
    r_inner: float = 13.0,
    r_outer: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = -60,
    fiber_angle_epi: float = +60,
    fiber_space: str = "P_1",
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.cylinder_racetrack(
        outdir=outdir,
        r_inner=r_inner,
        r_outer=r_outer,
        height=height,
        inner_flat_face_distance=inner_flat_face_distance,
        outer_flat_face_distance=outer_flat_face_distance,
        char_length=char_length,
        create_fibers=create_fibers,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
        fiber_space=fiber_space,
    )
    geo.save(outdir / "cylinder_racetrack.bp")


@click.command(help="Create D-shaped cylinder geometry")
@click.argument(
    "outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--char-length",
    default=10.0,
    type=float,
    help="Characteristic length of mesh",
    show_default=True,
)
@click.option(
    "--r-inner",
    default=13.0,
    type=float,
    help="Inner radius of the cylinder",
    show_default=True,
)
@click.option(
    "--r-outer",
    default=20.0,
    type=float,
    help="Outer radius of the cylinder",
    show_default=True,
)
@click.option(
    "--height",
    default=40.0,
    type=float,
    help="Height of the cylinder",
    show_default=True,
)
@click.option(
    "--inner-flat-face-distance",
    default=10.0,
    type=float,
    help="Distance from the inner flat face to the center",
    show_default=True,
)
@click.option(
    "--outer-flat-face-distance",
    default=17.0,
    type=float,
    help="Distance from the outer flat face to the center",
    show_default=True,
)
@click.option(
    "--create-fibers",
    default=False,
    is_flag=True,
    help="If True create analytic fibers",
    show_default=True,
)
@click.option(
    "--fiber-angle-endo",
    default=-60,
    type=float,
    help="Angle for the endocardium",
    show_default=True,
)
@click.option(
    "--fiber-angle-epi",
    default=+60,
    type=float,
    help="Angle for the epicardium",
    show_default=True,
)
@click.option(
    "--fiber-space",
    default="P_1",
    type=str,
    help="Function space for fibers of the form family_degree",
    show_default=True,
)
def cylinder_D_shaped(
    outdir: Path,
    char_length: float = 10.0,
    r_inner: float = 13.0,
    r_outer: float = 20.0,
    height: float = 40.0,
    inner_flat_face_distance: float = 10.0,
    outer_flat_face_distance: float = 17.0,
    create_fibers: bool = False,
    fiber_angle_endo: float = -60,
    fiber_angle_epi: float = +60,
    fiber_space: str = "P_1",
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    geo = mesh.cylinder_D_shaped(
        outdir=outdir,
        r_inner=r_inner,
        r_outer=r_outer,
        height=height,
        inner_flat_face_distance=inner_flat_face_distance,
        outer_flat_face_distance=outer_flat_face_distance,
        char_length=char_length,
        create_fibers=create_fibers,
        fiber_angle_endo=fiber_angle_endo,
        fiber_angle_epi=fiber_angle_epi,
        fiber_space=fiber_space,
    )
    geo.save(outdir / "cylinder_D_shaped.bp")


@click.command("gui")
def gui():
    # Make sure we can import the required packages
    from . import gui  # noqa: F401

    gui_path = Path(__file__).parent.joinpath("gui.py")
    import subprocess as sp

    sp.run(["streamlit", "run", gui_path.as_posix()])


@click.command(help="Rotate a mesh to align with a target normal")
@click.argument(
    "folder",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "-o",
    "--outdir",
    required=True,
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
)
@click.option(
    "--target-normal",
    required=True,
    type=float,
    nargs=3,
    help="Target normal vector to align the base normal with",
)
@click.option(
    "--base-marker",
    default="BASE",
    type=str,
    help="Marker name for the base",
    show_default=True,
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
    show_default=True,
)
def rotate(
    folder: Path,
    outdir: Path,
    target_normal: list[float],
    base_marker: str = "BASE",
    verbose: bool = False,
):
    from .geometry import Geometry

    init_logging(verbose=verbose)
    comm = MPI.COMM_WORLD

    geo = Geometry.from_folder(comm=comm, folder=folder)
    geo.rotate(target_normal=target_normal, base_marker=base_marker)
    geo.save_folder(Path(outdir))


app.add_command(lv_ellipsoid)
app.add_command(biv_ellipsoid)
app.add_command(slab)
app.add_command(slab_in_bath)
app.add_command(gui)
app.add_command(ukb)
app.add_command(cylinder)
app.add_command(cylinder_racetrack)
app.add_command(cylinder_D_shaped)
app.add_command(rotate)
