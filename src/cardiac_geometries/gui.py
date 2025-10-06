try:
    import streamlit as st
except ImportError:
    print("Please install streamlit - python3 -m pip install streamlit")
    exit(1)

try:
    import pyvista as pv
except ImportError:
    msg = (
        "Please install pyvista - python3 -m pip install pyvista. "
        "Note if you using ARM Mac, then check out the following link "
        "on how to install vtk: https://github.com/KitwareMedical/VTKPythonPackage/issues/42"
    )
    print(msg)
    exit(1)

try:
    from stpyvista import stpyvista
except ImportError:
    print("Please install stpyvista - python3 -m pip install stpyvista")
    exit(1)

import os

os.environ["GMSH_INTERRUPTIBLE"] = "0"

import math
from pathlib import Path

import mpi4py

import dolfinx

import cardiac_geometries


def return_none(*args, **kwargs):
    return None


def load_geometry(folder: str):
    comm = mpi4py.MPI.COMM_WORLD
    try:
        return cardiac_geometries.geometry.Geometry.from_folder(comm, folder)
    except Exception as e:
        st.error(f"Error loading geometry: {e}")
        return None


def plot_geometry(geo):
    V = dolfinx.fem.functionspace(geo.mesh, ("Lagrange", 1))

    # Plot the mesh with cell tags
    mesh_plotter = pv.Plotter()
    mesh_plotter.background_color = "white"
    mesh_plotter.window_size = [600, 400]

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)

    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    if geo.cfun is not None:
        grid.cell_data["Cell tags"] = geo.cfun.values
        grid.set_active_scalars("Cell tags")
    mesh_plotter.add_mesh(grid, show_edges=True)

    mesh_plotter.view_isometric()
    st.header("Mesh and cell tags")
    stpyvista(mesh_plotter)

    if geo.ffun is not None:
        vtk_bmesh = dolfinx.plot.vtk_mesh(geo.mesh, geo.ffun.dim, geo.ffun.indices)
        bgrid = pv.UnstructuredGrid(*vtk_bmesh)
        bgrid.cell_data["Facet tags"] = geo.ffun.values
        bgrid.set_active_scalars("Facet tags")
        facet_plotter = pv.Plotter()
        facet_plotter.background_color = "white"
        facet_plotter.window_size = [600, 400]
        facet_plotter.add_mesh(bgrid, show_edges=True)
        facet_plotter.view_isometric()
        st.header("Facet tags")
        stpyvista(facet_plotter)

    if geo.f0 is not None:
        st.header("Fibers")
        size_arrows_fibers = st.slider(
            "Arrow size fibers", min_value=0.1, max_value=10.0, value=2.0
        )
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(geo.f0.function_space)
        values = geo.f0.x.array.real.reshape((geometry.shape[0], len(geo.f0)))
        function_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        function_grid["u"] = values
        glyphs = function_grid.glyph(orient="u", factor=size_arrows_fibers)
        fiber_plotter = pv.Plotter()
        fiber_plotter.background_color = "white"
        fiber_plotter.window_size = [600, 400]
        fiber_plotter.add_mesh(glyphs, show_edges=True)
        fiber_plotter.view_isometric()
        stpyvista(fiber_plotter)

    if geo.s0 is not None:
        st.header("Sheets")
        size_arrows_sheets = st.slider(
            "Arrow size sheets", min_value=0.1, max_value=10.0, value=2.0
        )
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(geo.s0.function_space)
        values = geo.s0.x.array.real.reshape((geometry.shape[0], len(geo.s0)))
        function_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        function_grid["u"] = values
        glyphs = function_grid.glyph(orient="u", factor=size_arrows_sheets)
        sheet_plotter = pv.Plotter()
        sheet_plotter.background_color = "white"
        sheet_plotter.window_size = [600, 400]
        sheet_plotter.add_mesh(glyphs, show_edges=True)
        sheet_plotter.view_isometric()
        stpyvista(sheet_plotter)

    if geo.n0 is not None:
        st.header("Sheet Normal")
        size_arrows_normal = st.slider(
            "Arrow size sheet normal", min_value=0.1, max_value=10.0, value=2.0
        )
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(geo.n0.function_space)
        values = geo.n0.x.array.real.reshape((geometry.shape[0], len(geo.n0)))
        function_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        function_grid["u"] = values
        glyphs = function_grid.glyph(orient="u", factor=size_arrows_normal)
        normal_plotter = pv.Plotter()
        normal_plotter.background_color = "white"
        normal_plotter.window_size = [600, 400]
        normal_plotter.add_mesh(glyphs, show_edges=True)
        normal_plotter.view_isometric()
        stpyvista(normal_plotter)


def load():
    st.title("Load existing geometries")

    cwd = Path.cwd()
    folders = [f.name for f in cwd.iterdir() if f.is_dir()]
    # Select a folder

    folder = st.selectbox("Select a folder", folders)
    geo = load_geometry(folder)

    if geo is not None:
        plot_geometry(geo)

    return


def create_lv():
    st.title("Create lv geometry")

    outdir = st.text_input("Output directory", value="lv_ellipsoid")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Radius")
        r_short_endo = st.number_input("r_short_endo", value=7.0)
        r_short_epi = st.number_input("r_short_epi", value=10.0)
        r_long_endo = st.number_input("r_long_endo", value=17.0)
        r_long_epi = st.number_input("r_long_epi", value=20.0)
        p_size_ref = st.number_input("p_size_ref", value=3.0)

    with col2:
        st.header("Angles")
        mu_apex_endo = st.number_input("mu_apex_endo", value=-math.pi)
        mu_base_endo = st.number_input("mu_base_endo", value=-math.acos(5 / 17))
        mu_apex_epi = st.number_input("mu_apex_epi", value=-math.pi)
        mu_base_epi = st.number_input("mu_base_epi", value=-math.acos(5 / 20))

    with col3:
        st.header("Fibers")
        create_fibers = st.checkbox("Create fibers", value=True)
        if create_fibers:
            fiber_space = st.selectbox("Fiber space", ["P_1", "P_2"])
            fiber_angle_endo = st.number_input("fiber_angle_endo", value=60)
            fiber_angle_epi = st.number_input("fiber_angle_epi", value=-60)

    if st.button("Create"):
        args = [
            "geox",
            "lv-ellipsoid",
            "--r-short-endo",
            str(r_short_endo),
            "--r-short-epi",
            str(r_short_epi),
            "--r-long-endo",
            str(r_long_endo),
            "--r-long-epi",
            str(r_long_epi),
            "--mu-apex-endo",
            str(mu_apex_endo),
            "--mu-base-endo",
            str(mu_base_endo),
            "--mu-apex-epi",
            str(mu_apex_epi),
            "--mu-base-epi",
            str(mu_base_epi),
            "--psize-ref",
            str(p_size_ref),
        ]

        if create_fibers:
            args.extend(
                [
                    "--create-fibers",
                    "--fiber-space",
                    fiber_space,
                    "--fiber-angle-endo",
                    str(fiber_angle_endo),
                    "--fiber-angle-epi",
                    str(fiber_angle_epi),
                ]
            )

        args.append(str(outdir))
        st.markdown(f"```{' '.join(args)}```")

        if Path(outdir).exists():
            st.warning(f"Folder {outdir} already exists. Overwriting...")
            import shutil

            shutil.rmtree(outdir)

        import subprocess as sp

        ret = sp.run(
            args,
            capture_output=True,
        )

        st.markdown(f"```{ret.stdout.decode()}```")

    if st.button("Visualize folder"):
        geo = load_geometry(outdir)

        if geo is not None:
            plot_geometry(geo)

    return


# Page settings
st.set_page_config(page_title="simcardems")

# Sidebar settings
pages = {
    "Load": load,
    "Create LV geometry": create_lv,
}

st.sidebar.title("Cardiac geometries")

# Radio buttons to select desired option
page = st.sidebar.radio("Pages", tuple(pages.keys()))

pages[page]()

# About
st.sidebar.markdown(
    """
- [Source code](https://github.com/ComputationalPhysiology/cardiac-geometriesx)
- [Documentation](http://computationalphysiology.github.io/cardiac-geometriesx)


Copyright © 2024 Henrik Finsberg @ Simula Research Laboratory
""",
)
