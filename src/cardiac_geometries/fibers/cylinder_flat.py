from pathlib import Path

import dolfinx

from . import cylinder, slab, utils


def create_microstructure(
    mesh: dolfinx.mesh.Mesh,
    alpha_endo: float,
    alpha_epi: float,
    r_inner: float,
    r_outer: float,
    inner_flat_face_distance: float,
    ffun: dolfinx.mesh.MeshTags,
    markers: dict[str, tuple[int, int]],
    function_space: str = "P_1",
    outdir: str | Path | None = None,
    **kwargs,
) -> utils.Microstructure:
    """Generate microstructure for cylinder

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        A cylinder mesh
    alpha_endo : float
        Angle on the endocardium
    alpha_epi : float
        Angle on the epicardium
    r_inner : float
        Inner radius
    r_outer : float
        Outer radius
    function_space : str
        Function space to interpolate the fibers, by default P_1
    outdir : Optional[Union[str, Path]], optional
        Output directory to store the results, by default None.
        If no output directory is specified the results will not be stored,
        but only returned.

    Returns
    -------
    Microstructure
        Tuple with fiber, sheet and sheet normal
    """

    system_cylinder = cylinder.compute_system(
        mesh=mesh,
        function_space=function_space,
        r_inner=r_inner,
        r_outer=r_outer,
        alpha_endo=alpha_endo,
        alpha_epi=alpha_epi,
    )

    x = system_cylinder.f0.function_space.tabulate_dof_coordinates().T[0]

    f0_arr = system_cylinder.f0.x.array.reshape((-1, 3))
    s0_arr = system_cylinder.s0.x.array.reshape((-1, 3))
    n0_arr = system_cylinder.n0.x.array.reshape((-1, 3))

    if "INSIDE_FLAT" in markers:
        # This is the D-shaped cylinder
        t = utils.laplace(
            mesh,
            ffun,
            endo_marker=markers["INSIDE_FLAT"][0],
            epi_marker=markers["OUTSIDE_FLAT"][0],
            function_space=function_space,
        )

        if outdir is not None:
            try:
                with dolfinx.io.VTXWriter(
                    mesh.comm, Path(outdir) / "laplace_flat.bp", [t], engine="BP4"
                ) as file:
                    file.write(0.0)
            except RuntimeError:
                pass

        system_flat = slab.compute_system(
            t,
            alpha_endo=-alpha_endo,
            alpha_epi=-alpha_epi,
            endo_epi_axis="x",
        )
        flat_indices = x >= inner_flat_face_distance - 1e-8
        f0_arr[flat_indices, :] = system_flat.f0.x.array.reshape((-1, 3))[flat_indices, :]
        s0_arr[flat_indices, :] = system_flat.s0.x.array.reshape((-1, 3))[flat_indices, :]
        n0_arr[flat_indices, :] = system_flat.n0.x.array.reshape((-1, 3))[flat_indices, :]

    elif "INSIDE_FLAT1" in markers:
        # This is the racetrack

        # First side
        t = utils.laplace(
            mesh,
            ffun,
            endo_marker=markers["INSIDE_FLAT1"][0],
            epi_marker=markers["OUTSIDE_FLAT1"][0],
            function_space=function_space,
        )

        if outdir is not None:
            try:
                with dolfinx.io.VTXWriter(
                    mesh.comm, Path(outdir) / "laplace_flat1.bp", [t], engine="BP4"
                ) as file:
                    file.write(0.0)
            except RuntimeError:
                pass

        system_flat1 = slab.compute_system(
            t,
            alpha_endo=-alpha_endo,
            alpha_epi=-alpha_epi,
            endo_epi_axis="x",
        )
        flat_indices = x >= inner_flat_face_distance - 1e-8
        f0_arr[flat_indices, :] = system_flat1.f0.x.array.reshape((-1, 3))[flat_indices, :]
        s0_arr[flat_indices, :] = system_flat1.s0.x.array.reshape((-1, 3))[flat_indices, :]
        n0_arr[flat_indices, :] = system_flat1.n0.x.array.reshape((-1, 3))[flat_indices, :]

        # Second side
        t = utils.laplace(
            mesh,
            ffun,
            endo_marker=markers["INSIDE_FLAT2"][0],
            epi_marker=markers["OUTSIDE_FLAT2"][0],
            function_space=function_space,
        )

        if outdir is not None:
            try:
                with dolfinx.io.VTXWriter(
                    mesh.comm, Path(outdir) / "laplace_flat2.bp", [t], engine="BP4"
                ) as file:
                    file.write(0.0)
            except RuntimeError:
                pass

        system_flat1 = slab.compute_system(
            t,
            alpha_endo=alpha_endo,
            alpha_epi=alpha_epi,
            endo_epi_axis="x",
        )
        flat_indices = x <= -inner_flat_face_distance + 1e-8
        f0_arr[flat_indices, :] = system_flat1.f0.x.array.reshape((-1, 3))[flat_indices, :]
        s0_arr[flat_indices, :] = system_flat1.s0.x.array.reshape((-1, 3))[flat_indices, :]
        n0_arr[flat_indices, :] = system_flat1.n0.x.array.reshape((-1, 3))[flat_indices, :]
    else:
        raise ValueError(
            "Markers for flat faces not found. Please provide markers for the flat faces."
        )
    system_cylinder.f0.x.array[:] = f0_arr.reshape(-1)
    system_cylinder.s0.x.array[:] = s0_arr.reshape(-1)
    system_cylinder.n0.x.array[:] = n0_arr.reshape(-1)

    if outdir is not None:
        utils.save_microstructure(mesh, system_cylinder, outdir)

    return system_cylinder
