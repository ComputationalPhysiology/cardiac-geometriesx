import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mpi4py import MPI

import adios2
import adios4dolfinx
import dolfinx
import numpy as np
import ufl
from packaging.version import Version

from . import utils

logger = logging.getLogger(__name__)


@dataclass
class Geometry:
    mesh: dolfinx.mesh.Mesh
    markers: dict[str, tuple[int, int]] = field(default_factory=dict)
    cfun: dolfinx.mesh.MeshTags | None = None
    ffun: dolfinx.mesh.MeshTags | None = None
    efun: dolfinx.mesh.MeshTags | None = None
    vfun: dolfinx.mesh.MeshTags | None = None
    f0: dolfinx.fem.Function | None = None
    s0: dolfinx.fem.Function | None = None
    n0: dolfinx.fem.Function | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save the geometry to a file using adios4dolfinx.

        Parameters
        ----------
        path : str | Path
            The path to the file where the geometry will be saved.
            The file will be created if it does not exist, or overwritten if it does.
        """
        path = Path(path)

        shutil.rmtree(path, ignore_errors=True)
        self.mesh.comm.barrier()
        adios4dolfinx.write_mesh(mesh=self.mesh, filename=path)

        if Version(np.__version__) <= Version("2.11") or Version(adios2.__version__) >= Version(
            "2.10.2"
        ):
            # This is broken in adios < 2.10.2 and numpy >= 2.11
            adios4dolfinx.write_attributes(
                comm=self.mesh.comm,
                filename=path,
                name="markers",
                attributes={k: np.array(v, dtype=np.uint8) for k, v in self.markers.items()},
            )

        if self.cfun is not None:
            adios4dolfinx.write_meshtags(
                meshtags=self.cfun,
                mesh=self.mesh,
                filename=path,
                meshtag_name="Cell tags",
            )
        if self.ffun is not None:
            adios4dolfinx.write_meshtags(
                meshtags=self.ffun,
                mesh=self.mesh,
                filename=path,
                meshtag_name="Facet tags",
            )
        # if self.efun is not None:
        #     adios4dolfinx.write_meshtags(
        #         meshtags=self.efun,
        #         mesh=self.mesh,
        #         filename=path,
        #         meshtag_name="Edge tags",
        #     )
        # if self.vfun is not None:
        #     adios4dolfinx.write_meshtags(
        #         meshtags=self.vfun,
        #         mesh=self.mesh,
        #         filename=path,
        #         meshtag_name="Vertex tags",
        #     )

        if self.f0 is not None:
            el = self.f0.ufl_element()
            arr = utils.element2array(el)
            if Version(np.__version__) <= Version("2.11") or Version(adios2.__version__) >= Version(
                "2.10.2"
            ):
                # This is broken in adios for numpy >= 2.11
                adios4dolfinx.write_attributes(
                    comm=self.mesh.comm,
                    filename=path,
                    name="function_space",
                    attributes={k: arr for k in ("f0", "s0", "n0")},
                )
            adios4dolfinx.write_function(u=self.f0, filename=path, name="f0")
        if self.s0 is not None:
            adios4dolfinx.write_function(u=self.s0, filename=path, name="s0")
        if self.n0 is not None:
            adios4dolfinx.write_function(u=self.n0, filename=path, name="n0")

        self.mesh.comm.barrier()

    @property
    def dx(self):
        """Volume measure for the mesh using
        the cell function `cfun` if it exists as subdomain data.
        """
        return ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cfun)

    @property
    def ds(self):
        """Surface measure for the mesh using
        the facet function `ffun` if it exists as subdomain data.
        """
        return ufl.Measure("ds", domain=self.mesh, subdomain_data=self.ffun)

    @property
    def facet_normal(self) -> ufl.FacetNormal:
        """Facet normal vector for the mesh."""
        return ufl.FacetNormal(self.mesh)

    def refine(self, n=1, outdir: Path | None = None) -> "Geometry":
        """
        Refine the mesh and transfer the meshtags to new geometry.
        Also regenerate fibers if `self.info` is found.
        If `self.info` is not found, it currently raises a
        NotImplementedError, however fiber could be interpolated
        from the old mesh to the new mesh but this will result in a
        loss of information about the fiber orientation.

        Parameters
        ----------
        n : int, optional
            Number of times to refine the mesh, by default 1
        outdir : Path | None, optional
            Output directory to save the refined mesh and meshtags,
            by default None in which case the mesh is not saved.

        Returns
        -------
        Geometry
            A new Geometry object with the refined mesh and updated meshtags.

        Raises
        ------
        NotImplementedError
            If `self.info` is not found, indicating that fiber
            interpolation after refinement is not implemented yet.
        """
        mesh = self.mesh
        cfun = self.cfun
        ffun = self.ffun

        for _ in range(n):
            new_mesh, parent_cell, parent_facet = dolfinx.mesh.refine(
                mesh,
                partitioner=None,
                option=dolfinx.mesh.RefinementOption.parent_cell_and_facet,
            )
            new_mesh.name = mesh.name
            mesh = new_mesh
            new_mesh.topology.create_entities(1)
            new_mesh.topology.create_connectivity(2, 3)
            if cfun is not None:
                new_cfun = dolfinx.mesh.transfer_meshtag(cfun, new_mesh, parent_cell, parent_facet)
                new_cfun.name = cfun.name
                cfun = new_cfun
            else:
                new_cfun = None
            if ffun is not None:
                new_ffun = dolfinx.mesh.transfer_meshtag(ffun, new_mesh, parent_cell, parent_facet)
                new_ffun.name = ffun.name
                ffun = new_ffun
            else:
                new_ffun = None

        if outdir is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            with dolfinx.io.XDMFFile(new_mesh.comm, outdir / "mesh.xdmf", "w") as xdmf:
                xdmf.write_mesh(new_mesh)
                xdmf.write_meshtags(
                    cfun,
                    new_mesh.geometry,
                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{new_mesh.name}']/Geometry",
                )
                mesh.topology.create_connectivity(2, 3)
                xdmf.write_meshtags(
                    ffun,
                    new_mesh.geometry,
                    geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{new_mesh.name}']/Geometry",
                )

        if self.info is not None:
            info = self.info.copy()
            info["refinement"] = n
            info.pop("outdir", None)
            if outdir is not None:
                info["outdir"] = str(outdir)
            from .fibers import generate_fibers

            f0, s0, n0 = generate_fibers(mesh=new_mesh, ffun=new_ffun, markers=self.markers, **info)
        else:
            info = None
            # Interpolate fibers
            raise NotImplementedError(
                "Interpolating fibers after refinement is not implemented yet."
            )

        return Geometry(
            mesh=new_mesh,
            markers=self.markers,
            cfun=new_cfun,
            ffun=new_ffun,
            f0=f0,
            s0=s0,
            n0=n0,
        )

    @classmethod
    def from_file(
        cls,
        comm: MPI.Intracomm,
        path: str | Path,
        function_space_data: dict[str, np.ndarray] | None = None,
    ) -> "Geometry":
        """Read geometry from a file using adios4dolfinx.

        Parameters
        ----------
        comm : MPI.Intracomm
            The MPI communicator to use for reading the mesh.
        path : str | Path
            The path to the file containing the geometry data.
        function_space_data : dict[str, np.ndarray] | None, optional
            A dictionary containing function space data for the functions to be read.
            If None, it will be read from the file.

        Returns
        -------
        Geometry
            An instance of the Geometry class containing the mesh, markers, and functions.
        """

        path = Path(path)

        mesh = adios4dolfinx.read_mesh(comm=comm, filename=path)
        markers = adios4dolfinx.read_attributes(comm=comm, filename=path, name="markers")
        tags = {}
        for name, meshtag_name in (
            ("cfun", "Cell tags"),
            ("ffun", "Facet tags"),
            ("efun", "Edge tags"),
            ("vfun", "Vertex tags"),
        ):
            try:
                tags[name] = adios4dolfinx.read_meshtags(
                    mesh=mesh, meshtag_name=meshtag_name, filename=path
                )
            except KeyError:
                tags[name] = None

        functions = {}
        if function_space_data is None:
            function_space_data = adios4dolfinx.read_attributes(
                comm=comm, filename=path, name="function_space"
            )
        assert isinstance(function_space_data, dict), "function_space_data must be a dictionary"
        for name, el in function_space_data.items():
            element = utils.array2element(el)
            V = dolfinx.fem.functionspace(mesh, element)
            f = dolfinx.fem.Function(V, name=name)
            try:
                adios4dolfinx.read_function(u=f, filename=path, name=name)
            except KeyError:
                continue
            else:
                functions[name] = f

        return cls(
            mesh=mesh,
            markers=markers,
            **functions,
            **tags,
        )

    @classmethod
    def from_folder(cls, comm: MPI.Intracomm, folder: str | Path) -> "Geometry":
        """Read geometry from a folder containing mesh and markers files.

        Parameters
        ----------
        comm : MPI.Intracomm
            The MPI communicator to use for reading the mesh and markers.
        folder : str | Path
            The path to the folder containing the geometry data.
            The folder should contain the following files:
            - mesh.xdmf: The mesh file in XDMF format.
            - markers.json: A JSON file containing markers.
            - microstructure.json: A JSON file containing microstructure data (optional).
            - microstructure.bp: A BP file containing microstructure functions (optional).
            - info.json: A JSON file containing additional information (optional).

        Returns
        -------
        Geometry
            An instance of the Geometry class containing the mesh, markers, and functions.

        Raises
        ------
        ValueError
            If the required mesh file is not found in the specified folder.
        """
        folder = Path(folder)
        logger.info(f"Reading geometry from {folder}")
        # Read mesh
        if (folder / "mesh.xdmf").exists():
            logger.debug("Reading mesh")
            mesh, tags = utils.read_mesh(comm=comm, filename=folder / "mesh.xdmf")
        else:
            raise ValueError("No mesh file found")

        # Read markers
        if (folder / "markers.json").exists():
            logger.debug("Reading markers")
            if comm.rank == 0:
                markers = json.loads((folder / "markers.json").read_text())
            else:
                markers = {}
            markers = comm.bcast(markers, root=0)
        else:
            markers = {}

        if (folder / "microstructure.json").exists():
            if comm.rank == 0:
                microstructure = json.loads((folder / "microstructure.json").read_text())
            else:
                microstructure = {}
            microstructure = comm.bcast(microstructure, root=0)
        else:
            microstructure = {}

        functions = {}
        microstructure_path = folder / "microstructure.bp"
        if microstructure_path.exists():
            logger.debug("Reading microstructure")
            # function_space = adios4dolfinx.read_attributes(
            #     comm=MPI.COMM_WORLD, filename=microstructure_path, name="function_space"
            # )
            for name, el in microstructure.items():
                logger.debug(f"Reading {name}")
                element = utils.array2element(el)
                V = dolfinx.fem.functionspace(mesh, element)
                f = dolfinx.fem.Function(V, name=name)
                try:
                    adios4dolfinx.read_function(u=f, filename=microstructure_path, name=name)
                except KeyError:
                    continue
                else:
                    functions[name] = f

        if (folder / "info.json").exists():
            logger.debug("Reading info")
            if comm.rank == 0:
                info = json.loads((folder / "info.json").read_text())
            else:
                info = {}
            info = comm.bcast(info, root=0)
        else:
            info = {}

        return cls(
            mesh=mesh,
            markers=markers,
            info=info,
            **functions,
            **tags,
        )
