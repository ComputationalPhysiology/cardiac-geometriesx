import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mpi4py import MPI

import adios4dolfinx
import dolfinx
import numpy as np

from . import utils


@dataclass(frozen=True, slots=True)
class Geometry:
    mesh: dolfinx.mesh.Mesh
    markers: dict[str, tuple[str, str]] = field(default_factory=dict)
    ffun: dolfinx.mesh.MeshTags | None = None
    cfun: dolfinx.mesh.MeshTags | None = None
    f0: dolfinx.fem.Function | None = None
    s0: dolfinx.fem.Function | None = None
    n0: dolfinx.fem.Function | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)

        if path.exists() and self.mesh.comm.rank == 0:
            shutil.rmtree(path)
        self.mesh.comm.barrier()
        adios4dolfinx.write_mesh(mesh=self.mesh, filename=path)

        adios4dolfinx.write_attributes(
            comm=self.mesh.comm,
            filename=path,
            name="markers",
            attributes={k: np.array(v, dtype=np.uint8) for k, v in self.markers.items()},
        )

        if self.ffun is not None:
            adios4dolfinx.write_meshtags(
                meshtags=self.ffun,
                mesh=self.mesh,
                filename=path,
            )

        if self.f0 is not None:
            el = self.f0.ufl_element().basix_element
            arr = utils.element2array(el)

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

    @classmethod
    def from_file(cls, comm: MPI.Intracomm, path: str | Path) -> "Geometry":
        path = Path(path)
        mesh = adios4dolfinx.read_mesh(comm=comm, filename=path)
        markers = adios4dolfinx.read_attributes(comm=comm, filename=path, name="markers")
        ffun = adios4dolfinx.read_meshtags(mesh=mesh, meshtag_name="Facet tags", filename=path)
        function_space = adios4dolfinx.read_attributes(
            comm=comm, filename=path, name="function_space"
        )
        element = utils.array2element(function_space["f0"])
        # Assume same function space for all functions
        V = dolfinx.fem.functionspace(mesh, element)
        f0 = dolfinx.fem.Function(V, name="f0")
        s0 = dolfinx.fem.Function(V, name="s0")
        n0 = dolfinx.fem.Function(V, name="n0")

        adios4dolfinx.read_function(u=f0, filename=path, name="f0")
        adios4dolfinx.read_function(u=s0, filename=path, name="s0")
        adios4dolfinx.read_function(u=n0, filename=path, name="n0")
        return cls(mesh=mesh, markers=markers, ffun=ffun, f0=f0, s0=s0, n0=n0)

    @classmethod
    def from_folder(cls, comm: MPI.Intracomm, folder: str | Path) -> "Geometry":
        folder = Path(folder)

        # Read mesh
        if (folder / "mesh.xdmf").exists():
            mesh, cfun, ffun = utils.read_mesh(comm=comm, filename=folder / "mesh.xdmf")
        else:
            raise ValueError("No mesh file found")

        # Read markers
        if (folder / "markers.json").exists():
            if comm.rank == 0:
                markers = json.loads((folder / "markers.json").read_text())
            else:
                markers = {}
            markers = comm.bcast(markers, root=0)
        else:
            markers = {}

        microstructure_path = folder / "microstructure.bp"
        if microstructure_path.exists():
            function_space = adios4dolfinx.read_attributes(
                comm=MPI.COMM_WORLD, filename=microstructure_path, name="function_space"
            )
            # Assume same function space for all functions
            element = utils.array2element(function_space["f0"])
            V = dolfinx.fem.functionspace(mesh, element)
            f0 = dolfinx.fem.Function(V, name="f0")
            s0 = dolfinx.fem.Function(V, name="s0")
            n0 = dolfinx.fem.Function(V, name="n0")

            adios4dolfinx.read_function(u=f0, filename=microstructure_path, name="f0")
            adios4dolfinx.read_function(u=s0, filename=microstructure_path, name="s0")
            adios4dolfinx.read_function(u=n0, filename=microstructure_path, name="n0")
        else:
            f0 = s0 = n0 = None

        return cls(mesh=mesh, ffun=ffun, cfun=cfun, markers=markers, f0=f0, s0=s0, n0=n0)
