import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mpi4py import MPI

import adios4dolfinx
import dolfinx
import numpy as np

from . import utils


@dataclass  # (frozen=True, slots=True)
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

    def save(self, path: str | Path) -> None:
        path = Path(path)

        shutil.rmtree(path, ignore_errors=True)
        self.mesh.comm.barrier()
        adios4dolfinx.write_mesh(mesh=self.mesh, filename=path)

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
        if self.efun is not None:
            adios4dolfinx.write_meshtags(
                meshtags=self.efun,
                mesh=self.mesh,
                filename=path,
                meshtag_name="Edge tags",
            )
        if self.vfun is not None:
            adios4dolfinx.write_meshtags(
                meshtags=self.vfun,
                mesh=self.mesh,
                filename=path,
                meshtag_name="Vertex tags",
            )

        if self.f0 is not None:
            el = self.f0.ufl_element()
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

        self.mesh.comm.barrier()

    @classmethod
    def from_file(cls, comm: MPI.Intracomm, path: str | Path) -> "Geometry":
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
        function_space = adios4dolfinx.read_attributes(
            comm=comm, filename=path, name="function_space"
        )
        for name, el in function_space.items():
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
        folder = Path(folder)

        # Read mesh
        if (folder / "mesh.xdmf").exists():
            mesh, tags = utils.read_mesh(comm=comm, filename=folder / "mesh.xdmf")
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

        functions = {}
        microstructure_path = folder / "microstructure.bp"
        if microstructure_path.exists():
            function_space = adios4dolfinx.read_attributes(
                comm=MPI.COMM_WORLD, filename=microstructure_path, name="function_space"
            )
            for name, el in function_space.items():
                element = utils.array2element(el)
                V = dolfinx.fem.functionspace(mesh, element)
                f = dolfinx.fem.Function(V, name=name)
                try:
                    adios4dolfinx.read_function(u=f, filename=microstructure_path, name=name)
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
