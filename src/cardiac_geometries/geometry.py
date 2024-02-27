from dataclasses import dataclass, field
import json
from pathlib import Path
import dolfinx
from mpi4py import MPI
import shutil
import adios4dolfinx

from .utils import write_function


@dataclass(frozen=True, slots=True)
class Geometry:
    mesh: dolfinx.mesh.Mesh
    markers: dict[str, tuple[str, str]] = field(default_factory=dict)
    ffun: dolfinx.mesh.MeshTags | None = None
    f0: dolfinx.fem.Function | None = None
    s0: dolfinx.fem.Function | None = None
    n0: dolfinx.fem.Function | None = None

    def save(self, outdir: str | Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True)

        mesh_file = outdir / "mesh.bp"
        if mesh_file.exists():
            shutil.rmtree(mesh_file)
        adios4dolfinx.write_mesh(mesh=self.mesh, filename=mesh_file)

        if self.ffun is not None:
            ffun_file = outdir / "ffun.bp"
            if ffun_file.exists():
                shutil.rmtree(ffun_file)
            adios4dolfinx.write_meshtags(meshtags=self.ffun, mesh=self.mesh, filename=ffun_file)
        if self.f0 is not None:
            write_function(u=self.f0, filename=outdir / "f0.bp")
        if self.s0 is not None:
            write_function(u=self.s0, filename=outdir / "s0.bp")
        if self.n0 is not None:
            write_function(u=self.n0, filename=outdir / "n0.bp")
        if self.mesh.comm.rank == 0:
            (outdir / "markers.json").write_text(
                json.dumps({k: [int(vi) for vi in v] for k, v in self.markers.items()})
            )

    @classmethod
    def from_folder(cls, folder: str | Path, function_space="CG_1") -> "Geometry":
        folder = Path(folder)

        comm = MPI.COMM_WORLD

        # Read mesh
        if (folder / "mesh.bp").exists():
            mesh = adios4dolfinx.read_mesh(
                comm=comm,
                filename=folder / "mesh.bp",
                engine="BP4",
                ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            )
        elif (folder / "mesh.xdmf").exists():
            with dolfinx.io.XDMFFile(comm, folder / "mesh.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")
        else:
            raise ValueError("No mesh file found")

        # Read meshtags
        if (folder / "ffun.bp").exists():
            ffun = adios4dolfinx.read_meshtags(mesh=mesh, filename=folder / "ffun.bp")
        elif (folder / "ffun.xdmf").exists():
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            with dolfinx.io.XDMFFile(comm, folder / "ffun.xdmf", "r") as xdmf:
                ffun = xdmf.read_meshtags(mesh=mesh, name="Grid")
        else:
            ffun = None

        # Read markers
        if (folder / "markers.json").exists():
            if comm.rank == 0:
                markers = json.loads((folder / "markers.json").read_text())
            else:
                markers = {}
            markers = comm.bcast(markers, root=0)
        else:
            markers = {}

        # TODO: Read info about the function space from the file.
        # For now we just pass it in as an argument.

        family, degree = function_space.split("_")
        V = dolfinx.fem.functionspace(mesh, (family, int(degree), (mesh.geometry.dim,)))
        f0 = dolfinx.fem.Function(V)
        s0 = dolfinx.fem.Function(V)
        n0 = dolfinx.fem.Function(V)

        # Read fiber
        if (folder / "f0.bp").exists():
            adios4dolfinx.read_function(u=f0, filename=folder / "f0.bp")

        # Read sheet
        if (folder / "s0.bp").exists():
            adios4dolfinx.read_function(u=s0, filename=folder / "s0.bp")

        # Read sheet normal
        if (folder / "n0.bp").exists():
            adios4dolfinx.read_function(u=n0, filename=folder / "n0.bp")

        return cls(mesh=mesh, ffun=ffun, markers=markers, f0=f0, s0=s0, n0=n0)
