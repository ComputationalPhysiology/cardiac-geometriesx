# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/dolfinx:stable

ENV PYVISTA_JUPYTER_BACKEND="html"

# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 nodejs

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install ".[docs]"
