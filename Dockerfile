# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/lab:stable

ENV PYVISTA_JUPYTER_BACKEND="html"

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install ".[docs]"
ENTRYPOINT ["/bin/bash"]
