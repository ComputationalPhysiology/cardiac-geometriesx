FROM ghcr.io/fenics/dolfinx/dolfinx:v0.8.0

COPY . /repo/
WORKDIR /repo/

RUN python3 -m pip install --no-cache-dir .[demo]
RUN rm -rf /tmp
