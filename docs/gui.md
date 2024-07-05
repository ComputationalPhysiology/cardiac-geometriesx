
# Graphical user interface

`cardiac-geometries` also comes with a graphical user interface based on [`streamlit`](https://streamlit.io).

Here you see a quick demo of how it works

![_](_static/gui.mp4)


To install the requirements for running the GUI, you do
```
python3 -m pip install cardiac-geometries[gui]
```

To start the GUI simply run
```
geox gui
```

and then open the url <http://localhost:8501> in your browser.

Note that if you are running inside docker, you need to also forward the port 8501, i.e using the command

```
docker run --name geox -w /home/shared -v $PWD:/home/shared -p 8501:8501 -it ghcr.io/fenics/dolfinx/dolfinx:v0.8.0
```

when creating the container.
