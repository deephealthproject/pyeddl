# PyEDDL

Python bindings for EDDL.

Regenerate the binding code (if necessary, e.g., after making changes to the
code generation setup):

```
bash generate_bindings.sh
```

Build the Docker images:

```
bash build_docker.sh
```

Run the examples on the Docker image, e.g.:

```
docker run --rm -it pyeddl bash
cd examples
python3 eddl_mlp.py
```
