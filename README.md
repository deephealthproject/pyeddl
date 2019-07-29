# EDDL: Binder examples

This branch contains an example of
[Binder](https://github.com/RosettaCommons/binder)-generated code for part of
the EDDL library, as well as Python usage examples that reproduce EDDL
examples.

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
