# EDDL: Binder examples

This branch contains an example of
[Binder](https://github.com/RosettaCommons/binder)-generated code for part of
the EDDL library, as well as Python usage examples that reproduce EDDL
examples.

Regenerate the binding code (if necessary, e.g., after making changes to the
code generation setup):

```
bash build_bindings.sh
```

Build the Docker image, which builds EDDL and the Python bindings:

```
docker build -t binder_example .
```

Run the examples on the Docker image.
