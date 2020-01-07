# How to contribute

PyEDDL uses the [binder](https://github.com/RosettaCommons/binder) tool to
generate [pybind11](https://github.com/pybind/pybind11) bindings. The
resulting files are committed to the repository, and can be regenerated with:

```
bash generate_bindings.sh
```

Note that the above script uses the eddl Docker image to get the zlib
includes needed for bindings generation. Therefore, if you don't have it,
build the eddl image before running `generate_bindings`:

```
docker build -t eddl -f Dockerfile.eddl .
```

Regenerating the bindings can be necessary after updating the reference EDDL
revision or after making changes to the code generation setup (configuration
and include list). Note that add-ons are often necessary to complete the
bindings (see `codegen/config.cfg`).

To test the bindings on Docker, build the pyeddl image with:

```
docker build -t pyeddl .
```

Then you can run tests and/or examples on Docker, e.g.:

```
docker run --rm -it pyeddl bash
pytest tests
```
