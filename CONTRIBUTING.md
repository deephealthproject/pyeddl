# How to contribute

PyEDDL uses the [binder](https://github.com/RosettaCommons/binder) tool to
generate [pybind11](https://github.com/pybind/pybind11) bindings. The
resulting files are committed to the repository, and can be regenerated with:

```
bash generate_bindings.sh
```

Regenerating the bindings can be necessary after updating the reference EDDL
revision or after making changes to the code generation setup (configuration
and include list). Note that add-ons are often necessary to complete the
bindings (see `codegen/config.cfg`).

To test the bindings on Docker, build the EDDL and PyEDDL images with:

```
bash build_docker.sh
```

Then you can run tests and/or examples on Docker, e.g.:

```
docker run --rm -it pyeddl bash
cd examples
python3 eddl_mlp.py
```
