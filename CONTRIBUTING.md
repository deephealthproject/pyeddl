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


## Pybind11 notes

### Binding arrays of arbitrary objects

Binding `Net`'s `Xs` attribute turned out to be far from trivial. The
attribute is declared as:

```cpp
    vtensor Xs[MAX_THREADS];
```

Where vtensor is an alias for `vector<Tensor*>`.

Direct binding [does not work](https://stackoverflow.com/questions/52170192):

```cpp
    cl.def_readwrite("Xs", &Net::Xs);
```

Exposing it as a NumPy array does not work either:

```cpp
    cl.def_property_readonly("Xs", [](pybind11::object& obj) {
      Net& o = obj.cast<Net&>();
      vector<vector<Tensor*>> v(o.Xs, o.Xs + sizeof(o.Xs) / sizeof(o.Xs[0]));
      return v;
    });
```

You get an "Attempt to use a non-POD or unimplemented POD type as a numpy
dtype" error. Apparently [this can be made to work with structured
types](https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#structured-types),
but here we have an array of `vtensor`.

I finally decided to convert the array to a vector:

```cpp
    cl.def_property_readonly("Xs", [](pybind11::object& obj) {
      Net& o = obj.cast<Net&>();
      vector<vector<Tensor*>> v(o.Xs, o.Xs + sizeof(o.Xs) / sizeof(o.Xs[0]));
      return v;
    });
```

Better ideas are welcome!
