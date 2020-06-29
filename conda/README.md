# Conda recipes

This directory contains [Conda](https://docs.conda.io/en/latest/) recipes for
EDDL and PyEDDL, together with Dockerfiles for building and testing the
corresponding packages.

Note that the EDDL recipes change the upstream version from `v0.5.4a` to
`0.5.4a0`, in order to make it compatible with the Conda versioning scheme
(i.e., [PEP 440](https://www.python.org/dev/peps/pep-0440/)).

The `-gpu` packages support CUDA 10.1 (like Tensorflow).

To install, run:

```
conda install -c dhealth eddl-cpu  # eddl cpu-only version
conda install -c dhealth eddl-gpu  # eddl gpu-enabled version
conda install -c dhealth pyeddl-cpu  # pyeddl cpu-only version
conda install -c dhealth pyeddl-gpu  # pyeddl gpu-enabled version
```