#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <eddl/tensor/tensor.h>

void eddlT_addons(pybind11::module &m) {
    using array_t = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>;
    m.def("create", (class Tensor* (*)(const vector<int>&)) &eddlT::create, "C++: eddlT::create(const vector<int>&) --> class Tensor*", pybind11::arg("shape"));
    m.def("create", (class Tensor* (*)(const vector<int>&, int)) &eddlT::create, "C++: eddlT::create(const vector<int>&, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("dev"));
    // replaces Tensor* create(const vector<int> &shape, float *ptr)
    m.def("create", [](array_t array) -> class Tensor* {
        pybind11::buffer_info info = array.request();
        std::vector<int> shape(info.shape.begin(), info.shape.end());
        Tensor* t = new Tensor(shape);
        std::copy((float*)info.ptr, ((float*)info.ptr) + t->size, t->ptr);
        return t;
    }, "create(array) --> Tensor", pybind11::arg("array"));
    // replaces float* getptr(Tensor* A)
    m.def("getdata", [](class Tensor* t) -> array_t {
        pybind11::array_t<float> rval = pybind11::array_t<float>(t->size);
        pybind11::buffer_info info = rval.request();
        float* ptr = (float*)info.ptr;
        std::copy(t->ptr, t->ptr+t->size, ptr);
        rval.resize(t->shape);
        return rval;
    }, "getdata(Tensor) --> array", pybind11::arg("tensor"));
    m.def("randn", (class Tensor* (*)(const vector<int>&, int)) &eddlT::randn, "C++: eddlT::randn(const vector<int>&, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("dev") = DEV_CPU);
    m.def("load", (class Tensor* (*)(string, string)) &eddlT::load, "C++: eddlT::load(string, string) --> class Tensor*", pybind11::arg("fname"), pybind11::arg("format") = "");
    m.def("getShape", (vector<int> (*)(class Tensor*)) &eddlT::getShape, "C++: eddlT::getShape(class Tensor*) --> vector<int>", pybind11::arg("A"));
    m.def("reshape_", (void (*)(class Tensor*, vector<int>)) &eddlT::reshape_, "C++: eddlT::reshape_(class Tensor*, vector<int>) --> void", pybind11::arg("A"), pybind11::arg("indices"));
    m.def("save", (void (*)(class Tensor*, string, string)) &eddlT::save, "C++: eddlT::save(class Tensor*, string, string) --> void", pybind11::arg("A"), pybind11::arg("fname"), pybind11::arg("format") = "");
    m.attr("DEV_CPU") = pybind11::int_(DEV_CPU);
    m.attr("DEV_GPU") = pybind11::int_(DEV_GPU);
    m.attr("DEV_FPGA") = pybind11::int_(DEV_FPGA);
}
