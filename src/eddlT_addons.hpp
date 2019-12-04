#pragma once
#include <pybind11/pybind11.h>

void eddlT_addons(pybind11::module &m) {
    m.def("create", (class Tensor* (*)(const vector<int>&)) &eddlT::create, "C++: eddlT::create(const vector<int>&) --> class Tensor*", pybind11::arg("shape"));
    m.def("randn", (class Tensor* (*)(const vector<int>&, int)) &eddlT::randn, "C++: eddlT::randn(const vector<int>&, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("dev") = DEV_CPU);
    m.def("load", (class Tensor* (*)(string, string)) &eddlT::load, "C++: eddlT::load(string, string) --> class Tensor*", pybind11::arg("fname"), pybind11::arg("format") = "");
    m.def("getShape", (vector<int> (*)(class Tensor*)) &eddlT::getShape, "C++: eddlT::getShape(class Tensor*) --> vector<int>", pybind11::arg("A"));
    m.def("reshape_", (void (*)(class Tensor*, vector<int>)) &eddlT::reshape_, "C++: eddlT::reshape_(class Tensor*, vector<int>) --> void", pybind11::arg("A"), pybind11::arg("indices"));
    m.def("save", (void (*)(class Tensor*, string, string)) &eddlT::save, "C++: eddlT::save(class Tensor*, string, string) --> void", pybind11::arg("A"), pybind11::arg("fname"), pybind11::arg("format") = "");
}
