#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void lmaxpool_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<Layer*, PoolDescriptor*, std::string, int>(),
           pybind11::arg("parent"), pybind11::arg("D"), pybind11::arg("name"),
           pybind11::arg("dev"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
}
