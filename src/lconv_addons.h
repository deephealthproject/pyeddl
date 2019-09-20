#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void lconv_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<Layer*, int, const std::vector<int>&,
           const std::vector<int>&, string, int, const std::vector<int>&,
           bool, string, int>(),
           pybind11::arg("parent"), pybind11::arg("filters"),
           pybind11::arg("kernel_size"), pybind11::arg("strides"),
           pybind11::arg("padding"), pybind11::arg("groups"),
           pybind11::arg("dilation_rate"), pybind11::arg("use_bias"),
           pybind11::arg("name"), pybind11::arg("dev"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 4>(),
           pybind11::keep_alive<1, 5>(), pybind11::keep_alive<1, 8>());
}
