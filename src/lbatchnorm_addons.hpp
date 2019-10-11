#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void lbatchnorm_addons(pybind11::class_<type_, options...> &cl) {
  cl.def(pybind11::init<Layer*, float, float, bool, std::string, int>(),
           pybind11::arg("parent"), pybind11::arg("momentum"),
           pybind11::arg("epsilon"), pybind11::arg("affine"),
           pybind11::arg("name"), pybind11::arg("dev"),
           pybind11::keep_alive<1, 2>());
}
