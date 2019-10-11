#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void lrmean_addons(pybind11::class_<type_, options...> &cl) {
  cl.def(pybind11::init<Layer*, std::vector<int>, bool, std::string, int>(),
         pybind11::arg("l"), pybind11::arg("axis"), pybind11::arg("keepdims"),
         pybind11::arg("name"), pybind11::arg("dev"),
         pybind11::keep_alive<1, 2>());
}
