#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void lconcat_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<std::vector<Layer*>, std::string, int>(),
           pybind11::arg("parent"), pybind11::arg("name"), pybind11::arg("dev"),
           pybind11::keep_alive<1, 2>());
}
