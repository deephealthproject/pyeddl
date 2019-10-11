#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void lupsampling_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<Layer*, const std::vector<int>&, std::string,
	   std::string, int>(),
           pybind11::arg("parent"), pybind11::arg("size"),
           pybind11::arg("interpolation"), pybind11::arg("name"),
           pybind11::arg("dev"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
}
