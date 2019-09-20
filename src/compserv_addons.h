#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void compserv_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<int, std::vector<int>, std::vector<int>, int>(),
	   pybind11::arg("threads"), pybind11::arg("g"), pybind11::arg("f"),
	   pybind11::arg("lsb") = 1);
}
