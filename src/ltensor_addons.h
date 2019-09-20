#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void ltensor_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<string>());
    cl.def(pybind11::init<vector<int>, int>());
}
