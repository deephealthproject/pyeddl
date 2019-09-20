#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void linput_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<Tensor*, std::string, int>(),
           pybind11::keep_alive<1, 2>());
}
