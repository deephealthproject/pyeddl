#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void layer_addons(pybind11::class_<type_, options...> &cl) {
    cl.def_readwrite("input", &Layer::input);
    cl.def_readwrite("output", &Layer::output);
    cl.def_readwrite("target", &Layer::target);
    cl.def_readwrite("delta", &Layer::delta);
}
