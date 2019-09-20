#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void lembedding_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<int, int, std::string, int>());
}
