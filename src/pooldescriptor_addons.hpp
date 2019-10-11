#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void pooldescriptor_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<const std::vector<int>&, const std::vector<int>&,
           std::string>(),
           pybind11::arg("ks"), pybind11::arg("st"), pybind11::arg("p"),
           pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
    cl.def_readwrite("indX", &PoolDescriptor::indX);
    cl.def_readwrite("indY", &PoolDescriptor::indY);
}
