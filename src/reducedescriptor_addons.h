#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void reducedescriptor_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<Tensor*, std::vector<int>, std::string, bool>(),
           pybind11::arg("A"), pybind11::arg("axis"), pybind11::arg("mode"),
           pybind11::arg("keepdims"), pybind11::keep_alive<1, 2>());
    cl.def_readwrite("I", &ReduceDescriptor::I);
    cl.def_readwrite("O", &ReduceDescriptor::O);
    cl.def_readwrite("D", &ReduceDescriptor::D);
    cl.def_readwrite("ID", &ReduceDescriptor::ID);
    cl.def_readwrite("S", &ReduceDescriptor::S);
}
