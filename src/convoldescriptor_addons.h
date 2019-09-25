#pragma once

#include <pybind11/pybind11.h>
#include <eddl/tensor/tensor.h>

template <typename type_, typename... options>
void convoldescriptor_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<std::vector<int>, std::vector<int>,
	   std::vector<int>>());
    cl.def_readwrite("I", &ConvolDescriptor::I);
    cl.def_readwrite("ID", &ConvolDescriptor::ID);
    cl.def_readwrite("K", &ConvolDescriptor::K);
    cl.def_readwrite("bias", &ConvolDescriptor::bias);
    cl.def_readwrite("gK", &ConvolDescriptor::gK);
    cl.def_readwrite("gbias", &ConvolDescriptor::gbias);
    cl.def_readwrite("D", &ConvolDescriptor::D);
    cl.def_readwrite("O", &ConvolDescriptor::O);
}
