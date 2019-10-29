#pragma once
#include <pybind11/pybind11.h>

void eddl_addons(pybind11::module &m) {
    m.def("Input", (class Layer* (*)(const vector<int>&, string)) &eddl::Input, "C++: eddl::Input(const vector<int>&, string) --> class Layer*", pybind11::arg("shape"), pybind11::arg("name") = "");
    m.def("Activation", (class Layer* (*)(class Layer*, string, string)) &eddl::Activation, "C++: eddl::Activation(class Layer*, string, string) --> class Layer*", pybind11::arg("parent"), pybind11::arg("activation"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Dense", (class Layer* (*)(class Layer*, int, bool, class Regularizer*, string)) &eddl::Dense, "C++: eddl::Dense(class Layer*, int, bool, class Regularizer*, string) --> class Layer*", pybind11::arg("parent"), pybind11::arg("ndim"), pybind11::arg("use_bias") = true, pybind11::arg("reg") = nullptr, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 4>());
    m.def("Model", (class Net* (*)(vector<Layer*>, vector<Layer*>)) &eddl::Model, "C++: eddl::Model(vector<Layer*>, vector<Layer*>) --> class Net*", pybind11::arg("in"), pybind11::arg("out"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("summary", (string (*)(class Net*)) &eddl::summary, "C++: eddl::summary(class Net*) --> string", pybind11::arg("m"));
    m.def("plot", (void (*)(class Net*, string)) &eddl::plot, "C++: eddl::plot(class Net*, string) --> void", pybind11::arg("m"), pybind11::arg("fname"));
    m.def("build", (void (*)(class Net*, class Optimizer*, const vector<string>&, const vector<string>&, class CompServ*, class Initializer*)) &eddl::build, "C++: eddl::build(class Net*, class Optimizer*, const vector<string>&, const vector<string>&, class CompServ*, class Initializer*) --> void", pybind11::arg("net"), pybind11::arg("o"), pybind11::arg("lo"), pybind11::arg("me"), pybind11::arg("cs") = nullptr, pybind11::arg("init") = nullptr);
    m.def("fit", (void (*)(class Net*, const vector<Tensor*>&, const vector<Tensor*>&, int, int)) &eddl::fit, "C++: eddl::fit(class Net*, const vector<Tensor*>&, const vector<Tensor*>&, int, int) --> void", pybind11::arg("m"), pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("batch"), pybind11::arg("epochs"));
    m.def("BatchNormalization", (class Layer* (*)(class Layer*, float, float, bool, string)) &eddl::BatchNormalization, "C++: eddl::BatchNormalization(class Layer*, float, float, bool, string) --> class Layer*", pybind11::arg("parent"), pybind11::arg("momentum") = 0.9f, pybind11::arg("epsilon") = 0.001f, pybind11::arg("affine") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("evaluate", (void (*)(class Net*, const vector<Tensor*>&, const vector<Tensor*>&)) &eddl::evaluate, "C++: eddl::evaluate(class Net*, const vector<Tensor*>&, const vector<Tensor*>&) --> void", pybind11::arg("m"), pybind11::arg("in"), pybind11::arg("out"));
    m.def("CS_GPU", (class CompServ* (*)(const vector<int>&, int)) &eddl::CS_GPU, "C++: eddl::CS_GPU(const vector<int>&, int) --> class CompServ*", pybind11::arg("g"), pybind11::arg("lsb") = 1);
}
