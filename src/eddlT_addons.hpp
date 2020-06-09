// Copyright (c) 2019-2020, CRS4
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <eddl/tensor/tensor.h>

void eddlT_addons(pybind11::module &m) {
    using array_t = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>;
    // replaces Tensor* create(const vector<int> &shape, float *ptr)
    // needs to come before the array-based one, see
    // https://github.com/pybind/pybind11/issues/2027
    m.def("create", [](array_t array) -> class Tensor* {
        pybind11::buffer_info info = array.request();
        std::vector<int> shape(info.shape.begin(), info.shape.end());
        Tensor* t = new Tensor(shape);
        std::copy((float*)info.ptr, ((float*)info.ptr) + t->size, t->ptr);
        return t;
    }, "create(array) --> Tensor", pybind11::arg("array"));
    m.def("create", (class Tensor* (*)(const vector<int>&)) &eddlT::create, "C++: eddlT::create(const vector<int>&) --> class Tensor*", pybind11::arg("shape"));
    m.def("create", (class Tensor* (*)(const vector<int>&, int)) &eddlT::create, "C++: eddlT::create(const vector<int>&, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("dev"));
    // replaces float* getptr(Tensor* A)
    m.def("getdata", [](class Tensor* t) -> array_t {
        bool del = false;
        if (!t->isCPU()) {
            t = eddlT::toCPU(t);
            del = true;
        }
        pybind11::array_t<float> rval = pybind11::array_t<float>(t->size);
        pybind11::buffer_info info = rval.request();
        float* ptr = (float*)info.ptr;
        std::copy(t->ptr, t->ptr+t->size, ptr);
        rval.resize(t->shape);
        if (del) {
            delete t;
        }
        return rval;
    }, "getdata(Tensor) --> array", pybind11::arg("tensor"));
    m.def("randn", (class Tensor* (*)(const vector<int>&, int)) &eddlT::randn, "C++: eddlT::randn(const vector<int>&, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("dev") = DEV_CPU);
    m.def("load", (class Tensor* (*)(string, string)) &eddlT::load, "C++: eddlT::load(string, string) --> class Tensor*", pybind11::arg("fname"), pybind11::arg("format") = "");
    m.def("getShape", (vector<int> (*)(class Tensor*)) &eddlT::getShape, "C++: eddlT::getShape(class Tensor*) --> vector<int>", pybind11::arg("A"));
    m.def("reshape_", (void (*)(class Tensor*, vector<int>)) &eddlT::reshape_, "C++: eddlT::reshape_(class Tensor*, vector<int>) --> void", pybind11::arg("A"), pybind11::arg("indices"));
    m.def("save", (void (*)(class Tensor*, string, string)) &eddlT::save, "C++: eddlT::save(class Tensor*, string, string) --> void", pybind11::arg("A"), pybind11::arg("fname"), pybind11::arg("format") = "");
    m.def("zeros", (class Tensor* (*)(const vector<int>&, int)) &eddlT::zeros, "C++: eddlT::zeros(const vector<int>&, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("dev") = DEV_CPU);
    m.def("ones", (class Tensor* (*)(const vector<int>&, int)) &eddlT::ones, "C++: eddlT::ones(const vector<int>&, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("dev") = DEV_CPU);
    m.def("full", (class Tensor* (*)(const vector<int>&, float, int)) &eddlT::full, "C++: eddlT::full(const vector<int>&, float, int) --> class Tensor*", pybind11::arg("shape"), pybind11::arg("value"), pybind11::arg("dev") = DEV_CPU);
    m.def("set_", (void (*)(class Tensor*, vector<int>, float)) &eddlT::set_, "C++: eddlT::set_(class Tensor*, vector<int>, float) --> void", pybind11::arg("A"), pybind11::arg("indices"), pybind11::arg("value"));
    m.def("reduce_mean", (class Tensor* (*)(class Tensor*, vector<int>)) &eddlT::reduce_mean, "C++: eddlT::reduce_mean(class Tensor*, const vector<int>&) --> class Tensor*", pybind11::arg("A"), pybind11::arg("axis"));
    m.def("reduce_variance", (class Tensor* (*)(class Tensor*, vector<int>)) &eddlT::reduce_variance, "C++: eddlT::reduce_variance(class Tensor*, const vector<int>&) --> class Tensor*", pybind11::arg("A"), pybind11::arg("axis"));
    m.attr("DEV_CPU") = pybind11::int_(DEV_CPU);
    m.attr("DEV_GPU") = pybind11::int_(DEV_GPU);
    m.attr("DEV_FPGA") = pybind11::int_(DEV_FPGA);
}
