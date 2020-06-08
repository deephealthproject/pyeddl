// Copyright (c) 2019-2020, CRS4
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>


template <typename type_, typename... options>
void tensor_addons(pybind11::class_<type_, options...> &cl) {
    using array_t = pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast>;
    cl.def(pybind11::init([](array_t array, int device) {
        if (device >= DEV_FPGA) {
            throw std::invalid_argument("device not supported");
        }
        pybind11::buffer_info info = array.request();
        std::vector<int> shape(info.shape.begin(), info.shape.end());
        auto t = new Tensor(shape, DEV_CPU);
        std::copy((float*)info.ptr, ((float*)info.ptr) + t->size, t->ptr);
	if (device >= DEV_GPU) {
	    t->toGPU();
	}
        return t;
    }), pybind11::arg("buf"), pybind11::arg("dev") = DEV_CPU);
    cl.def(pybind11::init<const vector<int>&, int>(),
           pybind11::arg("shape"), pybind11::arg("dev") = DEV_CPU,
           pybind11::keep_alive<1, 2>());
    cl.def("getShape", &Tensor::getShape);
    cl.def_static("full", &Tensor::full, pybind11::arg("shape"),
		  pybind11::arg("value"), pybind11::arg("dev") = DEV_CPU);
    cl.def_static("ones", &Tensor::ones, pybind11::arg("shape"),
		  pybind11::arg("dev") = DEV_CPU);
    cl.def_static("randn", &Tensor::randn, pybind11::arg("shape"),
		  pybind11::arg("dev") = DEV_CPU);
    cl.def_static("transpose", &Tensor::transpose, pybind11::arg("A"),
		  pybind11::arg("B"), pybind11::arg("dims"));
    cl.def_static("zeros", &Tensor::zeros, pybind11::arg("shape"),
		  pybind11::arg("dev") = DEV_CPU);
    cl.def_static("load_uint8_t", &Tensor::load<uint8_t>,
		  pybind11::arg("filename"), pybind11::arg("format") = "");
    cl.def_static("permute", &Tensor::permute,
		  pybind11::arg("t"), pybind11::arg("dims"));
    cl.def_buffer([](Tensor &t) -> pybind11::buffer_info {
        if (!t.isCPU()) {
            std::cerr << "WARNING: converting tensor to CPU" << std::endl;
            t.toCPU();
        }
        std::vector<ssize_t> strides(t.ndim);
        ssize_t S = sizeof(float);
        for (int i = t.ndim - 1; i >=0; --i) {
            strides[i] = S;
            S *= t.shape[i];
        }
        return pybind11::buffer_info(
              t.ptr,
              sizeof(float),
              pybind11::format_descriptor<float>::format(),
              t.ndim,
              t.shape,
              strides
        );
    });
}
