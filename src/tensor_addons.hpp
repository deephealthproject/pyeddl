// Copyright (c) 2019-2021 CRS4
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
    cl.def("select", (Tensor* (Tensor::*)(const vector<string>&)) &Tensor::select, "C++: Tensor::select(const vector<string>&) --> Tensor*", pybind11::arg("indices"));
    cl.def("set_select", (void (Tensor::*)(const vector<string>&, Tensor*)) &Tensor::set_select, "C++: Tensor::set_select(const vector<string>&, Tensor*) --> void", pybind11::arg("indices"), pybind11::arg("A"));
    cl.def("scale", (Tensor* (Tensor::*)(vector<int>, WrappingMode, float, TransformationMode, bool)) &Tensor::scale, "Scale the tensor", pybind11::arg("new_shape"), pybind11::arg("mode")=WrappingMode::Constant, pybind11::arg("cval")=0.0f, pybind11::arg("coordinate_transformation_mode")=TransformationMode::Asymmetric, pybind11::arg("keep_size")=false);
    cl.def_static("full", &Tensor::full, pybind11::arg("shape"),
		  pybind11::arg("value"), pybind11::arg("dev") = DEV_CPU);
    cl.def_static("ones", &Tensor::ones, pybind11::arg("shape"),
		  pybind11::arg("dev") = DEV_CPU);
    cl.def_static("randu", &Tensor::randu, pybind11::arg("shape"),
		  pybind11::arg("dev") = DEV_CPU);
    cl.def_static("randn", &Tensor::randn, pybind11::arg("shape"),
		  pybind11::arg("dev") = DEV_CPU);
    cl.def_static("transpose", &Tensor::transpose, pybind11::arg("A"),
		  pybind11::arg("B"), pybind11::arg("dims"));
    cl.def_static("zeros", &Tensor::zeros, pybind11::arg("shape"),
		  pybind11::arg("dev") = DEV_CPU);
    cl.def_static("mult2D", [](Tensor* A, Tensor* B) {
	    Tensor *C = new Tensor({A->shape[0], B->shape[1]}, A->device);
	    Tensor::mult2D(A, 0, B, 0, C, 0);
	    return C;
	}, pybind11::arg("A"), pybind11::arg("B"));
    cl.def("save", &Tensor::save,
	   pybind11::arg("filename"), pybind11::arg("format") = "");
    cl.def_static("load", [](const string& filename, string format) {
	    return Tensor::load(filename, format);
	}, pybind11::arg("filename"), pybind11::arg("format") = "");
    cl.def("permute", (Tensor* (Tensor::*)(const vector<int>&)) &Tensor::permute, "In-place permutation of tensor dimensions", pybind11::arg("dims"));
    cl.def_static("permute_static", (Tensor* (*)(Tensor*, const vector<int>&)) &Tensor::permute, "Permutation of tensor dimensions", pybind11::arg("A"), pybind11::arg("dims"));
    cl.def("reshape_", (void (Tensor::*)(const vector<int>&)) &Tensor::reshape_, "C++: Tensor::reshape_(const vector<int>&) --> void", pybind11::arg("new_shape"));
    // Expose contents as a buffer object. Allows a = numpy.array(t).
    // Mostly useful for a = numpy.array(t, copy=False) (CPU only, of course).
    cl.def_buffer([](Tensor &t) -> pybind11::buffer_info {
        if (!t.isCPU()) {
          throw std::invalid_argument("device not supported");
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
    // get data as a NumPy array
    cl.def("getdata", [](Tensor* t) -> array_t {
        Tensor* aux = t;
        bool del = false;
        if (!t->isCPU()) {
            aux = t->clone();
            aux->toCPU();
            del = true;
        }
        pybind11::array_t<float> rval = pybind11::array_t<float>(aux->size);
        pybind11::buffer_info info = rval.request();
        float* ptr = (float*)info.ptr;
        std::copy(aux->ptr, aux->ptr + aux->size, ptr);
        rval.resize(aux->shape);
        if (del) {
            delete aux;
        }
        return rval;
    }, "getdata() --> array");
    // from the EDDL NLP examples
    cl.def_static("onehot", [](Tensor* in, int vocs) -> Tensor* {
	    int n = in->shape[0];
	    int l = in->shape[1];
	    Tensor *out = new Tensor({n, l, vocs});
	    out->fill_(0.0);
	    int p = 0;
	    for(int i = 0; i < n * l; i++, p += vocs) {
		int w = in->ptr[i];
		out->ptr[p+w] = 1.0;
	    }
	    return out;
	}, pybind11::arg("in"), pybind11::arg("vocs"));
}
