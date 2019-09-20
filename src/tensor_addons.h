#pragma once

#include <pybind11/pybind11.h>

template <typename type_, typename... options>
void tensor_addons(pybind11::class_<type_, options...> &cl) {
    cl.def(pybind11::init<const vector<int>&, int>(),
           pybind11::arg("shape"), pybind11::arg("dev"),
           pybind11::keep_alive<1, 2>());
    cl.def_buffer([](Tensor &t) -> pybind11::buffer_info {
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
    cl.def("__init__", [](Tensor &t, pybind11::buffer b, int device) {
        pybind11::buffer_info info = b.request();
        if (info.format != pybind11::format_descriptor<float>::format())
            throw std::runtime_error("Invalid format: expected a float array");
        bool have_simple_strides = true;
        std::vector<ssize_t> simple_strides(info.ndim);
        ssize_t S = sizeof(float);
        for (int i = info.ndim - 1; i >=0; --i) {
            simple_strides[i] = S;
            S *= info.shape[i];
        }
        for (int i = 0; i < info.ndim; ++i) {
            if (info.strides[i] != simple_strides[i]) {
                have_simple_strides = false;
                break;
            }
        }
        std::vector<int> shape(info.ndim);
        for (int i = 0; i < info.ndim; ++i) {
            shape[i] = info.shape[i];
        }
        new(&t) Tensor(shape, device);
        if (have_simple_strides) {
            std::copy((float*)info.ptr, ((float*)info.ptr) + t.size, t.ptr);
        } else {
            throw std::runtime_error("complex strides not supported");
        }
    });
}
