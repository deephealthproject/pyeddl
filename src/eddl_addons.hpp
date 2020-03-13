// Copyright (c) 2019-2020 CRS4
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
#include <pybind11/functional.h>
#ifdef EDDL_WITH_PROTOBUF
#include <eddl/serialization/onnx/eddl_onnx.h>
#endif

// Use return_value_policy::reference for objects that get deleted on the C++
// side. In particular, layers and optimizers are deleted by the Net destructor

void eddl_addons(pybind11::module &m) {

    // --- core layers ---
    m.def("Activation", (class Layer* (*)(class Layer*, string, vector<float>, string)) &eddl::Activation, "C++: eddl::Activation(class Layer*, string, vector<float>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("activation"), pybind11::arg("params") = vector<float>{}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Softmax", (class Layer* (*)(class Layer*, string)) &eddl::Softmax, "C++: eddl::Softmax(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Sigmoid", (class Layer* (*)(class Layer*, string)) &eddl::Sigmoid, "C++: eddl::Sigmoid(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("HardSigmoid", (class Layer* (*)(class Layer*, string)) &eddl::HardSigmoid, "C++: eddl::HardSigmoid(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("ReLu", (class Layer* (*)(class Layer*, string)) &eddl::ReLu, "C++: eddl::ReLu(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("ThresholdedReLu", (class Layer* (*)(class Layer*, float, string)) &eddl::ThresholdedReLu, "C++: eddl::ThresholdedReLu(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("alpha") = 1.0, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("LeakyReLu", (class Layer* (*)(class Layer*, float, string)) &eddl::LeakyReLu, "C++: eddl::LeakyReLu(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("alpha") = 0.01, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Elu", (class Layer* (*)(class Layer*, float, string)) &eddl::Elu, "C++: eddl::Elu(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("alpha") = 1.0, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Selu", (class Layer* (*)(class Layer*, string)) &eddl::Selu, "C++: eddl::Selu(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Exponential", (class Layer* (*)(class Layer*, string)) &eddl::Exponential, "C++: eddl::Exponential(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Softplus", (class Layer* (*)(class Layer*, string)) &eddl::Softplus, "C++: eddl::Softplus(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Softsign", (class Layer* (*)(class Layer*, string)) &eddl::Softsign, "C++: eddl::Softsign(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Linear", (class Layer* (*)(class Layer*, float, string)) &eddl::Linear, "C++: eddl::Linear(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("alpha") = 1.0, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Tanh", (class Layer* (*)(class Layer*, string)) &eddl::Tanh, "C++: eddl::Tanh(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Conv", (class Layer* (*)(class Layer*, int, const vector<int>&, const vector<int>&, string, bool, int, const vector<int>&, string)) &eddl::Conv, "C++: eddl::Conv(class Layer*, int, const vector<int>&, const vector<int> &, string, bool, int, const vector<int>&, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("strides") = vector<int>{1, 1}, pybind11::arg("padding") = "same", pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("ConvT", (class Layer* (*)(class Layer*, int, const vector<int>&, const vector<int>&, string, const vector<int>&, const vector<int>&, bool, string)) &eddl::ConvT, "C++: eddl::ConvT(class Layer*, int, const vector<int>&, const vector<int>&, string, const vector<int>&, const vector<int>&, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("output_padding"), pybind11::arg("padding") = "same", pybind11::arg("dilation_rate") = vector<int>{1, 1}, pybind11::arg("strides") = vector<int>{1, 1}, pybind11::arg("use_bias") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Dense", (class Layer* (*)(class Layer*, int, bool, string)) &eddl::Dense, "C++: eddl::Dense(class Layer*, int, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("ndim"), pybind11::arg("use_bias") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Embedding", (class Layer* (*)(int, int, string)) &eddl::Embedding, "C++: eddl::Embedding(int, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("input_dim"), pybind11::arg("output_dim"), pybind11::arg("name") = "");
    m.def("Input", (class Layer* (*)(const vector<int>&, string)) &eddl::Input, "C++: eddl::Input(const vector<int>&, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("shape"), pybind11::arg("name") = "");
    m.def("UpSampling", (class Layer* (*)(class Layer*, const vector<int>&, string, string)) &eddl::UpSampling, "C++: eddl::UpSampling(class Layer*, const vector<int>&, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("size"), pybind11::arg("interpolation") = "nearest", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Reshape", (class Layer* (*)(class Layer*, const vector<int>&, string)) &eddl::Reshape, "C++: eddl::Reshape(class Layer*, const vector<int>&, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("shape"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Flatten", (class Layer* (*)(class Layer*, string)) &eddl::Flatten, "C++: eddl::Flatten(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Transpose", (class Layer* (*)(class Layer*, string)) &eddl::Transpose, "C++: eddl::Transpose(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- transformations ---
    m.def("Shift", (class Layer* (*)(class Layer*, vector<int>, string, float, string)) &eddl::Shift, "C++: eddl::Shift(class Layer*, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("shift"), pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Rotate", (class Layer* (*)(class Layer*, float, vector<int>, string, float, string)) &eddl::Rotate, "C++: eddl::Rotate(class Layer*, float, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("angle"), pybind11::arg("offset_center") = vector<int>{0, 0}, pybind11::arg("da_mode") = "original", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Scale", (class Layer* (*)(class Layer*, vector<int>, bool, string, float, string)) &eddl::Scale, "C++: eddl::Scale(class Layer*, vector<int>, bool, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("new_shape"), pybind11::arg("reshape") = true, pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Flip", (class Layer* (*)(class Layer*, int, string)) &eddl::Flip, "C++: eddl::Flip(class Layer*, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("axis") = 0, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("HorizontalFlip", (class Layer* (*)(class Layer*, string)) &eddl::HorizontalFlip, "C++: eddl::HorizontalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("VerticalFlip", (class Layer* (*)(class Layer*, string)) &eddl::VerticalFlip, "C++: eddl::VerticalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Crop", (class Layer* (*)(class Layer*, vector<int>, vector<int>, bool, float, string)) &eddl::Crop, "C++: eddl::Crop(class Layer*, vector<int>, vector<int>, bool, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("from_coords"), pybind11::arg("to_coords"), pybind11::arg("reshape") = true, pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("CenteredCrop", (class Layer* (*)(class Layer*, vector<int>, bool, float, string)) &eddl::CenteredCrop, "C++: eddl::CenteredCrop(class Layer*, vector<int>, bool, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("size"), pybind11::arg("reshape") = true, pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("CropScale", (class Layer* (*)(class Layer*, vector<int>, vector<int>, string, float, string)) &eddl::CropScale, "C++: eddl::CropScale(class Layer*, vector<int>, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("from_coords"), pybind11::arg("to_coords"), pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Cutout", (class Layer* (*)(class Layer*, vector<int>, vector<int>, float, string)) &eddl::Cutout, "C++: eddl::Cutout(class Layer*, vector<int>, vector<int>, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("from_coords"), pybind11::arg("to_coords"), pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- data augmentation ---
    m.def("RandomShift", (class Layer* (*)(class Layer*, vector<float>, vector<float>, string, float, string)) &eddl::RandomShift, "C++: eddl::RandomShift(class Layer*, vector<float>, vector<float>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor_x"), pybind11::arg("factor_y"), pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomRotation", (class Layer* (*)(class Layer*, vector<float>, vector<int>, string, float, string)) &eddl::RandomRotation, "C++: eddl::RandomRotation(class Layer*, vector<float>, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor"), pybind11::arg("offset_center") = vector<int>{0, 0}, pybind11::arg("da_mode") = "original", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomScale", (class Layer* (*)(class Layer*, vector<float>, string, float, string)) &eddl::RandomScale, "C++: eddl::RandomScale(class Layer*, vector<float>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor"), pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomFlip", (class Layer* (*)(class Layer*, int, string)) &eddl::RandomFlip, "C++: eddl::RandomFlip(class Layer*, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("axis"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomHorizontalFlip", (class Layer* (*)(class Layer*, string)) &eddl::RandomHorizontalFlip, "C++: eddl::RandomHorizontalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomVerticalFlip", (class Layer* (*)(class Layer*, string)) &eddl::RandomVerticalFlip, "C++: eddl::RandomVerticalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomCrop", (class Layer* (*)(class Layer*, vector<int>, string)) &eddl::RandomCrop, "C++: eddl::RandomCrop(class Layer*, vector<int>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("new_shape"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomCropScale", (class Layer* (*)(class Layer*, vector<float>, string, string)) &eddl::RandomCropScale, "C++: eddl::RandomCropScale(class Layer*, vector<float>, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor"), pybind11::arg("da_mode") = "nearest", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomCutout", (class Layer* (*)(class Layer*, vector<float>, vector<float>, float, string)) &eddl::RandomCutout, "C++: eddl::RandomCutout(class Layer*, vector<float>, vector<float>, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor_x"), pybind11::arg("factor_y"), pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- losses ---
    m.def("getLoss", (class Loss* (*)(string)) &eddl::getLoss, "C++: eddl::getLoss(string) --> class Loss*", pybind11::return_value_policy::reference, pybind11::arg("type"));
    m.def("newloss", (class NetLoss* (*)(const std::function<Layer*(vector<Layer*>)>&, vector<Layer*>, string)) &eddl::newloss, "C++: eddl::newloss(const std::function<Layer*(vector<Layer*>)>&, vector<Layer*>, string) --> class NetLoss*");
    m.def("newloss", (class NetLoss* (*)(const std::function<Layer*(Layer*)>&, Layer*, string)) &eddl::newloss, "C++: eddl::newloss(const std::function<Layer*(Layer*)>&, Layer*, string) --> class NetLoss*");

    // --- metrics ---
    m.def("getMetric", (class Metric* (*)(string)) &eddl::getMetric, "C++: eddl::getMetric(string) --> class Metric*", pybind11::return_value_policy::reference, pybind11::arg("type"));

    // --- merge layers ---
    m.def("Add", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Add, "C++: eddl::Add(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Average", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Average, "C++: eddl::Average(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Concat", (class Layer* (*)(const vector<Layer*>, unsigned int, string)) &eddl::Concat, "C++: eddl::Concat(const vector<Layer*>, unsigned int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("axis") = 1, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("MatMul", (class Layer* (*)(const vector<Layer*>, string)) &eddl::MatMul, "C++: eddl::MatMul(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Maximum", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Maximum, "C++: eddl::Maximum(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Minimum", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Minimum, "C++: eddl::Minimum(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Subtract", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Subtract, "C++: eddl::Subtract(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- noise layers ---
    m.def("GaussianNoise", (class Layer* (*)(class Layer*, float, string)) &eddl::GaussianNoise, "C++: eddl::GaussianNoise(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("stddev"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- normalization layers ---
    m.def("BatchNormalization", (class Layer* (*)(class Layer*, float, float, bool, string)) &eddl::BatchNormalization, "C++: eddl::BatchNormalization(class Layer*, float, float, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("momentum") = 0.9f, pybind11::arg("epsilon") = 0.001f, pybind11::arg("affine") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("LayerNormalization", (class Layer* (*)(class Layer*, float, float, bool, string)) &eddl::LayerNormalization, "C++: eddl::LayerNormalization(class Layer*, float, float, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("momentum") = 0.9f, pybind11::arg("epsilon") = 0.001f, pybind11::arg("affine") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GroupNormalization", (class Layer* (*)(class Layer*, int, float, float, bool, string)) &eddl::GroupNormalization, "C++: eddl::GroupNormalization(class Layer*, int, float, float, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("groups"), pybind11::arg("momentum") = 0.9f, pybind11::arg("epsilon") = 0.001f, pybind11::arg("affine") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Norm", (class Layer* (*)(class Layer*, float, string)) &eddl::Norm, "C++: eddl::Norm(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("epsilon") = 0.001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("NormMax", (class Layer* (*)(class Layer*, float, string)) &eddl::NormMax, "C++: eddl::NormMax(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("epsilon") = 0.001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("NormMinMax", (class Layer* (*)(class Layer*, float, string)) &eddl::NormMinMax, "C++: eddl::NormMinMax(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("epsilon") = 0.001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Dropout", (class Layer* (*)(class Layer*, float, string)) &eddl::Dropout, "C++: eddl::Dropout(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("rate"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- operator layers ---
    m.def("Abs", (class Layer* (*)(class Layer*)) &eddl::Abs, "C++: eddl::Abs(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Diff", (class Layer* (*)(class Layer*, class Layer*)) &eddl::Diff, "C++: eddl::Diff(class Layer*, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Diff", (class Layer* (*)(class Layer*, float)) &eddl::Diff, "C++: eddl::Diff(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Diff", (class Layer* (*)(float, class Layer*)) &eddl::Diff, "C++: eddl::Diff(float, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("k"), pybind11::arg("l1"), pybind11::keep_alive<0, 2>());
    m.def("Div", (class Layer* (*)(class Layer*, class Layer*)) &eddl::Div, "C++: eddl::Div(class Layer*, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Div", (class Layer* (*)(class Layer*, float)) &eddl::Div, "C++: eddl::Div(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Div", (class Layer* (*)(float, class Layer*)) &eddl::Div, "C++: eddl::Div(float, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("k"), pybind11::arg("l1"), pybind11::keep_alive<0, 2>());
    m.def("Exp", (class Layer* (*)(class Layer*)) &eddl::Exp, "C++: eddl::Exp(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Log", (class Layer* (*)(class Layer*)) &eddl::Log, "C++: eddl::Log(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Log2", (class Layer* (*)(class Layer*)) &eddl::Log2, "C++: eddl::Log2(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Log10", (class Layer* (*)(class Layer*)) &eddl::Log10, "C++: eddl::Log10(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Mult", (class Layer* (*)(class Layer*, class Layer*)) &eddl::Mult, "C++: eddl::Mult(class Layer*, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Mult", (class Layer* (*)(class Layer*, float)) &eddl::Mult, "C++: eddl::Mult(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Mult", (class Layer* (*)(float, class Layer*)) &eddl::Mult, "C++: eddl::Mult(float, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("k"), pybind11::arg("l1"), pybind11::keep_alive<0, 2>());
    m.def("Pow", (class Layer* (*)(class Layer*, class Layer*)) &eddl::Pow, "C++: eddl::Pow(class Layer*, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Pow", (class Layer* (*)(class Layer*, float)) &eddl::Pow, "C++: eddl::Pow(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Sqrt", (class Layer* (*)(class Layer*)) &eddl::Sqrt, "C++: eddl::Sqrt(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Sum", (class Layer* (*)(class Layer*, class Layer*)) &eddl::Sum, "C++: eddl::Sum(class Layer*, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Sum", (class Layer* (*)(class Layer*, float)) &eddl::Sum, "C++: eddl::Sum(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Sum", (class Layer* (*)(float, class Layer*)) &eddl::Sum, "C++: eddl::Sum(float, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("k"), pybind11::arg("l1"), pybind11::keep_alive<0, 2>());
    m.def("Select", (class Layer* (*)(class Layer*, vector<string>, string)) &eddl::Select, "C++: eddl::Select(class Layer*, vector<string>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("indices"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Permute", (class Layer* (*)(class Layer*, vector<int>, string)) &eddl::Permute, "C++: eddl::Permute(class Layer*, vector<int>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("dims"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- reduction layers ---
    m.def("ReduceMean", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceMean, "C++: eddl::ReduceMean(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis") = std::vector<int>({0}), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceVar", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceVar, "C++: eddl::ReduceVar(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis") = std::vector<int>({0}), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceSum", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceSum, "C++: eddl::ReduceSum(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis") = std::vector<int>({0}), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceMax", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceMax, "C++: eddl::ReduceMax(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis") = std::vector<int>({0}), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceMin", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceMin, "C++: eddl::ReduceMin(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis") = std::vector<int>({0}), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());

    // --- generator layers ---
    m.def("GaussGenerator", (class Layer* (*)(float, float, vector<int>)) &eddl::GaussGenerator, "C++: eddl::GaussGenerator(float, float, vector<int>) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("mean"), pybind11::arg("stdev"), pybind11::arg("size"));
    m.def("UniformGenerator", (class Layer* (*)(float, float, vector<int>)) &eddl::UniformGenerator, "C++: eddl::UniformGenerator(float, float, vector<int>) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("low"), pybind11::arg("high"), pybind11::arg("size"));

    // --- optimizers ---
    m.def("setlr", (void (*)(class Net*, vector<float>)) &eddl::setlr, "C++: eddl::setlr(class Net*, vector<float>) --> void", pybind11::arg("net"), pybind11::arg("p"));
    m.def("adadelta", (class Optimizer* (*)(float, float, float, float)) &eddl::adadelta, "C++: eddl::adadelta(float, float, float, float) --> class Optimizer *", pybind11::return_value_policy::reference, pybind11::arg("lr"), pybind11::arg("rho"), pybind11::arg("epsilon"), pybind11::arg("weight_decay"));
    m.def("adam", (class Optimizer* (*)(float, float, float, float, float, bool)) &eddl::adam, "C++: eddl::adam(float, float, float, float, float, bool) --> class Optimizer *", pybind11::return_value_policy::reference, pybind11::arg("lr") = 0.01, pybind11::arg("beta_1") = 0.9, pybind11::arg("beta_2") = 0.999, pybind11::arg("epsilon")=0.000001, pybind11::arg("weight_decay") = 0, pybind11::arg("amsgrad") = false);
    m.def("adagrad", (class Optimizer* (*)(float, float, float)) &eddl::adagrad, "C++: eddl::adagrad(float, float, float) --> class Optimizer *", pybind11::return_value_policy::reference, pybind11::arg("lr"), pybind11::arg("epsilon"), pybind11::arg("weight_decay"));
    m.def("adamax", (class Optimizer* (*)(float, float, float, float, float)) &eddl::adamax, "C++: eddl::adamax(float, float, float, float, float) --> class Optimizer *", pybind11::return_value_policy::reference, pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("weight_decay"));
    m.def("nadam", (class Optimizer* (*)(float, float, float, float, float)) &eddl::nadam, "C++: eddl::nadam(float, float, float, float, float) --> class Optimizer *", pybind11::return_value_policy::reference, pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("schedule_decay"));
    m.def("rmsprop", (class Optimizer* (*)(float, float, float, float)) &eddl::rmsprop, "C++: eddl::rmsprop(float, float, float, float) --> class Optimizer *", pybind11::return_value_policy::reference, pybind11::arg("lr") = 0.01, pybind11::arg("rho") = 0.9, pybind11::arg("epsilon") = 0.00001, pybind11::arg("weight_decay") = 0.0);
    m.def("sgd", (class Optimizer* (*)(float, float, float, bool)) &eddl::sgd, "C++: eddl::sgd(float, float, float, bool) --> class Optimizer *", pybind11::return_value_policy::reference, pybind11::arg("lr") = 0.01f, pybind11::arg("momentum") = 0.0f, pybind11::arg("weight_decay") = 0.0f, pybind11::arg("nesterov") = false);

    // --- pooling layers ---
    m.def("AveragePool", (class Layer* (*)(class Layer*, const vector<int>&, const vector<int> &, string, string)) &eddl::AveragePool, "C++: eddl::AveragePool(class Layer*, const vector<int>&, const vector<int> &, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2}, pybind11::arg("strides") = vector<int>{2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalMaxPool", (class Layer* (*)(Layer*, string)) &eddl::GlobalMaxPool, "C++: eddl::GlobalMaxPool(Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAveragePool", (class Layer* (*)(Layer*, string)) &eddl::GlobalAveragePool, "C++: eddl::GlobalAveragePool(Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("MaxPool", (class Layer* (*)(class Layer*, const vector<int>&, const vector<int> &, string, string)) &eddl::MaxPool, "C++: eddl::MaxPool(class Layer*, const vector<int>&, const vector<int> &, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2}, pybind11::arg("strides") = vector<int>{2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- recurrent layers ---
    m.def("RNN", (class Layer* (*)(class Layer*, int, int, bool, float, bool, string)) &eddl::RNN, "C++: eddl::RNN(class Layer*, int, int, bool, float, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("units"), pybind11::arg("num_layers"), pybind11::arg("use_bias") = true, pybind11::arg("dropout") = 0.0f, pybind11::arg("bidirectional") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("LSTM", (class Layer* (*)(class Layer*, int, int, bool, float, bool, string)) &eddl::LSTM, "C++: eddl::LSTM(class Layer*, int, int, bool, float, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("units"), pybind11::arg("num_layers"), pybind11::arg("use_bias") = true, pybind11::arg("dropout") = 0.0f, pybind11::arg("bidirectional") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- initializers ---
    m.def("GlorotNormal", (class Layer* (*)(class Layer*, int)) &eddl::GlorotNormal, "C++: eddl::GlorotNormal(class Layer*, int) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("GlorotUniform", (class Layer* (*)(class Layer*, int)) &eddl::GlorotUniform, "C++: eddl::GlorotUniform(class Layer*, int) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("RandomNormal", (class Layer* (*)(class Layer*, float, float, float)) &eddl::RandomNormal, "C++: eddl::RandomNormal(class Layer*, float, float, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("m") = 0.0, pybind11::arg("s") = 0.1, pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("RandomUniform", (class Layer* (*)(class Layer*, float, float, float)) &eddl::RandomUniform, "C++: eddl::RandomUniform(class Layer*, float, float, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("min") = 0.0, pybind11::arg("max") = 0.1, pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("Constant", (class Layer* (*)(class Layer*, float)) &eddl::Constant, "C++: eddl::Constant(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("v") = 0.1, pybind11::keep_alive<0, 1>());

    // --- regularizers ---
    m.def("L2", (class Layer* (*)(class Layer*, float)) &eddl::L2, "C++: eddl::L2(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>());
    m.def("L1", (class Layer* (*)(class Layer*, float)) &eddl::L1, "C++: eddl::L1(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("l1"), pybind11::keep_alive<0, 1>());
    m.def("L1L2", (class Layer* (*)(class Layer*, float, float)) &eddl::L1L2, "C++: eddl::L1L2(class Layer*, float, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>());

    // --- computing services ---
    m.def("CS_CPU", (class CompServ* (*)(int, string)) &eddl::CS_CPU, "C++: eddl::CS_CPU(int, string) --> class CompServ*", pybind11::arg("th") = -1, pybind11::arg("mem") = "low_mem");
    m.def("CS_GPU", (class CompServ* (*)(const vector<int>, string)) &eddl::CS_GPU, "C++: eddl::CS_GPU(const vector<int>&, string) --> class CompServ*", pybind11::arg("g") = vector<int>{1}, pybind11::arg("mem") = "low_mem");
    m.def("CS_GPU", (class CompServ* (*)(const vector<int>, int, string)) &eddl::CS_GPU, "C++: eddl::CS_GPU(const vector<int>&, int, string) --> class CompServ*", pybind11::arg("g") = vector<int>{1}, pybind11::arg("lsb") = 1, pybind11::arg("mem") = "low_mem");
    m.def("CS_FGPA", (class CompServ* (*)(const vector<int>&, int)) &eddl::CS_FGPA, "C++: eddl::CS_FGPA(const vector<int>&, int) --> class CompServ*", pybind11::arg("f"), pybind11::arg("lsb") = 1);
    m.def("CS_COMPSS", (class CompServ* (*)(string)) &eddl::CS_COMPSS, "C++: eddl::CS_COMPSS(string) --> class CompServ*", pybind11::arg("filename"));
    m.def("exist", (bool (*)(string)) &eddl::exist, "C++: eddl::exist(string) --> bool", pybind11::arg("name"));

    // --- fine-grained methods ---
    m.def("random_indices", (vector<int> (*)(int, int)) &eddl::random_indices, "C++: eddl::random_indices(int, int) --> vector<int>", pybind11::arg("batch_size"), pybind11::arg("num_samples"));
    m.def("train_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>)) &eddl::train_batch, "C++: eddl::train_batch(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("indices"));
    m.def("train_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>)) &eddl::train_batch, "C++: eddl::train_batch(class Net*, vector<Tensor*>, vector<Tensor*>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"));
    m.def("eval_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>)) &eddl::eval_batch, "C++: eddl::eval_batch(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("indices"));
    m.def("eval_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>)) &eddl::eval_batch, "C++: eddl::eval_batch(class Net*, vector<Tensor*>, vector<Tensor*>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"));
    m.def("next_batch", (void (*)(vector<Tensor*>, vector<Tensor*>)) &eddl::next_batch, "C++: eddl::next_batch(vector<Tensor*>, vector<Tensor*>) --> void", pybind11::arg("in"), pybind11::arg("out"));
    m.def("forward", (vector<Layer*> (*)(class Net*, vector<Layer*>)) &eddl::forward, "C++: eddl::forward(class Net*, vector<Layer*>) --> vector<Layer*>", pybind11::arg("m"), pybind11::arg("in"));
    m.def("forward", (vector<Layer*> (*)(class Net*, vector<Tensor*>)) &eddl::forward, "C++: eddl::forward(class Net*, vector<Tensor*>) --> vector<Layer*>", pybind11::arg("m"), pybind11::arg("in"));
    m.def("forward", (vector<Layer*> (*)(class Net*)) &eddl::forward, "C++: eddl::forward(class Net*) --> vector<Layer*>", pybind11::arg("m"));
    m.def("forward", (vector<Layer*> (*)(class Net*, int)) &eddl::forward, "C++: eddl::forward(class Net*, int) --> vector<Layer*>", pybind11::arg("m"), pybind11::arg("b"));
    m.def("detach", (class Layer* (*)(class Layer*)) &eddl::detach, "C++: eddl::detach(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"));
    m.def("detach", (class vector<Layer*> (*)(class vector<Layer*>)) &eddl::detach, "C++: eddl::detach(class vector<Layer*>) --> class vector<Layer*>", pybind11::return_value_policy::reference, pybind11::arg("l"));
    m.def("backward", (void (*)(class Net*, vector<Tensor*>)) &eddl::backward, "C++: eddl::backward(class Net*, vector<Tensor*>) --> void", pybind11::arg("m"), pybind11::arg("target"));
    m.def("getTensor", (class Tensor* (*)(class Layer*)) &eddl::getTensor, "C++: eddl::getTensor(class Layer*) --> class Tensor*", pybind11::return_value_policy::reference, pybind11::arg("l"));
    m.def("getOut", (vector<Layer*> (*)(class Net*)) &eddl::getOut, "C++: eddl::getOut(class Net*) --> vector<Layer*>", pybind11::arg("net"));

    // --- model methods ---
    m.def("Model", (class Net* (*)(vector<Layer*>, vector<Layer*>)) &eddl::Model, "C++: eddl::Model(vector<Layer*>, vector<Layer*>) --> class Net*", pybind11::arg("in"), pybind11::arg("out"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("build", (void (*)(class Net*, class Optimizer*, const vector<string>&, const vector<string>&, class CompServ*, bool)) &eddl::build, "C++: eddl::build(class Net*, class Optimizer*, const vector<string>&, const vector<string>&, class CompServ*, bool) --> void", pybind11::arg("net"), pybind11::arg("o"), pybind11::arg("lo"), pybind11::arg("me"), pybind11::arg("cs") = nullptr, pybind11::arg("init_weights") = true, pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 5>());
    m.def("toGPU", (void (*)(class Net*, vector<int>, int)) &eddl::toGPU, "C++: eddl::toGPU(class Net*, vector<int>, int) --> void", pybind11::arg("net"), pybind11::arg("g"), pybind11::arg("lsb"));
    m.def("toGPU", (void (*)(class Net*, vector<int>, string)) &eddl::toGPU, "C++: eddl::toGPU(class Net*, vector<int>, string) --> void", pybind11::arg("net"), pybind11::arg("g"), pybind11::arg("mem"));
    m.def("toGPU", (void (*)(class Net*, vector<int>, int, string)) &eddl::toGPU, "C++: eddl::toGPU(class Net*, vector<int>, int, string) --> void", pybind11::arg("net"), pybind11::arg("g"), pybind11::arg("lsb"), pybind11::arg("mem"));
    m.def("toGPU", (void (*)(class Net*, vector<int>)) &eddl::toGPU, "C++: eddl::toGPU(class Net*, vector<int>) --> void", pybind11::arg("net"), pybind11::arg("g"));
    m.def("toGPU", (void (*)(class Net*, string)) &eddl::toGPU, "C++: eddl::toGPU(class Net*, string) --> void", pybind11::arg("net"), pybind11::arg("mem"));
    m.def("setlogfile", (void (*)(class Net*, string)) &eddl::setlogfile, "C++: eddl::setlogfile(class Net*, string) --> void", pybind11::arg("net"), pybind11::arg("fname"));
    m.def("load", (void (*)(class Net*, const string&, string)) &eddl::load, "C++: eddl::load(class Net*, const string&, string) --> void", pybind11::arg("m"), pybind11::arg("fname"), pybind11::arg("format") = "bin");
    m.def("save", (void (*)(class Net*, const string&, string)) &eddl::save, "C++: eddl::save(class Net*, const string&, string) --> void", pybind11::arg("m"), pybind11::arg("fname"), pybind11::arg("format") = "bin");
    m.def("plot", (void (*)(class Net*, string, string)) &eddl::plot, "C++: eddl::plot(class Net*, string, string) --> void", pybind11::arg("m"), pybind11::arg("fname"), pybind11::arg("string") = "LR");
    m.def("fit", (void (*)(class Net*, const vector<Tensor*>&, const vector<Tensor*>&, int, int)) &eddl::fit, "C++: eddl::fit(class Net*, const vector<Tensor*>&, const vector<Tensor*>&, int, int) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("m"), pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("batch"), pybind11::arg("epochs"));
    m.def("evaluate", (void (*)(class Net*, const vector<Tensor*>&, const vector<Tensor*>&)) &eddl::evaluate, "C++: eddl::evaluate(class Net*, const vector<Tensor*>&, const vector<Tensor*>&) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("m"), pybind11::arg("in"), pybind11::arg("out"));

    // not implemented upstream:
    //   Affine
    //   ColorJitter
    //   Grayscale
    //   Normalize
    //   Pad
    //   RandomAffine
    //   RandomCenteredCrop
    //   RandomGrayscale

#ifdef EDDL_WITH_PROTOBUF
    // --- serialization ---
    m.def("save_net_to_onnx_file", (void (*)(class Net*, string)) &eddl::save_net_to_onnx_file, "C++: eddl::save_net_to_onnx_file(class Net *, string) --> void", pybind11::arg("net"), pybind11::arg("path"));
    m.def("import_net_from_onnx_file", (class Net* (*)(string)) &eddl::import_net_from_onnx_file, "C++: eddl::import_net_from_onnx_file(string) --> class Net*", pybind11::arg("path"));
    m.def("serialize_net_to_onnx_string", [](Net* net, bool gradients) -> pybind11::bytes {
      string* s = eddl::serialize_net_to_onnx_string(net, gradients);
      return pybind11::bytes(*s);
    }, pybind11::arg("net"), pybind11::arg("gradients"));
    m.def("import_net_from_onnx_string", [](pybind11::bytes model_string) -> Net* {
      string s = string(model_string);
      return eddl::import_net_from_onnx_string(&s, s.size());
    }, pybind11::arg("model_string"));
#endif
}
