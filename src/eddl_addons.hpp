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
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#ifdef EDDL_WITH_PROTOBUF
#include <eddl/serialization/onnx/eddl_onnx.h>
#endif

PYBIND11_MAKE_OPAQUE(std::vector<Layer*>);

// Use return_value_policy::reference for objects that get deleted on the C++
// side. In particular, layers and optimizers are deleted by the Net destructor

void eddl_addons(pybind11::module &m) {

    // Avoid "Could not allocate weak reference" error when returning [layer]
    pybind11::bind_vector<std::vector<Layer*>>(m, "VLayer");
    pybind11::implicitly_convertible<pybind11::list, std::vector<Layer*>>();

    // --- specific layer classes ---

    // Minimal bindings for specific layers. These are not meant to be used
    // directly, but their presence allows pybind11 to return the relevant
    // specific layer type from functions like getLayer.

    // tier 1
    pybind11::class_<LinLayer, std::shared_ptr<LinLayer>, Layer>(m, "LinLayer", "");
    pybind11::class_<MLayer, std::shared_ptr<MLayer>, Layer>(m, "MLayer", "");
    pybind11::class_<OperatorLayer, std::shared_ptr<OperatorLayer>, Layer>(m, "OperatorLayer", "");
    pybind11::class_<ReductionLayer, std::shared_ptr<ReductionLayer>, Layer>(m, "ReductionLayer", "");
    pybind11::class_<ReductionLayer2, std::shared_ptr<ReductionLayer2>, Layer>(m, "ReductionLayer2", "");

    // tier 2
    pybind11::class_<GeneratorLayer, std::shared_ptr<GeneratorLayer>, LinLayer>(m, "GeneratorLayer", "");
    pybind11::class_<LActivation, std::shared_ptr<LActivation>, LinLayer>(m, "LActivation", "");
    pybind11::class_<LBatchNorm, std::shared_ptr<LBatchNorm>, LinLayer>(m, "LBatchNorm", "");
    pybind11::class_<LConstOfTensor, std::shared_ptr<LConstOfTensor>, LinLayer>(m, "LConstOfTensor", "");
    pybind11::class_<LConv, std::shared_ptr<LConv>, LinLayer>(m, "LConv", "");
    pybind11::class_<LConv1D, std::shared_ptr<LConv1D>, LinLayer>(m, "LConv1D", "");
    pybind11::class_<LConv2dActivation, std::shared_ptr<LConv2dActivation>, LinLayer>(m, "LConv2dActivation", "");
    pybind11::class_<LConv3D, std::shared_ptr<LConv3D>, LinLayer>(m, "LConv3D", "");
    pybind11::class_<LConvT2D, std::shared_ptr<LConvT2D>, LinLayer>(m, "LConvT2D", "");
    pybind11::class_<LConvT3D, std::shared_ptr<LConvT3D>, LinLayer>(m, "LConvT3D", "");
    pybind11::class_<LDataAugmentation, std::shared_ptr<LDataAugmentation>, LinLayer>(m, "LDataAugmentation", "");
    pybind11::class_<LDense, std::shared_ptr<LDense>, LinLayer>(m, "LDense", "");
    pybind11::class_<LDropout, std::shared_ptr<LDropout>, LinLayer>(m, "LDropout", "");
    pybind11::class_<LEmbedding, std::shared_ptr<LEmbedding>, LinLayer>(m, "LEmbedding", "");
    pybind11::class_<LEqual, std::shared_ptr<LEqual>, LinLayer>(m, "LEqual", "");
    pybind11::class_<LExpand, std::shared_ptr<LExpand>, LinLayer>(m, "LExpand", "");
    pybind11::class_<LGather, std::shared_ptr<LGather>, LinLayer>(m, "LGather", "");
    pybind11::class_<LGaussianNoise, std::shared_ptr<LGaussianNoise>, LinLayer>(m, "LGaussianNoise", "");
    pybind11::class_<LGroupNorm, std::shared_ptr<LGroupNorm>, LinLayer>(m, "LGroupNorm", "");
    pybind11::class_<LInput, std::shared_ptr<LInput>, LinLayer>(m, "LInput", "");
    pybind11::class_<LLayerNorm, std::shared_ptr<LLayerNorm>, LinLayer>(m, "LLayerNorm", "");
    pybind11::class_<LNorm, std::shared_ptr<LNorm>, LinLayer>(m, "LNorm", "");
    pybind11::class_<LNormMax, std::shared_ptr<LNormMax>, LinLayer>(m, "LNormMax", "");
    pybind11::class_<LNormMinMax, std::shared_ptr<LNormMinMax>, LinLayer>(m, "LNormMinMax", "");
    pybind11::class_<LPad, std::shared_ptr<LPad>, LinLayer>(m, "LPad", "");
    pybind11::class_<LPermute, std::shared_ptr<LPermute>, LinLayer>(m, "LPermute", "");
    pybind11::class_<LPool, std::shared_ptr<LPool>, LinLayer>(m, "LPool", "");
    pybind11::class_<LPool1D, std::shared_ptr<LPool1D>, LinLayer>(m, "LPool1D", "");
    pybind11::class_<LPool3D, std::shared_ptr<LPool3D>, LinLayer>(m, "LPool3D", "");
    pybind11::class_<LReshape, std::shared_ptr<LReshape>, LinLayer>(m, "LReshape", "");
    pybind11::class_<LResize, std::shared_ptr<LResize>, LinLayer>(m, "LResize", "");
    pybind11::class_<LSelect, std::shared_ptr<LSelect>, LinLayer>(m, "LSelect", "");
    pybind11::class_<LShape, std::shared_ptr<LShape>, LinLayer>(m, "LShape", "");
    pybind11::class_<LSplit, std::shared_ptr<LSplit>, LinLayer>(m, "LSplit", "");
    pybind11::class_<LSqueeze, std::shared_ptr<LSqueeze>, LinLayer>(m, "LSqueeze", "");
    pybind11::class_<LTensor, std::shared_ptr<LTensor>, LinLayer>(m, "LTensor", "");
    pybind11::class_<LTranspose, std::shared_ptr<LTranspose>, LinLayer>(m, "LTranspose", "");
    pybind11::class_<LUnsqueeze, std::shared_ptr<LUnsqueeze>, LinLayer>(m, "LUnsqueeze", "");
    pybind11::class_<LUpSampling, std::shared_ptr<LUpSampling>, LinLayer>(m, "LUpSampling", "");
    pybind11::class_<LUpSampling3D, std::shared_ptr<LUpSampling3D>, LinLayer>(m, "LUpSampling3D", "");

    pybind11::class_<LAdd, std::shared_ptr<LAdd>, MLayer>(m, "LAdd", "");
    pybind11::class_<LAverage, std::shared_ptr<LAverage>, MLayer>(m, "LAverage", "");
    pybind11::class_<LConcat, std::shared_ptr<LConcat>, MLayer>(m, "LConcat", "");
    pybind11::class_<LCopyStates, std::shared_ptr<LCopyStates>, MLayer>(m, "LCopyStates", "");
    pybind11::class_<LGRU, std::shared_ptr<LGRU>, MLayer>(m, "LGRU", "");
    pybind11::class_<LLSTM, std::shared_ptr<LLSTM>, MLayer>(m, "LLSTM", "");
    pybind11::class_<LMatMul, std::shared_ptr<LMatMul>, MLayer>(m, "LMatMul", "");
    pybind11::class_<LMaximum, std::shared_ptr<LMaximum>, MLayer>(m, "LMaximum", "");
    pybind11::class_<LMinimum, std::shared_ptr<LMinimum>, MLayer>(m, "LMinimum", "");
    pybind11::class_<LRNN, std::shared_ptr<LRNN>, MLayer>(m, "LRNN", "");
    pybind11::class_<LSubtract, std::shared_ptr<LSubtract>, MLayer>(m, "LSubtract", "");
    pybind11::class_<LStates, std::shared_ptr<LStates>, MLayer>(m, "LStates", "");
    pybind11::class_<LWhere, std::shared_ptr<LWhere>, MLayer>(m, "LWhere", "");

    pybind11::class_<LAbs, std::shared_ptr<LAbs>, OperatorLayer>(m, "LAbs", "");
    pybind11::class_<LClamp, std::shared_ptr<LClamp>, OperatorLayer>(m, "LClamp", "");
    pybind11::class_<LDiff, std::shared_ptr<LDiff>, OperatorLayer>(m, "LDiff", "");
    pybind11::class_<LDiv, std::shared_ptr<LDiv>, OperatorLayer>(m, "LDiv", "");
    pybind11::class_<LExp, std::shared_ptr<LExp>, OperatorLayer>(m, "LExp", "");
    pybind11::class_<LLog, std::shared_ptr<LLog>, OperatorLayer>(m, "LLog", "");
    pybind11::class_<LLog10, std::shared_ptr<LLog10>, OperatorLayer>(m, "LLog10", "");
    pybind11::class_<LLog2, std::shared_ptr<LLog2>, OperatorLayer>(m, "LLog2", "");
    pybind11::class_<LMult, std::shared_ptr<LMult>, OperatorLayer>(m, "LMult", "");
    pybind11::class_<LPow, std::shared_ptr<LPow>, OperatorLayer>(m, "LPow", "");
    pybind11::class_<LSqrt, std::shared_ptr<LSqrt>, OperatorLayer>(m, "LSqrt", "");
    pybind11::class_<LSum, std::shared_ptr<LSum>, OperatorLayer>(m, "LSum", "");

    pybind11::class_<LRMax, std::shared_ptr<LRMax>, ReductionLayer>(m, "LRMax", "");
    pybind11::class_<LRMean, std::shared_ptr<LRMean>, ReductionLayer>(m, "LRMean", "");
    pybind11::class_<LRMin, std::shared_ptr<LRMin>, ReductionLayer>(m, "LRMin", "");
    pybind11::class_<LRSum, std::shared_ptr<LRSum>, ReductionLayer>(m, "LRSum", "");
    pybind11::class_<LRVar, std::shared_ptr<LRVar>, ReductionLayer>(m, "LRVar", "");

    pybind11::class_<LRArgmax, std::shared_ptr<LRArgmax>, ReductionLayer2>(m, "LRArgmax", "");

    // tier 3
    pybind11::class_<LGauss, std::shared_ptr<LGauss>, GeneratorLayer>(m, "LGauss", "");
    pybind11::class_<LUniform, std::shared_ptr<LUniform>, GeneratorLayer>(m, "LUniform", "");

    pybind11::class_<LCrop, std::shared_ptr<LCrop>, LDataAugmentation>(m, "LCrop", "");
    pybind11::class_<LCropRandom, std::shared_ptr<LCropRandom>, LDataAugmentation>(m, "LCropRandom", "");
    pybind11::class_<LCropScaleRandom, std::shared_ptr<LCropScaleRandom>, LDataAugmentation>(m, "LCropScaleRandom", "");
    pybind11::class_<LCutout, std::shared_ptr<LCutout>, LDataAugmentation>(m, "LCutout", "");
    pybind11::class_<LCutoutRandom, std::shared_ptr<LCutoutRandom>, LDataAugmentation>(m, "LCutoutRandom", "");
    pybind11::class_<LFlip, std::shared_ptr<LFlip>, LDataAugmentation>(m, "LFlip", "");
    pybind11::class_<LFlipRandom, std::shared_ptr<LFlipRandom>, LDataAugmentation>(m, "LFlipRandom", "");
    pybind11::class_<LRotate, std::shared_ptr<LRotate>, LDataAugmentation>(m, "LRotate", "");
    pybind11::class_<LRotateRandom, std::shared_ptr<LRotateRandom>, LDataAugmentation>(m, "LRotateRandom", "");
    pybind11::class_<LScale, std::shared_ptr<LScale>, LDataAugmentation>(m, "LScale", "");
    pybind11::class_<LScaleRandom, std::shared_ptr<LScaleRandom>, LDataAugmentation>(m, "LScaleRandom", "");
    pybind11::class_<LShift, std::shared_ptr<LShift>, LDataAugmentation>(m, "LShift", "");
    pybind11::class_<LShiftRandom, std::shared_ptr<LShiftRandom>, LDataAugmentation>(m, "LShiftRandom", "");

    pybind11::class_<LAveragePool, std::shared_ptr<LAveragePool>, LPool>(m, "LAveragePool", "");
    pybind11::class_<LMaxPool, std::shared_ptr<LMaxPool>, LPool>(m, "LMaxPool", "");

    pybind11::class_<LAveragePool1D, std::shared_ptr<LAveragePool1D>, LPool1D>(m, "LAveragePool1D", "");
    pybind11::class_<LMaxPool1D, std::shared_ptr<LMaxPool1D>, LPool1D>(m, "LMaxPool1D", "");

    pybind11::class_<LAveragePool3D, std::shared_ptr<LAveragePool3D>, LPool3D>(m, "LAveragePool3D", "");
    pybind11::class_<LMaxPool3D, std::shared_ptr<LMaxPool3D>, LPool3D>(m, "LMaxPool3D", "");

    // tier 4
    pybind11::class_<LCropScale, std::shared_ptr<LCropScale>, LCrop>(m, "LCropScale", "");


    // --- core layers ---
    m.def("Activation", (class Layer* (*)(class Layer*, string, vector<float>, string)) &eddl::Activation, "C++: eddl::Activation(class Layer*, string, vector<float>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("activation"), pybind11::arg("params") = vector<float>{}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Softmax", (class Layer* (*)(class Layer*, int, string)) &eddl::Softmax, "C++: eddl::Softmax(class Layer*, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("axis") = -1, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
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
    m.def("Conv1D", (class Layer* (*)(class Layer*, int, vector<int>, vector<int>, string, bool, int, const vector<int>, string)) &eddl::Conv1D, "C++: eddl::Conv1D(class Layer*, int, vector<int>, vector<int>, string, bool, int, const vector<int>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("strides") = vector<int>{1}, pybind11::arg("padding") = "same", pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Conv2D", (Layer* (*)(Layer*, int, const vector<int>&, const vector<int>&, string, bool, int, const vector<int>&, string)) &eddl::Conv2D, "2D convolution layer", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("strides") = vector<int>{1, 1}, pybind11::arg("padding") = "same", pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Conv3D", (Layer* (*)(Layer*, int, const vector<int>&, const vector<int>&, string, bool, int, const vector<int>&, string)) &eddl::Conv3D, "3D convolution layer", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("strides") = vector<int>{1, 1, 1}, pybind11::arg("padding") = "same", pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("PointwiseConv", (class Layer* (*)(class Layer*, int, const vector<int>&, bool, int, const vector<int>&, string)) &eddl::PointwiseConv, "Pointwise convolution", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("strides") = vector<int>{1, 1}, pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("PointwiseConv2D", (Layer* (*)(Layer*, int, const vector<int>&, bool, int, const vector<int>&, string)) &eddl::PointwiseConv2D, "Pointwise 2D convolution layer", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("strides") = vector<int>{1, 1}, pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("ConvT2D", (Layer* (*)(Layer*, int, const vector<int>&, const vector<int>&, string, bool, int, const vector<int>&, string)) &eddl::ConvT2D, "2D transposed convolution layer", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("strides") = vector<int>{1, 1}, pybind11::arg("padding") = "same", pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("ConvT3D", (Layer* (*)(Layer*, int, const vector<int>&, const vector<int>&, string, bool, int, const vector<int>&, string)) &eddl::ConvT3D, "3D transposed convolution layer", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("strides") = vector<int>{1, 1, 1}, pybind11::arg("padding") = "same", pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Dense", (class Layer* (*)(class Layer*, int, bool, string)) &eddl::Dense, "C++: eddl::Dense(class Layer*, int, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("ndim"), pybind11::arg("use_bias") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Embedding", (class Layer* (*)(class Layer*, int, int, int, bool, string)) &eddl::Embedding, "C++: eddl::Embedding(class Layer*, int, int, int, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("vocsize"), pybind11::arg("length"), pybind11::arg("output_dim"), pybind11::arg("mask_zeros") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Input", (class Layer* (*)(const vector<int>&, string)) &eddl::Input, "C++: eddl::Input(const vector<int>&, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("shape"), pybind11::arg("name") = "");
    m.def("UpSampling", (class Layer* (*)(class Layer*, const vector<int>&, string, string)) &eddl::UpSampling, "C++: eddl::UpSampling(class Layer*, const vector<int>&, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("size"), pybind11::arg("interpolation") = "nearest", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("UpSampling2D", (Layer* (*)(Layer*, const vector<int>&, string, string)) &eddl::UpSampling2D, "C++: eddl::UpSampling2D(class Layer*, const vector<int>&, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("size"), pybind11::arg("interpolation") = "nearest", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("UpSampling3D", (Layer* (*)(Layer*, vector<int>, bool, string, float, string, string)) &eddl::UpSampling3D, "3D Upsampling layer", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("new_shape"), pybind11::arg("reshape") = true, pybind11::arg("da_mode") = "constant", pybind11::arg("constant") = 0.0f, pybind11::arg("coordinate_transformation_mode") = "asymmetric", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Resize", (Layer* (*)(Layer*, vector<int>, bool, string, float, string, string)) &eddl::Resize, "Resize the input image", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("new_shape"), pybind11::arg("reshape") = true, pybind11::arg("da_mode") = "constant", pybind11::arg("constant") = 0.0f, pybind11::arg("coordinate_transformation_mode") = "asymmetric", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Reshape", (class Layer* (*)(class Layer*, const vector<int>&, string)) &eddl::Reshape, "C++: eddl::Reshape(class Layer*, const vector<int>&, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("shape"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Flatten", (class Layer* (*)(class Layer*, string)) &eddl::Flatten, "C++: eddl::Flatten(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Squeeze", (Layer* (*)(Layer*, int, string)) &eddl::Squeeze, "C++: eddl::Squeeze(class Layer*, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("axis") = -1, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Unsqueeze", (Layer* (*)(Layer*, int, string)) &eddl::Unsqueeze, "C++: eddl::Unsqueeze(class Layer*, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("axis") = 0, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Transpose", (class Layer* (*)(class Layer*, string)) &eddl::Transpose, "C++: eddl::Transpose(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("ConstOfTensor", (Layer* (*)(Tensor*, string)) &eddl::ConstOfTensor, "Repeat tensor for each batch", pybind11::return_value_policy::reference, pybind11::arg("t"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Where", (Layer* (*)(Layer*, Layer*, Layer*, string)) &eddl::Where, "Choose elements from layers depending on a condition", pybind11::return_value_policy::reference, pybind11::arg("parent1"), pybind11::arg("parent2"), pybind11::arg("condition"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- transformations ---
    m.def("Shift", (class Layer* (*)(class Layer*, vector<int>, string, float, string)) &eddl::Shift, "C++: eddl::Shift(class Layer*, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("shift"), pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Rotate", (class Layer* (*)(class Layer*, float, vector<int>, string, float, string)) &eddl::Rotate, "C++: eddl::Rotate(class Layer*, float, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("angle"), pybind11::arg("offset_center") = vector<int>{0, 0}, pybind11::arg("da_mode") = "original", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Scale", (class Layer* (*)(class Layer*, vector<int>, bool, string, float, string, string)) &eddl::Scale, "Resize the input image", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("new_shape"), pybind11::arg("reshape") = true, pybind11::arg("da_mode") = "constant", pybind11::arg("constant") = 0.0f, pybind11::arg("coordinate_transformation_mode") = "asymmetric", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Flip", (class Layer* (*)(class Layer*, int, string)) &eddl::Flip, "C++: eddl::Flip(class Layer*, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("axis") = 0, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("HorizontalFlip", (class Layer* (*)(class Layer*, string)) &eddl::HorizontalFlip, "C++: eddl::HorizontalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Pad", (Layer* (*)(Layer*, vector<int>, float, string)) &eddl::Pad, "Pad image on all sides", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("padding"), pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("VerticalFlip", (class Layer* (*)(class Layer*, string)) &eddl::VerticalFlip, "C++: eddl::VerticalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Crop", (class Layer* (*)(class Layer*, vector<int>, vector<int>, bool, float, string)) &eddl::Crop, "C++: eddl::Crop(class Layer*, vector<int>, vector<int>, bool, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("from_coords"), pybind11::arg("to_coords"), pybind11::arg("reshape") = true, pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("CenteredCrop", (class Layer* (*)(class Layer*, vector<int>, bool, float, string)) &eddl::CenteredCrop, "C++: eddl::CenteredCrop(class Layer*, vector<int>, bool, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("size"), pybind11::arg("reshape") = true, pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("CropScale", (class Layer* (*)(class Layer*, vector<int>, vector<int>, string, float, string)) &eddl::CropScale, "C++: eddl::CropScale(class Layer*, vector<int>, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("from_coords"), pybind11::arg("to_coords"), pybind11::arg("da_mode") = "constant", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Cutout", (class Layer* (*)(class Layer*, vector<int>, vector<int>, float, string)) &eddl::Cutout, "C++: eddl::Cutout(class Layer*, vector<int>, vector<int>, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("from_coords"), pybind11::arg("to_coords"), pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- data augmentation ---
    m.def("RandomShift", (class Layer* (*)(class Layer*, vector<float>, vector<float>, string, float, string)) &eddl::RandomShift, "C++: eddl::RandomShift(class Layer*, vector<float>, vector<float>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor_x"), pybind11::arg("factor_y"), pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomRotation", (class Layer* (*)(class Layer*, vector<float>, vector<int>, string, float, string)) &eddl::RandomRotation, "C++: eddl::RandomRotation(class Layer*, vector<float>, vector<int>, string, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor"), pybind11::arg("offset_center") = vector<int>{0, 0}, pybind11::arg("da_mode") = "original", pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomScale", (class Layer* (*)(class Layer*, vector<float>, string, float, string, string)) &eddl::RandomScale, "Resize the input image randomly", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor"), pybind11::arg("da_mode") = "nearest", pybind11::arg("constant") = 0.0f, pybind11::arg("coordinate_transformation_mode") = "asymmetric", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomFlip", (class Layer* (*)(class Layer*, int, string)) &eddl::RandomFlip, "C++: eddl::RandomFlip(class Layer*, int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("axis"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomHorizontalFlip", (class Layer* (*)(class Layer*, string)) &eddl::RandomHorizontalFlip, "C++: eddl::RandomHorizontalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomVerticalFlip", (class Layer* (*)(class Layer*, string)) &eddl::RandomVerticalFlip, "C++: eddl::RandomVerticalFlip(class Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomCrop", (class Layer* (*)(class Layer*, vector<int>, string)) &eddl::RandomCrop, "C++: eddl::RandomCrop(class Layer*, vector<int>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("new_shape"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomCropScale", (class Layer* (*)(class Layer*, vector<float>, string, string)) &eddl::RandomCropScale, "C++: eddl::RandomCropScale(class Layer*, vector<float>, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor"), pybind11::arg("da_mode") = "nearest", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("RandomCutout", (class Layer* (*)(class Layer*, vector<float>, vector<float>, float, string)) &eddl::RandomCutout, "C++: eddl::RandomCutout(class Layer*, vector<float>, vector<float>, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("factor_x"), pybind11::arg("factor_y"), pybind11::arg("constant") = 0.0f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- losses ---
    m.def("getLoss", (class Loss* (*)(string)) &eddl::getLoss, "C++: eddl::getLoss(string) --> class Loss*", pybind11::return_value_policy::reference, pybind11::arg("type"));
    m.def("newloss", (class NetLoss* (*)(const std::function<Layer*(vector<Layer*>)>&, vector<Layer*>, string)) &eddl::newloss, "C++: eddl::newloss(const std::function<Layer*(vector<Layer*>)>&, vector<Layer*>, string) --> class NetLoss*", pybind11::return_value_policy::reference);
    m.def("newloss", (class NetLoss* (*)(const std::function<Layer*(Layer*)>&, Layer*, string)) &eddl::newloss, "C++: eddl::newloss(const std::function<Layer*(Layer*)>&, Layer*, string) --> class NetLoss*", pybind11::return_value_policy::reference);

    // --- metrics ---
    m.def("getMetric", (class Metric* (*)(string)) &eddl::getMetric, "C++: eddl::getMetric(string) --> class Metric*", pybind11::return_value_policy::reference, pybind11::arg("type"));

    // --- merge layers ---
    m.def("Add", (class Layer* (*)(const vector<Layer*>&, string)) &eddl::Add, "C++: eddl::Add(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Add", (class Layer * (*)(class Layer *, class Layer *)) &eddl::Add, "Layer that computes the sum of two layers.\n\n  \n  Layer\n  \n\n  Layer\n  \n\n     The result after computing the sum between layers l1 and l2\n\nC++: eddl::Add(class Layer *, class Layer *) --> class Layer *", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Add", (class Layer * (*)(class Layer *, float)) &eddl::Add, "Layer that computes the sum of a float number and a layer.\n\n  \n  Parent layer\n  \n\n  Number\n  \n\n     Parent layer l1 after computing his sum with k\n\nC++: eddl::Add(class Layer *, float) --> class Layer *", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Add", (class Layer * (*)(float, class Layer *)) &eddl::Add, "C++: eddl::Add(float, class Layer *) --> class Layer *", pybind11::return_value_policy::reference, pybind11::arg("k"), pybind11::arg("l1"), pybind11::keep_alive<0, 2>());
    m.def("Average", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Average, "C++: eddl::Average(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Concat", (class Layer* (*)(const vector<Layer*>, unsigned int, string)) &eddl::Concat, "C++: eddl::Concat(const vector<Layer*>, unsigned int, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("axis") = 0, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("MatMul", (class Layer* (*)(const vector<Layer*>, string)) &eddl::MatMul, "C++: eddl::MatMul(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Maximum", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Maximum, "C++: eddl::Maximum(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Minimum", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Minimum, "C++: eddl::Minimum(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Subtract", (class Layer* (*)(const vector<Layer*>, string)) &eddl::Subtract, "C++: eddl::Subtract(const vector<Layer*>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("layers"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- noise layers ---
    m.def("GaussianNoise", (class Layer* (*)(class Layer*, float, string)) &eddl::GaussianNoise, "C++: eddl::GaussianNoise(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("stddev"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- normalization layers ---
    m.def("BatchNormalization", (class Layer* (*)(class Layer*, bool, float, float, string)) &eddl::BatchNormalization, "C++: eddl::BatchNormalization(class Layer*, bool, float, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("affine"), pybind11::arg("momentum") = 0.99f, pybind11::arg("epsilon") = 0.001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("LayerNormalization", (class Layer* (*)(class Layer*, bool, float, string)) &eddl::LayerNormalization, "C++: eddl::LayerNormalization(class Layer*, bool, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("affine"), pybind11::arg("epsilon") = 0.00001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GroupNormalization", (class Layer* (*)(class Layer*, int, float, bool, string)) &eddl::GroupNormalization, "C++: eddl::GroupNormalization(class Layer*, int, float, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("groups"), pybind11::arg("epsilon") = 0.001f, pybind11::arg("affine") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Norm", (class Layer* (*)(class Layer*, float, string)) &eddl::Norm, "C++: eddl::Norm(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("epsilon") = 0.001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("NormMax", (class Layer* (*)(class Layer*, float, string)) &eddl::NormMax, "C++: eddl::NormMax(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("epsilon") = 0.001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("NormMinMax", (class Layer* (*)(class Layer*, float, string)) &eddl::NormMinMax, "C++: eddl::NormMinMax(class Layer*, float, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("epsilon") = 0.001f, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Dropout", (class Layer* (*)(class Layer*, float, bool, string)) &eddl::Dropout, "C++: eddl::Dropout(class Layer*, float, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("rate"), pybind11::arg("iw") = true, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- operator layers ---
    m.def("Abs", (class Layer* (*)(class Layer*)) &eddl::Abs, "C++: eddl::Abs(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Sub", (class Layer* (*)(class Layer*, class Layer*)) &eddl::Sub, "C++: eddl::Sub(class Layer*, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Sub", (class Layer* (*)(class Layer*, float)) &eddl::Sub, "C++: eddl::Sub(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Sub", (class Layer* (*)(float, class Layer*)) &eddl::Sub, "C++: eddl::Sub(float, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("k"), pybind11::arg("l1"), pybind11::keep_alive<0, 2>());
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
    m.def("Pow", (class Layer* (*)(class Layer*, float)) &eddl::Pow, "C++: eddl::Pow(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Sqrt", (class Layer* (*)(class Layer*)) &eddl::Sqrt, "C++: eddl::Sqrt(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::keep_alive<0, 1>());
    m.def("Sum", (class Layer* (*)(class Layer*, class Layer*)) &eddl::Sum, "C++: eddl::Sum(class Layer*, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("Sum", (class Layer* (*)(class Layer*, float)) &eddl::Sum, "C++: eddl::Sum(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l1"), pybind11::arg("k"), pybind11::keep_alive<0, 1>());
    m.def("Sum", (class Layer* (*)(float, class Layer*)) &eddl::Sum, "C++: eddl::Sum(float, class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("k"), pybind11::arg("l1"), pybind11::keep_alive<0, 2>());
    m.def("Select", (class Layer* (*)(class Layer*, vector<string>, string)) &eddl::Select, "C++: eddl::Select(class Layer*, vector<string>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("indices"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Slice", (class Layer* (*)(class Layer*, vector<string>, string)) &eddl::Slice, "Alias for Select", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("indices"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Expand", (Layer* (*)(Layer*, int, string)) &eddl::Expand, "Expand singleton dimensions", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("size"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Split", (vector<Layer*> (*)(Layer*, vector<int>, int, bool, string)) &eddl::Split, "Split layer into list of tensor layers", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("indexes"), pybind11::arg("axis") = -1, pybind11::arg("merge_sublayers") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("Permute", (class Layer* (*)(class Layer*, vector<int>, string)) &eddl::Permute, "C++: eddl::Permute(class Layer*, vector<int>, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("dims"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- reduction layers ---
    m.def("ReduceMean", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceMean, "C++: eddl::ReduceMean(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis"), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceVar", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceVar, "C++: eddl::ReduceVar(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis"), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceSum", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceSum, "C++: eddl::ReduceSum(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis"), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceMax", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceMax, "C++: eddl::ReduceMax(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis"), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceMin", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceMin, "C++: eddl::ReduceMin(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis"), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());
    m.def("ReduceArgMax", (class Layer* (*)(class Layer*, vector<int>, bool)) &eddl::ReduceArgMax, "C++: eddl::ReduceArgMax(class Layer*, vector<int>, bool) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("axis"), pybind11::arg("keepdims") = false, pybind11::keep_alive<0, 1>());

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
    m.def("AvgPool", (Layer* (*)(Layer*, const vector<int>&, const vector<int> &, string, string)) &eddl::AvgPool, "Alias for AveragePool", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2}, pybind11::arg("strides") = vector<int>{2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("AveragePool1D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::AveragePool1D, "1D average pooling", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2}, pybind11::arg("strides") = vector<int>{2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("AvgPool1D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::AvgPool1D, "Alias for AveragePool1D", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2}, pybind11::arg("strides") = vector<int>{2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("AveragePool2D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::AveragePool2D, "2D average pooling", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2}, pybind11::arg("strides") = vector<int>{2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("AvgPool2D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::AvgPool2D, "Alias for AveragePool2D", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2}, pybind11::arg("strides") = vector<int>{2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("AveragePool3D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::AveragePool3D, "3D average pooling", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2, 2}, pybind11::arg("strides") = vector<int>{2, 2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("AvgPool3D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::AvgPool3D, "Alias for AveragePool3D", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2, 2}, pybind11::arg("strides") = vector<int>{2, 2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalMaxPool", (class Layer* (*)(Layer*, string)) &eddl::GlobalMaxPool, "C++: eddl::GlobalMaxPool(Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalMaxPool1D", (class Layer* (*)(Layer*, string)) &eddl::GlobalMaxPool1D, "GlobalMaxPooling1D operation.", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalMaxPool2D", (class Layer* (*)(Layer*, string)) &eddl::GlobalMaxPool2D, "GlobalMaxPooling2D operation.", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalMaxPool3D", (class Layer* (*)(Layer*, string)) &eddl::GlobalMaxPool3D, "GlobalMaxPooling3D operation.", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAveragePool", (class Layer* (*)(Layer*, string)) &eddl::GlobalAveragePool, "C++: eddl::GlobalAveragePool(Layer*, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAvgPool", (class Layer* (*)(Layer*, string)) &eddl::GlobalAvgPool, "Alias for GlobalAveragePool", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAveragePool1D", (Layer* (*)(Layer*, string)) &eddl::GlobalAveragePool1D, "C++: eddl::GlobalAveragePool1D(Layer*, string) --> Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAvgPool1D", (Layer* (*)(Layer*, string)) &eddl::GlobalAvgPool1D, "Alias for GlobalAveragePool1D", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAveragePool2D", (Layer* (*)(Layer*, string)) &eddl::GlobalAveragePool2D, "C++: eddl::GlobalAveragePool2D(Layer*, string) --> Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAvgPool2D", (Layer* (*)(Layer*, string)) &eddl::GlobalAvgPool2D, "Alias for GlobalAveragePool2D", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAveragePool3D", (Layer* (*)(Layer*, string)) &eddl::GlobalAveragePool3D, "C++: eddl::GlobalAveragePool3D(Layer*, string) --> Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GlobalAvgPool3D", (Layer* (*)(Layer*, string)) &eddl::GlobalAvgPool3D, "Alias for GlobalAveragePool3D", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("MaxPool", (class Layer* (*)(class Layer*, const vector<int>&, const vector<int> &, string, string)) &eddl::MaxPool, "C++: eddl::MaxPool(class Layer*, const vector<int>&, const vector<int> &, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2}, pybind11::arg("strides") = vector<int>{2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("MaxPool1D", (class Layer* (*)(class Layer*, vector<int>, vector<int> &, string, string)) &eddl::MaxPool1D, "C++: eddl::MaxPool1D(class Layer*, vector<int>, vector<int>, string, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2}, pybind11::arg("strides") = vector<int>{2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("MaxPool2D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::MaxPool2D, "MaxPooling2D operation.", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2}, pybind11::arg("strides") = vector<int>{2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("MaxPool3D", (Layer* (*)(Layer*, vector<int>, vector<int>, string, string)) &eddl::MaxPool3D, "MaxPooling3D operation.", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("pool_size") = vector<int>{2, 2, 2}, pybind11::arg("strides") = vector<int>{2, 2, 2}, pybind11::arg("padding") = "none", pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- recurrent layers ---
    m.def("RNN", (class Layer* (*)(class Layer*, int, string, bool, bool, string)) &eddl::RNN, "C++: eddl::RNN(class Layer*, int, string, bool, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("units"), pybind11::arg("activation") = "tanh", pybind11::arg("use_bias") = true, pybind11::arg("bidirectional") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("LSTM", (class Layer* (*)(class Layer*, int, bool, bool, string)) &eddl::LSTM, "C++: eddl::LSTM(class Layer*, int, bool, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("units"), pybind11::arg("mask_zeros") = false, pybind11::arg("bidirectional") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("LSTM", (class Layer* (*)(vector<Layer*>, int, bool, bool, string)) &eddl::LSTM, "C++: eddl::LSTM(vector<Layer*>, int, bool, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("units"), pybind11::arg("mask_zeros") = false, pybind11::arg("bidirectional") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("States", (Layer* (*)(const vector<int>&, string)) &eddl::States, "C++: eddl::States(const vector<int>&, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("shape"), pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GRU", (Layer* (*)(Layer*, int, bool, bool, string)) &eddl::GRU, "C++: eddl::GRU(Layer*, int, bool, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("units"), pybind11::arg("mask_zeros") = false, pybind11::arg("bidirectional") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GRU", (Layer* (*)(vector<Layer*>, int, bool, bool, string)) &eddl::GRU, "C++: eddl::GRU(vector<Layer*>, int, bool, bool, string) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("units"), pybind11::arg("mask_zeros") = false, pybind11::arg("bidirectional") = false, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());
    m.def("GetStates", (Layer* (*)(Layer*)) &eddl::GetStates, "C++: eddl::GetStates(Layer*) --> Layer*", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::keep_alive<0, 1>());
    m.def("setDecoder", (void (*)(Layer*)) &eddl::setDecoder, "C++: eddl::setDecoder(Layer*) --> void", pybind11::arg("l"));

    // --- initializers ---
    m.def("GlorotNormal", (class Layer* (*)(class Layer*, int)) &eddl::GlorotNormal, "C++: eddl::GlorotNormal(class Layer*, int) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("GlorotUniform", (class Layer* (*)(class Layer*, int)) &eddl::GlorotUniform, "C++: eddl::GlorotUniform(class Layer*, int) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("HeNormal", (class Layer* (*)(class Layer*, int)) &eddl::HeNormal, "C++: eddl::HeNormal(class Layer*, int) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("HeUniform", (class Layer* (*)(class Layer*, int)) &eddl::HeUniform, "C++: eddl::HeUniform(class Layer*, int) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("RandomNormal", (class Layer* (*)(class Layer*, float, float, float)) &eddl::RandomNormal, "C++: eddl::RandomNormal(class Layer*, float, float, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("m") = 0.0, pybind11::arg("s") = 0.1, pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("RandomUniform", (class Layer* (*)(class Layer*, float, float, float)) &eddl::RandomUniform, "C++: eddl::RandomUniform(class Layer*, float, float, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("min") = 0.0, pybind11::arg("max") = 0.1, pybind11::arg("seed") = 1234, pybind11::keep_alive<0, 1>());
    m.def("Constant", (class Layer* (*)(class Layer*, float)) &eddl::Constant, "C++: eddl::Constant(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("v") = 0.1, pybind11::keep_alive<0, 1>());

    // --- regularizers ---
    m.def("L2", (class Layer* (*)(class Layer*, float)) &eddl::L2, "C++: eddl::L2(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>());
    m.def("L1", (class Layer* (*)(class Layer*, float)) &eddl::L1, "C++: eddl::L1(class Layer*, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("l1"), pybind11::keep_alive<0, 1>());
    m.def("L1L2", (class Layer* (*)(class Layer*, float, float)) &eddl::L1L2, "C++: eddl::L1L2(class Layer*, float, float) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"), pybind11::arg("l1"), pybind11::arg("l2"), pybind11::keep_alive<0, 1>());

    // --- fused layers ---
    m.def("Conv2dActivation", (Layer* (*)(Layer*, string, int, const vector<int>&, const vector<int>&, string, bool, int, const vector<int>&, string)) &eddl::Conv2dActivation, "Convolution + Activation layer", pybind11::return_value_policy::reference, pybind11::arg("parent"), pybind11::arg("act"), pybind11::arg("filters"), pybind11::arg("kernel_size"), pybind11::arg("strides") = vector<int>{1, 1}, pybind11::arg("padding") = "same", pybind11::arg("use_bias") = true, pybind11::arg("groups") = 1, pybind11::arg("dilation_rate") = vector<int>{1, 1}, pybind11::arg("name") = "", pybind11::keep_alive<0, 1>());

    // --- computing services ---
    m.def("CS_CPU", (class CompServ* (*)(int, string)) &eddl::CS_CPU, "C++: eddl::CS_CPU(int, string) --> class CompServ*", pybind11::return_value_policy::reference, pybind11::arg("th"), pybind11::arg("mem"));
    m.def("CS_GPU", (class CompServ* (*)(const vector<int>)) &eddl::CS_GPU, "C++: eddl::CS_GPU(const vector<int>) --> class CompServ*", pybind11::return_value_policy::reference, pybind11::arg("g"));
    m.def("CS_GPU", (class CompServ* (*)(const vector<int>, string)) &eddl::CS_GPU, "C++: eddl::CS_GPU(const vector<int>, string) --> class CompServ*", pybind11::return_value_policy::reference, pybind11::arg("g"), pybind11::arg("mem"));
    m.def("CS_GPU", (class CompServ* (*)(const vector<int>, int)) &eddl::CS_GPU, "C++: eddl::CS_GPU(const vector<int>, int) --> class CompServ*", pybind11::return_value_policy::reference, pybind11::arg("g"), pybind11::arg("lsb"));
    m.def("CS_GPU", (class CompServ* (*)(const vector<int>, int, string)) &eddl::CS_GPU, "C++: eddl::CS_GPU(const vector<int>, int, string) --> class CompServ*", pybind11::return_value_policy::reference, pybind11::arg("g"), pybind11::arg("lsb"), pybind11::arg("mem"));
    m.def("CS_FPGA", (class CompServ* (*)(const vector<int>&, int)) &eddl::CS_FPGA, "C++: eddl::CS_FPGA(const vector<int>&, int) --> class CompServ*", pybind11::return_value_policy::reference, pybind11::arg("f"), pybind11::arg("lsb") = 1);
    m.def("CS_COMPSS", (class CompServ* (*)(string)) &eddl::CS_COMPSS, "C++: eddl::CS_COMPSS(string) --> class CompServ*", pybind11::return_value_policy::reference, pybind11::arg("filename"));
    m.def("exist", (bool (*)(string)) &eddl::exist, "C++: eddl::exist(string) --> bool", pybind11::arg("name"));

    // --- fine-grained methods ---
    m.def("random_indices", (vector<int> (*)(int, int)) &eddl::random_indices, "C++: eddl::random_indices(int, int) --> vector<int>", pybind11::arg("batch_size"), pybind11::arg("num_samples"));
    m.def("train_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>)) &eddl::train_batch, "C++: eddl::train_batch(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("indices"));
    m.def("train_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>)) &eddl::train_batch, "C++: eddl::train_batch(class Net*, vector<Tensor*>, vector<Tensor*>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"));
    m.def("eval_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>)) &eddl::eval_batch, "C++: eddl::eval_batch(class Net*, vector<Tensor*>, vector<Tensor*>, vector<int>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("indices"));
    m.def("eval_batch", (void (*)(class Net*, vector<Tensor*>, vector<Tensor*>)) &eddl::eval_batch, "C++: eddl::eval_batch(class Net*, vector<Tensor*>, vector<Tensor*>) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("net"), pybind11::arg("in"), pybind11::arg("out"));
    m.def("next_batch", (void (*)(vector<Tensor*>, vector<Tensor*>)) &eddl::next_batch, "C++: eddl::next_batch(vector<Tensor*>, vector<Tensor*>) --> void", pybind11::arg("in"), pybind11::arg("out"));
    m.def("forward", (vector<Layer*> (*)(class Net*, vector<Layer*>)) &eddl::forward, "C++: eddl::forward(class Net*, vector<Layer*>) --> vector<Layer*>", pybind11::return_value_policy::reference, pybind11::arg("m"), pybind11::arg("in"));
    m.def("forward", (vector<Layer*> (*)(class Net*, vector<Tensor*>)) &eddl::forward, "C++: eddl::forward(class Net*, vector<Tensor*>) --> vector<Layer*>", pybind11::return_value_policy::reference, pybind11::arg("m"), pybind11::arg("in"));
    m.def("forward", (vector<Layer*> (*)(class Net*)) &eddl::forward, "C++: eddl::forward(class Net*) --> vector<Layer*>", pybind11::return_value_policy::reference, pybind11::arg("m"));
    m.def("forward", (vector<Layer*> (*)(class Net*, int)) &eddl::forward, "C++: eddl::forward(class Net*, int) --> vector<Layer*>", pybind11::return_value_policy::reference, pybind11::arg("m"), pybind11::arg("b"));
    m.def("detach", (class Layer* (*)(class Layer*)) &eddl::detach, "C++: eddl::detach(class Layer*) --> class Layer*", pybind11::return_value_policy::reference, pybind11::arg("l"));
    m.def("detach", (class vector<Layer*> (*)(class vector<Layer*>)) &eddl::detach, "C++: eddl::detach(class vector<Layer*>) --> class vector<Layer*>", pybind11::return_value_policy::reference, pybind11::arg("l"));
    m.def("backward", (void (*)(class Net*, vector<Tensor*>)) &eddl::backward, "C++: eddl::backward(class Net*, vector<Tensor*>) --> void", pybind11::arg("m"), pybind11::arg("target"));
    m.def("optimize", (void (*)(vector<NetLoss*>)) &eddl::optimize, "C++: eddl::optimize(vector<NetLoss*>) --> void", pybind11::arg("l"));
    m.def("getOut", (vector<Layer*> (*)(class Net*)) &eddl::getOut, "C++: eddl::getOut(class Net*) --> vector<Layer*>", pybind11::return_value_policy::reference, pybind11::arg("net"));
    m.def("get_losses", (vector<float> (*)(class Net*)) &eddl::get_losses, "Get model losses", pybind11::arg("m"));
    m.def("get_metrics", (vector<float> (*)(class Net*)) &eddl::get_metrics, "Get model metrics", pybind11::arg("m"));

    // --- manage tensors inside layers ---
    m.def("getParams", (vector<Tensor*> (*)(class Layer *)) &eddl::getParams, "C++: eddl::getParams(class Layer *) --> vector<Tensor*>", pybind11::arg("l1"));
    m.def("getGradients", (vector<Tensor*> (*)(class Layer *)) &eddl::getGradients, "C++: eddl::getGradients(class Layer *) --> vector<Tensor*>", pybind11::arg("l1"));
    m.def("getStates", (vector<Tensor*> (*)(class Layer *)) &eddl::getStates, "C++: eddl::getStates(class Layer *) --> vector<Tensor*>", pybind11::arg("l1"));

    // --- model methods ---
    m.def("Model", (class Net* (*)(vector<Layer*>, vector<Layer*>)) &eddl::Model, "C++: eddl::Model(vector<Layer*>, vector<Layer*>) --> class Net*", pybind11::arg("in"), pybind11::arg("out"), pybind11::keep_alive<0, 1>(), pybind11::keep_alive<0, 2>());
    m.def("setName", (void (*)(class Net*, string)) &eddl::setName, "C++: eddl::setName(class Net*, string) --> void", pybind11::arg("m"), pybind11::arg("name"));
    m.def("getLayer", (Layer* (*)(class Net*, string)) &eddl::getLayer, "C++: eddl::getLayer(class Net*, string) --> Layer*", pybind11::return_value_policy::reference, pybind11::arg("net"), pybind11::arg("l"));
    m.def("removeLayer", (void (*)(class Net*, string)) &eddl::removeLayer, "C++: eddl::removeLayer(class Net*, string) --> void", pybind11::arg("net"), pybind11::arg("l"));
    m.def("initializeLayer", (void (*)(class Net*, string)) &eddl::initializeLayer, "C++: eddl::initializeLayer(class Net*, string) --> void", pybind11::arg("net"), pybind11::arg("l"));
    m.def("setTrainable", (void (*)(class Net*, string, bool)) &eddl::setTrainable, "C++: eddl::setTrainable(class Net*, string, bool) --> void", pybind11::arg("net"), pybind11::arg("lanme"), pybind11::arg("val"));
    m.def("get_parameters", (vector<vector<Tensor*>> (*)(class Net*, bool)) &eddl::get_parameters, "C++: eddl::get_parameters(class Net*, bool) --> vector<vector<Tensor*>>", pybind11::return_value_policy::reference, pybind11::arg("net"), pybind11::arg("deepcopy")=false);
    m.def("set_parameters", (void (*)(class Net*, const vector<vector<Tensor*>>&)) &eddl::set_parameters, "C++: eddl::set_parameters(class Net*, const vector<vector<Tensor*>>&) --> void", pybind11::arg("net"), pybind11::arg("params"));
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
    m.def("evaluate", (void (*)(class Net*, const vector<Tensor*>&, const vector<Tensor*>&, int)) &eddl::evaluate, "C++: eddl::evaluate(class Net*, const vector<Tensor*>&, const vector<Tensor*>&, int) --> void", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("m"), pybind11::arg("in"), pybind11::arg("out"), pybind11::arg("bs")=-1);
    m.def("predict", (vector<Tensor*> (*)(class Net*, const vector<Tensor*>&)) &eddl::predict, "C++: eddl::predict(class Net*, const vector<Tensor*>&) --> vector<Tensor*>", pybind11::call_guard<pybind11::gil_scoped_release>(), pybind11::arg("m"), pybind11::arg("in"));
    // not implemented upstream:
    //   Affine
    //   ColorJitter
    //   Grayscale
    //   Normalize
    //   RandomAffine
    //   RandomGrayscale

    // --- get models ---
    m.def("download_model", (void (*)(string, string)) &eddl::download_model, "C++: eddl::download_model(string, string) --> void", pybind11::arg("name"), pybind11::arg("link"));
    m.def("download_vgg16", (Net* (*)(bool, vector<int>)) &eddl::download_vgg16, "C++: eddl::download_vgg16(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_vgg16_bn", (Net* (*)(bool, vector<int>)) &eddl::download_vgg16_bn, "C++: eddl::download_vgg16_bn(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_vgg19", (Net* (*)(bool, vector<int>)) &eddl::download_vgg19, "C++: eddl::download_vgg19(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_vgg19_bn", (Net* (*)(bool, vector<int>)) &eddl::download_vgg19_bn, "C++: eddl::download_vgg19_bn(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_resnet18", (Net* (*)(bool, vector<int>)) &eddl::download_resnet18, "C++: eddl::download_resnet18(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_resnet34", (Net* (*)(bool, vector<int>)) &eddl::download_resnet34, "C++: eddl::download_resnet34(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_resnet50", (Net* (*)(bool, vector<int>)) &eddl::download_resnet50, "C++: eddl::download_resnet50(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_resnet101", (Net* (*)(bool, vector<int>)) &eddl::download_resnet101, "C++: eddl::download_resnet101(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_resnet152", (Net* (*)(bool, vector<int>)) &eddl::download_resnet152, "C++: eddl::download_resnet152(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});
    m.def("download_densenet121", (Net* (*)(bool, vector<int>)) &eddl::download_densenet121, "C++: eddl::download_densenet121(string, string) --> Net*", pybind11::arg("top") = true, pybind11::arg("input_shape") = vector<int>{});


#ifdef EDDL_WITH_PROTOBUF

    pybind11::enum_<LOG_LEVEL>(m, "LOG_LEVEL", "ONNX log level")
	.value("TRACE", LOG_LEVEL::TRACE)
	.value("DEBUG", LOG_LEVEL::DEBUG)
	.value("INFO", LOG_LEVEL::INFO)
	.value("WARN", LOG_LEVEL::WARN)
	.value("ERROR", LOG_LEVEL::ERROR)
	.value("NO_LOGS", LOG_LEVEL::NO_LOGS);

    // --- serialization ---
    m.def("save_net_to_onnx_file", (void (*)(class Net*, string)) &save_net_to_onnx_file, "C++: eddl::save_net_to_onnx_file(class Net *, string) --> void", pybind11::arg("net"), pybind11::arg("path"));
    m.def("import_net_from_onnx_file", (class Net* (*)(string, int, LOG_LEVEL)) &import_net_from_onnx_file, "Imports ONNX Net from file", pybind11::arg("path"), pybind11::arg("mem") = 0, pybind11::arg("log_level") = LOG_LEVEL::INFO);
    m.def("import_net_from_onnx_file", (class Net* (*)(string, vector<int>, int, LOG_LEVEL)) &import_net_from_onnx_file, "Imports ONNX Net from file and changes its input shape", pybind11::arg("path"), pybind11::arg("input_shape"), pybind11::arg("mem") = 0, pybind11::arg("log_level") = LOG_LEVEL::INFO);
    m.def("serialize_net_to_onnx_string", [](Net* net, bool gradients) -> pybind11::bytes {
      string* s = serialize_net_to_onnx_string(net, gradients);
      return pybind11::bytes(*s);
    }, pybind11::arg("net"), pybind11::arg("gradients"));
    m.def("import_net_from_onnx_string", [](pybind11::bytes model_string, int mem = 0) -> Net* {
      string s = string(model_string);
      return import_net_from_onnx_string(&s, mem);
    }, pybind11::arg("model_string"), pybind11::arg("mem") = 0);
#endif

    // --- constants ---
    m.attr("DEV_CPU") = pybind11::int_(DEV_CPU);
    m.attr("DEV_GPU") = pybind11::int_(DEV_GPU);
    m.attr("DEV_FPGA") = pybind11::int_(DEV_FPGA);
    m.attr("MAX_THREADS") = pybind11::int_(MAX_THREADS);
}
