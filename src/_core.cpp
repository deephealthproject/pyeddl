// File: eddl/tensor/tensor.cpp
#include <eddl/tensor/tensor.h>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <sstream> // __str__
#include <string>
#include <tensor_addons.hpp>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_tensor_tensor(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tensor file:eddl/tensor/tensor.h line:58
		pybind11::class_<Tensor, std::shared_ptr<Tensor>> cl(M(""), "Tensor", pybind11::buffer_protocol());
		cl.def( pybind11::init( [](){ return new Tensor(); } ) );
		cl.def( pybind11::init( [](Tensor const &o){ return new Tensor(o); } ) );
		cl.def_readwrite("device", &Tensor::device);
		cl.def_readwrite("ndim", &Tensor::ndim);
		cl.def_readwrite("size", &Tensor::size);
		cl.def_readwrite("shape", &Tensor::shape);
		cl.def_readwrite("stride", &Tensor::stride);
		cl.def_readwrite("gpu_device", &Tensor::gpu_device);
		cl.def("toCPU", [](Tensor &o) -> void { return o.toCPU(); }, "");
		cl.def("toCPU", (void (Tensor::*)(int)) &Tensor::toCPU, "C++: Tensor::toCPU(int) --> void", pybind11::arg("dev"));
		cl.def("toGPU", [](Tensor &o) -> void { return o.toGPU(); }, "");
		cl.def("toGPU", (void (Tensor::*)(int)) &Tensor::toGPU, "C++: Tensor::toGPU(int) --> void", pybind11::arg("dev"));
		cl.def("clone", (class Tensor * (Tensor::*)()) &Tensor::clone, "C++: Tensor::clone() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("resize", (void (Tensor::*)(int, float *)) &Tensor::resize, "C++: Tensor::resize(int, float *) --> void", pybind11::arg("b"), pybind11::arg("fptr"));
		cl.def("resize", (void (Tensor::*)(int)) &Tensor::resize, "C++: Tensor::resize(int) --> void", pybind11::arg("b"));
		cl.def("resize", (void (Tensor::*)(int, class Tensor *)) &Tensor::resize, "C++: Tensor::resize(int, class Tensor *) --> void", pybind11::arg("b"), pybind11::arg("T"));
		cl.def("isCPU", (int (Tensor::*)()) &Tensor::isCPU, "C++: Tensor::isCPU() --> int");
		cl.def("isGPU", (int (Tensor::*)()) &Tensor::isGPU, "C++: Tensor::isGPU() --> int");
		cl.def("isFPGA", (int (Tensor::*)()) &Tensor::isFPGA, "C++: Tensor::isFPGA() --> int");
		cl.def("info", (void (Tensor::*)()) &Tensor::info, "C++: Tensor::info() --> void");
		cl.def("print", (void (Tensor::*)()) &Tensor::print, "C++: Tensor::print() --> void");
		cl.def("fill_", (void (Tensor::*)(float)) &Tensor::fill_, "C++: Tensor::fill_(float) --> void", pybind11::arg("v"));
		cl.def_static("arange", [](float const & a0, float const & a1) -> Tensor * { return Tensor::arange(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("arange", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::arange(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
		cl.def_static("arange", (class Tensor * (*)(float, float, float, int)) &Tensor::arange, "C++: Tensor::arange(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("range", [](float const & a0, float const & a1) -> Tensor * { return Tensor::range(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("range", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::range(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
		cl.def_static("range", (class Tensor * (*)(float, float, float, int)) &Tensor::range, "C++: Tensor::range(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("linspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::linspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("linspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::linspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("linspace", (class Tensor * (*)(float, float, int, int)) &Tensor::linspace, "C++: Tensor::linspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));
		cl.def_static("logspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::logspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::logspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2, float const & a3) -> Tensor * { return Tensor::logspace(a0, a1, a2, a3); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"));
		cl.def_static("logspace", (class Tensor * (*)(float, float, int, float, int)) &Tensor::logspace, "C++: Tensor::logspace(float, float, int, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"), pybind11::arg("dev"));
		cl.def_static("eye", [](int const & a0) -> Tensor * { return Tensor::eye(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("size"));
		cl.def_static("eye", (class Tensor * (*)(int, int)) &Tensor::eye, "C++: Tensor::eye(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("size"), pybind11::arg("dev"));
		cl.def_static("flip_random", (void (*)(class Tensor *, class Tensor *, int)) &Tensor::flip_random, "C++: Tensor::flip_random(class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"));
		cl.def_static("crop_random", (void (*)(class Tensor *, class Tensor *)) &Tensor::crop_random, "C++: Tensor::crop_random(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("interpolate", (class Tensor * (*)(float, class Tensor *, float, class Tensor *)) &Tensor::interpolate, "C++: Tensor::interpolate(float, class Tensor *, float, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("factor1"), pybind11::arg("A"), pybind11::arg("factor2"), pybind11::arg("B"));
		cl.def("abs_", (void (Tensor::*)()) &Tensor::abs_, "C++: Tensor::abs_() --> void");
		cl.def_static("abs", (class Tensor * (*)(class Tensor *)) &Tensor::abs, "C++: Tensor::abs(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("acos_", (void (Tensor::*)()) &Tensor::acos_, "C++: Tensor::acos_() --> void");
		cl.def_static("acos", (class Tensor * (*)(class Tensor *)) &Tensor::acos, "C++: Tensor::acos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("add_", (void (Tensor::*)(float)) &Tensor::add_, "C++: Tensor::add_(float) --> void", pybind11::arg("v"));
		cl.def("add_", (void (Tensor::*)(class Tensor *)) &Tensor::add_, "C++: Tensor::add_(class Tensor *) --> void", pybind11::arg("A"));
		cl.def_static("add", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::add, "C++: Tensor::add(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("add", (void (*)(float, class Tensor *, float, class Tensor *, class Tensor *, int)) &Tensor::add, "C++: Tensor::add(float, class Tensor *, float, class Tensor *, class Tensor *, int) --> void", pybind11::arg("scA"), pybind11::arg("A"), pybind11::arg("scB"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("add", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::add, "C++: Tensor::add(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("inc", (void (*)(class Tensor *, class Tensor *)) &Tensor::inc, "C++: Tensor::inc(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("asin_", (void (Tensor::*)()) &Tensor::asin_, "C++: Tensor::asin_() --> void");
		cl.def_static("asin", (class Tensor * (*)(class Tensor *)) &Tensor::asin, "C++: Tensor::asin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("atan_", (void (Tensor::*)()) &Tensor::atan_, "C++: Tensor::atan_() --> void");
		cl.def_static("atan", (class Tensor * (*)(class Tensor *)) &Tensor::atan, "C++: Tensor::atan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("ceil_", (void (Tensor::*)()) &Tensor::ceil_, "C++: Tensor::ceil_() --> void");
		cl.def_static("ceil", (class Tensor * (*)(class Tensor *)) &Tensor::ceil, "C++: Tensor::ceil(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("clamp_", (void (Tensor::*)(float, float)) &Tensor::clamp_, "C++: Tensor::clamp_(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("clamp", (class Tensor * (*)(class Tensor *, float, float)) &Tensor::clamp, "C++: Tensor::clamp(class Tensor *, float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));
		cl.def("clampmax_", (void (Tensor::*)(float)) &Tensor::clampmax_, "C++: Tensor::clampmax_(float) --> void", pybind11::arg("max"));
		cl.def_static("clampmax", (class Tensor * (*)(class Tensor *, float)) &Tensor::clampmax, "C++: Tensor::clampmax(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("max"));
		cl.def("clampmin_", (void (Tensor::*)(float)) &Tensor::clampmin_, "C++: Tensor::clampmin_(float) --> void", pybind11::arg("min"));
		cl.def_static("clampmin", (class Tensor * (*)(class Tensor *, float)) &Tensor::clampmin, "C++: Tensor::clampmin(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"));
		cl.def("cos_", (void (Tensor::*)()) &Tensor::cos_, "C++: Tensor::cos_() --> void");
		cl.def_static("cos", (class Tensor * (*)(class Tensor *)) &Tensor::cos, "C++: Tensor::cos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("cosh_", (void (Tensor::*)()) &Tensor::cosh_, "C++: Tensor::cosh_() --> void");
		cl.def_static("cosh", (class Tensor * (*)(class Tensor *)) &Tensor::cosh, "C++: Tensor::cosh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("inv_", (void (Tensor::*)()) &Tensor::inv_, "C++: Tensor::inv_() --> void");
		cl.def("div_", (void (Tensor::*)(float)) &Tensor::div_, "C++: Tensor::div_(float) --> void", pybind11::arg("v"));
		cl.def_static("div", (class Tensor * (*)(class Tensor *, float)) &Tensor::div, "C++: Tensor::div(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def_static("el_div", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_div, "C++: Tensor::el_div(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def("exp_", (void (Tensor::*)()) &Tensor::exp_, "C++: Tensor::exp_() --> void");
		cl.def_static("exp", (class Tensor * (*)(class Tensor *)) &Tensor::exp, "C++: Tensor::exp(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("floor_", (void (Tensor::*)()) &Tensor::floor_, "C++: Tensor::floor_() --> void");
		cl.def_static("floor", (class Tensor * (*)(class Tensor *)) &Tensor::floor, "C++: Tensor::floor(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("log_", (void (Tensor::*)()) &Tensor::log_, "C++: Tensor::log_() --> void");
		cl.def_static("log", (class Tensor * (*)(class Tensor *)) &Tensor::log, "C++: Tensor::log(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("log2_", (void (Tensor::*)()) &Tensor::log2_, "C++: Tensor::log2_() --> void");
		cl.def_static("log2", (class Tensor * (*)(class Tensor *)) &Tensor::log2, "C++: Tensor::log2(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("log10_", (void (Tensor::*)()) &Tensor::log10_, "C++: Tensor::log10_() --> void");
		cl.def_static("log10", (class Tensor * (*)(class Tensor *)) &Tensor::log10, "C++: Tensor::log10(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("logn_", (void (Tensor::*)(float)) &Tensor::logn_, "C++: Tensor::logn_(float) --> void", pybind11::arg("n"));
		cl.def_static("logn", (class Tensor * (*)(class Tensor *, float)) &Tensor::logn, "C++: Tensor::logn(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("n"));
		cl.def("max", (float (Tensor::*)()) &Tensor::max, "C++: Tensor::max() --> float");
		cl.def("min", (float (Tensor::*)()) &Tensor::min, "C++: Tensor::min() --> float");
		cl.def("mod_", (void (Tensor::*)(float)) &Tensor::mod_, "C++: Tensor::mod_(float) --> void", pybind11::arg("v"));
		cl.def_static("mod", (class Tensor * (*)(class Tensor *, float)) &Tensor::mod, "C++: Tensor::mod(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def("mult_", (void (Tensor::*)(float)) &Tensor::mult_, "C++: Tensor::mult_(float) --> void", pybind11::arg("v"));
		cl.def_static("mult", (class Tensor * (*)(class Tensor *, float)) &Tensor::mult, "C++: Tensor::mult(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def_static("mult2D", (void (*)(class Tensor *, int, class Tensor *, int, class Tensor *, int)) &Tensor::mult2D, "C++: Tensor::mult2D(class Tensor *, int, class Tensor *, int, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("tA"), pybind11::arg("B"), pybind11::arg("tB"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("el_mult", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_mult, "C++: Tensor::el_mult(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def("neg_", (void (Tensor::*)()) &Tensor::neg_, "C++: Tensor::neg_() --> void");
		cl.def_static("neg", (class Tensor * (*)(class Tensor *)) &Tensor::neg, "C++: Tensor::neg(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("normalize_", [](Tensor &o) -> void { return o.normalize_(); }, "");
		cl.def("normalize_", [](Tensor &o, float const & a0) -> void { return o.normalize_(a0); }, "", pybind11::arg("min"));
		cl.def("normalize_", (void (Tensor::*)(float, float)) &Tensor::normalize_, "C++: Tensor::normalize_(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def("pow_", (void (Tensor::*)(float)) &Tensor::pow_, "C++: Tensor::pow_(float) --> void", pybind11::arg("exp"));
		cl.def_static("pow", (class Tensor * (*)(class Tensor *, float)) &Tensor::pow, "C++: Tensor::pow(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("exp"));
		cl.def("powb_", (void (Tensor::*)(float)) &Tensor::powb_, "C++: Tensor::powb_(float) --> void", pybind11::arg("base"));
		cl.def_static("powb", (class Tensor * (*)(class Tensor *, float)) &Tensor::powb, "C++: Tensor::powb(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("base"));
		cl.def("reciprocal_", (void (Tensor::*)()) &Tensor::reciprocal_, "C++: Tensor::reciprocal_() --> void");
		cl.def_static("reciprocal", (class Tensor * (*)(class Tensor *)) &Tensor::reciprocal, "C++: Tensor::reciprocal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("remainder_", (void (Tensor::*)(float)) &Tensor::remainder_, "C++: Tensor::remainder_(float) --> void", pybind11::arg("v"));
		cl.def_static("remainder", (class Tensor * (*)(class Tensor *, float)) &Tensor::remainder, "C++: Tensor::remainder(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def("round_", (void (Tensor::*)()) &Tensor::round_, "C++: Tensor::round_() --> void");
		cl.def_static("round", (class Tensor * (*)(class Tensor *)) &Tensor::round, "C++: Tensor::round(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("rsqrt_", (void (Tensor::*)()) &Tensor::rsqrt_, "C++: Tensor::rsqrt_() --> void");
		cl.def_static("rsqrt", (class Tensor * (*)(class Tensor *)) &Tensor::rsqrt, "C++: Tensor::rsqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sigmoid_", (void (Tensor::*)()) &Tensor::sigmoid_, "C++: Tensor::sigmoid_() --> void");
		cl.def_static("sigmoid", (class Tensor * (*)(class Tensor *)) &Tensor::sigmoid, "C++: Tensor::sigmoid(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sign_", (void (Tensor::*)()) &Tensor::sign_, "C++: Tensor::sign_() --> void");
		cl.def_static("sign", (class Tensor * (*)(class Tensor *)) &Tensor::sign, "C++: Tensor::sign(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("sign", (void (*)(class Tensor *, class Tensor *)) &Tensor::sign, "C++: Tensor::sign(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sin_", (void (Tensor::*)()) &Tensor::sin_, "C++: Tensor::sin_() --> void");
		cl.def_static("sin", (class Tensor * (*)(class Tensor *)) &Tensor::sin, "C++: Tensor::sin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sinh_", (void (Tensor::*)()) &Tensor::sinh_, "C++: Tensor::sinh_() --> void");
		cl.def_static("sinh", (class Tensor * (*)(class Tensor *)) &Tensor::sinh, "C++: Tensor::sinh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sqr_", (void (Tensor::*)()) &Tensor::sqr_, "C++: Tensor::sqr_() --> void");
		cl.def_static("sqr", (class Tensor * (*)(class Tensor *)) &Tensor::sqr, "C++: Tensor::sqr(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sqrt_", (void (Tensor::*)()) &Tensor::sqrt_, "C++: Tensor::sqrt_() --> void");
		cl.def_static("sqrt", (class Tensor * (*)(class Tensor *)) &Tensor::sqrt, "C++: Tensor::sqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("sub_", (void (Tensor::*)(float)) &Tensor::sub_, "C++: Tensor::sub_(float) --> void", pybind11::arg("v"));
		cl.def_static("sub", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::sub, "C++: Tensor::sub(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sum", (float (Tensor::*)()) &Tensor::sum, "C++: Tensor::sum() --> float");
		cl.def_static("sum2D_rowwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_rowwise, "C++: Tensor::sum2D_rowwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("sum2D_colwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_colwise, "C++: Tensor::sum2D_colwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("tan_", (void (Tensor::*)()) &Tensor::tan_, "C++: Tensor::tan_() --> void");
		cl.def_static("tan", (class Tensor * (*)(class Tensor *)) &Tensor::tan, "C++: Tensor::tan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("tanh_", (void (Tensor::*)()) &Tensor::tanh_, "C++: Tensor::tanh_() --> void");
		cl.def_static("tanh", (class Tensor * (*)(class Tensor *)) &Tensor::tanh, "C++: Tensor::tanh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("trunc_", (void (Tensor::*)()) &Tensor::trunc_, "C++: Tensor::trunc_() --> void");
		cl.def_static("trunc", (class Tensor * (*)(class Tensor *)) &Tensor::trunc, "C++: Tensor::trunc(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("reduce_sum2D", (void (*)(class Tensor *, class Tensor *, int, int)) &Tensor::reduce_sum2D, "C++: Tensor::reduce_sum2D(class Tensor *, class Tensor *, int, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"), pybind11::arg("incB"));
		cl.def_static("reduceTosum", (void (*)(class Tensor *, class Tensor *, int)) &Tensor::reduceTosum, "C++: Tensor::reduceTosum(class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"));
		cl.def_static("eqsize", (int (*)(class Tensor *, class Tensor *)) &Tensor::eqsize, "C++: Tensor::eqsize(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equal", [](class Tensor * a0, class Tensor * a1) -> int { return Tensor::equal(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equal", (int (*)(class Tensor *, class Tensor *, float)) &Tensor::equal, "C++: Tensor::equal(class Tensor *, class Tensor *, float) --> int", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("epsilon"));
		cl.def_static("copy", (void (*)(class Tensor *, class Tensor *)) &Tensor::copy, "C++: Tensor::copy(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("fill", (void (*)(class Tensor *, int, int, class Tensor *, int, int, int)) &Tensor::fill, "C++: Tensor::fill(class Tensor *, int, int, class Tensor *, int, int, int) --> void", pybind11::arg("A"), pybind11::arg("aini"), pybind11::arg("aend"), pybind11::arg("B"), pybind11::arg("bini"), pybind11::arg("bend"), pybind11::arg("inc"));
		cl.def_static("tile", (void (*)(class Tensor *, class Tensor *)) &Tensor::tile, "C++: Tensor::tile(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("rand_uniform", (void (Tensor::*)(float)) &Tensor::rand_uniform, "C++: Tensor::rand_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_signed_uniform", (void (Tensor::*)(float)) &Tensor::rand_signed_uniform, "C++: Tensor::rand_signed_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_normal", [](Tensor &o, float const & a0, float const & a1) -> void { return o.rand_normal(a0, a1); }, "", pybind11::arg("m"), pybind11::arg("s"));
		cl.def("rand_normal", (void (Tensor::*)(float, float, bool)) &Tensor::rand_normal, "C++: Tensor::rand_normal(float, float, bool) --> void", pybind11::arg("m"), pybind11::arg("s"), pybind11::arg("fast_math"));
		cl.def("rand_binary", (void (Tensor::*)(float)) &Tensor::rand_binary, "C++: Tensor::rand_binary(float) --> void", pybind11::arg("v"));

		tensor_addons(cl);
	}
}


// File: eddl/net/compserv.cpp
#include <eddl/initializers/initializer.h>
#include <eddl/layers/layer.h>
#include <eddl/losses/loss.h>
#include <eddl/metrics/metric.h>
#include <eddl/net/compserv.h>
#include <eddl/net/netloss.h>
#include <eddl/optimizers/optim.h>
#include <eddl/regularizers/regularizer.h>
#include <eddl/tensor/tensor.h>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <sstream> // __str__
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// Layer file:eddl/layers/layer.h line:32
struct PyCallBack_Layer : public Layer {
	using Layer::Layer;

	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::info();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::resize(a0);
	}
	void reset() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "reset");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset();
	}
	void zeroGrads() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "zeroGrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::zeroGrads();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::backward();
	}
};

// Optimizer file:eddl/optimizers/optim.h line:27
struct PyCallBack_Optimizer : public Optimizer {
	using Optimizer::Optimizer;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Optimizer *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Optimizer::applygrads(a0);
	}
	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Optimizer *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::clone();
	}
};

void bind_eddl_net_compserv(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // CompServ file:eddl/net/compserv.h line:21
		pybind11::class_<CompServ, std::shared_ptr<CompServ>> cl(M(""), "CompServ", "");
		cl.def( pybind11::init( [](CompServ const &o){ return new CompServ(o); } ) );
		cl.def_readwrite("type", &CompServ::type);
		cl.def_readwrite("local_threads", &CompServ::local_threads);
		cl.def_readwrite("local_gpus", &CompServ::local_gpus);
		cl.def_readwrite("local_fpgas", &CompServ::local_fpgas);
		cl.def_readwrite("lsb", &CompServ::lsb);
	}
	{ // Layer file:eddl/layers/layer.h line:32
		pybind11::class_<Layer, std::shared_ptr<Layer>, PyCallBack_Layer> cl(M(""), "Layer", "");
		cl.def( pybind11::init( [](PyCallBack_Layer const &o){ return new PyCallBack_Layer(o); } ) );
		cl.def( pybind11::init( [](Layer const &o){ return new Layer(o); } ) );
		cl.def_readwrite("name", &Layer::name);
		cl.def_readwrite("params", &Layer::params);
		cl.def_readwrite("gradients", &Layer::gradients);
		cl.def_readwrite("parent", &Layer::parent);
		cl.def_readwrite("child", &Layer::child);
		cl.def_readwrite("mode", &Layer::mode);
		cl.def_readwrite("dev", &Layer::dev);
		cl.def_readwrite("lin", &Layer::lin);
		cl.def_readwrite("lout", &Layer::lout);
		cl.def_readwrite("delta_bp", &Layer::delta_bp);
		cl.def_readwrite("detached", &Layer::detached);
		cl.def("initialize", (void (Layer::*)()) &Layer::initialize, "C++: Layer::initialize() --> void");
		cl.def("info", (void (Layer::*)()) &Layer::info, "C++: Layer::info() --> void");
		cl.def("setmode", (void (Layer::*)(int)) &Layer::setmode, "C++: Layer::setmode(int) --> void", pybind11::arg("m"));
		cl.def("detach", (void (Layer::*)(class Layer *)) &Layer::detach, "C++: Layer::detach(class Layer *) --> void", pybind11::arg("l"));
		cl.def("getWeights", (class Tensor * (Layer::*)()) &Layer::getWeights, "C++: Layer::getWeights() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setWeights", (class Tensor * (Layer::*)(class Tensor)) &Layer::setWeights, "C++: Layer::setWeights(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("getBias", (class Tensor * (Layer::*)()) &Layer::getBias, "C++: Layer::getBias() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setBias", (class Tensor * (Layer::*)(class Tensor)) &Layer::setBias, "C++: Layer::setBias(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("clamp", (void (Layer::*)(float, float)) &Layer::clamp, "C++: Layer::clamp(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def("setdetach", (void (Layer::*)()) &Layer::setdetach, "C++: Layer::setdetach() --> void");
		cl.def("resize", (void (Layer::*)(int)) &Layer::resize, "C++: Layer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("reset", (void (Layer::*)()) &Layer::reset, "C++: Layer::reset() --> void");
		cl.def("zeroGrads", (void (Layer::*)()) &Layer::zeroGrads, "C++: Layer::zeroGrads() --> void");
		cl.def("addchild", (void (Layer::*)(class Layer *)) &Layer::addchild, "C++: Layer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (Layer::*)(class Layer *)) &Layer::addparent, "C++: Layer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("forward", (void (Layer::*)()) &Layer::forward, "C++: Layer::forward() --> void");
		cl.def("backward", (void (Layer::*)()) &Layer::backward, "C++: Layer::backward() --> void");
		cl.def("assign", (class Layer & (Layer::*)(const class Layer &)) &Layer::operator=, "C++: Layer::operator=(const class Layer &) --> class Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Optimizer file:eddl/optimizers/optim.h line:27
		pybind11::class_<Optimizer, std::shared_ptr<Optimizer>, PyCallBack_Optimizer> cl(M(""), "Optimizer", "");
		cl.def( pybind11::init( [](){ return new Optimizer(); }, [](){ return new PyCallBack_Optimizer(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Optimizer const &o){ return new PyCallBack_Optimizer(o); } ) );
		cl.def( pybind11::init( [](Optimizer const &o){ return new Optimizer(o); } ) );
		cl.def_readwrite("name", &Optimizer::name);
		cl.def_readwrite("layers", &Optimizer::layers);
		cl.def("applygrads", (void (Optimizer::*)(int)) &Optimizer::applygrads, "C++: Optimizer::applygrads(int) --> void", pybind11::arg("batch"));
		cl.def("clone", (class Optimizer * (Optimizer::*)()) &Optimizer::clone, "C++: Optimizer::clone() --> class Optimizer *", pybind11::return_value_policy::automatic);
		cl.def("assign", (class Optimizer & (Optimizer::*)(const class Optimizer &)) &Optimizer::operator=, "C++: Optimizer::operator=(const class Optimizer &) --> class Optimizer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Net file: line:41
		pybind11::class_<Net, std::shared_ptr<Net>> cl(M(""), "Net", "");
		cl.def( pybind11::init( [](Net const &o){ return new Net(o); } ) );
		cl.def_readwrite("name", &Net::name);
		cl.def_readwrite("dev", &Net::dev);
		cl.def_readwrite("batch_size", &Net::batch_size);
		cl.def_readwrite("tr_batches", &Net::tr_batches);
		cl.def_readwrite("inferenced_samples", &Net::inferenced_samples);
		cl.def_readwrite("trmode", &Net::trmode);
		cl.def_readwrite("devsel", &Net::devsel);
		cl.def_readwrite("layers", &Net::layers);
		cl.def_readwrite("lin", &Net::lin);
		cl.def_readwrite("lout", &Net::lout);
		cl.def_readwrite("vfts", &Net::vfts);
		cl.def_readwrite("vbts", &Net::vbts);
		cl.def_readwrite("netinput", &Net::netinput);
		cl.def_readwrite("losses", &Net::losses);
		cl.def_readwrite("metrics", &Net::metrics);
		cl.def_readwrite("fiterr", &Net::fiterr);
		cl.def_readwrite("total_loss", &Net::total_loss);
		cl.def_readwrite("total_metric", &Net::total_metric);
		cl.def_readwrite("snets", &Net::snets);
		cl.def("toCPU", (void (Net::*)(int)) &Net::toCPU, "C++: Net::toCPU(int) --> void", pybind11::arg("t"));
		cl.def("fts", (void (Net::*)()) &Net::fts, "C++: Net::fts() --> void");
		cl.def("bts", (void (Net::*)()) &Net::bts, "C++: Net::bts() --> void");
		cl.def("split", (void (Net::*)(int, int)) &Net::split, "C++: Net::split(int, int) --> void", pybind11::arg("c"), pybind11::arg("todev"));
		cl.def("inNet", (int (Net::*)(class Layer *)) &Net::inNet, "C++: Net::inNet(class Layer *) --> int", pybind11::arg("l"));
		cl.def("walk", (void (Net::*)(class Layer *)) &Net::walk, "C++: Net::walk(class Layer *) --> void", pybind11::arg("l"));
		cl.def("walk_back", (void (Net::*)(class Layer *)) &Net::walk_back, "C++: Net::walk_back(class Layer *) --> void", pybind11::arg("l"));
		cl.def("resize", (void (Net::*)(int)) &Net::resize, "C++: Net::resize(int) --> void", pybind11::arg("batch"));
		cl.def("setmode", (void (Net::*)(int)) &Net::setmode, "C++: Net::setmode(int) --> void", pybind11::arg("m"));
		cl.def("do_initialize", (void (Net::*)()) &Net::do_initialize, "C++: Net::do_initialize() --> void");
		cl.def("do_reset", (void (Net::*)()) &Net::do_reset, "C++: Net::do_reset() --> void");
		cl.def("do_reset_grads", (void (Net::*)()) &Net::do_reset_grads, "C++: Net::do_reset_grads() --> void");
		cl.def("do_forward", (void (Net::*)()) &Net::do_forward, "C++: Net::do_forward() --> void");
		cl.def("do_delta", (void (Net::*)()) &Net::do_delta, "C++: Net::do_delta() --> void");
		cl.def("do_compute_loss", (void (Net::*)()) &Net::do_compute_loss, "C++: Net::do_compute_loss() --> void");
		cl.def("do_backward", (void (Net::*)()) &Net::do_backward, "C++: Net::do_backward() --> void");
		cl.def("do_applygrads", (void (Net::*)()) &Net::do_applygrads, "C++: Net::do_applygrads() --> void");
		cl.def("sync_weights", (void (Net::*)()) &Net::sync_weights, "C++: Net::sync_weights() --> void");
		cl.def("forward", (void (Net::*)()) &Net::forward, "C++: Net::forward() --> void");
		cl.def("reset_loss", (void (Net::*)()) &Net::reset_loss, "C++: Net::reset_loss() --> void");
		cl.def("print_loss", (void (Net::*)(int)) &Net::print_loss, "C++: Net::print_loss(int) --> void", pybind11::arg("b"));
		cl.def("backward", (void (Net::*)()) &Net::backward, "C++: Net::backward() --> void");
		cl.def("delta", (void (Net::*)()) &Net::delta, "C++: Net::delta() --> void");
		cl.def("reset", (void (Net::*)()) &Net::reset, "C++: Net::reset() --> void");
		cl.def("reset_grads", (void (Net::*)()) &Net::reset_grads, "C++: Net::reset_grads() --> void");
		cl.def("update", (void (Net::*)()) &Net::update, "C++: Net::update() --> void");
		cl.def("compute_loss", (void (Net::*)()) &Net::compute_loss, "C++: Net::compute_loss() --> void");
		cl.def("clamp", (void (Net::*)(float, float)) &Net::clamp, "C++: Net::clamp(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
	}
	{ // NetLoss file:eddl/net/netloss.h line:24
		pybind11::class_<NetLoss, std::shared_ptr<NetLoss>> cl(M(""), "NetLoss", "");
		cl.def( pybind11::init( [](NetLoss const &o){ return new NetLoss(o); } ) );
		cl.def_readwrite("name", &NetLoss::name);
		cl.def_readwrite("value", &NetLoss::value);
		cl.def_readwrite("input", &NetLoss::input);
		cl.def_readwrite("ginput", &NetLoss::ginput);
		cl.def("compute", (float (NetLoss::*)()) &NetLoss::compute, "C++: NetLoss::compute() --> float");
	}
}


// File: eddl/apis/eddl.cpp
#include <eddl/apis/eddl.h>
#include <eddl/layers/layer.h>
#include <eddl/losses/loss.h>
#include <eddl/metrics/metric.h>
#include <eddl/net/compserv.h>
#include <eddl/net/netloss.h>
#include <eddl/optimizers/optim.h>
#include <eddl/tensor/tensor.h>
#include <eddl_addons.hpp>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_apis_eddl(std::function< pybind11::module &(std::string const &namespace_) > &M)
{

	eddl_addons(M("eddl"));
	// eddl::CS_CPU(int) file:eddl/apis/eddl.h line:234
	M("eddl").def("CS_CPU", []() -> CompServ * { return eddl::CS_CPU(); }, "", pybind11::return_value_policy::automatic);
	M("eddl").def("CS_CPU", (class CompServ * (*)(int)) &eddl::CS_CPU, "C++: eddl::CS_CPU(int) --> class CompServ *", pybind11::return_value_policy::automatic, pybind11::arg("th"));

	// eddl::resize_model(class Net *, int) file:eddl/apis/eddl.h line:243
	M("eddl").def("resize_model", (void (*)(class Net *, int)) &eddl::resize_model, "C++: eddl::resize_model(class Net *, int) --> void", pybind11::arg("net"), pybind11::arg("batch_size"));

	// eddl::set_mode(class Net *, int) file:eddl/apis/eddl.h line:245
	M("eddl").def("set_mode", (void (*)(class Net *, int)) &eddl::set_mode, "C++: eddl::set_mode(class Net *, int) --> void", pybind11::arg("net"), pybind11::arg("mode"));

	// eddl::clamp(class Net *, float, float) file:eddl/apis/eddl.h line:257
	M("eddl").def("clamp", (void (*)(class Net *, float, float)) &eddl::clamp, "C++: eddl::clamp(class Net *, float, float) --> void", pybind11::arg("m"), pybind11::arg("min"), pybind11::arg("max"));

	// eddl::print_loss(class Net *, int) file:eddl/apis/eddl.h line:261
	M("eddl").def("print_loss", (void (*)(class Net *, int)) &eddl::print_loss, "C++: eddl::print_loss(class Net *, int) --> void", pybind11::arg("m"), pybind11::arg("batch"));

	// eddl::reset_loss(class Net *) file:eddl/apis/eddl.h line:263
	M("eddl").def("reset_loss", (void (*)(class Net *)) &eddl::reset_loss, "C++: eddl::reset_loss(class Net *) --> void", pybind11::arg("m"));

	// eddl::zeroGrads(class Net *) file:eddl/apis/eddl.h line:264
	M("eddl").def("zeroGrads", (void (*)(class Net *)) &eddl::zeroGrads, "C++: eddl::zeroGrads(class Net *) --> void", pybind11::arg("m"));

	// eddl::backward(class Net *) file:eddl/apis/eddl.h line:267
	M("eddl").def("backward", (void (*)(class Net *)) &eddl::backward, "C++: eddl::backward(class Net *) --> void", pybind11::arg("net"));

	// eddl::backward(class NetLoss *) file:eddl/apis/eddl.h line:268
	M("eddl").def("backward", (void (*)(class NetLoss *)) &eddl::backward, "C++: eddl::backward(class NetLoss *) --> void", pybind11::arg("l"));

	// eddl::compute_loss(class NetLoss *) file:eddl/apis/eddl.h line:270
	M("eddl").def("compute_loss", (float (*)(class NetLoss *)) &eddl::compute_loss, "C++: eddl::compute_loss(class NetLoss *) --> float", pybind11::arg("L"));

	// eddl::update(class Net *) file:eddl/apis/eddl.h line:272
	M("eddl").def("update", (void (*)(class Net *)) &eddl::update, "C++: eddl::update(class Net *) --> void", pybind11::arg("m"));

	// eddl::copyTensor(class Layer *, class Layer *) file:eddl/apis/eddl.h line:274
	M("eddl").def("copyTensor", (void (*)(class Layer *, class Layer *)) &eddl::copyTensor, "C++: eddl::copyTensor(class Layer *, class Layer *) --> void", pybind11::arg("l1"), pybind11::arg("l2"));

	// eddl::copyGrad(class Layer *, class Layer *) file:eddl/apis/eddl.h line:275
	M("eddl").def("copyGrad", (void (*)(class Layer *, class Layer *)) &eddl::copyGrad, "C++: eddl::copyGrad(class Layer *, class Layer *) --> void", pybind11::arg("l1"), pybind11::arg("l2"));

	// eddl::getGrad(class Layer *) file:eddl/apis/eddl.h line:278
	M("eddl").def("getGrad", (class Tensor * (*)(class Layer *)) &eddl::getGrad, "C++: eddl::getGrad(class Layer *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("l"));

	// eddl::build(class Net *, class Optimizer *, class CompServ *) file:eddl/apis/eddl.h line:285
	M("eddl").def("build", [](class Net * a0) -> void { return eddl::build(a0); }, "", pybind11::arg("net"));
	M("eddl").def("build", [](class Net * a0, class Optimizer * a1) -> void { return eddl::build(a0, a1); }, "", pybind11::arg("net"), pybind11::arg("o"));
	M("eddl").def("build", (void (*)(class Net *, class Optimizer *, class CompServ *)) &eddl::build, "C++: eddl::build(class Net *, class Optimizer *, class CompServ *) --> void", pybind11::arg("net"), pybind11::arg("o"), pybind11::arg("cs"));

	// eddl::toCPU(class Net *, int) file:eddl/apis/eddl.h line:288
	M("eddl").def("toCPU", [](class Net * a0) -> void { return eddl::toCPU(a0); }, "", pybind11::arg("net"));
	M("eddl").def("toCPU", (void (*)(class Net *, int)) &eddl::toCPU, "C++: eddl::toCPU(class Net *, int) --> void", pybind11::arg("net"), pybind11::arg("t"));

	// eddl::summary(class Net *) file:eddl/apis/eddl.h line:292
	M("eddl").def("summary", (void (*)(class Net *)) &eddl::summary, "C++: eddl::summary(class Net *) --> void", pybind11::arg("m"));

	// eddl::download_mnist() file:eddl/apis/eddl.h line:308
	M("eddl").def("download_mnist", (void (*)()) &eddl::download_mnist, "C++: eddl::download_mnist() --> void");

	// eddl::download_cifar10() file:eddl/apis/eddl.h line:309
	M("eddl").def("download_cifar10", (void (*)()) &eddl::download_cifar10, "C++: eddl::download_cifar10() --> void");

}


// File: eddl/apis/eddlT.cpp
#include <eddl/apis/eddlT.h>
#include <eddl/tensor/tensor.h>
#include <eddlT_addons.hpp>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_apis_eddlT(std::function< pybind11::module &(std::string const &namespace_) > &M)
{

	eddlT_addons(M("eddlT"));
	// eddlT::arange(float, float, float, int) file:eddl/apis/eddlT.h line:30
	M("eddlT").def("arange", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return eddlT::arange(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
	M("eddlT").def("arange", (class Tensor * (*)(float, float, float, int)) &eddlT::arange, "C++: eddlT::arange(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));

	// eddlT::range(float, float, float, int) file:eddl/apis/eddlT.h line:31
	M("eddlT").def("range", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return eddlT::range(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
	M("eddlT").def("range", (class Tensor * (*)(float, float, float, int)) &eddlT::range, "C++: eddlT::range(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));

	// eddlT::linspace(float, float, int, int) file:eddl/apis/eddlT.h line:32
	M("eddlT").def("linspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return eddlT::linspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
	M("eddlT").def("linspace", (class Tensor * (*)(float, float, int, int)) &eddlT::linspace, "C++: eddlT::linspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));

	// eddlT::logspace(float, float, int, float, int) file:eddl/apis/eddlT.h line:33
	M("eddlT").def("logspace", [](float const & a0, float const & a1, int const & a2, float const & a3) -> Tensor * { return eddlT::logspace(a0, a1, a2, a3); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"));
	M("eddlT").def("logspace", (class Tensor * (*)(float, float, int, float, int)) &eddlT::logspace, "C++: eddlT::logspace(float, float, int, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"), pybind11::arg("dev"));

	// eddlT::eye(int, int) file:eddl/apis/eddlT.h line:34
	M("eddlT").def("eye", [](int const & a0) -> Tensor * { return eddlT::eye(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("size"));
	M("eddlT").def("eye", (class Tensor * (*)(int, int)) &eddlT::eye, "C++: eddlT::eye(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("size"), pybind11::arg("dev"));

	// eddlT::toCPU_(class Tensor *) file:eddl/apis/eddlT.h line:39
	M("eddlT").def("toCPU_", (void (*)(class Tensor *)) &eddlT::toCPU_, "C++: eddlT::toCPU_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::toGPU_(class Tensor *) file:eddl/apis/eddlT.h line:40
	M("eddlT").def("toGPU_", (void (*)(class Tensor *)) &eddlT::toGPU_, "C++: eddlT::toGPU_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::toCPU(class Tensor *) file:eddl/apis/eddlT.h line:41
	M("eddlT").def("toCPU", (class Tensor * (*)(class Tensor *)) &eddlT::toCPU, "C++: eddlT::toCPU(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::toGPU(class Tensor *) file:eddl/apis/eddlT.h line:42
	M("eddlT").def("toGPU", (class Tensor * (*)(class Tensor *)) &eddlT::toGPU, "C++: eddlT::toGPU(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::clone(class Tensor *) file:eddl/apis/eddlT.h line:43
	M("eddlT").def("clone", (class Tensor * (*)(class Tensor *)) &eddlT::clone, "C++: eddlT::clone(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::select(class Tensor *, int) file:eddl/apis/eddlT.h line:44
	M("eddlT").def("select", (class Tensor * (*)(class Tensor *, int)) &eddlT::select, "C++: eddlT::select(class Tensor *, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("i"));

	// eddlT::copyTensor(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:45
	M("eddlT").def("copyTensor", (void (*)(class Tensor *, class Tensor *)) &eddlT::copyTensor, "C++: eddlT::copyTensor(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::fill_(class Tensor *, float) file:eddl/apis/eddlT.h line:48
	M("eddlT").def("fill_", (void (*)(class Tensor *, float)) &eddlT::fill_, "C++: eddlT::fill_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::print(class Tensor *) file:eddl/apis/eddlT.h line:56
	M("eddlT").def("print", (void (*)(class Tensor *)) &eddlT::print, "C++: eddlT::print(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::info(class Tensor *) file:eddl/apis/eddlT.h line:57
	M("eddlT").def("info", (void (*)(class Tensor *)) &eddlT::info, "C++: eddlT::info(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::abs_(class Tensor *) file:eddl/apis/eddlT.h line:66
	M("eddlT").def("abs_", (void (*)(class Tensor *)) &eddlT::abs_, "C++: eddlT::abs_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::abs(class Tensor *) file:eddl/apis/eddlT.h line:67
	M("eddlT").def("abs", (class Tensor * (*)(class Tensor *)) &eddlT::abs, "C++: eddlT::abs(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::acos_(class Tensor *) file:eddl/apis/eddlT.h line:69
	M("eddlT").def("acos_", (void (*)(class Tensor *)) &eddlT::acos_, "C++: eddlT::acos_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::acos(class Tensor *) file:eddl/apis/eddlT.h line:70
	M("eddlT").def("acos", (class Tensor * (*)(class Tensor *)) &eddlT::acos, "C++: eddlT::acos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::add_(class Tensor *, float) file:eddl/apis/eddlT.h line:72
	M("eddlT").def("add_", (void (*)(class Tensor *, float)) &eddlT::add_, "C++: eddlT::add_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::add(class Tensor *, float) file:eddl/apis/eddlT.h line:73
	M("eddlT").def("add", (class Tensor * (*)(class Tensor *, float)) &eddlT::add, "C++: eddlT::add(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::add_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:74
	M("eddlT").def("add_", (void (*)(class Tensor *, class Tensor *)) &eddlT::add_, "C++: eddlT::add_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::add(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:75
	M("eddlT").def("add", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::add, "C++: eddlT::add(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::asin_(class Tensor *) file:eddl/apis/eddlT.h line:79
	M("eddlT").def("asin_", (void (*)(class Tensor *)) &eddlT::asin_, "C++: eddlT::asin_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::asin(class Tensor *) file:eddl/apis/eddlT.h line:80
	M("eddlT").def("asin", (class Tensor * (*)(class Tensor *)) &eddlT::asin, "C++: eddlT::asin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::atan_(class Tensor *) file:eddl/apis/eddlT.h line:82
	M("eddlT").def("atan_", (void (*)(class Tensor *)) &eddlT::atan_, "C++: eddlT::atan_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::atan(class Tensor *) file:eddl/apis/eddlT.h line:83
	M("eddlT").def("atan", (class Tensor * (*)(class Tensor *)) &eddlT::atan, "C++: eddlT::atan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::ceil_(class Tensor *) file:eddl/apis/eddlT.h line:85
	M("eddlT").def("ceil_", (void (*)(class Tensor *)) &eddlT::ceil_, "C++: eddlT::ceil_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::ceil(class Tensor *) file:eddl/apis/eddlT.h line:86
	M("eddlT").def("ceil", (class Tensor * (*)(class Tensor *)) &eddlT::ceil, "C++: eddlT::ceil(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::clamp_(class Tensor *, float, float) file:eddl/apis/eddlT.h line:88
	M("eddlT").def("clamp_", (void (*)(class Tensor *, float, float)) &eddlT::clamp_, "C++: eddlT::clamp_(class Tensor *, float, float) --> void", pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::clamp(class Tensor *, float, float) file:eddl/apis/eddlT.h line:89
	M("eddlT").def("clamp", (class Tensor * (*)(class Tensor *, float, float)) &eddlT::clamp, "C++: eddlT::clamp(class Tensor *, float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::clampmax_(class Tensor *, float) file:eddl/apis/eddlT.h line:91
	M("eddlT").def("clampmax_", (void (*)(class Tensor *, float)) &eddlT::clampmax_, "C++: eddlT::clampmax_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("max"));

	// eddlT::clampmax(class Tensor *, float) file:eddl/apis/eddlT.h line:92
	M("eddlT").def("clampmax", (class Tensor * (*)(class Tensor *, float)) &eddlT::clampmax, "C++: eddlT::clampmax(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("max"));

	// eddlT::clampmin_(class Tensor *, float) file:eddl/apis/eddlT.h line:94
	M("eddlT").def("clampmin_", (void (*)(class Tensor *, float)) &eddlT::clampmin_, "C++: eddlT::clampmin_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("min"));

	// eddlT::clampmin(class Tensor *, float) file:eddl/apis/eddlT.h line:95
	M("eddlT").def("clampmin", (class Tensor * (*)(class Tensor *, float)) &eddlT::clampmin, "C++: eddlT::clampmin(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"));

	// eddlT::cos_(class Tensor *) file:eddl/apis/eddlT.h line:97
	M("eddlT").def("cos_", (void (*)(class Tensor *)) &eddlT::cos_, "C++: eddlT::cos_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::cos(class Tensor *) file:eddl/apis/eddlT.h line:98
	M("eddlT").def("cos", (class Tensor * (*)(class Tensor *)) &eddlT::cos, "C++: eddlT::cos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::cosh_(class Tensor *) file:eddl/apis/eddlT.h line:100
	M("eddlT").def("cosh_", (void (*)(class Tensor *)) &eddlT::cosh_, "C++: eddlT::cosh_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::cosh(class Tensor *) file:eddl/apis/eddlT.h line:101
	M("eddlT").def("cosh", (class Tensor * (*)(class Tensor *)) &eddlT::cosh, "C++: eddlT::cosh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::div_(class Tensor *, float) file:eddl/apis/eddlT.h line:105
	M("eddlT").def("div_", (void (*)(class Tensor *, float)) &eddlT::div_, "C++: eddlT::div_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::div(class Tensor *, float) file:eddl/apis/eddlT.h line:106
	M("eddlT").def("div", (class Tensor * (*)(class Tensor *, float)) &eddlT::div, "C++: eddlT::div(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::div_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:107
	M("eddlT").def("div_", (void (*)(class Tensor *, class Tensor *)) &eddlT::div_, "C++: eddlT::div_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::div(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:108
	M("eddlT").def("div", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::div, "C++: eddlT::div(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::exp_(class Tensor *) file:eddl/apis/eddlT.h line:110
	M("eddlT").def("exp_", (void (*)(class Tensor *)) &eddlT::exp_, "C++: eddlT::exp_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::exp(class Tensor *) file:eddl/apis/eddlT.h line:111
	M("eddlT").def("exp", (class Tensor * (*)(class Tensor *)) &eddlT::exp, "C++: eddlT::exp(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::floor_(class Tensor *) file:eddl/apis/eddlT.h line:113
	M("eddlT").def("floor_", (void (*)(class Tensor *)) &eddlT::floor_, "C++: eddlT::floor_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::floor(class Tensor *) file:eddl/apis/eddlT.h line:114
	M("eddlT").def("floor", (class Tensor * (*)(class Tensor *)) &eddlT::floor, "C++: eddlT::floor(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::inc_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:116
	M("eddlT").def("inc_", (void (*)(class Tensor *, class Tensor *)) &eddlT::inc_, "C++: eddlT::inc_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::log_(class Tensor *) file:eddl/apis/eddlT.h line:118
	M("eddlT").def("log_", (void (*)(class Tensor *)) &eddlT::log_, "C++: eddlT::log_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::log(class Tensor *) file:eddl/apis/eddlT.h line:119
	M("eddlT").def("log", (class Tensor * (*)(class Tensor *)) &eddlT::log, "C++: eddlT::log(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::log2_(class Tensor *) file:eddl/apis/eddlT.h line:121
	M("eddlT").def("log2_", (void (*)(class Tensor *)) &eddlT::log2_, "C++: eddlT::log2_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::log2(class Tensor *) file:eddl/apis/eddlT.h line:122
	M("eddlT").def("log2", (class Tensor * (*)(class Tensor *)) &eddlT::log2, "C++: eddlT::log2(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::log10_(class Tensor *) file:eddl/apis/eddlT.h line:124
	M("eddlT").def("log10_", (void (*)(class Tensor *)) &eddlT::log10_, "C++: eddlT::log10_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::log10(class Tensor *) file:eddl/apis/eddlT.h line:125
	M("eddlT").def("log10", (class Tensor * (*)(class Tensor *)) &eddlT::log10, "C++: eddlT::log10(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::logn_(class Tensor *, float) file:eddl/apis/eddlT.h line:127
	M("eddlT").def("logn_", (void (*)(class Tensor *, float)) &eddlT::logn_, "C++: eddlT::logn_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("n"));

	// eddlT::logn(class Tensor *, float) file:eddl/apis/eddlT.h line:128
	M("eddlT").def("logn", (class Tensor * (*)(class Tensor *, float)) &eddlT::logn, "C++: eddlT::logn(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("n"));

	// eddlT::max(class Tensor *) file:eddl/apis/eddlT.h line:130
	M("eddlT").def("max", (float (*)(class Tensor *)) &eddlT::max, "C++: eddlT::max(class Tensor *) --> float", pybind11::arg("A"));

	// eddlT::min(class Tensor *) file:eddl/apis/eddlT.h line:131
	M("eddlT").def("min", (float (*)(class Tensor *)) &eddlT::min, "C++: eddlT::min(class Tensor *) --> float", pybind11::arg("A"));

	// eddlT::mod_(class Tensor *, float) file:eddl/apis/eddlT.h line:133
	M("eddlT").def("mod_", (void (*)(class Tensor *, float)) &eddlT::mod_, "C++: eddlT::mod_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::mod(class Tensor *, float) file:eddl/apis/eddlT.h line:134
	M("eddlT").def("mod", (class Tensor * (*)(class Tensor *, float)) &eddlT::mod, "C++: eddlT::mod(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

}


// File: eddl/apis/eddlT_1.cpp
#include <eddl/apis/eddlT.h>
#include <eddl/tensor/tensor.h>
#include <eddlT_addons.hpp>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_apis_eddlT_1(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	// eddlT::mult_(class Tensor *, float) file:eddl/apis/eddlT.h line:136
	M("eddlT").def("mult_", (void (*)(class Tensor *, float)) &eddlT::mult_, "C++: eddlT::mult_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::mult(class Tensor *, float) file:eddl/apis/eddlT.h line:137
	M("eddlT").def("mult", (class Tensor * (*)(class Tensor *, float)) &eddlT::mult, "C++: eddlT::mult(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::mult_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:138
	M("eddlT").def("mult_", (void (*)(class Tensor *, class Tensor *)) &eddlT::mult_, "C++: eddlT::mult_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::mult(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:139
	M("eddlT").def("mult", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::mult, "C++: eddlT::mult(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::mult2D(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:140
	M("eddlT").def("mult2D", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::mult2D, "C++: eddlT::mult2D(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::neg_(class Tensor *) file:eddl/apis/eddlT.h line:142
	M("eddlT").def("neg_", (void (*)(class Tensor *)) &eddlT::neg_, "C++: eddlT::neg_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::neg(class Tensor *) file:eddl/apis/eddlT.h line:143
	M("eddlT").def("neg", (class Tensor * (*)(class Tensor *)) &eddlT::neg, "C++: eddlT::neg(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::normalize_(class Tensor *, float, float) file:eddl/apis/eddlT.h line:145
	M("eddlT").def("normalize_", (void (*)(class Tensor *, float, float)) &eddlT::normalize_, "C++: eddlT::normalize_(class Tensor *, float, float) --> void", pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::normalize(class Tensor *, float, float) file:eddl/apis/eddlT.h line:146
	M("eddlT").def("normalize", (class Tensor * (*)(class Tensor *, float, float)) &eddlT::normalize, "C++: eddlT::normalize(class Tensor *, float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::reciprocal_(class Tensor *) file:eddl/apis/eddlT.h line:151
	M("eddlT").def("reciprocal_", (void (*)(class Tensor *)) &eddlT::reciprocal_, "C++: eddlT::reciprocal_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::reciprocal(class Tensor *) file:eddl/apis/eddlT.h line:152
	M("eddlT").def("reciprocal", (class Tensor * (*)(class Tensor *)) &eddlT::reciprocal, "C++: eddlT::reciprocal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::round_(class Tensor *) file:eddl/apis/eddlT.h line:157
	M("eddlT").def("round_", (void (*)(class Tensor *)) &eddlT::round_, "C++: eddlT::round_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::round(class Tensor *) file:eddl/apis/eddlT.h line:158
	M("eddlT").def("round", (class Tensor * (*)(class Tensor *)) &eddlT::round, "C++: eddlT::round(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::rsqrt_(class Tensor *) file:eddl/apis/eddlT.h line:160
	M("eddlT").def("rsqrt_", (void (*)(class Tensor *)) &eddlT::rsqrt_, "C++: eddlT::rsqrt_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::rsqrt(class Tensor *) file:eddl/apis/eddlT.h line:161
	M("eddlT").def("rsqrt", (class Tensor * (*)(class Tensor *)) &eddlT::rsqrt, "C++: eddlT::rsqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sigmoid_(class Tensor *) file:eddl/apis/eddlT.h line:163
	M("eddlT").def("sigmoid_", (void (*)(class Tensor *)) &eddlT::sigmoid_, "C++: eddlT::sigmoid_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sigmoid(class Tensor *) file:eddl/apis/eddlT.h line:164
	M("eddlT").def("sigmoid", (class Tensor * (*)(class Tensor *)) &eddlT::sigmoid, "C++: eddlT::sigmoid(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sign_(class Tensor *) file:eddl/apis/eddlT.h line:166
	M("eddlT").def("sign_", (void (*)(class Tensor *)) &eddlT::sign_, "C++: eddlT::sign_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sign(class Tensor *) file:eddl/apis/eddlT.h line:167
	M("eddlT").def("sign", (class Tensor * (*)(class Tensor *)) &eddlT::sign, "C++: eddlT::sign(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sin_(class Tensor *) file:eddl/apis/eddlT.h line:170
	M("eddlT").def("sin_", (void (*)(class Tensor *)) &eddlT::sin_, "C++: eddlT::sin_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sin(class Tensor *) file:eddl/apis/eddlT.h line:171
	M("eddlT").def("sin", (class Tensor * (*)(class Tensor *)) &eddlT::sin, "C++: eddlT::sin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sinh_(class Tensor *) file:eddl/apis/eddlT.h line:173
	M("eddlT").def("sinh_", (void (*)(class Tensor *)) &eddlT::sinh_, "C++: eddlT::sinh_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sinh(class Tensor *) file:eddl/apis/eddlT.h line:174
	M("eddlT").def("sinh", (class Tensor * (*)(class Tensor *)) &eddlT::sinh, "C++: eddlT::sinh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sqr_(class Tensor *) file:eddl/apis/eddlT.h line:176
	M("eddlT").def("sqr_", (void (*)(class Tensor *)) &eddlT::sqr_, "C++: eddlT::sqr_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sqr(class Tensor *) file:eddl/apis/eddlT.h line:177
	M("eddlT").def("sqr", (class Tensor * (*)(class Tensor *)) &eddlT::sqr, "C++: eddlT::sqr(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sqrt_(class Tensor *) file:eddl/apis/eddlT.h line:179
	M("eddlT").def("sqrt_", (void (*)(class Tensor *)) &eddlT::sqrt_, "C++: eddlT::sqrt_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sqrt(class Tensor *) file:eddl/apis/eddlT.h line:180
	M("eddlT").def("sqrt", (class Tensor * (*)(class Tensor *)) &eddlT::sqrt, "C++: eddlT::sqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sub_(class Tensor *, float) file:eddl/apis/eddlT.h line:182
	M("eddlT").def("sub_", (void (*)(class Tensor *, float)) &eddlT::sub_, "C++: eddlT::sub_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::sub(class Tensor *, float) file:eddl/apis/eddlT.h line:183
	M("eddlT").def("sub", (class Tensor * (*)(class Tensor *, float)) &eddlT::sub, "C++: eddlT::sub(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::sub_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:184
	M("eddlT").def("sub_", (void (*)(class Tensor *, class Tensor *)) &eddlT::sub_, "C++: eddlT::sub_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::sub(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:185
	M("eddlT").def("sub", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::sub, "C++: eddlT::sub(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::tan_(class Tensor *) file:eddl/apis/eddlT.h line:190
	M("eddlT").def("tan_", (void (*)(class Tensor *)) &eddlT::tan_, "C++: eddlT::tan_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::tan(class Tensor *) file:eddl/apis/eddlT.h line:191
	M("eddlT").def("tan", (class Tensor * (*)(class Tensor *)) &eddlT::tan, "C++: eddlT::tan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::tanh_(class Tensor *) file:eddl/apis/eddlT.h line:193
	M("eddlT").def("tanh_", (void (*)(class Tensor *)) &eddlT::tanh_, "C++: eddlT::tanh_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::tanh(class Tensor *) file:eddl/apis/eddlT.h line:194
	M("eddlT").def("tanh", (class Tensor * (*)(class Tensor *)) &eddlT::tanh, "C++: eddlT::tanh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::trunc_(class Tensor *) file:eddl/apis/eddlT.h line:196
	M("eddlT").def("trunc_", (void (*)(class Tensor *)) &eddlT::trunc_, "C++: eddlT::trunc_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::trunc(class Tensor *) file:eddl/apis/eddlT.h line:197
	M("eddlT").def("trunc", (class Tensor * (*)(class Tensor *)) &eddlT::trunc, "C++: eddlT::trunc(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

}


#include <map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_eddl_tensor_tensor(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_net_compserv(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_apis_eddl(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_apis_eddlT(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_apis_eddlT_1(std::function< pybind11::module &(std::string const &namespace_) > &M);


PYBIND11_MODULE(_core, root_module) {
	root_module.doc() = "_core module";

	std::map <std::string, pybind11::module> modules;
	ModuleGetter M = [&](std::string const &namespace_) -> pybind11::module & {
		auto it = modules.find(namespace_);
		if( it == modules.end() ) throw std::runtime_error("Attempt to access pybind11::module for namespace " + namespace_ + " before it was created!!!");
		return it->second;
	};

	modules[""] = root_module;

	std::vector< std::pair<std::string, std::string> > sub_modules {
		{"", "eddl"},
		{"", "eddlT"},
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule(p.second.c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_eddl_tensor_tensor(M);
	bind_eddl_net_compserv(M);
	bind_eddl_apis_eddl(M);
	bind_eddl_apis_eddlT(M);
	bind_eddl_apis_eddlT_1(M);

}

// Source list file: /pyeddl/codegen/bindings/_core.sources
// _core.cpp
// eddl/tensor/tensor.cpp
// eddl/net/compserv.cpp
// eddl/apis/eddl.cpp
// eddl/apis/eddlT.cpp
// eddl/apis/eddlT_1.cpp

// Modules list file: /pyeddl/codegen/bindings/_core.modules
// eddl eddlT 
