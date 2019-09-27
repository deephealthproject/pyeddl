// File: bits/libio.cpp
#include <eddl/utils.h>
#include <sstream> // __str__
#include <stdio.h>

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

void bind_bits_libio(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // _IO_FILE file:bits/libio.h line:245
		pybind11::class_<_IO_FILE, std::shared_ptr<_IO_FILE>> cl(M(""), "_IO_FILE", "");

		cl.def( pybind11::init( [](){ return new _IO_FILE(); } ) );
		cl.def_readwrite("_flags", &_IO_FILE::_flags);
		cl.def_readwrite("_fileno", &_IO_FILE::_fileno);
		cl.def_readwrite("_flags2", &_IO_FILE::_flags2);
		cl.def_readwrite("_old_offset", &_IO_FILE::_old_offset);
		cl.def_readwrite("_cur_column", &_IO_FILE::_cur_column);
		cl.def_readwrite("_vtable_offset", &_IO_FILE::_vtable_offset);
		cl.def_readwrite("_offset", &_IO_FILE::_offset);
		cl.def_readwrite("__pad5", &_IO_FILE::__pad5);
		cl.def_readwrite("_mode", &_IO_FILE::_mode);
	}
	// get_fmem(int, char *) file:eddl/utils.h line:28
	M("").def("get_fmem", (float * (*)(int, char *)) &get_fmem, "C++: get_fmem(int, char *) --> float *", pybind11::return_value_policy::automatic, pybind11::arg("size"), pybind11::arg("str"));

	// humanSize(unsigned long) file:eddl/utils.h line:30
	M("").def("humanSize", (char * (*)(unsigned long)) &humanSize, "C++: humanSize(unsigned long) --> char *", pybind11::return_value_policy::automatic, pybind11::arg("bytes"));

	// get_free_mem() file:eddl/utils.h line:32
	M("").def("get_free_mem", (unsigned long (*)()) &get_free_mem, "C++: get_free_mem() --> unsigned long");

}


// File: eddl/tensor/tensor.cpp
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <memory>
#include <sstream> // __str__
#include <stdio.h>
#include <string>
#include <tensor_addons.h>
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
	{ // Tensor file:eddl/tensor/tensor.h line:68
		pybind11::class_<Tensor, std::shared_ptr<Tensor>> cl(M(""), "Tensor", pybind11::buffer_protocol());

		cl.def( pybind11::init( [](){ return new Tensor(); } ) );
		cl.def_readwrite("device", &Tensor::device);
		cl.def_readwrite("ndim", &Tensor::ndim);
		cl.def_readwrite("size", &Tensor::size);
		cl.def_readwrite("shape", &Tensor::shape);
		cl.def_readwrite("stride", &Tensor::stride);
		cl.def_readwrite("gpu_device", &Tensor::gpu_device);
		cl.def("isCPU", (int (Tensor::*)()) &Tensor::isCPU, "C++: Tensor::isCPU() --> int");
		cl.def("isGPU", (int (Tensor::*)()) &Tensor::isGPU, "C++: Tensor::isGPU() --> int");
		cl.def("isFPGA", (int (Tensor::*)()) &Tensor::isFPGA, "C++: Tensor::isFPGA() --> int");
		cl.def("info", (void (Tensor::*)()) &Tensor::info, "C++: Tensor::info() --> void");
		cl.def("print", (void (Tensor::*)()) &Tensor::print, "C++: Tensor::print() --> void");
		cl.def("numel", (int (Tensor::*)()) &Tensor::numel, "C++: Tensor::numel() --> int");
		cl.def("set", (void (Tensor::*)(float)) &Tensor::set, "C++: Tensor::set(float) --> void", pybind11::arg("v"));
		cl.def_static("arange", [](float const & a0, float const & a1) -> Tensor * { return Tensor::arange(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("arange", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::arange(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"));
		cl.def_static("arange", (class Tensor * (*)(float, float, float, int)) &Tensor::arange, "C++: Tensor::arange(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("range", [](float const & a0, float const & a1) -> Tensor * { return Tensor::range(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("range", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::range(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"));
		cl.def_static("range", (class Tensor * (*)(float, float, float, int)) &Tensor::range, "C++: Tensor::range(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("linspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::linspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("linspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::linspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("linspace", (class Tensor * (*)(float, float, int, int)) &Tensor::linspace, "C++: Tensor::linspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));
		cl.def_static("logspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::logspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::logspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2, float const & a3) -> Tensor * { return Tensor::logspace(a0, a1, a2, a3); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"));
		cl.def_static("logspace", (class Tensor * (*)(float, float, int, float, int)) &Tensor::logspace, "C++: Tensor::logspace(float, float, int, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"), pybind11::arg("dev"));
		cl.def_static("eye", [](int const & a0) -> Tensor * { return Tensor::eye(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("size"));
		cl.def_static("eye", (class Tensor * (*)(int, int)) &Tensor::eye, "C++: Tensor::eye(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("size"), pybind11::arg("dev"));
		cl.def("abs_", (void (Tensor::*)()) &Tensor::abs_, "C++: Tensor::abs_() --> void");
		cl.def_static("abs", (class Tensor * (*)(class Tensor *)) &Tensor::abs, "C++: Tensor::abs(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("acos_", (void (Tensor::*)()) &Tensor::acos_, "C++: Tensor::acos_() --> void");
		cl.def_static("acos", (class Tensor * (*)(class Tensor *)) &Tensor::acos, "C++: Tensor::acos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("add_", (void (Tensor::*)(float)) &Tensor::add_, "C++: Tensor::add_(float) --> void", pybind11::arg("v"));
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
		cl.def("clamp_", (void (Tensor::*)()) &Tensor::clamp_, "C++: Tensor::clamp_() --> void");
		cl.def_static("clamp", (class Tensor * (*)(class Tensor *)) &Tensor::clamp, "C++: Tensor::clamp(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("cos_", (void (Tensor::*)()) &Tensor::cos_, "C++: Tensor::cos_() --> void");
		cl.def_static("cos", (class Tensor * (*)(class Tensor *)) &Tensor::cos, "C++: Tensor::cos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("cosh_", (void (Tensor::*)()) &Tensor::cosh_, "C++: Tensor::cosh_() --> void");
		cl.def_static("cosh", (class Tensor * (*)(class Tensor *)) &Tensor::cosh, "C++: Tensor::cosh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("div_", (void (Tensor::*)(float)) &Tensor::div_, "C++: Tensor::div_(float) --> void", pybind11::arg("v"));
		cl.def_static("div", (class Tensor * (*)(class Tensor *)) &Tensor::div, "C++: Tensor::div(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
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
		cl.def_static("logn", (class Tensor * (*)(class Tensor *)) &Tensor::logn, "C++: Tensor::logn(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("mod_", (void (Tensor::*)()) &Tensor::mod_, "C++: Tensor::mod_() --> void");
		cl.def_static("mod", (class Tensor * (*)(class Tensor *)) &Tensor::mod, "C++: Tensor::mod(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("mult_", (void (Tensor::*)(float)) &Tensor::mult_, "C++: Tensor::mult_(float) --> void", pybind11::arg("v"));
		cl.def_static("mult", (class Tensor * (*)(class Tensor *)) &Tensor::mult, "C++: Tensor::mult(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("mult2D", (void (*)(class Tensor *, int, class Tensor *, int, class Tensor *, int)) &Tensor::mult2D, "C++: Tensor::mult2D(class Tensor *, int, class Tensor *, int, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("tA"), pybind11::arg("B"), pybind11::arg("tB"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("el_mult", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_mult, "C++: Tensor::el_mult(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def("neg_", (void (Tensor::*)()) &Tensor::neg_, "C++: Tensor::neg_() --> void");
		cl.def_static("neg", (class Tensor * (*)(class Tensor *)) &Tensor::neg, "C++: Tensor::neg(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("pow_", (void (Tensor::*)(float)) &Tensor::pow_, "C++: Tensor::pow_(float) --> void", pybind11::arg("exp"));
		cl.def_static("pow", (class Tensor * (*)(class Tensor *)) &Tensor::pow, "C++: Tensor::pow(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("reciprocal_", (void (Tensor::*)()) &Tensor::reciprocal_, "C++: Tensor::reciprocal_() --> void");
		cl.def_static("reciprocal", (class Tensor * (*)(class Tensor *)) &Tensor::reciprocal, "C++: Tensor::reciprocal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("remainder_", (void (Tensor::*)()) &Tensor::remainder_, "C++: Tensor::remainder_() --> void");
		cl.def_static("remainder", (class Tensor * (*)(class Tensor *)) &Tensor::remainder, "C++: Tensor::remainder(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
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
		cl.def("sum_", (float (Tensor::*)()) &Tensor::sum_, "C++: Tensor::sum_() --> float");
		cl.def_static("sum", (class Tensor * (*)(class Tensor *)) &Tensor::sum, "C++: Tensor::sum(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("sum2D_rowwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_rowwise, "C++: Tensor::sum2D_rowwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("sum2D_colwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_colwise, "C++: Tensor::sum2D_colwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("sum_abs_", (float (Tensor::*)()) &Tensor::sum_abs_, "C++: Tensor::sum_abs_() --> float");
		cl.def_static("sum_abs", (class Tensor * (*)(class Tensor *)) &Tensor::sum_abs, "C++: Tensor::sum_abs(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("tan", (class Tensor * (*)(class Tensor *)) &Tensor::tan, "C++: Tensor::tan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("tanh", (class Tensor * (*)(class Tensor *)) &Tensor::tanh, "C++: Tensor::tanh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("reduce_sum2D", (void (*)(class Tensor *, class Tensor *, int, int)) &Tensor::reduce_sum2D, "C++: Tensor::reduce_sum2D(class Tensor *, class Tensor *, int, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"), pybind11::arg("incB"));
		cl.def_static("reduceTosum", (void (*)(class Tensor *, class Tensor *, int)) &Tensor::reduceTosum, "C++: Tensor::reduceTosum(class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"));
		cl.def_static("eqsize", (int (*)(class Tensor *, class Tensor *)) &Tensor::eqsize, "C++: Tensor::eqsize(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equal", (int (*)(class Tensor *, class Tensor *)) &Tensor::equal, "C++: Tensor::equal(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("copy", (void (*)(class Tensor *, class Tensor *)) &Tensor::copy, "C++: Tensor::copy(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("fill", (void (*)(class Tensor *, int, int, class Tensor *, int, int, int)) &Tensor::fill, "C++: Tensor::fill(class Tensor *, int, int, class Tensor *, int, int, int) --> void", pybind11::arg("A"), pybind11::arg("aini"), pybind11::arg("aend"), pybind11::arg("B"), pybind11::arg("bini"), pybind11::arg("bend"), pybind11::arg("inc"));
		cl.def("rand_uniform", (void (Tensor::*)(float)) &Tensor::rand_uniform, "C++: Tensor::rand_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_signed_uniform", (void (Tensor::*)(float)) &Tensor::rand_signed_uniform, "C++: Tensor::rand_signed_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_normal", [](Tensor &o, float const & a0, float const & a1) -> void { return o.rand_normal(a0, a1); }, "", pybind11::arg("m"), pybind11::arg("s"));
		cl.def("rand_normal", (void (Tensor::*)(float, float, bool)) &Tensor::rand_normal, "C++: Tensor::rand_normal(float, float, bool) --> void", pybind11::arg("m"), pybind11::arg("s"), pybind11::arg("fast_math"));
		cl.def("rand_binary", (void (Tensor::*)(float)) &Tensor::rand_binary, "C++: Tensor::rand_binary(float) --> void", pybind11::arg("v"));

		tensor_addons(cl);
	}
}


// File: eddl/descriptors/descriptors.cpp
#include <compserv_addons.h>
#include <convoldescriptor_addons.h>
#include <eddl/compserv.h>
#include <eddl/descriptors/descriptors.h>
#include <eddl/layers/layer.h>
#include <eddl/losses/loss.h>
#include <eddl/metrics/metric.h>
#include <eddl/tensor/nn/tensor_nn.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <layer_addons.h>
#include <memory>
#include <pooldescriptor_addons.h>
#include <sstream> // __str__
#include <stdio.h>
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

// Loss file:eddl/losses/loss.h line:32
struct PyCallBack_Loss : public Loss {
	using Loss::Loss;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Loss *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Loss::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Loss *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return Loss::value(a0, a1);
	}
};

// LMeanSquaredError file:eddl/losses/loss.h line:43
struct PyCallBack_LMeanSquaredError : public LMeanSquaredError {
	using LMeanSquaredError::LMeanSquaredError;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMeanSquaredError *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMeanSquaredError::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMeanSquaredError *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return LMeanSquaredError::value(a0, a1);
	}
};

// LCrossEntropy file:eddl/losses/loss.h line:52
struct PyCallBack_LCrossEntropy : public LCrossEntropy {
	using LCrossEntropy::LCrossEntropy;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LCrossEntropy *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LCrossEntropy::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LCrossEntropy *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return LCrossEntropy::value(a0, a1);
	}
};

// LSoftCrossEntropy file:eddl/losses/loss.h line:61
struct PyCallBack_LSoftCrossEntropy : public LSoftCrossEntropy {
	using LSoftCrossEntropy::LSoftCrossEntropy;

	void delta(class Tensor * a0, class Tensor * a1, class Tensor * a2) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSoftCrossEntropy *>(this), "delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1, a2);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LSoftCrossEntropy::delta(a0, a1, a2);
	}
	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LSoftCrossEntropy *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return LSoftCrossEntropy::value(a0, a1);
	}
};

// Metric file:eddl/metrics/metric.h line:34
struct PyCallBack_Metric : public Metric {
	using Metric::Metric;

	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Metric *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return Metric::value(a0, a1);
	}
};

// MMeanSquaredError file:eddl/metrics/metric.h line:44
struct PyCallBack_MMeanSquaredError : public MMeanSquaredError {
	using MMeanSquaredError::MMeanSquaredError;

	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MMeanSquaredError *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return MMeanSquaredError::value(a0, a1);
	}
};

// MCategoricalAccuracy file:eddl/metrics/metric.h line:52
struct PyCallBack_MCategoricalAccuracy : public MCategoricalAccuracy {
	using MCategoricalAccuracy::MCategoricalAccuracy;

	float value(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MCategoricalAccuracy *>(this), "value");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<float>::value) {
				static pybind11::detail::overload_caster_t<float> caster;
				return pybind11::detail::cast_ref<float>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<float>(std::move(o));
		}
		return MCategoricalAccuracy::value(a0, a1);
	}
};

// Layer file:eddl/layers/layer.h line:35
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

// LinLayer file:eddl/layers/layer.h line:99
struct PyCallBack_LinLayer : public LinLayer {
	using LinLayer::LinLayer;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LinLayer *>(this), "info");
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
};

void bind_eddl_descriptors_descriptors(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // ConvolDescriptor file:eddl/descriptors/descriptors.h line:20
		pybind11::class_<ConvolDescriptor, std::shared_ptr<ConvolDescriptor>> cl(M(""), "ConvolDescriptor", "");

		cl.def( pybind11::init( [](){ return new ConvolDescriptor(); } ) );
		cl.def_readwrite("ksize", &ConvolDescriptor::ksize);
		cl.def_readwrite("stride", &ConvolDescriptor::stride);
		cl.def_readwrite("pad", &ConvolDescriptor::pad);
		cl.def_readwrite("nk", &ConvolDescriptor::nk);
		cl.def_readwrite("kr", &ConvolDescriptor::kr);
		cl.def_readwrite("kc", &ConvolDescriptor::kc);
		cl.def_readwrite("kz", &ConvolDescriptor::kz);
		cl.def_readwrite("sr", &ConvolDescriptor::sr);
		cl.def_readwrite("sc", &ConvolDescriptor::sc);
		cl.def_readwrite("ir", &ConvolDescriptor::ir);
		cl.def_readwrite("ic", &ConvolDescriptor::ic);
		cl.def_readwrite("iz", &ConvolDescriptor::iz);
		cl.def_readwrite("r", &ConvolDescriptor::r);
		cl.def_readwrite("c", &ConvolDescriptor::c);
		cl.def_readwrite("z", &ConvolDescriptor::z);
		cl.def_readwrite("padr", &ConvolDescriptor::padr);
		cl.def_readwrite("padc", &ConvolDescriptor::padc);
		cl.def_readwrite("matI", &ConvolDescriptor::matI);
		cl.def_readwrite("matK", &ConvolDescriptor::matK);
		cl.def_readwrite("matO", &ConvolDescriptor::matO);
		cl.def_readwrite("matD", &ConvolDescriptor::matD);
		cl.def_readwrite("matgK", &ConvolDescriptor::matgK);
		cl.def("build", (void (ConvolDescriptor::*)(class Tensor *)) &ConvolDescriptor::build, "C++: ConvolDescriptor::build(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("resize", (void (ConvolDescriptor::*)(class Tensor *)) &ConvolDescriptor::resize, "C++: ConvolDescriptor::resize(class Tensor *) --> void", pybind11::arg("A"));

		convoldescriptor_addons(cl);
	}
	{ // PoolDescriptor file:eddl/descriptors/descriptors.h line:70
		pybind11::class_<PoolDescriptor, std::shared_ptr<PoolDescriptor>, ConvolDescriptor> cl(M(""), "PoolDescriptor", "");

		cl.def("build", (void (PoolDescriptor::*)(class Tensor *)) &PoolDescriptor::build, "C++: PoolDescriptor::build(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("resize", (void (PoolDescriptor::*)(class Tensor *)) &PoolDescriptor::resize, "C++: PoolDescriptor::resize(class Tensor *) --> void", pybind11::arg("A"));

		pooldescriptor_addons(cl);
	}
	// cent(class Tensor *, class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:12
	M("").def("cent", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &cent, "C++: cent(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));

	// accuracy(class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:15
	M("").def("accuracy", (int (*)(class Tensor *, class Tensor *)) &accuracy, "C++: accuracy(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));

	// ReLu(class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:20
	M("").def("ReLu", (void (*)(class Tensor *, class Tensor *)) &ReLu, "C++: ReLu(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// D_ReLu(class Tensor *, class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:21
	M("").def("D_ReLu", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &D_ReLu, "C++: D_ReLu(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("D"), pybind11::arg("I"), pybind11::arg("PD"));

	// Softmax(class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:24
	M("").def("Softmax", (void (*)(class Tensor *, class Tensor *)) &Softmax, "C++: Softmax(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// D_Softmax(class Tensor *, class Tensor *, class Tensor *) file:eddl/tensor/nn/tensor_nn.h line:25
	M("").def("D_Softmax", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &D_Softmax, "C++: D_Softmax(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("D"), pybind11::arg("I"), pybind11::arg("PD"));

	// Conv2D(class ConvolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:30
	M("").def("Conv2D", (void (*)(class ConvolDescriptor *)) &Conv2D, "C++: Conv2D(class ConvolDescriptor *) --> void", pybind11::arg("D"));

	// Conv2D_grad(class ConvolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:31
	M("").def("Conv2D_grad", (void (*)(class ConvolDescriptor *)) &Conv2D_grad, "C++: Conv2D_grad(class ConvolDescriptor *) --> void", pybind11::arg("D"));

	// Conv2D_back(class ConvolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:32
	M("").def("Conv2D_back", (void (*)(class ConvolDescriptor *)) &Conv2D_back, "C++: Conv2D_back(class ConvolDescriptor *) --> void", pybind11::arg("D"));

	// MPool2D(class PoolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:35
	M("").def("MPool2D", (void (*)(class PoolDescriptor *)) &MPool2D, "C++: MPool2D(class PoolDescriptor *) --> void", pybind11::arg("D"));

	// MPool2D_back(class PoolDescriptor *) file:eddl/tensor/nn/tensor_nn.h line:36
	M("").def("MPool2D_back", (void (*)(class PoolDescriptor *)) &MPool2D_back, "C++: MPool2D_back(class PoolDescriptor *) --> void", pybind11::arg("D"));

	{ // CompServ file:eddl/compserv.h line:30
		pybind11::class_<CompServ, std::shared_ptr<CompServ>> cl(M(""), "CompServ", "");

		cl.def( pybind11::init<struct _IO_FILE *>(), pybind11::arg("csspec") );

		cl.def_readwrite("type", &CompServ::type);
		cl.def_readwrite("local_threads", &CompServ::local_threads);
		cl.def_readwrite("local_gpus", &CompServ::local_gpus);
		cl.def_readwrite("local_fpgas", &CompServ::local_fpgas);
		cl.def_readwrite("lsb", &CompServ::lsb);

		compserv_addons(cl);
	}
	{ // Loss file:eddl/losses/loss.h line:32
		pybind11::class_<Loss, std::shared_ptr<Loss>, PyCallBack_Loss> cl(M(""), "Loss", "");

		cl.def_readwrite("name", &Loss::name);
		cl.def("delta", (void (Loss::*)(class Tensor *, class Tensor *, class Tensor *)) &Loss::delta, "C++: Loss::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (Loss::*)(class Tensor *, class Tensor *)) &Loss::value, "C++: Loss::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class Loss & (Loss::*)(const class Loss &)) &Loss::operator=, "C++: Loss::operator=(const class Loss &) --> class Loss &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LMeanSquaredError file:eddl/losses/loss.h line:43
		pybind11::class_<LMeanSquaredError, std::shared_ptr<LMeanSquaredError>, PyCallBack_LMeanSquaredError, Loss> cl(M(""), "LMeanSquaredError", "");

		cl.def( pybind11::init( [](){ return new LMeanSquaredError(); }, [](){ return new PyCallBack_LMeanSquaredError(); } ) );
		cl.def("delta", (void (LMeanSquaredError::*)(class Tensor *, class Tensor *, class Tensor *)) &LMeanSquaredError::delta, "C++: LMeanSquaredError::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (LMeanSquaredError::*)(class Tensor *, class Tensor *)) &LMeanSquaredError::value, "C++: LMeanSquaredError::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class LMeanSquaredError & (LMeanSquaredError::*)(const class LMeanSquaredError &)) &LMeanSquaredError::operator=, "C++: LMeanSquaredError::operator=(const class LMeanSquaredError &) --> class LMeanSquaredError &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LCrossEntropy file:eddl/losses/loss.h line:52
		pybind11::class_<LCrossEntropy, std::shared_ptr<LCrossEntropy>, PyCallBack_LCrossEntropy, Loss> cl(M(""), "LCrossEntropy", "");

		cl.def( pybind11::init( [](){ return new LCrossEntropy(); }, [](){ return new PyCallBack_LCrossEntropy(); } ) );
		cl.def("delta", (void (LCrossEntropy::*)(class Tensor *, class Tensor *, class Tensor *)) &LCrossEntropy::delta, "C++: LCrossEntropy::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (LCrossEntropy::*)(class Tensor *, class Tensor *)) &LCrossEntropy::value, "C++: LCrossEntropy::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class LCrossEntropy & (LCrossEntropy::*)(const class LCrossEntropy &)) &LCrossEntropy::operator=, "C++: LCrossEntropy::operator=(const class LCrossEntropy &) --> class LCrossEntropy &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LSoftCrossEntropy file:eddl/losses/loss.h line:61
		pybind11::class_<LSoftCrossEntropy, std::shared_ptr<LSoftCrossEntropy>, PyCallBack_LSoftCrossEntropy, Loss> cl(M(""), "LSoftCrossEntropy", "");

		cl.def( pybind11::init( [](){ return new LSoftCrossEntropy(); }, [](){ return new PyCallBack_LSoftCrossEntropy(); } ) );
		cl.def("delta", (void (LSoftCrossEntropy::*)(class Tensor *, class Tensor *, class Tensor *)) &LSoftCrossEntropy::delta, "C++: LSoftCrossEntropy::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (LSoftCrossEntropy::*)(class Tensor *, class Tensor *)) &LSoftCrossEntropy::value, "C++: LSoftCrossEntropy::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class LSoftCrossEntropy & (LSoftCrossEntropy::*)(const class LSoftCrossEntropy &)) &LSoftCrossEntropy::operator=, "C++: LSoftCrossEntropy::operator=(const class LSoftCrossEntropy &) --> class LSoftCrossEntropy &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Metric file:eddl/metrics/metric.h line:34
		pybind11::class_<Metric, std::shared_ptr<Metric>, PyCallBack_Metric> cl(M(""), "Metric", "");

		cl.def_readwrite("name", &Metric::name);
		cl.def("value", (float (Metric::*)(class Tensor *, class Tensor *)) &Metric::value, "C++: Metric::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class Metric & (Metric::*)(const class Metric &)) &Metric::operator=, "C++: Metric::operator=(const class Metric &) --> class Metric &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // MMeanSquaredError file:eddl/metrics/metric.h line:44
		pybind11::class_<MMeanSquaredError, std::shared_ptr<MMeanSquaredError>, PyCallBack_MMeanSquaredError, Metric> cl(M(""), "MMeanSquaredError", "");

		cl.def( pybind11::init( [](){ return new MMeanSquaredError(); }, [](){ return new PyCallBack_MMeanSquaredError(); } ) );
		cl.def("value", (float (MMeanSquaredError::*)(class Tensor *, class Tensor *)) &MMeanSquaredError::value, "C++: MMeanSquaredError::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class MMeanSquaredError & (MMeanSquaredError::*)(const class MMeanSquaredError &)) &MMeanSquaredError::operator=, "C++: MMeanSquaredError::operator=(const class MMeanSquaredError &) --> class MMeanSquaredError &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // MCategoricalAccuracy file:eddl/metrics/metric.h line:52
		pybind11::class_<MCategoricalAccuracy, std::shared_ptr<MCategoricalAccuracy>, PyCallBack_MCategoricalAccuracy, Metric> cl(M(""), "MCategoricalAccuracy", "");

		cl.def( pybind11::init( [](){ return new MCategoricalAccuracy(); }, [](){ return new PyCallBack_MCategoricalAccuracy(); } ) );
		cl.def("value", (float (MCategoricalAccuracy::*)(class Tensor *, class Tensor *)) &MCategoricalAccuracy::value, "C++: MCategoricalAccuracy::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class MCategoricalAccuracy & (MCategoricalAccuracy::*)(const class MCategoricalAccuracy &)) &MCategoricalAccuracy::operator=, "C++: MCategoricalAccuracy::operator=(const class MCategoricalAccuracy &) --> class MCategoricalAccuracy &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Layer file:eddl/layers/layer.h line:35
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
		cl.def_readwrite("isplot", &Layer::isplot);
		cl.def("initialize", (void (Layer::*)()) &Layer::initialize, "C++: Layer::initialize() --> void");
		cl.def("reset", (void (Layer::*)()) &Layer::reset, "C++: Layer::reset() --> void");
		cl.def("info", (void (Layer::*)()) &Layer::info, "C++: Layer::info() --> void");
		cl.def("setmode", (void (Layer::*)(int)) &Layer::setmode, "C++: Layer::setmode(int) --> void", pybind11::arg("m"));
		cl.def("getWeights", (class Tensor * (Layer::*)()) &Layer::getWeights, "C++: Layer::getWeights() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setWeights", (class Tensor * (Layer::*)(class Tensor)) &Layer::setWeights, "C++: Layer::setWeights(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("getBias", (class Tensor * (Layer::*)()) &Layer::getBias, "C++: Layer::getBias() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setBias", (class Tensor * (Layer::*)(class Tensor)) &Layer::setBias, "C++: Layer::setBias(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("resize", (void (Layer::*)(int)) &Layer::resize, "C++: Layer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("addchild", (void (Layer::*)(class Layer *)) &Layer::addchild, "C++: Layer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (Layer::*)(class Layer *)) &Layer::addparent, "C++: Layer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("forward", (void (Layer::*)()) &Layer::forward, "C++: Layer::forward() --> void");
		cl.def("backward", (void (Layer::*)()) &Layer::backward, "C++: Layer::backward() --> void");
		cl.def("assign", (class Layer & (Layer::*)(const class Layer &)) &Layer::operator=, "C++: Layer::operator=(const class Layer &) --> class Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		layer_addons(cl);
	}
	{ // LinLayer file:eddl/layers/layer.h line:99
		pybind11::class_<LinLayer, std::shared_ptr<LinLayer>, PyCallBack_LinLayer, Layer> cl(M(""), "LinLayer", "//////////////////////////////////////\n//////////////////////////////////////");

		cl.def( pybind11::init( [](PyCallBack_LinLayer const &o){ return new PyCallBack_LinLayer(o); } ) );
		cl.def( pybind11::init( [](LinLayer const &o){ return new LinLayer(o); } ) );
		cl.def("addchild", (void (LinLayer::*)(class Layer *)) &LinLayer::addchild, "C++: LinLayer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (LinLayer::*)(class Layer *)) &LinLayer::addparent, "C++: LinLayer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("resize", (void (LinLayer::*)(int)) &LinLayer::resize, "C++: LinLayer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("forward", (void (LinLayer::*)()) &LinLayer::forward, "C++: LinLayer::forward() --> void");
		cl.def("backward", (void (LinLayer::*)()) &LinLayer::backward, "C++: LinLayer::backward() --> void");
		cl.def("assign", (class LinLayer & (LinLayer::*)(const class LinLayer &)) &LinLayer::operator=, "C++: LinLayer::operator=(const class LinLayer &) --> class LinLayer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: eddl/layers/layer.cpp
#include <eddl/descriptors/descriptors.h>
#include <eddl/layers/conv/layer_conv.h>
#include <eddl/layers/core/layer_core.h>
#include <eddl/layers/layer.h>
#include <eddl/layers/noise/layer_noise.h>
#include <eddl/layers/pool/layer_pool.h>
#include <eddl/optimizers/optim.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <lactivation_addons.h>
#include <lconv_addons.h>
#include <ldense_addons.h>
#include <lembedding_addons.h>
#include <lgaussiannoise_addons.h>
#include <linput_addons.h>
#include <lmaxpool_addons.h>
#include <lreshape_addons.h>
#include <ltensor_addons.h>
#include <memory>
#include <sstream> // __str__
#include <stdio.h>
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

// MLayer file:eddl/layers/layer.h line:127
struct PyCallBack_MLayer : public MLayer {
	using MLayer::MLayer;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return MLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const MLayer *>(this), "info");
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
};

// LTensor file:eddl/layers/core/layer_core.h line:35
struct PyCallBack_LTensor : public LTensor {
	using LTensor::LTensor;

	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "info");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::info();
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTensor::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTensor *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
};

// LInput file:eddl/layers/core/layer_core.h line:70
struct PyCallBack_LInput : public LInput {
	using LInput::LInput;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LInput::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LInput::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LInput::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LInput *>(this), "info");
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
};

// LEmbedding file:eddl/layers/core/layer_core.h line:91
struct PyCallBack_LEmbedding : public LEmbedding {
	using LEmbedding::LEmbedding;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LEmbedding::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LEmbedding::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LEmbedding::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LEmbedding *>(this), "info");
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
};

// LDense file:eddl/layers/core/layer_core.h line:114
struct PyCallBack_LDense : public LDense {
	using LDense::LDense;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDense::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDense::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDense::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDense *>(this), "info");
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
};

// LActivation file:eddl/layers/core/layer_core.h line:143
struct PyCallBack_LActivation : public LActivation {
	using LActivation::LActivation;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LActivation::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LActivation::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LActivation::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LActivation *>(this), "info");
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
};

// LReshape file:eddl/layers/core/layer_core.h line:165
struct PyCallBack_LReshape : public LReshape {
	using LReshape::LReshape;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LReshape::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LReshape::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LReshape::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LReshape *>(this), "info");
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
};

// LTranspose file:eddl/layers/core/layer_core.h line:190
struct PyCallBack_LTranspose : public LTranspose {
	using LTranspose::LTranspose;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTranspose::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTranspose::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LTranspose::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LTranspose *>(this), "info");
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
};

// LDropout file:eddl/layers/core/layer_core.h line:215
struct PyCallBack_LDropout : public LDropout {
	using LDropout::LDropout;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDropout::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDropout::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LDropout::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LDropout *>(this), "info");
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
};

// LBatchNorm file:eddl/layers/core/layer_core.h line:241
struct PyCallBack_LBatchNorm : public LBatchNorm {
	using LBatchNorm::LBatchNorm;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LBatchNorm::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LBatchNorm::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LBatchNorm::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LBatchNorm *>(this), "info");
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
};

// LConv file:eddl/layers/conv/layer_conv.h line:36
struct PyCallBack_LConv : public LConv {
	using LConv::LConv;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConv::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConv::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LConv::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConv *>(this), "info");
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
};

// LConvT file:eddl/layers/conv/layer_conv.h line:68
struct PyCallBack_LConvT : public LConvT {
	using LConvT::LConvT;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LConvT *>(this), "info");
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
};

// LUpSampling file:eddl/layers/conv/layer_conv.h line:95
struct PyCallBack_LUpSampling : public LUpSampling {
	using LUpSampling::LUpSampling;

	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LUpSampling *>(this), "info");
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
};

// LPool file:eddl/layers/pool/layer_pool.h line:37
struct PyCallBack_LPool : public LPool {
	using LPool::LPool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LPool *>(this), "info");
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
};

// LMaxPool file:eddl/layers/pool/layer_pool.h line:49
struct PyCallBack_LMaxPool : public LMaxPool {
	using LMaxPool::LMaxPool;

	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaxPool::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaxPool::backward();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LMaxPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LMaxPool *>(this), "info");
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
};

// LAveragePool file:eddl/layers/pool/layer_pool.h line:78
struct PyCallBack_LAveragePool : public LAveragePool {
	using LAveragePool::LAveragePool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LAveragePool *>(this), "info");
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
};

// LGlobalMaxPool file:eddl/layers/pool/layer_pool.h line:102
struct PyCallBack_LGlobalMaxPool : public LGlobalMaxPool {
	using LGlobalMaxPool::LGlobalMaxPool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalMaxPool *>(this), "info");
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
};

// LGlobalAveragePool file:eddl/layers/pool/layer_pool.h line:125
struct PyCallBack_LGlobalAveragePool : public LGlobalAveragePool {
	using LGlobalAveragePool::LGlobalAveragePool;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LPool::resize(a0);
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::backward();
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGlobalAveragePool *>(this), "info");
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
};

// LGaussianNoise file:eddl/layers/noise/layer_noise.h line:36
struct PyCallBack_LGaussianNoise : public LGaussianNoise {
	using LGaussianNoise::LGaussianNoise;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LGaussianNoise::resize(a0);
	}
	void forward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "forward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LGaussianNoise::forward();
	}
	void backward() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "backward");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LGaussianNoise::backward();
	}
	void addchild(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "addchild");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addchild(a0);
	}
	void addparent(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "addparent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return LinLayer::addparent(a0);
	}
	void info() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const LGaussianNoise *>(this), "info");
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
};

// Optimizer file:eddl/optimizers/optim.h line:38
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

// SGD file:eddl/optimizers/optim.h line:55
struct PyCallBack_SGD : public SGD {
	using SGD::SGD;

	class Optimizer * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const SGD *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return SGD::clone();
	}
	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const SGD *>(this), "applygrads");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SGD::applygrads(a0);
	}
};

void bind_eddl_layers_layer(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // MLayer file:eddl/layers/layer.h line:127
		pybind11::class_<MLayer, std::shared_ptr<MLayer>, PyCallBack_MLayer, Layer> cl(M(""), "MLayer", "//////////////////////////////////////\n//////////////////////////////////////");

		cl.def("addchild", (void (MLayer::*)(class Layer *)) &MLayer::addchild, "C++: MLayer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (MLayer::*)(class Layer *)) &MLayer::addparent, "C++: MLayer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("resize", (void (MLayer::*)(int)) &MLayer::resize, "C++: MLayer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("forward", (void (MLayer::*)()) &MLayer::forward, "C++: MLayer::forward() --> void");
		cl.def("backward", (void (MLayer::*)()) &MLayer::backward, "C++: MLayer::backward() --> void");
		cl.def("assign", (class MLayer & (MLayer::*)(const class MLayer &)) &MLayer::operator=, "C++: MLayer::operator=(const class MLayer &) --> class MLayer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LTensor file:eddl/layers/core/layer_core.h line:35
		pybind11::class_<LTensor, std::shared_ptr<LTensor>, PyCallBack_LTensor, LinLayer> cl(M(""), "LTensor", "Tensor Layer");

		cl.def( pybind11::init<class Layer *>(), pybind11::arg("l") );

		cl.def("info", (void (LTensor::*)()) &LTensor::info, "C++: LTensor::info() --> void");
		cl.def("forward", (void (LTensor::*)()) &LTensor::forward, "C++: LTensor::forward() --> void");
		cl.def("backward", (void (LTensor::*)()) &LTensor::backward, "C++: LTensor::backward() --> void");
		cl.def("resize", (void (LTensor::*)(int)) &LTensor::resize, "C++: LTensor::resize(int) --> void", pybind11::arg("batch"));
		cl.def("__add__", (class LTensor (LTensor::*)(class LTensor)) &LTensor::operator+, "C++: LTensor::operator+(class LTensor) --> class LTensor", pybind11::arg("L"));
		cl.def("assign", (class LTensor & (LTensor::*)(const class LTensor &)) &LTensor::operator=, "C++: LTensor::operator=(const class LTensor &) --> class LTensor &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		ltensor_addons(cl);
	}
	{ // LInput file:eddl/layers/core/layer_core.h line:70
		pybind11::class_<LInput, std::shared_ptr<LInput>, PyCallBack_LInput, LinLayer> cl(M(""), "LInput", "INPUT Layer");

		cl.def("forward", (void (LInput::*)()) &LInput::forward, "C++: LInput::forward() --> void");
		cl.def("backward", (void (LInput::*)()) &LInput::backward, "C++: LInput::backward() --> void");
		cl.def("resize", (void (LInput::*)(int)) &LInput::resize, "C++: LInput::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LInput & (LInput::*)(const class LInput &)) &LInput::operator=, "C++: LInput::operator=(const class LInput &) --> class LInput &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		linput_addons(cl);
	}
	{ // LEmbedding file:eddl/layers/core/layer_core.h line:91
		pybind11::class_<LEmbedding, std::shared_ptr<LEmbedding>, PyCallBack_LEmbedding, LinLayer> cl(M(""), "LEmbedding", "EMBEDDING Layer");

		cl.def_readwrite("input_dim", &LEmbedding::input_dim);
		cl.def_readwrite("output_dim", &LEmbedding::output_dim);
		cl.def("forward", (void (LEmbedding::*)()) &LEmbedding::forward, "C++: LEmbedding::forward() --> void");
		cl.def("backward", (void (LEmbedding::*)()) &LEmbedding::backward, "C++: LEmbedding::backward() --> void");
		cl.def("resize", (void (LEmbedding::*)(int)) &LEmbedding::resize, "C++: LEmbedding::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LEmbedding & (LEmbedding::*)(const class LEmbedding &)) &LEmbedding::operator=, "C++: LEmbedding::operator=(const class LEmbedding &) --> class LEmbedding &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lembedding_addons(cl);
	}
	{ // LDense file:eddl/layers/core/layer_core.h line:114
		pybind11::class_<LDense, std::shared_ptr<LDense>, PyCallBack_LDense, LinLayer> cl(M(""), "LDense", "Dense Layer");

		cl.def_readwrite("ndim", &LDense::ndim);
		cl.def_readwrite("use_bias", &LDense::use_bias);
		cl.def("forward", (void (LDense::*)()) &LDense::forward, "C++: LDense::forward() --> void");
		cl.def("backward", (void (LDense::*)()) &LDense::backward, "C++: LDense::backward() --> void");
		cl.def("resize", (void (LDense::*)(int)) &LDense::resize, "C++: LDense::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LDense & (LDense::*)(const class LDense &)) &LDense::operator=, "C++: LDense::operator=(const class LDense &) --> class LDense &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		ldense_addons(cl);
	}
	{ // LActivation file:eddl/layers/core/layer_core.h line:143
		pybind11::class_<LActivation, std::shared_ptr<LActivation>, PyCallBack_LActivation, LinLayer> cl(M(""), "LActivation", "Activation Layer");

		cl.def_readwrite("act", &LActivation::act);
		cl.def("forward", (void (LActivation::*)()) &LActivation::forward, "C++: LActivation::forward() --> void");
		cl.def("backward", (void (LActivation::*)()) &LActivation::backward, "C++: LActivation::backward() --> void");
		cl.def("resize", (void (LActivation::*)(int)) &LActivation::resize, "C++: LActivation::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LActivation & (LActivation::*)(const class LActivation &)) &LActivation::operator=, "C++: LActivation::operator=(const class LActivation &) --> class LActivation &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lactivation_addons(cl);
	}
	{ // LReshape file:eddl/layers/core/layer_core.h line:165
		pybind11::class_<LReshape, std::shared_ptr<LReshape>, PyCallBack_LReshape, LinLayer> cl(M(""), "LReshape", "Reshape Layer");

		cl.def_readwrite("ls", &LReshape::ls);
		cl.def("forward", (void (LReshape::*)()) &LReshape::forward, "C++: LReshape::forward() --> void");
		cl.def("backward", (void (LReshape::*)()) &LReshape::backward, "C++: LReshape::backward() --> void");
		cl.def("resize", (void (LReshape::*)(int)) &LReshape::resize, "C++: LReshape::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LReshape & (LReshape::*)(const class LReshape &)) &LReshape::operator=, "C++: LReshape::operator=(const class LReshape &) --> class LReshape &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lreshape_addons(cl);
	}
	{ // LTranspose file:eddl/layers/core/layer_core.h line:190
		pybind11::class_<LTranspose, std::shared_ptr<LTranspose>, PyCallBack_LTranspose, LinLayer> cl(M(""), "LTranspose", "Transpose Layer");

		cl.def_readwrite("dims", &LTranspose::dims);
		cl.def_readwrite("rdims", &LTranspose::rdims);
		cl.def("forward", (void (LTranspose::*)()) &LTranspose::forward, "C++: LTranspose::forward() --> void");
		cl.def("backward", (void (LTranspose::*)()) &LTranspose::backward, "C++: LTranspose::backward() --> void");
		cl.def("resize", (void (LTranspose::*)(int)) &LTranspose::resize, "C++: LTranspose::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LTranspose & (LTranspose::*)(const class LTranspose &)) &LTranspose::operator=, "C++: LTranspose::operator=(const class LTranspose &) --> class LTranspose &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LDropout file:eddl/layers/core/layer_core.h line:215
		pybind11::class_<LDropout, std::shared_ptr<LDropout>, PyCallBack_LDropout, LinLayer> cl(M(""), "LDropout", "Drop-out Layer");

		cl.def_readwrite("df", &LDropout::df);
		cl.def("forward", (void (LDropout::*)()) &LDropout::forward, "C++: LDropout::forward() --> void");
		cl.def("backward", (void (LDropout::*)()) &LDropout::backward, "C++: LDropout::backward() --> void");
		cl.def("resize", (void (LDropout::*)(int)) &LDropout::resize, "C++: LDropout::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LDropout & (LDropout::*)(const class LDropout &)) &LDropout::operator=, "C++: LDropout::operator=(const class LDropout &) --> class LDropout &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LBatchNorm file:eddl/layers/core/layer_core.h line:241
		pybind11::class_<LBatchNorm, std::shared_ptr<LBatchNorm>, PyCallBack_LBatchNorm, LinLayer> cl(M(""), "LBatchNorm", "BatchNormalization Layer");

		cl.def_readwrite("momentum", &LBatchNorm::momentum);
		cl.def_readwrite("epsilon", &LBatchNorm::epsilon);
		cl.def_readwrite("affine", &LBatchNorm::affine);
		cl.def("forward", (void (LBatchNorm::*)()) &LBatchNorm::forward, "C++: LBatchNorm::forward() --> void");
		cl.def("backward", (void (LBatchNorm::*)()) &LBatchNorm::backward, "C++: LBatchNorm::backward() --> void");
		cl.def("resize", (void (LBatchNorm::*)(int)) &LBatchNorm::resize, "C++: LBatchNorm::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LBatchNorm & (LBatchNorm::*)(const class LBatchNorm &)) &LBatchNorm::operator=, "C++: LBatchNorm::operator=(const class LBatchNorm &) --> class LBatchNorm &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LConv file:eddl/layers/conv/layer_conv.h line:36
		pybind11::class_<LConv, std::shared_ptr<LConv>, PyCallBack_LConv, LinLayer> cl(M(""), "LConv", "Conv2D Layer");

		cl.def("forward", (void (LConv::*)()) &LConv::forward, "C++: LConv::forward() --> void");
		cl.def("backward", (void (LConv::*)()) &LConv::backward, "C++: LConv::backward() --> void");
		cl.def("resize", (void (LConv::*)(int)) &LConv::resize, "C++: LConv::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LConv & (LConv::*)(const class LConv &)) &LConv::operator=, "C++: LConv::operator=(const class LConv &) --> class LConv &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lconv_addons(cl);
	}
	{ // LConvT file:eddl/layers/conv/layer_conv.h line:68
		pybind11::class_<LConvT, std::shared_ptr<LConvT>, PyCallBack_LConvT, LinLayer> cl(M(""), "LConvT", "ConvT2D Layer");

		cl.def("assign", (class LConvT & (LConvT::*)(const class LConvT &)) &LConvT::operator=, "C++: LConvT::operator=(const class LConvT &) --> class LConvT &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LUpSampling file:eddl/layers/conv/layer_conv.h line:95
		pybind11::class_<LUpSampling, std::shared_ptr<LUpSampling>, PyCallBack_LUpSampling, LinLayer> cl(M(""), "LUpSampling", "UpSampling2D Layer");

		cl.def_readwrite("size", &LUpSampling::size);
		cl.def_readwrite("interpolation", &LUpSampling::interpolation);
		cl.def("assign", (class LUpSampling & (LUpSampling::*)(const class LUpSampling &)) &LUpSampling::operator=, "C++: LUpSampling::operator=(const class LUpSampling &) --> class LUpSampling &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LPool file:eddl/layers/pool/layer_pool.h line:37
		pybind11::class_<LPool, std::shared_ptr<LPool>, PyCallBack_LPool, LinLayer> cl(M(""), "LPool", "Pool2D Layer");

		cl.def("resize", (void (LPool::*)(int)) &LPool::resize, "C++: LPool::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LPool & (LPool::*)(const class LPool &)) &LPool::operator=, "C++: LPool::operator=(const class LPool &) --> class LPool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LMaxPool file:eddl/layers/pool/layer_pool.h line:49
		pybind11::class_<LMaxPool, std::shared_ptr<LMaxPool>, PyCallBack_LMaxPool, LPool> cl(M(""), "LMaxPool", "MaxPool2D Layer");

		cl.def("forward", (void (LMaxPool::*)()) &LMaxPool::forward, "C++: LMaxPool::forward() --> void");
		cl.def("backward", (void (LMaxPool::*)()) &LMaxPool::backward, "C++: LMaxPool::backward() --> void");
		cl.def("resize", (void (LMaxPool::*)(int)) &LMaxPool::resize, "C++: LMaxPool::resize(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class LMaxPool & (LMaxPool::*)(const class LMaxPool &)) &LMaxPool::operator=, "C++: LMaxPool::operator=(const class LMaxPool &) --> class LMaxPool &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lmaxpool_addons(cl);
	}
	{ // LAveragePool file:eddl/layers/pool/layer_pool.h line:78
		pybind11::class_<LAveragePool, std::shared_ptr<LAveragePool>, PyCallBack_LAveragePool, LPool> cl(M(""), "LAveragePool", "AveragePool2D Layer");

		cl.def("assign", (class LAveragePool & (LAveragePool::*)(const class LAveragePool &)) &LAveragePool::operator=, "C++: LAveragePool::operator=(const class LAveragePool &) --> class LAveragePool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LGlobalMaxPool file:eddl/layers/pool/layer_pool.h line:102
		pybind11::class_<LGlobalMaxPool, std::shared_ptr<LGlobalMaxPool>, PyCallBack_LGlobalMaxPool, LPool> cl(M(""), "LGlobalMaxPool", "GlobalMaxPool2D Layer");

		cl.def("assign", (class LGlobalMaxPool & (LGlobalMaxPool::*)(const class LGlobalMaxPool &)) &LGlobalMaxPool::operator=, "C++: LGlobalMaxPool::operator=(const class LGlobalMaxPool &) --> class LGlobalMaxPool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LGlobalAveragePool file:eddl/layers/pool/layer_pool.h line:125
		pybind11::class_<LGlobalAveragePool, std::shared_ptr<LGlobalAveragePool>, PyCallBack_LGlobalAveragePool, LPool> cl(M(""), "LGlobalAveragePool", "GlobalAveragePool2D Layer");

		cl.def("assign", (class LGlobalAveragePool & (LGlobalAveragePool::*)(const class LGlobalAveragePool &)) &LGlobalAveragePool::operator=, "C++: LGlobalAveragePool::operator=(const class LGlobalAveragePool &) --> class LGlobalAveragePool &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // LGaussianNoise file:eddl/layers/noise/layer_noise.h line:36
		pybind11::class_<LGaussianNoise, std::shared_ptr<LGaussianNoise>, PyCallBack_LGaussianNoise, LinLayer> cl(M(""), "LGaussianNoise", "GaussianNoise Layer");

		cl.def_readwrite("stdev", &LGaussianNoise::stdev);
		cl.def("resize", (void (LGaussianNoise::*)(int)) &LGaussianNoise::resize, "C++: LGaussianNoise::resize(int) --> void", pybind11::arg("batch"));
		cl.def("forward", (void (LGaussianNoise::*)()) &LGaussianNoise::forward, "C++: LGaussianNoise::forward() --> void");
		cl.def("backward", (void (LGaussianNoise::*)()) &LGaussianNoise::backward, "C++: LGaussianNoise::backward() --> void");
		cl.def("assign", (class LGaussianNoise & (LGaussianNoise::*)(const class LGaussianNoise &)) &LGaussianNoise::operator=, "C++: LGaussianNoise::operator=(const class LGaussianNoise &) --> class LGaussianNoise &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		lgaussiannoise_addons(cl);
	}
	{ // Optimizer file:eddl/optimizers/optim.h line:38
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
	{ // SGD file:eddl/optimizers/optim.h line:55
		pybind11::class_<SGD, std::shared_ptr<SGD>, PyCallBack_SGD, Optimizer> cl(M(""), "SGD", "");

		cl.def( pybind11::init( [](){ return new SGD(); }, [](){ return new PyCallBack_SGD(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new SGD(a0); }, [](float const & a0){ return new PyCallBack_SGD(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new SGD(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_SGD(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new SGD(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_SGD(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init<float, float, float, bool>(), pybind11::arg("lr"), pybind11::arg("momentum"), pybind11::arg("weight_decay"), pybind11::arg("nesterov") );

		cl.def_readwrite("lr", &SGD::lr);
		cl.def_readwrite("mu", &SGD::mu);
		cl.def_readwrite("weight_decay", &SGD::weight_decay);
		cl.def_readwrite("nesterov", &SGD::nesterov);
		cl.def_readwrite("mT", &SGD::mT);
		cl.def("clone", (class Optimizer * (SGD::*)()) &SGD::clone, "C++: SGD::clone() --> class Optimizer *", pybind11::return_value_policy::automatic);
		cl.def("applygrads", (void (SGD::*)(int)) &SGD::applygrads, "C++: SGD::applygrads(int) --> void", pybind11::arg("batch"));
		cl.def("assign", (class SGD & (SGD::*)(const class SGD &)) &SGD::operator=, "C++: SGD::operator=(const class SGD &) --> class SGD &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: eddl/optimizers/optim.cpp
#include <eddl/compserv.h>
#include <eddl/layers/layer.h>
#include <eddl/losses/loss.h>
#include <eddl/metrics/metric.h>
#include <eddl/net.h>
#include <eddl/optimizers/optim.h>
#include <eddl/tensor/tensor.h>
#include <iterator>
#include <memory>
#include <net_addons.h>
#include <sstream> // __str__
#include <stdio.h>
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

// Adam file:eddl/optimizers/optim.h line:76
struct PyCallBack_Adam : public Adam {
	using Adam::Adam;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adam *>(this), "applygrads");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Adam *>(this), "clone");
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

// AdaDelta file:eddl/optimizers/optim.h line:100
struct PyCallBack_AdaDelta : public AdaDelta {
	using AdaDelta::AdaDelta;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const AdaDelta *>(this), "applygrads");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const AdaDelta *>(this), "clone");
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

// Adagrad file:eddl/optimizers/optim.h line:121
struct PyCallBack_Adagrad : public Adagrad {
	using Adagrad::Adagrad;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adagrad *>(this), "applygrads");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Adagrad *>(this), "clone");
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

// Adamax file:eddl/optimizers/optim.h line:141
struct PyCallBack_Adamax : public Adamax {
	using Adamax::Adamax;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Adamax *>(this), "applygrads");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Adamax *>(this), "clone");
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

// Nadam file:eddl/optimizers/optim.h line:163
struct PyCallBack_Nadam : public Nadam {
	using Nadam::Nadam;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Nadam *>(this), "applygrads");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const Nadam *>(this), "clone");
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

// RMSProp file:eddl/optimizers/optim.h line:185
struct PyCallBack_RMSProp : public RMSProp {
	using RMSProp::RMSProp;

	void applygrads(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const RMSProp *>(this), "applygrads");
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
		pybind11::function overload = pybind11::get_overload(static_cast<const RMSProp *>(this), "clone");
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

void bind_eddl_optimizers_optim(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Adam file:eddl/optimizers/optim.h line:76
		pybind11::class_<Adam, std::shared_ptr<Adam>, PyCallBack_Adam, Optimizer> cl(M(""), "Adam", "");

		cl.def( pybind11::init( [](){ return new Adam(); }, [](){ return new PyCallBack_Adam(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Adam(a0); }, [](float const & a0){ return new PyCallBack_Adam(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Adam(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Adam(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new Adam(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_Adam(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3){ return new Adam(a0, a1, a2, a3); }, [](float const & a0, float const & a1, float const & a2, float const & a3){ return new PyCallBack_Adam(a0, a1, a2, a3); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3, float const & a4){ return new Adam(a0, a1, a2, a3, a4); }, [](float const & a0, float const & a1, float const & a2, float const & a3, float const & a4){ return new PyCallBack_Adam(a0, a1, a2, a3, a4); } ), "doc");
		cl.def( pybind11::init<float, float, float, float, float, bool>(), pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("weight_decay"), pybind11::arg("amsgrad") );

		cl.def_readwrite("lr", &Adam::lr);
		cl.def_readwrite("beta_1", &Adam::beta_1);
		cl.def_readwrite("beta_2", &Adam::beta_2);
		cl.def_readwrite("epsilon", &Adam::epsilon);
		cl.def_readwrite("weight_decay", &Adam::weight_decay);
		cl.def_readwrite("amsgrad", &Adam::amsgrad);
		cl.def_readwrite("mT", &Adam::mT);
		cl.def("assign", (class Adam & (Adam::*)(const class Adam &)) &Adam::operator=, "C++: Adam::operator=(const class Adam &) --> class Adam &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // AdaDelta file:eddl/optimizers/optim.h line:100
		pybind11::class_<AdaDelta, std::shared_ptr<AdaDelta>, PyCallBack_AdaDelta, Optimizer> cl(M(""), "AdaDelta", "");

		cl.def( pybind11::init( [](){ return new AdaDelta(); }, [](){ return new PyCallBack_AdaDelta(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new AdaDelta(a0); }, [](float const & a0){ return new PyCallBack_AdaDelta(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new AdaDelta(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_AdaDelta(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new AdaDelta(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_AdaDelta(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init<float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("rho"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def_readwrite("lr", &AdaDelta::lr);
		cl.def_readwrite("rho", &AdaDelta::rho);
		cl.def_readwrite("epsilon", &AdaDelta::epsilon);
		cl.def_readwrite("weight_decay", &AdaDelta::weight_decay);
		cl.def_readwrite("mT", &AdaDelta::mT);
		cl.def("assign", (class AdaDelta & (AdaDelta::*)(const class AdaDelta &)) &AdaDelta::operator=, "C++: AdaDelta::operator=(const class AdaDelta &) --> class AdaDelta &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Adagrad file:eddl/optimizers/optim.h line:121
		pybind11::class_<Adagrad, std::shared_ptr<Adagrad>, PyCallBack_Adagrad, Optimizer> cl(M(""), "Adagrad", "");

		cl.def( pybind11::init( [](){ return new Adagrad(); }, [](){ return new PyCallBack_Adagrad(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Adagrad(a0); }, [](float const & a0){ return new PyCallBack_Adagrad(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Adagrad(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Adagrad(a0, a1); } ), "doc");
		cl.def( pybind11::init<float, float, float>(), pybind11::arg("lr"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def_readwrite("lr", &Adagrad::lr);
		cl.def_readwrite("epsilon", &Adagrad::epsilon);
		cl.def_readwrite("weight_decay", &Adagrad::weight_decay);
		cl.def_readwrite("mT", &Adagrad::mT);
		cl.def("assign", (class Adagrad & (Adagrad::*)(const class Adagrad &)) &Adagrad::operator=, "C++: Adagrad::operator=(const class Adagrad &) --> class Adagrad &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Adamax file:eddl/optimizers/optim.h line:141
		pybind11::class_<Adamax, std::shared_ptr<Adamax>, PyCallBack_Adamax, Optimizer> cl(M(""), "Adamax", "");

		cl.def( pybind11::init( [](){ return new Adamax(); }, [](){ return new PyCallBack_Adamax(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Adamax(a0); }, [](float const & a0){ return new PyCallBack_Adamax(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Adamax(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Adamax(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new Adamax(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_Adamax(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3){ return new Adamax(a0, a1, a2, a3); }, [](float const & a0, float const & a1, float const & a2, float const & a3){ return new PyCallBack_Adamax(a0, a1, a2, a3); } ), "doc");
		cl.def( pybind11::init<float, float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def_readwrite("lr", &Adamax::lr);
		cl.def_readwrite("beta_1", &Adamax::beta_1);
		cl.def_readwrite("beta_2", &Adamax::beta_2);
		cl.def_readwrite("epsilon", &Adamax::epsilon);
		cl.def_readwrite("weight_decay", &Adamax::weight_decay);
		cl.def_readwrite("mT", &Adamax::mT);
		cl.def("assign", (class Adamax & (Adamax::*)(const class Adamax &)) &Adamax::operator=, "C++: Adamax::operator=(const class Adamax &) --> class Adamax &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Nadam file:eddl/optimizers/optim.h line:163
		pybind11::class_<Nadam, std::shared_ptr<Nadam>, PyCallBack_Nadam, Optimizer> cl(M(""), "Nadam", "");

		cl.def( pybind11::init( [](){ return new Nadam(); }, [](){ return new PyCallBack_Nadam(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new Nadam(a0); }, [](float const & a0){ return new PyCallBack_Nadam(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new Nadam(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_Nadam(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new Nadam(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_Nadam(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2, float const & a3){ return new Nadam(a0, a1, a2, a3); }, [](float const & a0, float const & a1, float const & a2, float const & a3){ return new PyCallBack_Nadam(a0, a1, a2, a3); } ), "doc");
		cl.def( pybind11::init<float, float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("beta_1"), pybind11::arg("beta_2"), pybind11::arg("epsilon"), pybind11::arg("schedule_decay") );

		cl.def_readwrite("lr", &Nadam::lr);
		cl.def_readwrite("beta_1", &Nadam::beta_1);
		cl.def_readwrite("beta_2", &Nadam::beta_2);
		cl.def_readwrite("epsilon", &Nadam::epsilon);
		cl.def_readwrite("schedule_decay", &Nadam::schedule_decay);
		cl.def_readwrite("mT", &Nadam::mT);
		cl.def("assign", (class Nadam & (Nadam::*)(const class Nadam &)) &Nadam::operator=, "C++: Nadam::operator=(const class Nadam &) --> class Nadam &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // RMSProp file:eddl/optimizers/optim.h line:185
		pybind11::class_<RMSProp, std::shared_ptr<RMSProp>, PyCallBack_RMSProp, Optimizer> cl(M(""), "RMSProp", "");

		cl.def( pybind11::init( [](){ return new RMSProp(); }, [](){ return new PyCallBack_RMSProp(); } ), "doc");
		cl.def( pybind11::init( [](float const & a0){ return new RMSProp(a0); }, [](float const & a0){ return new PyCallBack_RMSProp(a0); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1){ return new RMSProp(a0, a1); }, [](float const & a0, float const & a1){ return new PyCallBack_RMSProp(a0, a1); } ), "doc");
		cl.def( pybind11::init( [](float const & a0, float const & a1, float const & a2){ return new RMSProp(a0, a1, a2); }, [](float const & a0, float const & a1, float const & a2){ return new PyCallBack_RMSProp(a0, a1, a2); } ), "doc");
		cl.def( pybind11::init<float, float, float, float>(), pybind11::arg("lr"), pybind11::arg("rho"), pybind11::arg("epsilon"), pybind11::arg("weight_decay") );

		cl.def_readwrite("lr", &RMSProp::lr);
		cl.def_readwrite("rho", &RMSProp::rho);
		cl.def_readwrite("epsilon", &RMSProp::epsilon);
		cl.def_readwrite("weight_decay", &RMSProp::weight_decay);
		cl.def_readwrite("mT", &RMSProp::mT);
		cl.def("assign", (class RMSProp & (RMSProp::*)(const class RMSProp &)) &RMSProp::operator=, "C++: RMSProp::operator=(const class RMSProp &) --> class RMSProp &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	// train_batch_t(void *) file:eddl/net.h line:46
	M("").def("train_batch_t", (void * (*)(void *)) &train_batch_t, "C++: train_batch_t(void *) --> void *", pybind11::return_value_policy::automatic, pybind11::arg("targs"));

	{ // Net file:eddl/net.h line:52
		pybind11::class_<Net, std::shared_ptr<Net>> cl(M(""), "Net", "");

		cl.def_readwrite("name", &Net::name);
		cl.def_readwrite("dev", &Net::dev);
		cl.def_readwrite("batch_size", &Net::batch_size);
		cl.def_readwrite("tr_batches", &Net::tr_batches);
		cl.def_readwrite("devsel", &Net::devsel);
		cl.def_readwrite("layers", &Net::layers);
		cl.def_readwrite("lin", &Net::lin);
		cl.def_readwrite("lout", &Net::lout);
		cl.def_readwrite("vfts", &Net::vfts);
		cl.def_readwrite("vbts", &Net::vbts);
		cl.def_readwrite("losses", &Net::losses);
		cl.def_readwrite("metrics", &Net::metrics);
		cl.def_readwrite("fiterr", &Net::fiterr);
		cl.def_readwrite("snets", &Net::snets);
		cl.def("initialize", (void (Net::*)()) &Net::initialize, "C++: Net::initialize() --> void");
		cl.def("reset", (void (Net::*)()) &Net::reset, "C++: Net::reset() --> void");
		cl.def("forward", (void (Net::*)()) &Net::forward, "C++: Net::forward() --> void");
		cl.def("delta", (void (Net::*)()) &Net::delta, "C++: Net::delta() --> void");
		cl.def("loss", (void (Net::*)()) &Net::loss, "C++: Net::loss() --> void");
		cl.def("backward", (void (Net::*)()) &Net::backward, "C++: Net::backward() --> void");
		cl.def("applygrads", (void (Net::*)()) &Net::applygrads, "C++: Net::applygrads() --> void");
		cl.def("split", (void (Net::*)(int, int)) &Net::split, "C++: Net::split(int, int) --> void", pybind11::arg("c"), pybind11::arg("todev"));
		cl.def("inNet", (int (Net::*)(class Layer *)) &Net::inNet, "C++: Net::inNet(class Layer *) --> int", pybind11::arg("l"));
		cl.def("walk", (void (Net::*)(class Layer *)) &Net::walk, "C++: Net::walk(class Layer *) --> void", pybind11::arg("l"));
		cl.def("fts", (void (Net::*)()) &Net::fts, "C++: Net::fts() --> void");
		cl.def("bts", (void (Net::*)()) &Net::bts, "C++: Net::bts() --> void");
		cl.def("resize", (void (Net::*)(int)) &Net::resize, "C++: Net::resize(int) --> void", pybind11::arg("batch"));
		cl.def("setmode", (void (Net::*)(int)) &Net::setmode, "C++: Net::setmode(int) --> void", pybind11::arg("m"));
		cl.def("sync_weights", (void (Net::*)()) &Net::sync_weights, "C++: Net::sync_weights() --> void");
		cl.def("clean_fiterr", (void (Net::*)()) &Net::clean_fiterr, "C++: Net::clean_fiterr() --> void");

		net_addons(cl);
	}
}


#include <map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_bits_libio(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_tensor_tensor(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_descriptors_descriptors(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_layers_layer(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_optimizers_optim(std::function< pybind11::module &(std::string const &namespace_) > &M);


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
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule(p.second.c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_bits_libio(M);
	bind_eddl_tensor_tensor(M);
	bind_eddl_descriptors_descriptors(M);
	bind_eddl_layers_layer(M);
	bind_eddl_optimizers_optim(M);

}

// Source list file: /pyeddl/codegen/bindings/_core.sources
// _core.cpp
// bits/libio.cpp
// eddl/tensor/tensor.cpp
// eddl/descriptors/descriptors.cpp
// eddl/layers/layer.cpp
// eddl/optimizers/optim.cpp

// Modules list file: /pyeddl/codegen/bindings/_core.modules
// 
