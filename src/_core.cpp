// File: eddl/descriptors/tensor_descriptors.cpp
#include <eddl/descriptors/tensor_descriptors.h>
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

// TensorDescriptor file:eddl/descriptors/tensor_descriptors.h line:24
struct PyCallBack_TensorDescriptor : public TensorDescriptor {
	using TensorDescriptor::TensorDescriptor;

	void build() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const TensorDescriptor *>(this), "build");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TensorDescriptor::build();
	}
	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const TensorDescriptor *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TensorDescriptor::resize(a0);
	}
};

// SelDescriptor file:eddl/descriptors/tensor_descriptors.h line:41
struct PyCallBack_SelDescriptor : public SelDescriptor {
	using SelDescriptor::SelDescriptor;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const SelDescriptor *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SelDescriptor::resize(a0);
	}
	void build_indices() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const SelDescriptor *>(this), "build_indices");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return SelDescriptor::build_indices();
	}
	void build() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const SelDescriptor *>(this), "build");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return TensorDescriptor::build();
	}
};

void bind_eddl_descriptors_tensor_descriptors(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // TensorDescriptor file:eddl/descriptors/tensor_descriptors.h line:24
		pybind11::class_<TensorDescriptor, std::shared_ptr<TensorDescriptor>, PyCallBack_TensorDescriptor> cl(M(""), "TensorDescriptor", "");
		cl.def( pybind11::init( [](){ return new TensorDescriptor(); }, [](){ return new PyCallBack_TensorDescriptor(); } ), "doc");
		cl.def( pybind11::init<int>(), pybind11::arg("dev") );

		cl.def( pybind11::init( [](PyCallBack_TensorDescriptor const &o){ return new PyCallBack_TensorDescriptor(o); } ) );
		cl.def( pybind11::init( [](TensorDescriptor const &o){ return new TensorDescriptor(o); } ) );
		cl.def_readwrite("device", &TensorDescriptor::device);
		cl.def("build", (void (TensorDescriptor::*)()) &TensorDescriptor::build, "C++: TensorDescriptor::build() --> void");
		cl.def("resize", (void (TensorDescriptor::*)(int)) &TensorDescriptor::resize, "C++: TensorDescriptor::resize(int) --> void", pybind11::arg("b"));
		cl.def("free_memory", (void (TensorDescriptor::*)()) &TensorDescriptor::free_memory, "C++: TensorDescriptor::free_memory() --> void");
		cl.def("assign", (class TensorDescriptor & (TensorDescriptor::*)(const class TensorDescriptor &)) &TensorDescriptor::operator=, "C++: TensorDescriptor::operator=(const class TensorDescriptor &) --> class TensorDescriptor &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // SelDescriptor file:eddl/descriptors/tensor_descriptors.h line:41
		pybind11::class_<SelDescriptor, std::shared_ptr<SelDescriptor>, PyCallBack_SelDescriptor, TensorDescriptor> cl(M(""), "SelDescriptor", "");
		cl.def( pybind11::init<int>(), pybind11::arg("dev") );

		cl.def( pybind11::init( [](PyCallBack_SelDescriptor const &o){ return new PyCallBack_SelDescriptor(o); } ) );
		cl.def( pybind11::init( [](SelDescriptor const &o){ return new SelDescriptor(o); } ) );
		cl.def_readwrite("ishape", &SelDescriptor::ishape);
		cl.def_readwrite("oshape", &SelDescriptor::oshape);
		cl.def_readwrite("idxs_range", &SelDescriptor::idxs_range);
		cl.def_readwrite("indices", &SelDescriptor::indices);
		cl.def("resize", (void (SelDescriptor::*)(int)) &SelDescriptor::resize, "C++: SelDescriptor::resize(int) --> void", pybind11::arg("b"));
		cl.def("build_indices", (void (SelDescriptor::*)()) &SelDescriptor::build_indices, "C++: SelDescriptor::build_indices() --> void");
		cl.def("assign", (class SelDescriptor & (SelDescriptor::*)(const class SelDescriptor &)) &SelDescriptor::operator=, "C++: SelDescriptor::operator=(const class SelDescriptor &) --> class SelDescriptor &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
}


// File: eddl/tensor/tensor.cpp
#include <eddl/descriptors/tensor_descriptors.h>
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
	{ // Tensor file:eddl/tensor/tensor.h line:60
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
		cl.def("toCPU", (void (Tensor::*)(int)) &Tensor::toCPU, "Clone a tensor to the CPU.\n\nC++: Tensor::toCPU(int) --> void", pybind11::arg("dev"));
		cl.def("toGPU", [](Tensor &o) -> void { return o.toGPU(); }, "");
		cl.def("toGPU", (void (Tensor::*)(int)) &Tensor::toGPU, "Clone a tensor to the GPU.\n\nC++: Tensor::toGPU(int) --> void", pybind11::arg("dev"));
		cl.def("clone", (class Tensor * (Tensor::*)()) &Tensor::clone, "Clone a tensor (same device).\n\n  \n    Tensor\n\nC++: Tensor::clone() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("deleteData", (void (Tensor::*)()) &Tensor::deleteData, "C++: Tensor::deleteData() --> void");
		cl.def("resize", (void (Tensor::*)(int, float *)) &Tensor::resize, "C++: Tensor::resize(int, float *) --> void", pybind11::arg("b"), pybind11::arg("fptr"));
		cl.def("resize", (void (Tensor::*)(int)) &Tensor::resize, "C++: Tensor::resize(int) --> void", pybind11::arg("b"));
		cl.def("resize", (void (Tensor::*)(int, class Tensor *)) &Tensor::resize, "C++: Tensor::resize(int, class Tensor *) --> void", pybind11::arg("b"), pybind11::arg("T"));
		cl.def("isCPU", (int (Tensor::*)()) &Tensor::isCPU, "Check if the tensor is in CPU.\n\n  \n int\n\nC++: Tensor::isCPU() --> int");
		cl.def("isGPU", (int (Tensor::*)()) &Tensor::isGPU, "Check if the tensor is in GPU.\n\n  \n int\n\nC++: Tensor::isGPU() --> int");
		cl.def("isFPGA", (int (Tensor::*)()) &Tensor::isFPGA, "Check if the tensor is in FPGA.\n\n  \n int\n\nC++: Tensor::isFPGA() --> int");
		cl.def("info", (void (Tensor::*)()) &Tensor::info, "Print shape, device and size information.\n\n  \n    void\n\nC++: Tensor::info() --> void");
		cl.def("print", [](Tensor &o) -> void { return o.print(); }, "");
		cl.def("print", [](Tensor &o, int const & a0) -> void { return o.print(a0); }, "", pybind11::arg("precision"));
		cl.def("print", (void (Tensor::*)(int, bool)) &Tensor::print, "Print the tensor values.\n\n  \n    void\n\nC++: Tensor::print(int, bool) --> void", pybind11::arg("precision"), pybind11::arg("raw"));
		cl.def_static("isSquared", (bool (*)(class Tensor *)) &Tensor::isSquared, "Check if all dimensions in the tensor are the same.\n\n  \n   Tensor\n  \n\n    bool\n\nC++: Tensor::isSquared(class Tensor *) --> bool", pybind11::arg("A"));
		cl.def_static("moveaxis", (class Tensor * (*)(class Tensor *, int, int)) &Tensor::moveaxis, "C++: Tensor::moveaxis(class Tensor *, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("t"), pybind11::arg("source"), pybind11::arg("destination"));
		cl.def_static("swapaxis", (class Tensor * (*)(class Tensor *, int, int)) &Tensor::swapaxis, "C++: Tensor::swapaxis(class Tensor *, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("t"), pybind11::arg("axis1"), pybind11::arg("axis2"));
		cl.def("fill_", (void (Tensor::*)(float)) &Tensor::fill_, "C++: Tensor::fill_(float) --> void", pybind11::arg("v"));
		cl.def_static("flatten", (class Tensor * (*)(class Tensor *)) &Tensor::flatten, "C++: Tensor::flatten(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("squeeze_", (void (Tensor::*)()) &Tensor::squeeze_, "C++: Tensor::squeeze_() --> void");
		cl.def_static("squeeze", (class Tensor * (*)(class Tensor *)) &Tensor::squeeze, "C++: Tensor::squeeze(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("unsqueeze_", (void (Tensor::*)()) &Tensor::unsqueeze_, "C++: Tensor::unsqueeze_() --> void");
		cl.def_static("unsqueeze", (class Tensor * (*)(class Tensor *)) &Tensor::unsqueeze, "C++: Tensor::unsqueeze(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
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
		cl.def_static("geomspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::geomspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("geomspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::geomspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("geomspace", (class Tensor * (*)(float, float, int, int)) &Tensor::geomspace, "C++: Tensor::geomspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));
		cl.def_static("eye", [](int const & a0) -> Tensor * { return Tensor::eye(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("rows"));
		cl.def_static("eye", [](int const & a0, int const & a1) -> Tensor * { return Tensor::eye(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("rows"), pybind11::arg("offset"));
		cl.def_static("eye", (class Tensor * (*)(int, int, int)) &Tensor::eye, "Number of rows of the tensor.\n  \n\n  \n  \n\n    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.\n  \n\n     Tensor of the specified shape filled with the value\n\nC++: Tensor::eye(int, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("rows"), pybind11::arg("offset"), pybind11::arg("dev"));
		cl.def_static("identity", [](int const & a0) -> Tensor * { return Tensor::identity(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("rows"));
		cl.def_static("identity", (class Tensor * (*)(int, int)) &Tensor::identity, "Create a tensor representing the identity matrix. Equivalent to calling function ``eye`` with ``offset = 0``.\n\n  \n  Shape of the tensor to create.\n  \n\n  Value to use to fill the tensor.\n  \n\n    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.\n  \n\n     Tensor of the specified shape filled with the value\n\nC++: Tensor::identity(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("rows"), pybind11::arg("dev"));
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
		cl.def("div_", (void (Tensor::*)(float)) &Tensor::div_, "C++: Tensor::div_(float) --> void", pybind11::arg("v"));
		cl.def_static("div", (class Tensor * (*)(class Tensor *, float)) &Tensor::div, "C++: Tensor::div(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def_static("div", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::div, "C++: Tensor::div(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("el_div", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_div, "C++: Tensor::el_div(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def("inv_", [](Tensor &o) -> void { return o.inv_(); }, "");
		cl.def("inv_", (void (Tensor::*)(float)) &Tensor::inv_, "C++: Tensor::inv_(float) --> void", pybind11::arg("v"));
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
		cl.def_static("mult", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::mult, "C++: Tensor::mult(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
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
		cl.def_static("all", (bool (*)(class Tensor *)) &Tensor::all, "Test whether all elements evaluate to True.\n\n  \n   Tensor to evaluate\n  \n\n    bool\n\nC++: Tensor::all(class Tensor *) --> bool", pybind11::arg("A"));
		cl.def_static("any", (bool (*)(class Tensor *)) &Tensor::any, "Test whether any element evaluates to True.\n\n  \n   Tensor to evaluate\n  \n\n    bool\n\nC++: Tensor::any(class Tensor *) --> bool", pybind11::arg("A"));
		cl.def_static("isfinite", (void (*)(class Tensor *, class Tensor *)) &Tensor::isfinite, "Test element-wise for finiteness (not infinity or not Not a Number).\n\n  \n   Tensor to evaluate\n  \n\n   Tensor to store the results of the test as booleans\n  \n\n    void\n\nC++: Tensor::isfinite(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("isinf", (void (*)(class Tensor *, class Tensor *)) &Tensor::isinf, "Test element-wise for positive or negative infinity.\n\n  \n   Tensor to evaluate\n  \n\n   Tensor to store the results of the test as booleans\n  \n\n    void\n\nC++: Tensor::isinf(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("isnan", (void (*)(class Tensor *, class Tensor *)) &Tensor::isnan, "Test element-wise for Nan.\n\n  \n   Tensor to evaluate\n  \n\n   Tensor to store the results of the test as booleans\n  \n\n    void\n\nC++: Tensor::isnan(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("isneginf", (void (*)(class Tensor *, class Tensor *)) &Tensor::isneginf, "Test element-wise for negative infinity.\n\n  \n   Tensor to evaluate\n  \n\n   Tensor to store the results of the test as booleans\n  \n\n    void\n\nC++: Tensor::isneginf(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("isposinf", (void (*)(class Tensor *, class Tensor *)) &Tensor::isposinf, "Test element-wise for positive infinity.\n\n  \n   Tensor to evaluate\n  \n\n   Tensor to store the results of the test as booleans\n  \n\n    void\n\nC++: Tensor::isposinf(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("logical_and", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::logical_and, "Compute the truth value of ``A and B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor to store the results of the operation\n  \n\n    void\n\nC++: Tensor::logical_and(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("logical_or", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::logical_or, "Compute the truth value of ``A or B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor to store the results of the operation\n  \n\n    void\n\nC++: Tensor::logical_or(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("logical_not", (void (*)(class Tensor *, class Tensor *)) &Tensor::logical_not, "Compute the truth value of ``not A`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor to store the results of the operation\n  \n\n    void\n\nC++: Tensor::logical_not(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("logical_xor", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::logical_xor, "Compute the truth value of ``A xor B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor to store the results of the operation\n  \n\n    void\n\nC++: Tensor::logical_xor(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("allclose", [](class Tensor * a0, class Tensor * a1) -> bool { return Tensor::allclose(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("allclose", [](class Tensor * a0, class Tensor * a1, float const & a2) -> bool { return Tensor::allclose(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rtol"));
		cl.def_static("allclose", [](class Tensor * a0, class Tensor * a1, float const & a2, float const & a3) -> bool { return Tensor::allclose(a0, a1, a2, a3); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rtol"), pybind11::arg("atol"));
		cl.def_static("allclose", (bool (*)(class Tensor *, class Tensor *, float, float, bool)) &Tensor::allclose, "Returns True if two arrays are element-wise equal within a tolerance.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n\n  \n\n\n  \n\n\n  \n\n    void\n\nC++: Tensor::allclose(class Tensor *, class Tensor *, float, float, bool) --> bool", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rtol"), pybind11::arg("atol"), pybind11::arg("equal_nan"));
		cl.def_static("isclose", [](class Tensor * a0, class Tensor * a1, class Tensor * a2) -> void { return Tensor::isclose(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("isclose", [](class Tensor * a0, class Tensor * a1, class Tensor * a2, float const & a3) -> void { return Tensor::isclose(a0, a1, a2, a3); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("rtol"));
		cl.def_static("isclose", [](class Tensor * a0, class Tensor * a1, class Tensor * a2, float const & a3, float const & a4) -> void { return Tensor::isclose(a0, a1, a2, a3, a4); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("rtol"), pybind11::arg("atol"));
		cl.def_static("isclose", (void (*)(class Tensor *, class Tensor *, class Tensor *, float, float, bool)) &Tensor::isclose, "Returns a boolean array where two arrays are element-wise equal within a tolerance.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor\n  \n\n\n  \n\n\n  \n\n\n  \n\n    void\n\nC++: Tensor::isclose(class Tensor *, class Tensor *, class Tensor *, float, float, bool) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("rtol"), pybind11::arg("atol"), pybind11::arg("equal_nan"));
		cl.def_static("greater", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::greater, "Return the truth value of ``A > B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::greater(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("greater_equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::greater_equal, "Return the truth value of ``A >= B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::greater_equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("less", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::less, "Return the truth value of ``A < B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::less(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("less_equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::less_equal, "Return the truth value of ``A <= B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::less_equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::equal, "Return the truth value of ``A == B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("not_equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::not_equal, "Return the truth value of ``A != B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::not_equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("eqsize", (int (*)(class Tensor *, class Tensor *)) &Tensor::eqsize, "C++: Tensor::eqsize(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equal2", [](class Tensor * a0, class Tensor * a1) -> int { return Tensor::equal2(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equal2", (int (*)(class Tensor *, class Tensor *, float)) &Tensor::equal2, "C++: Tensor::equal2(class Tensor *, class Tensor *, float) --> int", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("epsilon"));
		cl.def_static("select", (void (*)(class Tensor *, class Tensor *, class SelDescriptor *)) &Tensor::select, "C++: Tensor::select(class Tensor *, class Tensor *, class SelDescriptor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("sd"));
		cl.def_static("set_select", (void (*)(class Tensor *, class Tensor *, class SelDescriptor *)) &Tensor::set_select, "C++: Tensor::set_select(class Tensor *, class Tensor *, class SelDescriptor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("sd"));
		cl.def_static("set_select_back", (void (*)(class Tensor *, class Tensor *, class SelDescriptor *)) &Tensor::set_select_back, "C++: Tensor::set_select_back(class Tensor *, class Tensor *, class SelDescriptor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("sd"));
		cl.def_static("copy", (void (*)(class Tensor *, class Tensor *)) &Tensor::copy, "Copy data from tensor A to B.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n    void\n\nC++: Tensor::copy(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("fill", (void (*)(class Tensor *, int, int, class Tensor *, int, int, int)) &Tensor::fill, "C++: Tensor::fill(class Tensor *, int, int, class Tensor *, int, int, int) --> void", pybind11::arg("A"), pybind11::arg("aini"), pybind11::arg("aend"), pybind11::arg("B"), pybind11::arg("bini"), pybind11::arg("bend"), pybind11::arg("inc"));
		cl.def_static("select_back", (void (*)(class Tensor *, class Tensor *, class SelDescriptor *)) &Tensor::select_back, "C++: Tensor::select_back(class Tensor *, class Tensor *, class SelDescriptor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("sd"));
		cl.def_static("tile", (void (*)(class Tensor *, class Tensor *)) &Tensor::tile, "C++: Tensor::tile(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("rand_uniform", (void (Tensor::*)(float)) &Tensor::rand_uniform, "C++: Tensor::rand_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_signed_uniform", (void (Tensor::*)(float)) &Tensor::rand_signed_uniform, "C++: Tensor::rand_signed_uniform(float) --> void", pybind11::arg("v"));
		cl.def("rand_normal", [](Tensor &o, float const & a0, float const & a1) -> void { return o.rand_normal(a0, a1); }, "", pybind11::arg("m"), pybind11::arg("s"));
		cl.def("rand_normal", (void (Tensor::*)(float, float, bool)) &Tensor::rand_normal, "C++: Tensor::rand_normal(float, float, bool) --> void", pybind11::arg("m"), pybind11::arg("s"), pybind11::arg("fast_math"));
		cl.def("rand_binary", (void (Tensor::*)(float)) &Tensor::rand_binary, "C++: Tensor::rand_binary(float) --> void", pybind11::arg("v"));

		tensor_addons(cl);
	}
}


// File: eddl/losses/loss.cpp
#include <eddl/descriptors/tensor_descriptors.h>
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
#include <loss_addons.hpp>
#include <memory>
#include <metric_addons.hpp>
#include <net_addons.hpp>
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

// Loss file:eddl/losses/loss.h line:22
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

// Metric file:eddl/metrics/metric.h line:23
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

// Layer file:eddl/layers/layer.h line:32
struct PyCallBack_Layer : public Layer {
	using Layer::Layer;

	void initialize() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "initialize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::initialize();
	}
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
	void mem_delta_parent() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "mem_delta_parent");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::mem_delta_parent();
	}
	void mem_delta() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "mem_delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::mem_delta();
	}
	void free_delta() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "free_delta");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::free_delta();
	}
	void copy(class Layer * a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "copy");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::copy(a0);
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
	void set_trainable(bool a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "set_trainable");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::set_trainable(a0);
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
	int get_trainable_params_count() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "get_trainable_params_count");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<int>::value) {
				static pybind11::detail::overload_caster_t<int> caster;
				return pybind11::detail::cast_ref<int>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<int>(std::move(o));
		}
		return Layer::get_trainable_params_count();
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
	void update_weights(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "update_weights");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::update_weights(a0, a1);
	}
	void accumulate_accumulated_gradients(class Tensor * a0, class Tensor * a1) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "accumulate_accumulated_gradients");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0, a1);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::accumulate_accumulated_gradients(a0, a1);
	}
	void reset_accumulated_gradients() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "reset_accumulated_gradients");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::reset_accumulated_gradients();
	}
	void apply_accumulated_gradients() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "apply_accumulated_gradients");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::apply_accumulated_gradients();
	}
	void enable_distributed() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "enable_distributed");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::enable_distributed();
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

void bind_eddl_losses_loss(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Loss file:eddl/losses/loss.h line:22
		pybind11::class_<Loss, std::unique_ptr<Loss, pybind11::nodelete>, PyCallBack_Loss> cl(M(""), "Loss", "");
		cl.def( pybind11::init( [](PyCallBack_Loss const &o){ return new PyCallBack_Loss(o); } ) );
		cl.def( pybind11::init( [](Loss const &o){ return new Loss(o); } ) );
		cl.def_readwrite("name", &Loss::name);
		cl.def("delta", (void (Loss::*)(class Tensor *, class Tensor *, class Tensor *)) &Loss::delta, "C++: Loss::delta(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("T"), pybind11::arg("Y"), pybind11::arg("D"));
		cl.def("value", (float (Loss::*)(class Tensor *, class Tensor *)) &Loss::value, "C++: Loss::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class Loss & (Loss::*)(const class Loss &)) &Loss::operator=, "C++: Loss::operator=(const class Loss &) --> class Loss &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		loss_addons(cl);
	}
	{ // CompServ file:eddl/net/compserv.h line:21
		pybind11::class_<CompServ, std::shared_ptr<CompServ>> cl(M(""), "CompServ", "");
		cl.def( pybind11::init( [](CompServ const &o){ return new CompServ(o); } ) );
		cl.def_readwrite("type", &CompServ::type);
		cl.def_readwrite("local_threads", &CompServ::local_threads);
		cl.def_readwrite("local_gpus", &CompServ::local_gpus);
		cl.def_readwrite("local_fpgas", &CompServ::local_fpgas);
		cl.def_readwrite("lsb", &CompServ::lsb);
		cl.def_readwrite("mem_level", &CompServ::mem_level);
	}
	{ // Metric file:eddl/metrics/metric.h line:23
		pybind11::class_<Metric, std::unique_ptr<Metric, pybind11::nodelete>, PyCallBack_Metric> cl(M(""), "Metric", "");
		cl.def( pybind11::init( [](PyCallBack_Metric const &o){ return new PyCallBack_Metric(o); } ) );
		cl.def( pybind11::init( [](Metric const &o){ return new Metric(o); } ) );
		cl.def_readwrite("name", &Metric::name);
		cl.def("value", (float (Metric::*)(class Tensor *, class Tensor *)) &Metric::value, "C++: Metric::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("assign", (class Metric & (Metric::*)(const class Metric &)) &Metric::operator=, "C++: Metric::operator=(const class Metric &) --> class Metric &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		metric_addons(cl);
	}
	{ // Layer file:eddl/layers/layer.h line:32
		pybind11::class_<Layer, std::shared_ptr<Layer>, PyCallBack_Layer> cl(M(""), "Layer", "");
		cl.def( pybind11::init( [](PyCallBack_Layer const &o){ return new PyCallBack_Layer(o); } ) );
		cl.def( pybind11::init( [](Layer const &o){ return new Layer(o); } ) );
		cl.def_readwrite("name", &Layer::name);
		cl.def_readwrite("trainable", &Layer::trainable);
		cl.def_readwrite("mem_level", &Layer::mem_level);
		cl.def_readwrite("params", &Layer::params);
		cl.def_readwrite("gradients", &Layer::gradients);
		cl.def_readwrite("acc_gradients", &Layer::acc_gradients);
		cl.def_readwrite("parent", &Layer::parent);
		cl.def_readwrite("child", &Layer::child);
		cl.def_readwrite("mode", &Layer::mode);
		cl.def_readwrite("dev", &Layer::dev);
		cl.def_readwrite("lin", &Layer::lin);
		cl.def_readwrite("lout", &Layer::lout);
		cl.def_readwrite("delta_bp", &Layer::delta_bp);
		cl.def_readwrite("detached", &Layer::detached);
		cl.def_readwrite("verbosity_level", &Layer::verbosity_level);
		cl.def("initialize", (void (Layer::*)()) &Layer::initialize, "C++: Layer::initialize() --> void");
		cl.def("info", (void (Layer::*)()) &Layer::info, "C++: Layer::info() --> void");
		cl.def("setmode", (void (Layer::*)(int)) &Layer::setmode, "C++: Layer::setmode(int) --> void", pybind11::arg("m"));
		cl.def("check_target", (void (Layer::*)()) &Layer::check_target, "C++: Layer::check_target() --> void");
		cl.def("detach", (void (Layer::*)(class Layer *)) &Layer::detach, "C++: Layer::detach(class Layer *) --> void", pybind11::arg("l"));
		cl.def("getWeights", (class Tensor * (Layer::*)()) &Layer::getWeights, "C++: Layer::getWeights() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setWeights", (class Tensor * (Layer::*)(class Tensor)) &Layer::setWeights, "C++: Layer::setWeights(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("getBias", (class Tensor * (Layer::*)()) &Layer::getBias, "C++: Layer::getBias() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("setBias", (class Tensor * (Layer::*)(class Tensor)) &Layer::setBias, "C++: Layer::setBias(class Tensor) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("bias"));
		cl.def("clamp", (void (Layer::*)(float, float)) &Layer::clamp, "C++: Layer::clamp(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def("set_detach", (void (Layer::*)()) &Layer::set_detach, "C++: Layer::set_detach() --> void");
		cl.def("set_mem_level", (void (Layer::*)(int)) &Layer::set_mem_level, "C++: Layer::set_mem_level(int) --> void", pybind11::arg("mem"));
		cl.def("mem_delta_parent", (void (Layer::*)()) &Layer::mem_delta_parent, "C++: Layer::mem_delta_parent() --> void");
		cl.def("mem_delta", (void (Layer::*)()) &Layer::mem_delta, "C++: Layer::mem_delta() --> void");
		cl.def("free_delta", (void (Layer::*)()) &Layer::free_delta, "C++: Layer::free_delta() --> void");
		cl.def("copy", (void (Layer::*)(class Layer *)) &Layer::copy, "C++: Layer::copy(class Layer *) --> void", pybind11::arg("l2"));
		cl.def("resize", (void (Layer::*)(int)) &Layer::resize, "C++: Layer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("set_trainable", (void (Layer::*)(bool)) &Layer::set_trainable, "C++: Layer::set_trainable(bool) --> void", pybind11::arg("value"));
		cl.def("reset", (void (Layer::*)()) &Layer::reset, "C++: Layer::reset() --> void");
		cl.def("get_trainable_params_count", (int (Layer::*)()) &Layer::get_trainable_params_count, "C++: Layer::get_trainable_params_count() --> int");
		cl.def("zeroGrads", (void (Layer::*)()) &Layer::zeroGrads, "C++: Layer::zeroGrads() --> void");
		cl.def("addchild", (void (Layer::*)(class Layer *)) &Layer::addchild, "C++: Layer::addchild(class Layer *) --> void", pybind11::arg("l"));
		cl.def("addparent", (void (Layer::*)(class Layer *)) &Layer::addparent, "C++: Layer::addparent(class Layer *) --> void", pybind11::arg("l"));
		cl.def("forward", (void (Layer::*)()) &Layer::forward, "C++: Layer::forward() --> void");
		cl.def("backward", (void (Layer::*)()) &Layer::backward, "C++: Layer::backward() --> void");
		cl.def("update_weights", (void (Layer::*)(class Tensor *, class Tensor *)) &Layer::update_weights, "C++: Layer::update_weights(class Tensor *, class Tensor *) --> void", pybind11::arg("w"), pybind11::arg("bias"));
		cl.def("accumulate_accumulated_gradients", (void (Layer::*)(class Tensor *, class Tensor *)) &Layer::accumulate_accumulated_gradients, "C++: Layer::accumulate_accumulated_gradients(class Tensor *, class Tensor *) --> void", pybind11::arg("gw"), pybind11::arg("gbias"));
		cl.def("reset_accumulated_gradients", (void (Layer::*)()) &Layer::reset_accumulated_gradients, "C++: Layer::reset_accumulated_gradients() --> void");
		cl.def("apply_accumulated_gradients", (void (Layer::*)()) &Layer::apply_accumulated_gradients, "C++: Layer::apply_accumulated_gradients() --> void");
		cl.def("enable_distributed", (void (Layer::*)()) &Layer::enable_distributed, "C++: Layer::enable_distributed() --> void");
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
		cl.def_readwrite("mem_level", &Net::mem_level);
		cl.def_readwrite("verbosity_level", &Net::verbosity_level);
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
		cl.def("enable_distributed", (void (Net::*)()) &Net::enable_distributed, "C++: Net::enable_distributed() --> void");
		cl.def("setmode", (void (Net::*)(int)) &Net::setmode, "C++: Net::setmode(int) --> void", pybind11::arg("m"));
		cl.def("do_initialize", (void (Net::*)()) &Net::do_initialize, "C++: Net::do_initialize() --> void");
		cl.def("do_reset", (void (Net::*)()) &Net::do_reset, "C++: Net::do_reset() --> void");
		cl.def("do_reset_grads", (void (Net::*)()) &Net::do_reset_grads, "C++: Net::do_reset_grads() --> void");
		cl.def("do_forward", (void (Net::*)()) &Net::do_forward, "C++: Net::do_forward() --> void");
		cl.def("do_delta", (void (Net::*)()) &Net::do_delta, "C++: Net::do_delta() --> void");
		cl.def("do_compute_loss", (void (Net::*)()) &Net::do_compute_loss, "C++: Net::do_compute_loss() --> void");
		cl.def("do_backward", (void (Net::*)()) &Net::do_backward, "C++: Net::do_backward() --> void");
		cl.def("do_applygrads", (void (Net::*)()) &Net::do_applygrads, "C++: Net::do_applygrads() --> void");
		cl.def("reset_accumulated_gradients", (void (Net::*)()) &Net::reset_accumulated_gradients, "C++: Net::reset_accumulated_gradients() --> void");
		cl.def("apply_accumulated_gradients", (void (Net::*)()) &Net::apply_accumulated_gradients, "C++: Net::apply_accumulated_gradients() --> void");
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

		net_addons(cl);
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
#include <eddl/descriptors/tensor_descriptors.h>
#include <eddl/layers/layer.h>
#include <eddl/losses/loss.h>
#include <eddl/metrics/metric.h>
#include <eddl/net/compserv.h>
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
	// eddl::build(class Net *, class Optimizer *, class CompServ *, bool) file:eddl/apis/eddl.h line:65
	M("eddl").def("build", [](class Net * a0) -> void { return eddl::build(a0); }, "", pybind11::arg("net"));
	M("eddl").def("build", [](class Net * a0, class Optimizer * a1) -> void { return eddl::build(a0, a1); }, "", pybind11::arg("net"), pybind11::arg("o"));
	M("eddl").def("build", [](class Net * a0, class Optimizer * a1, class CompServ * a2) -> void { return eddl::build(a0, a1, a2); }, "", pybind11::arg("net"), pybind11::arg("o"), pybind11::arg("cs"));
	M("eddl").def("build", (void (*)(class Net *, class Optimizer *, class CompServ *, bool)) &eddl::build, "C++: eddl::build(class Net *, class Optimizer *, class CompServ *, bool) --> void", pybind11::arg("net"), pybind11::arg("o"), pybind11::arg("cs"), pybind11::arg("init_weigths"));

	// eddl::toGPU(class Net *) file:eddl/apis/eddl.h line:93
	M("eddl").def("toGPU", (void (*)(class Net *)) &eddl::toGPU, "C++: eddl::toGPU(class Net *) --> void", pybind11::arg("net"));

	// eddl::toCPU(class Net *, int) file:eddl/apis/eddl.h line:102
	M("eddl").def("toCPU", [](class Net * a0) -> void { return eddl::toCPU(a0); }, "", pybind11::arg("net"));
	M("eddl").def("toCPU", (void (*)(class Net *, int)) &eddl::toCPU, "Assign model operations to the CPU.\n\n  \n  Model\n  \n\n  CPU Threads\n  \n\n     (void)\n\nC++: eddl::toCPU(class Net *, int) --> void", pybind11::arg("net"), pybind11::arg("t"));

	// eddl::summary(class Net *) file:eddl/apis/eddl.h line:167
	M("eddl").def("summary", (void (*)(class Net *)) &eddl::summary, "Prints a summary representation of your model.\n\n  \n  Model to train\n  \n\n     (void) Prints the model\n\nC++: eddl::summary(class Net *) --> void", pybind11::arg("m"));

	// eddl::set_mode(class Net *, int) file:eddl/apis/eddl.h line:357
	M("eddl").def("set_mode", (void (*)(class Net *, int)) &eddl::set_mode, "Set model mode.\n\n  \n  Model\n  \n\n  Train 1, Test 0\n  \n\n     (void)\n\nC++: eddl::set_mode(class Net *, int) --> void", pybind11::arg("net"), pybind11::arg("mode"));

	// eddl::reset_loss(class Net *) file:eddl/apis/eddl.h line:364
	M("eddl").def("reset_loss", (void (*)(class Net *)) &eddl::reset_loss, "Resets model loss.\n\n  \n  Model\n  \n\n     (void)\n\nC++: eddl::reset_loss(class Net *) --> void", pybind11::arg("m"));

	// eddl::zeroGrads(class Net *) file:eddl/apis/eddl.h line:375
	M("eddl").def("zeroGrads", (void (*)(class Net *)) &eddl::zeroGrads, "Set model gradients to zero.\n\n  \n  Model\n  \n\n     (void)\n\nC++: eddl::zeroGrads(class Net *) --> void", pybind11::arg("m"));

	// eddl::backward(class Net *) file:eddl/apis/eddl.h line:384
	M("eddl").def("backward", (void (*)(class Net *)) &eddl::backward, "C++: eddl::backward(class Net *) --> void", pybind11::arg("net"));

	// eddl::backward(class NetLoss *) file:eddl/apis/eddl.h line:385
	M("eddl").def("backward", (void (*)(class NetLoss *)) &eddl::backward, "C++: eddl::backward(class NetLoss *) --> void", pybind11::arg("l"));

	// eddl::update(class Net *) file:eddl/apis/eddl.h line:386
	M("eddl").def("update", (void (*)(class Net *)) &eddl::update, "C++: eddl::update(class Net *) --> void", pybind11::arg("m"));

	// eddl::print_loss(class Net *, int) file:eddl/apis/eddl.h line:394
	M("eddl").def("print_loss", (void (*)(class Net *, int)) &eddl::print_loss, "Prints model loss at some batch.\n\n  \n  Model\n  \n\n  Batch number\n  \n\n     (void)\n\nC++: eddl::print_loss(class Net *, int) --> void", pybind11::arg("m"), pybind11::arg("batch"));

	// eddl::clamp(class Net *, float, float) file:eddl/apis/eddl.h line:405
	M("eddl").def("clamp", (void (*)(class Net *, float, float)) &eddl::clamp, "Model parameters values clipping.\n\n  \n  Model\n  \n\n  Minimum value\n  \n\n   Maximum value\n  \n\n     (void) Performs model clamp between min and max\n\nC++: eddl::clamp(class Net *, float, float) --> void", pybind11::arg("m"), pybind11::arg("min"), pybind11::arg("max"));

	// eddl::compute_loss(class NetLoss *) file:eddl/apis/eddl.h line:408
	M("eddl").def("compute_loss", (float (*)(class NetLoss *)) &eddl::compute_loss, "C++: eddl::compute_loss(class NetLoss *) --> float", pybind11::arg("L"));

	// eddl::compute_metric(class NetLoss *) file:eddl/apis/eddl.h line:409
	M("eddl").def("compute_metric", (float (*)(class NetLoss *)) &eddl::compute_metric, "C++: eddl::compute_metric(class NetLoss *) --> float", pybind11::arg("L"));

	// eddl::set_trainable(class Layer *, bool) file:eddl/apis/eddl.h line:1376
	M("eddl").def("set_trainable", (void (*)(class Layer *, bool)) &eddl::set_trainable, "C++: eddl::set_trainable(class Layer *, bool) --> void", pybind11::arg("l"), pybind11::arg("val"));

	// eddl::copyTensor(class Layer *, class Layer *) file:eddl/apis/eddl.h line:1377
	M("eddl").def("copyTensor", (void (*)(class Layer *, class Layer *)) &eddl::copyTensor, "C++: eddl::copyTensor(class Layer *, class Layer *) --> void", pybind11::arg("l1"), pybind11::arg("l2"));

	// eddl::copyGrad(class Layer *, class Layer *) file:eddl/apis/eddl.h line:1378
	M("eddl").def("copyGrad", (void (*)(class Layer *, class Layer *)) &eddl::copyGrad, "C++: eddl::copyGrad(class Layer *, class Layer *) --> void", pybind11::arg("l1"), pybind11::arg("l2"));

	// eddl::getGrad(class Layer *) file:eddl/apis/eddl.h line:1381
	M("eddl").def("getGrad", (class Tensor * (*)(class Layer *)) &eddl::getGrad, "C++: eddl::getGrad(class Layer *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("l"));

	// eddl::download_mnist() file:eddl/apis/eddl.h line:1487
	M("eddl").def("download_mnist", (void (*)()) &eddl::download_mnist, "Downloads MNIST Dataset.\n\n  \n   http://yann.lecun.com/exdb/mnist/\n\n  \n     (void) The binary files of MNIST\n\nC++: eddl::download_mnist() --> void");

	// eddl::download_cifar10() file:eddl/apis/eddl.h line:1495
	M("eddl").def("download_cifar10", (void (*)()) &eddl::download_cifar10, "Downloads CIFAR-10 Dataset.\n\n  \n   https://www.cs.toronto.edu/~kriz/cifar.html\n\n  \n     (void) The binary files of CIFAR-10\n\nC++: eddl::download_cifar10() --> void");

	// eddl::download_drive() file:eddl/apis/eddl.h line:1503
	M("eddl").def("download_drive", (void (*)()) &eddl::download_drive, "Downloads DRIVE Dataset.\n\n  \n   https://drive.grand-challenge.org/\n\n  \n     (void) The numpy files of DRIVE\n\nC++: eddl::download_drive() --> void");

}


// File: eddl/apis/eddlT.cpp
#include <eddl/apis/eddlT.h>
#include <eddl/descriptors/tensor_descriptors.h>
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
	// eddlT::arange(float, float, float, int) file:eddl/apis/eddlT.h line:59
	M("eddlT").def("arange", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return eddlT::arange(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
	M("eddlT").def("arange", (class Tensor * (*)(float, float, float, int)) &eddlT::arange, "C++: eddlT::arange(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));

	// eddlT::range(float, float, float, int) file:eddl/apis/eddlT.h line:60
	M("eddlT").def("range", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return eddlT::range(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
	M("eddlT").def("range", (class Tensor * (*)(float, float, float, int)) &eddlT::range, "C++: eddlT::range(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));

	// eddlT::linspace(float, float, int, int) file:eddl/apis/eddlT.h line:61
	M("eddlT").def("linspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return eddlT::linspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
	M("eddlT").def("linspace", (class Tensor * (*)(float, float, int, int)) &eddlT::linspace, "C++: eddlT::linspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));

	// eddlT::logspace(float, float, int, float, int) file:eddl/apis/eddlT.h line:62
	M("eddlT").def("logspace", [](float const & a0, float const & a1, int const & a2, float const & a3) -> Tensor * { return eddlT::logspace(a0, a1, a2, a3); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"));
	M("eddlT").def("logspace", (class Tensor * (*)(float, float, int, float, int)) &eddlT::logspace, "C++: eddlT::logspace(float, float, int, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"), pybind11::arg("dev"));

	// eddlT::eye(int, int) file:eddl/apis/eddlT.h line:63
	M("eddlT").def("eye", [](int const & a0) -> Tensor * { return eddlT::eye(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("size"));
	M("eddlT").def("eye", (class Tensor * (*)(int, int)) &eddlT::eye, "C++: eddlT::eye(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("size"), pybind11::arg("dev"));

	// eddlT::toCPU_(class Tensor *) file:eddl/apis/eddlT.h line:68
	M("eddlT").def("toCPU_", (void (*)(class Tensor *)) &eddlT::toCPU_, "C++: eddlT::toCPU_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::toGPU_(class Tensor *) file:eddl/apis/eddlT.h line:69
	M("eddlT").def("toGPU_", (void (*)(class Tensor *)) &eddlT::toGPU_, "C++: eddlT::toGPU_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::toCPU(class Tensor *) file:eddl/apis/eddlT.h line:70
	M("eddlT").def("toCPU", (class Tensor * (*)(class Tensor *)) &eddlT::toCPU, "C++: eddlT::toCPU(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::toGPU(class Tensor *) file:eddl/apis/eddlT.h line:71
	M("eddlT").def("toGPU", (class Tensor * (*)(class Tensor *)) &eddlT::toGPU, "C++: eddlT::toGPU(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::clone(class Tensor *) file:eddl/apis/eddlT.h line:72
	M("eddlT").def("clone", (class Tensor * (*)(class Tensor *)) &eddlT::clone, "C++: eddlT::clone(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::select(class Tensor *, int) file:eddl/apis/eddlT.h line:73
	M("eddlT").def("select", (class Tensor * (*)(class Tensor *, int)) &eddlT::select, "C++: eddlT::select(class Tensor *, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("i"));

	// eddlT::copyTensor(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:74
	M("eddlT").def("copyTensor", (void (*)(class Tensor *, class Tensor *)) &eddlT::copyTensor, "C++: eddlT::copyTensor(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::fill_(class Tensor *, float) file:eddl/apis/eddlT.h line:77
	M("eddlT").def("fill_", (void (*)(class Tensor *, float)) &eddlT::fill_, "C++: eddlT::fill_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::print(class Tensor *) file:eddl/apis/eddlT.h line:85
	M("eddlT").def("print", (void (*)(class Tensor *)) &eddlT::print, "C++: eddlT::print(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::info(class Tensor *) file:eddl/apis/eddlT.h line:86
	M("eddlT").def("info", (void (*)(class Tensor *)) &eddlT::info, "C++: eddlT::info(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::abs_(class Tensor *) file:eddl/apis/eddlT.h line:105
	M("eddlT").def("abs_", (void (*)(class Tensor *)) &eddlT::abs_, "C++: eddlT::abs_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::abs(class Tensor *) file:eddl/apis/eddlT.h line:106
	M("eddlT").def("abs", (class Tensor * (*)(class Tensor *)) &eddlT::abs, "C++: eddlT::abs(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::acos_(class Tensor *) file:eddl/apis/eddlT.h line:108
	M("eddlT").def("acos_", (void (*)(class Tensor *)) &eddlT::acos_, "C++: eddlT::acos_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::acos(class Tensor *) file:eddl/apis/eddlT.h line:109
	M("eddlT").def("acos", (class Tensor * (*)(class Tensor *)) &eddlT::acos, "C++: eddlT::acos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::add_(class Tensor *, float) file:eddl/apis/eddlT.h line:111
	M("eddlT").def("add_", (void (*)(class Tensor *, float)) &eddlT::add_, "C++: eddlT::add_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::add(class Tensor *, float) file:eddl/apis/eddlT.h line:112
	M("eddlT").def("add", (class Tensor * (*)(class Tensor *, float)) &eddlT::add, "C++: eddlT::add(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::add_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:113
	M("eddlT").def("add_", (void (*)(class Tensor *, class Tensor *)) &eddlT::add_, "C++: eddlT::add_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::add(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:114
	M("eddlT").def("add", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::add, "C++: eddlT::add(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::asin_(class Tensor *) file:eddl/apis/eddlT.h line:118
	M("eddlT").def("asin_", (void (*)(class Tensor *)) &eddlT::asin_, "C++: eddlT::asin_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::asin(class Tensor *) file:eddl/apis/eddlT.h line:119
	M("eddlT").def("asin", (class Tensor * (*)(class Tensor *)) &eddlT::asin, "C++: eddlT::asin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::atan_(class Tensor *) file:eddl/apis/eddlT.h line:121
	M("eddlT").def("atan_", (void (*)(class Tensor *)) &eddlT::atan_, "C++: eddlT::atan_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::atan(class Tensor *) file:eddl/apis/eddlT.h line:122
	M("eddlT").def("atan", (class Tensor * (*)(class Tensor *)) &eddlT::atan, "C++: eddlT::atan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::ceil_(class Tensor *) file:eddl/apis/eddlT.h line:124
	M("eddlT").def("ceil_", (void (*)(class Tensor *)) &eddlT::ceil_, "C++: eddlT::ceil_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::ceil(class Tensor *) file:eddl/apis/eddlT.h line:125
	M("eddlT").def("ceil", (class Tensor * (*)(class Tensor *)) &eddlT::ceil, "C++: eddlT::ceil(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::clamp_(class Tensor *, float, float) file:eddl/apis/eddlT.h line:127
	M("eddlT").def("clamp_", (void (*)(class Tensor *, float, float)) &eddlT::clamp_, "C++: eddlT::clamp_(class Tensor *, float, float) --> void", pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::clamp(class Tensor *, float, float) file:eddl/apis/eddlT.h line:128
	M("eddlT").def("clamp", (class Tensor * (*)(class Tensor *, float, float)) &eddlT::clamp, "C++: eddlT::clamp(class Tensor *, float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::clampmax_(class Tensor *, float) file:eddl/apis/eddlT.h line:130
	M("eddlT").def("clampmax_", (void (*)(class Tensor *, float)) &eddlT::clampmax_, "C++: eddlT::clampmax_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("max"));

	// eddlT::clampmax(class Tensor *, float) file:eddl/apis/eddlT.h line:131
	M("eddlT").def("clampmax", (class Tensor * (*)(class Tensor *, float)) &eddlT::clampmax, "C++: eddlT::clampmax(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("max"));

	// eddlT::clampmin_(class Tensor *, float) file:eddl/apis/eddlT.h line:133
	M("eddlT").def("clampmin_", (void (*)(class Tensor *, float)) &eddlT::clampmin_, "C++: eddlT::clampmin_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("min"));

	// eddlT::clampmin(class Tensor *, float) file:eddl/apis/eddlT.h line:134
	M("eddlT").def("clampmin", (class Tensor * (*)(class Tensor *, float)) &eddlT::clampmin, "C++: eddlT::clampmin(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"));

	// eddlT::cos_(class Tensor *) file:eddl/apis/eddlT.h line:136
	M("eddlT").def("cos_", (void (*)(class Tensor *)) &eddlT::cos_, "C++: eddlT::cos_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::cos(class Tensor *) file:eddl/apis/eddlT.h line:137
	M("eddlT").def("cos", (class Tensor * (*)(class Tensor *)) &eddlT::cos, "C++: eddlT::cos(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::cosh_(class Tensor *) file:eddl/apis/eddlT.h line:139
	M("eddlT").def("cosh_", (void (*)(class Tensor *)) &eddlT::cosh_, "C++: eddlT::cosh_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::cosh(class Tensor *) file:eddl/apis/eddlT.h line:140
	M("eddlT").def("cosh", (class Tensor * (*)(class Tensor *)) &eddlT::cosh, "C++: eddlT::cosh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::div_(class Tensor *, float) file:eddl/apis/eddlT.h line:144
	M("eddlT").def("div_", (void (*)(class Tensor *, float)) &eddlT::div_, "C++: eddlT::div_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::div(class Tensor *, float) file:eddl/apis/eddlT.h line:145
	M("eddlT").def("div", (class Tensor * (*)(class Tensor *, float)) &eddlT::div, "C++: eddlT::div(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::div_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:146
	M("eddlT").def("div_", (void (*)(class Tensor *, class Tensor *)) &eddlT::div_, "C++: eddlT::div_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::div(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:147
	M("eddlT").def("div", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::div, "C++: eddlT::div(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::exp_(class Tensor *) file:eddl/apis/eddlT.h line:149
	M("eddlT").def("exp_", (void (*)(class Tensor *)) &eddlT::exp_, "C++: eddlT::exp_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::exp(class Tensor *) file:eddl/apis/eddlT.h line:150
	M("eddlT").def("exp", (class Tensor * (*)(class Tensor *)) &eddlT::exp, "C++: eddlT::exp(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::floor_(class Tensor *) file:eddl/apis/eddlT.h line:152
	M("eddlT").def("floor_", (void (*)(class Tensor *)) &eddlT::floor_, "C++: eddlT::floor_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::floor(class Tensor *) file:eddl/apis/eddlT.h line:153
	M("eddlT").def("floor", (class Tensor * (*)(class Tensor *)) &eddlT::floor, "C++: eddlT::floor(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::inc_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:155
	M("eddlT").def("inc_", (void (*)(class Tensor *, class Tensor *)) &eddlT::inc_, "C++: eddlT::inc_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::log_(class Tensor *) file:eddl/apis/eddlT.h line:157
	M("eddlT").def("log_", (void (*)(class Tensor *)) &eddlT::log_, "C++: eddlT::log_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::log(class Tensor *) file:eddl/apis/eddlT.h line:158
	M("eddlT").def("log", (class Tensor * (*)(class Tensor *)) &eddlT::log, "C++: eddlT::log(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::log2_(class Tensor *) file:eddl/apis/eddlT.h line:160
	M("eddlT").def("log2_", (void (*)(class Tensor *)) &eddlT::log2_, "C++: eddlT::log2_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::log2(class Tensor *) file:eddl/apis/eddlT.h line:161
	M("eddlT").def("log2", (class Tensor * (*)(class Tensor *)) &eddlT::log2, "C++: eddlT::log2(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::log10_(class Tensor *) file:eddl/apis/eddlT.h line:163
	M("eddlT").def("log10_", (void (*)(class Tensor *)) &eddlT::log10_, "C++: eddlT::log10_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::log10(class Tensor *) file:eddl/apis/eddlT.h line:164
	M("eddlT").def("log10", (class Tensor * (*)(class Tensor *)) &eddlT::log10, "C++: eddlT::log10(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::logn_(class Tensor *, float) file:eddl/apis/eddlT.h line:166
	M("eddlT").def("logn_", (void (*)(class Tensor *, float)) &eddlT::logn_, "C++: eddlT::logn_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("n"));

	// eddlT::logn(class Tensor *, float) file:eddl/apis/eddlT.h line:167
	M("eddlT").def("logn", (class Tensor * (*)(class Tensor *, float)) &eddlT::logn, "C++: eddlT::logn(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("n"));

	// eddlT::max(class Tensor *) file:eddl/apis/eddlT.h line:169
	M("eddlT").def("max", (float (*)(class Tensor *)) &eddlT::max, "C++: eddlT::max(class Tensor *) --> float", pybind11::arg("A"));

	// eddlT::min(class Tensor *) file:eddl/apis/eddlT.h line:170
	M("eddlT").def("min", (float (*)(class Tensor *)) &eddlT::min, "C++: eddlT::min(class Tensor *) --> float", pybind11::arg("A"));

	// eddlT::mod_(class Tensor *, float) file:eddl/apis/eddlT.h line:172
	M("eddlT").def("mod_", (void (*)(class Tensor *, float)) &eddlT::mod_, "C++: eddlT::mod_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::mod(class Tensor *, float) file:eddl/apis/eddlT.h line:173
	M("eddlT").def("mod", (class Tensor * (*)(class Tensor *, float)) &eddlT::mod, "C++: eddlT::mod(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

}


// File: eddl/apis/eddlT_1.cpp
#include <eddl/apis/eddlT.h>
#include <eddl/descriptors/tensor_descriptors.h>
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
	// eddlT::mult_(class Tensor *, float) file:eddl/apis/eddlT.h line:175
	M("eddlT").def("mult_", (void (*)(class Tensor *, float)) &eddlT::mult_, "C++: eddlT::mult_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::mult(class Tensor *, float) file:eddl/apis/eddlT.h line:176
	M("eddlT").def("mult", (class Tensor * (*)(class Tensor *, float)) &eddlT::mult, "C++: eddlT::mult(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::mult_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:177
	M("eddlT").def("mult_", (void (*)(class Tensor *, class Tensor *)) &eddlT::mult_, "C++: eddlT::mult_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::mult(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:178
	M("eddlT").def("mult", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::mult, "C++: eddlT::mult(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::mult2D(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:179
	M("eddlT").def("mult2D", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::mult2D, "C++: eddlT::mult2D(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::neg_(class Tensor *) file:eddl/apis/eddlT.h line:181
	M("eddlT").def("neg_", (void (*)(class Tensor *)) &eddlT::neg_, "C++: eddlT::neg_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::neg(class Tensor *) file:eddl/apis/eddlT.h line:182
	M("eddlT").def("neg", (class Tensor * (*)(class Tensor *)) &eddlT::neg, "C++: eddlT::neg(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::normalize_(class Tensor *, float, float) file:eddl/apis/eddlT.h line:184
	M("eddlT").def("normalize_", (void (*)(class Tensor *, float, float)) &eddlT::normalize_, "C++: eddlT::normalize_(class Tensor *, float, float) --> void", pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::normalize(class Tensor *, float, float) file:eddl/apis/eddlT.h line:185
	M("eddlT").def("normalize", (class Tensor * (*)(class Tensor *, float, float)) &eddlT::normalize, "C++: eddlT::normalize(class Tensor *, float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("min"), pybind11::arg("max"));

	// eddlT::reciprocal_(class Tensor *) file:eddl/apis/eddlT.h line:190
	M("eddlT").def("reciprocal_", (void (*)(class Tensor *)) &eddlT::reciprocal_, "C++: eddlT::reciprocal_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::reciprocal(class Tensor *) file:eddl/apis/eddlT.h line:191
	M("eddlT").def("reciprocal", (class Tensor * (*)(class Tensor *)) &eddlT::reciprocal, "C++: eddlT::reciprocal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::round_(class Tensor *) file:eddl/apis/eddlT.h line:196
	M("eddlT").def("round_", (void (*)(class Tensor *)) &eddlT::round_, "C++: eddlT::round_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::round(class Tensor *) file:eddl/apis/eddlT.h line:197
	M("eddlT").def("round", (class Tensor * (*)(class Tensor *)) &eddlT::round, "C++: eddlT::round(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::rsqrt_(class Tensor *) file:eddl/apis/eddlT.h line:199
	M("eddlT").def("rsqrt_", (void (*)(class Tensor *)) &eddlT::rsqrt_, "C++: eddlT::rsqrt_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::rsqrt(class Tensor *) file:eddl/apis/eddlT.h line:200
	M("eddlT").def("rsqrt", (class Tensor * (*)(class Tensor *)) &eddlT::rsqrt, "C++: eddlT::rsqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sigmoid_(class Tensor *) file:eddl/apis/eddlT.h line:202
	M("eddlT").def("sigmoid_", (void (*)(class Tensor *)) &eddlT::sigmoid_, "C++: eddlT::sigmoid_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sigmoid(class Tensor *) file:eddl/apis/eddlT.h line:203
	M("eddlT").def("sigmoid", (class Tensor * (*)(class Tensor *)) &eddlT::sigmoid, "C++: eddlT::sigmoid(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sign_(class Tensor *) file:eddl/apis/eddlT.h line:205
	M("eddlT").def("sign_", (void (*)(class Tensor *)) &eddlT::sign_, "C++: eddlT::sign_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sign(class Tensor *) file:eddl/apis/eddlT.h line:206
	M("eddlT").def("sign", (class Tensor * (*)(class Tensor *)) &eddlT::sign, "C++: eddlT::sign(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sin_(class Tensor *) file:eddl/apis/eddlT.h line:209
	M("eddlT").def("sin_", (void (*)(class Tensor *)) &eddlT::sin_, "C++: eddlT::sin_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sin(class Tensor *) file:eddl/apis/eddlT.h line:210
	M("eddlT").def("sin", (class Tensor * (*)(class Tensor *)) &eddlT::sin, "C++: eddlT::sin(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sinh_(class Tensor *) file:eddl/apis/eddlT.h line:212
	M("eddlT").def("sinh_", (void (*)(class Tensor *)) &eddlT::sinh_, "C++: eddlT::sinh_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sinh(class Tensor *) file:eddl/apis/eddlT.h line:213
	M("eddlT").def("sinh", (class Tensor * (*)(class Tensor *)) &eddlT::sinh, "C++: eddlT::sinh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sqr_(class Tensor *) file:eddl/apis/eddlT.h line:215
	M("eddlT").def("sqr_", (void (*)(class Tensor *)) &eddlT::sqr_, "C++: eddlT::sqr_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sqr(class Tensor *) file:eddl/apis/eddlT.h line:216
	M("eddlT").def("sqr", (class Tensor * (*)(class Tensor *)) &eddlT::sqr, "C++: eddlT::sqr(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sqrt_(class Tensor *) file:eddl/apis/eddlT.h line:218
	M("eddlT").def("sqrt_", (void (*)(class Tensor *)) &eddlT::sqrt_, "C++: eddlT::sqrt_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::sqrt(class Tensor *) file:eddl/apis/eddlT.h line:219
	M("eddlT").def("sqrt", (class Tensor * (*)(class Tensor *)) &eddlT::sqrt, "C++: eddlT::sqrt(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::sub_(class Tensor *, float) file:eddl/apis/eddlT.h line:221
	M("eddlT").def("sub_", (void (*)(class Tensor *, float)) &eddlT::sub_, "C++: eddlT::sub_(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::sub(class Tensor *, float) file:eddl/apis/eddlT.h line:222
	M("eddlT").def("sub", (class Tensor * (*)(class Tensor *, float)) &eddlT::sub, "C++: eddlT::sub(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));

	// eddlT::sub_(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:223
	M("eddlT").def("sub_", (void (*)(class Tensor *, class Tensor *)) &eddlT::sub_, "C++: eddlT::sub_(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::sub(class Tensor *, class Tensor *) file:eddl/apis/eddlT.h line:224
	M("eddlT").def("sub", (class Tensor * (*)(class Tensor *, class Tensor *)) &eddlT::sub, "C++: eddlT::sub(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));

	// eddlT::tan_(class Tensor *) file:eddl/apis/eddlT.h line:229
	M("eddlT").def("tan_", (void (*)(class Tensor *)) &eddlT::tan_, "C++: eddlT::tan_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::tan(class Tensor *) file:eddl/apis/eddlT.h line:230
	M("eddlT").def("tan", (class Tensor * (*)(class Tensor *)) &eddlT::tan, "C++: eddlT::tan(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::tanh_(class Tensor *) file:eddl/apis/eddlT.h line:232
	M("eddlT").def("tanh_", (void (*)(class Tensor *)) &eddlT::tanh_, "C++: eddlT::tanh_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::tanh(class Tensor *) file:eddl/apis/eddlT.h line:233
	M("eddlT").def("tanh", (class Tensor * (*)(class Tensor *)) &eddlT::tanh, "C++: eddlT::tanh(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

	// eddlT::trunc_(class Tensor *) file:eddl/apis/eddlT.h line:235
	M("eddlT").def("trunc_", (void (*)(class Tensor *)) &eddlT::trunc_, "C++: eddlT::trunc_(class Tensor *) --> void", pybind11::arg("A"));

	// eddlT::trunc(class Tensor *) file:eddl/apis/eddlT.h line:236
	M("eddlT").def("trunc", (class Tensor * (*)(class Tensor *)) &eddlT::trunc, "C++: eddlT::trunc(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));

}


#include <map>
#include <memory>
#include <stdexcept>
#include <functional>
#include <string>

#include <pybind11/pybind11.h>

typedef std::function< pybind11::module & (std::string const &) > ModuleGetter;

void bind_eddl_descriptors_tensor_descriptors(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_tensor_tensor(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_losses_loss(std::function< pybind11::module &(std::string const &namespace_) > &M);
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

	bind_eddl_descriptors_tensor_descriptors(M);
	bind_eddl_tensor_tensor(M);
	bind_eddl_losses_loss(M);
	bind_eddl_apis_eddl(M);
	bind_eddl_apis_eddlT(M);
	bind_eddl_apis_eddlT_1(M);

}

// Source list file: /pyeddl/codegen/bindings/_core.sources
// _core.cpp
// eddl/descriptors/tensor_descriptors.cpp
// eddl/tensor/tensor.cpp
// eddl/losses/loss.cpp
// eddl/apis/eddl.cpp
// eddl/apis/eddlT.cpp
// eddl/apis/eddlT_1.cpp

// Modules list file: /pyeddl/codegen/bindings/_core.modules
// eddl eddlT 
