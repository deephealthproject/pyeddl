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
#include <utils_addons.hpp>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

// TensorDescriptor file:eddl/descriptors/tensor_descriptors.h line:25
struct PyCallBack_TensorDescriptor : public TensorDescriptor {
	using TensorDescriptor::TensorDescriptor;

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

// SelDescriptor file:eddl/descriptors/tensor_descriptors.h line:47
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
};

// ReduceDescriptor2 file:eddl/descriptors/tensor_descriptors.h line:77
struct PyCallBack_ReduceDescriptor2 : public ReduceDescriptor2 {
	using ReduceDescriptor2::ReduceDescriptor2;

	void resize(int a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const ReduceDescriptor2 *>(this), "resize");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return ReduceDescriptor2::resize(a0);
	}
};

void bind_eddl_descriptors_tensor_descriptors(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // TensorDescriptor file:eddl/descriptors/tensor_descriptors.h line:25
		pybind11::class_<TensorDescriptor, std::shared_ptr<TensorDescriptor>, PyCallBack_TensorDescriptor> cl(M(""), "TensorDescriptor", "");
		cl.def( pybind11::init<int>(), pybind11::arg("dev") );

		cl.def( pybind11::init( [](PyCallBack_TensorDescriptor const &o){ return new PyCallBack_TensorDescriptor(o); } ) );
		cl.def( pybind11::init( [](TensorDescriptor const &o){ return new TensorDescriptor(o); } ) );
		cl.def_readwrite("device", &TensorDescriptor::device);
		cl.def("resize", (void (TensorDescriptor::*)(int)) &TensorDescriptor::resize, "C++: TensorDescriptor::resize(int) --> void", pybind11::arg("b"));
		cl.def("free_memory", (void (TensorDescriptor::*)()) &TensorDescriptor::free_memory, "C++: TensorDescriptor::free_memory() --> void");
		cl.def("assign", (class TensorDescriptor & (TensorDescriptor::*)(const class TensorDescriptor &)) &TensorDescriptor::operator=, "C++: TensorDescriptor::operator=(const class TensorDescriptor &) --> class TensorDescriptor &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // SelDescriptor file:eddl/descriptors/tensor_descriptors.h line:47
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
	{ // ReduceDescriptor2 file:eddl/descriptors/tensor_descriptors.h line:77
		pybind11::class_<ReduceDescriptor2, std::shared_ptr<ReduceDescriptor2>, PyCallBack_ReduceDescriptor2, TensorDescriptor> cl(M(""), "ReduceDescriptor2", "");
		cl.def( pybind11::init( [](PyCallBack_ReduceDescriptor2 const &o){ return new PyCallBack_ReduceDescriptor2(o); } ) );
		cl.def( pybind11::init( [](ReduceDescriptor2 const &o){ return new ReduceDescriptor2(o); } ) );
		cl.def_readwrite("axis", &ReduceDescriptor2::axis);
		cl.def_readwrite("keepdims", &ReduceDescriptor2::keepdims);
		cl.def_readwrite("index", &ReduceDescriptor2::index);
		cl.def_readwrite("ishape", &ReduceDescriptor2::ishape);
		cl.def_readwrite("oshape", &ReduceDescriptor2::oshape);
		cl.def_readwrite("size_reduction", &ReduceDescriptor2::size_reduction);
		cl.def("resize", (void (ReduceDescriptor2::*)(int)) &ReduceDescriptor2::resize, "C++: ReduceDescriptor2::resize(int) --> void", pybind11::arg("b"));
		cl.def("build_map", [](ReduceDescriptor2 &o) -> void { return o.build_map(); }, "");
		cl.def("build_map", (void (ReduceDescriptor2::*)(bool)) &ReduceDescriptor2::build_map, "C++: ReduceDescriptor2::build_map(bool) --> void", pybind11::arg("reverse"));
		cl.def("assign", (class ReduceDescriptor2 & (ReduceDescriptor2::*)(const class ReduceDescriptor2 &)) &ReduceDescriptor2::operator=, "C++: ReduceDescriptor2::operator=(const class ReduceDescriptor2 &) --> class ReduceDescriptor2 &", pybind11::return_value_policy::automatic, pybind11::arg(""));
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
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <utils_addons.hpp>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_tensor_tensor(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
	{ // Tensor file:eddl/tensor/tensor.h line:76
		pybind11::class_<Tensor, std::shared_ptr<Tensor>> cl(M(""), "Tensor", pybind11::buffer_protocol());
		cl.def( pybind11::init( [](){ return new Tensor(); } ) );
		cl.def( pybind11::init( [](Tensor const &o){ return new Tensor(o); } ) );
		cl.def_readwrite("device", &Tensor::device);
		cl.def_readwrite("isshared", &Tensor::isshared);
		cl.def_readwrite("ndim", &Tensor::ndim);
		cl.def_readwrite("size", &Tensor::size);
		cl.def_readwrite("shape", &Tensor::shape);
		cl.def_readwrite("stride", &Tensor::stride);
		cl.def_readwrite("gpu_device", &Tensor::gpu_device);
		cl.def("updateDevice", (void (Tensor::*)(int)) &Tensor::updateDevice, "C++: Tensor::updateDevice(int) --> void", pybind11::arg("dev"));
		cl.def("updateSize", (void (Tensor::*)()) &Tensor::updateSize, "C++: Tensor::updateSize() --> void");
		cl.def("updateStrides", (void (Tensor::*)()) &Tensor::updateStrides, "C++: Tensor::updateStrides() --> void");
		cl.def("updateData", [](Tensor &o, float * a0) -> void { return o.updateData(a0); }, "", pybind11::arg("ptr"));
		cl.def("updateData", [](Tensor &o, float * a0, void * a1) -> void { return o.updateData(a0, a1); }, "", pybind11::arg("ptr"), pybind11::arg("ptr2"));
		cl.def("updateData", (void (Tensor::*)(float *, void *, bool)) &Tensor::updateData, "C++: Tensor::updateData(float *, void *, bool) --> void", pybind11::arg("ptr"), pybind11::arg("ptr2"), pybind11::arg("setshared"));
		cl.def("deleteData", (void (Tensor::*)()) &Tensor::deleteData, "C++: Tensor::deleteData() --> void");
		cl.def("toCPU", [](Tensor &o) -> void { return o.toCPU(); }, "");
		cl.def("toCPU", (void (Tensor::*)(int)) &Tensor::toCPU, "Clone a tensor to the CPU.\n\nC++: Tensor::toCPU(int) --> void", pybind11::arg("dev"));
		cl.def("toGPU", [](Tensor &o) -> void { return o.toGPU(); }, "");
		cl.def("toGPU", (void (Tensor::*)(int)) &Tensor::toGPU, "Clone a tensor to the GPU.\n\nC++: Tensor::toGPU(int) --> void", pybind11::arg("dev"));
		cl.def("toFPGA", [](Tensor &o) -> void { return o.toFPGA(); }, "");
		cl.def("toFPGA", (void (Tensor::*)(int)) &Tensor::toFPGA, "Clone a tensor to the GFPGA.\n\nC++: Tensor::toFPGA(int) --> void", pybind11::arg("dev"));
		cl.def("toDevice", (void (Tensor::*)(int)) &Tensor::toDevice, "Clone a tensor to a specific device.\n\nC++: Tensor::toDevice(int) --> void", pybind11::arg("dev"));
		cl.def("isCPU", (int (Tensor::*)()) &Tensor::isCPU, "Check if the tensor is in CPU.\n\n  \n int\n\nC++: Tensor::isCPU() --> int");
		cl.def("isGPU", (int (Tensor::*)()) &Tensor::isGPU, "Check if the tensor is in GPU.\n\n  \n int\n\nC++: Tensor::isGPU() --> int");
		cl.def("isFPGA", (int (Tensor::*)()) &Tensor::isFPGA, "Check if the tensor is in FPGA.\n\n  \n int\n\nC++: Tensor::isFPGA() --> int");
		cl.def("info", (void (Tensor::*)()) &Tensor::info, "Print shape, device and size information.\n\n  \n    void\n\nC++: Tensor::info() --> void");
		cl.def("print", [](Tensor &o) -> void { return o.print(); }, "");
		cl.def("print", [](Tensor &o, int const & a0) -> void { return o.print(a0); }, "", pybind11::arg("precision"));
		cl.def("print", (void (Tensor::*)(int, bool)) &Tensor::print, "Print the tensor values.\n\n  \n    void\n\nC++: Tensor::print(int, bool) --> void", pybind11::arg("precision"), pybind11::arg("raw"));
		cl.def("getDeviceID", (int (Tensor::*)(int) const) &Tensor::getDeviceID, "Returns the device name given a device number\n\n  \n    string\n\nC++: Tensor::getDeviceID(int) const --> int", pybind11::arg("dev"));
		cl.def("numel", (unsigned int (Tensor::*)()) &Tensor::numel, "C++: Tensor::numel() --> unsigned int");
		cl.def_static("isSquared", (bool (*)(class Tensor *)) &Tensor::isSquared, "Check if all dimensions in the tensor are the same.\n\n  \n   Tensor\n  \n\n    bool\n\nC++: Tensor::isSquared(class Tensor *) --> bool", pybind11::arg("A"));
		cl.def_static("load_from_ptr", (class Tensor * (*)(void *)) &Tensor::load_from_ptr, "Load tensor from a void pointer.\n\n  \n    Void pointer to the serialized tensor.\n  \n\n    Tensor\n\nC++: Tensor::load_from_ptr(void *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("src"));
		cl.def_static("empty_like", (class Tensor * (*)(class Tensor *)) &Tensor::empty_like, "Create a tensor with the shape and device of another one, but empty\n\n  \n  Input tensor from wich to take shape and device.\n  \n\n     Empty initialized A-shaped tensor\n\nC++: Tensor::empty_like(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("zeros_like", (class Tensor * (*)(class Tensor *)) &Tensor::zeros_like, "Create a tensor with the shape and device of another one, initialized with zeros\n\n  \n  Input tensor from wich to take shape and device.\n  \n\n     Zeros initialized A-shaped tensor\n\nC++: Tensor::zeros_like(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("ones_like", (class Tensor * (*)(class Tensor *)) &Tensor::ones_like, "Create a tensor with the shape and device of another one, initialized with ones\n\n  \n  Input tensor from wich to take shape and device.\n  \n\n     Ones initialized A-shaped tensor\n\nC++: Tensor::ones_like(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("full_like", (class Tensor * (*)(class Tensor *, float)) &Tensor::full_like, "Create a tensor with the shape an device of the input tensor and filled with a specific value.\n\n  \n  Input tensor from wich to take shape and device.\n  \n\n  Value to use to fill the tensor\n  \n\n     Value initialized A-shaped tensor\n\nC++: Tensor::full_like(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("value"));
		cl.def_static("arange", [](float const & a0, float const & a1) -> Tensor * { return Tensor::arange(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("arange", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::arange(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
		cl.def_static("arange", (class Tensor * (*)(float, float, float, int)) &Tensor::arange, "Create a 1-D tensor of size ceil(end - start) with values from start to end with step step.\n\n   \n Start index\n   \n\n  End index\n   \n\n  The gap between two values in the tensor.\n   \n\n One of ``DEV_CPU``or ``DEV_GPU``\n   \n\n The new tensor\n\nC++: Tensor::arange(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("range", [](float const & a0, float const & a1) -> Tensor * { return Tensor::range(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("range", [](float const & a0, float const & a1, float const & a2) -> Tensor * { return Tensor::range(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"));
		cl.def_static("range", (class Tensor * (*)(float, float, float, int)) &Tensor::range, "Creates a 1-D tensor of size floor(end - start)/ + 1 with values from start to end with step step.\n\n   \n Start value\n   \n\n  End value\n   \n\n  The gap between two values in the tensor.\n   \n\n One of ``DEV_CPU``or ``DEV_GPU``\n   \n\n The new tensor\n\nC++: Tensor::range(float, float, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("step"), pybind11::arg("dev"));
		cl.def_static("linspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::linspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("linspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::linspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("linspace", (class Tensor * (*)(float, float, int, int)) &Tensor::linspace, "Creates a 1-D tensor with a sequence of num evenly-spaced values starting at start. If steps > 1, the values in the sequence increase by end - start / steps - 1, so that the last one is exactly end.\n   \n\n Start value\n   \n\n  End value\n   \n\n  The gap between two values in the tensor.\n   \n\n One of ``DEV_CPU``or ``DEV_GPU``\n   \n\n The new tensor\n\nC++: Tensor::linspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));
		cl.def_static("logspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::logspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::logspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("logspace", [](float const & a0, float const & a1, int const & a2, float const & a3) -> Tensor * { return Tensor::logspace(a0, a1, a2, a3); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"));
		cl.def_static("logspace", (class Tensor * (*)(float, float, int, float, int)) &Tensor::logspace, "Creates a 1-D tensor with a sequence of num  logarithmic spaced values starting at start. If steps > 1, the values in the sequence increase by end - start / steps - 1, so that the last one is exactly end.\n   \n\n Start value\n   \n\n  End value\n   \n\n  The gap between two values in the tensor.\n   \n\n  The base of the logarithm to apply.\n   \n\n One of ``DEV_CPU``or ``DEV_GPU``\n   \n\n The new tensor\n\nC++: Tensor::logspace(float, float, int, float, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("base"), pybind11::arg("dev"));
		cl.def_static("geomspace", [](float const & a0, float const & a1) -> Tensor * { return Tensor::geomspace(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"));
		cl.def_static("geomspace", [](float const & a0, float const & a1, int const & a2) -> Tensor * { return Tensor::geomspace(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"));
		cl.def_static("geomspace", (class Tensor * (*)(float, float, int, int)) &Tensor::geomspace, "Creates a 1-D tensor with a sequence of num  geometrically spaced values starting at start. If steps > 1, the values in the sequence increase by end - start / steps - 1, so that the last one is exactly end.\n   \n\n Start value\n   \n\n  End value\n   \n\n  The gap between two values in the tensor.\n   \n\n One of ``DEV_CPU``or ``DEV_GPU``\n   \n\n The new tensor\n\nC++: Tensor::geomspace(float, float, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("start"), pybind11::arg("end"), pybind11::arg("steps"), pybind11::arg("dev"));
		cl.def_static("eye", [](int const & a0) -> Tensor * { return Tensor::eye(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("rows"));
		cl.def_static("eye", [](int const & a0, int const & a1) -> Tensor * { return Tensor::eye(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("rows"), pybind11::arg("offset"));
		cl.def_static("eye", (class Tensor * (*)(int, int, int)) &Tensor::eye, "Number of rows of the tensor.\n  \n\n\n  \n\n    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.\n  \n\n     Tensor of the specified shape filled with the value\n\nC++: Tensor::eye(int, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("rows"), pybind11::arg("offset"), pybind11::arg("dev"));
		cl.def_static("identity", [](int const & a0) -> Tensor * { return Tensor::identity(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("rows"));
		cl.def_static("identity", (class Tensor * (*)(int, int)) &Tensor::identity, "Create a tensor representing the identity matrix. Equivalent to calling function ``eye`` with ``offset = 0``.\n\n  \n  Shape of the tensor to create.\n  \n\n    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.\n  \n\n     Tensor of the specified shape filled with the value\n\nC++: Tensor::identity(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("rows"), pybind11::arg("dev"));
		cl.def("diag_", [](Tensor &o) -> void { return o.diag_(); }, "");
		cl.def("diag_", (void (Tensor::*)(int)) &Tensor::diag_, "Extract a diagonal or construct a diagonal array. In-place operation.\n\n  \n  Offset. If k=0, main diagonal is selected. If k>0, a diagonal above the main diagonal is selected. If k<0, a diagonal below the main diagonal is selected.\n\nC++: Tensor::diag_(int) --> void", pybind11::arg("k"));
		cl.def("diag", [](Tensor &o) -> Tensor * { return o.diag(); }, "", pybind11::return_value_policy::automatic);
		cl.def("diag", (class Tensor * (Tensor::*)(int)) &Tensor::diag, "Extract a diagonal or construct a diagonal array.\n\n  \n  Offset. If k=0, main diagonal is selected. If k>0, a diagonal above the main diagonal is selected. If k<0, a diagonal below the main diagonal is selected.\n  \n\n  A new tensor with the elements on the selected diagonal.\n\nC++: Tensor::diag(int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("k"));
		cl.def_static("static_diag", [](class Tensor * a0, class Tensor * a1) -> void { return Tensor::diag(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_diag", (void (*)(class Tensor *, class Tensor *, int)) &Tensor::diag, "Extract a diagonal or construct a diagonal array.\n\n  \n  Input matrix.\n  \n\n  Output matrix.\n  \n\n  Offset. If k=0, main diagonal is selected. If k>0, a diagonal above the main diagonal is selected. If k<0, a diagonal below the main diagonal is selected.\n\nC++: Tensor::diag(class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("k"));
		cl.def("maximum", (class Tensor * (Tensor::*)(float)) &Tensor::maximum, "Apply a lower bound to the elements in a tensor.\n\n  \n  Lower bound.\n  \n\n A new tensor with the values lower than v set to v.\n\nC++: Tensor::maximum(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_maximum", (class Tensor * (*)(class Tensor *, float)) &Tensor::maximum, "Apply a lower bound to the elements in a tensor.\n\n  \n  Input tensor.\n  \n\n  Lower bound.\n  \n\n A new tensor with the values of A lower than v set to v.\n\nC++: Tensor::maximum(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def_static("static_maximum", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::maximum, "Apply a lower bound to the elements in a tensor.\n\n  \n  Input tensor.\n  \n\n  Output tensor.\n  \n\n  Lower bound.\n\nC++: Tensor::maximum(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def_static("static_maximum", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::maximum, "Element-wise selection of the maximum values in the same position in two tensors.\n\n  \n  Input tensor.\n  \n\n  Input tensor.\n  \n\n  A tensor with the higher value in the same position between A and B.\n\nC++: Tensor::maximum(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_maximum", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::maximum, "Element-wise selection of the maximum values in the same position in two tensors.\n\n  \n  Input tensor.\n  \n\n  Input tensor.\n  \n\n  Output tensor with the higher value in the same position between A and B.\n\nC++: Tensor::maximum(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("minimum", (class Tensor * (Tensor::*)(float)) &Tensor::minimum, "Apply a upper bound to the elements in a tensor.\n\n  \n  Lower bound.\n  \n\n A new tensor with the values higher than v set to v.\n\nC++: Tensor::minimum(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_minimum", (class Tensor * (*)(class Tensor *, float)) &Tensor::minimum, "Apply a upper bound to the elements in a tensor.\n\n  \n  Input tensor.\n  \n\n  Lower bound.\n  \n\n A new tensor with the values of A higher than v set to v.\n\nC++: Tensor::minimum(class Tensor *, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("v"));
		cl.def_static("static_minimum", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::minimum, "Apply a upper bound to the elements in a tensor.\n\n  \n  Input tensor.\n  \n\n  Output tensor.\n  \n\n  Upper bound.\n\nC++: Tensor::minimum(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def_static("static_minimum", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::minimum, "Element-wise selection of the minimum values in the same position in two tensors.\n\n  \n  Input tensor.\n  \n\n  Input tensor.\n  \n\n  A tensor with the lower value in the same position between A and B.\n\nC++: Tensor::minimum(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_minimum", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::minimum, "Element-wise selection of the minimum values in the same position in two tensors.\n\n  \n  Input tensor.\n  \n\n  Input tensor.\n  \n\n  Output tensor with the lower value in the same position between A and B.\n\nC++: Tensor::minimum(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("max", (float (Tensor::*)()) &Tensor::max, "Obtain the maximum value in the tensor\n   \n\n float. The maximum value in the tensor\n\nC++: Tensor::max() --> float");
		cl.def_static("static_max", (float (*)(class Tensor *)) &Tensor::max, "Obtain the maximum value in a tensor\n   \n\n The tensor where the operation is applied\n   \n\n The maximum value in A\n\nC++: Tensor::max(class Tensor *) --> float", pybind11::arg("A"));
		cl.def_static("static_max", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::max, "C++: Tensor::max(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("argmax", (int (Tensor::*)()) &Tensor::argmax, "Obtain the index of the maximum value in the tensor\n   \n\n The desired index.\n\nC++: Tensor::argmax() --> int");
		cl.def_static("static_argmax", (int (*)(class Tensor *)) &Tensor::argmax, "Obtain the index of the maximum value in the tensor\n   \n\n The tensor where the operation is applied.\n   \n\n The desired index.\n\nC++: Tensor::argmax(class Tensor *) --> int", pybind11::arg("A"));
		cl.def_static("static_argmax", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::argmax, "C++: Tensor::argmax(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def_static("argmax_d", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::argmax_d, "C++: Tensor::argmax_d(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("D"), pybind11::arg("O"), pybind11::arg("PD"));
		cl.def("min", (float (Tensor::*)()) &Tensor::min, "Obtain the minimum value in the tensor\n   \n\n float. The minimum value in the tensor\n\nC++: Tensor::min() --> float");
		cl.def_static("static_min", (float (*)(class Tensor *)) &Tensor::min, "Obtain the minimum value in a tensor\n   \n\n The tensor where the operation is applied\n   \n\n The minimum value in A\n\nC++: Tensor::min(class Tensor *) --> float", pybind11::arg("A"));
		cl.def_static("static_min", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::min, "C++: Tensor::min(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("argmin", (int (Tensor::*)()) &Tensor::argmin, "Obtain the index of the minimum value in the tensor\n   \n\n The desired index.\n\nC++: Tensor::argmin() --> int");
		cl.def_static("static_argmin", (int (*)(class Tensor *)) &Tensor::argmin, "Obtain the index of the minimum value in the tensor\n   \n\n The tensor where the operation is applied.\n   \n\n The desired index.\n\nC++: Tensor::argmin(class Tensor *) --> int", pybind11::arg("A"));
		cl.def_static("static_argmin", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::argmin, "C++: Tensor::argmin(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("sum", (float (Tensor::*)()) &Tensor::sum, "Obtain the sum of all the values in the tensor.\n   \n\n The sum of all the elements in the tensor.\n\nC++: Tensor::sum() --> float");
		cl.def_static("static_sum", (float (*)(class Tensor *)) &Tensor::sum, "Obtain the sum of all the values in a tensor.\n   \n\n Input tensor.\n   \n\n The sum of all the elements in the input tensor.\n\nC++: Tensor::sum(class Tensor *) --> float", pybind11::arg("A"));
		cl.def_static("static_sum", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::sum, "C++: Tensor::sum(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("sum_abs", (float (Tensor::*)()) &Tensor::sum_abs, "Obtain the absolute value sum of all the values in the tensor.\n   \n\n The absolute value sum of all the elements in the tensor.\n\nC++: Tensor::sum_abs() --> float");
		cl.def_static("static_sum_abs", (float (*)(class Tensor *)) &Tensor::sum_abs, "Obtain the absolute value sum of all the values in a tensor.\n   \n\n Input tensor.\n   \n\n The absolute value sum of all the elements in the input tensor.\n\nC++: Tensor::sum_abs(class Tensor *) --> float", pybind11::arg("A"));
		cl.def_static("static_sum_abs", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::sum_abs, "C++: Tensor::sum_abs(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("prod", (float (Tensor::*)()) &Tensor::prod, "Obtain the product of all the values in the tensor.\n   \n\n The product of all the elements in the tensor.\n\nC++: Tensor::prod() --> float");
		cl.def_static("static_prod", (float (*)(class Tensor *)) &Tensor::prod, "Obtain the product of all the values in a tensor.\n   \n\n Input tensor.\n   \n\n The product of all the elements in the input tensor.\n\nC++: Tensor::prod(class Tensor *) --> float", pybind11::arg("A"));
		cl.def_static("static_prod", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::prod, "C++: Tensor::prod(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("mean", (float (Tensor::*)()) &Tensor::mean, "Obtain the mean of all the values in the tensor.\n   \n\n The mean of all the elements in the tensor.\n\nC++: Tensor::mean() --> float");
		cl.def_static("static_mean", (float (*)(class Tensor *)) &Tensor::mean, "Obtain the mean of all the values in a tensor.\n   \n\n Input tensor.\n   \n\n The mean of all the elements in the input tensor.\n\nC++: Tensor::mean(class Tensor *) --> float", pybind11::arg("A"));
		cl.def_static("static_mean", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::mean, "C++: Tensor::mean(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("median", (float (Tensor::*)()) &Tensor::median, "Obtain the median value of all the elements in the tensor\n   \n\n float The median value.\n\nC++: Tensor::median() --> float");
		cl.def_static("static_median", (float (*)(class Tensor *)) &Tensor::median, "Obtain the median value of all the elements in the tensor\n   \n\n The tensor from which to extract the median of its values\n   \n\n float. The median value.\n\nC++: Tensor::median(class Tensor *) --> float", pybind11::arg("A"));
		cl.def_static("static_median", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::median, "C++: Tensor::median(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("std", [](Tensor &o) -> float { return o.std(); }, "");
		cl.def("std", (float (Tensor::*)(bool)) &Tensor::std, "Obtain the standard deviation of all the values in the tensor.\n   \n\n Whether the standard deviation is computed using the unbiased estimation or not.\n   \n\n The standard deviation of all the elements in the tensor.\n\nC++: Tensor::std(bool) --> float", pybind11::arg("unbiased"));
		cl.def_static("static_std", [](class Tensor * a0) -> float { return Tensor::std(a0); }, "", pybind11::arg("A"));
		cl.def_static("static_std", (float (*)(class Tensor *, bool)) &Tensor::std, "Obtain the standard deviation of all the values in a tensor.\n   \n\n Input tensor.\n   \n\n Whether the standard deviation is computed using the unbiased estimation or not.\n   \n\n The standard deviation of all the elements in the input tensor.\n\nC++: Tensor::std(class Tensor *, bool) --> float", pybind11::arg("A"), pybind11::arg("unbiased"));
		cl.def_static("static_std", [](class Tensor * a0, class Tensor * a1, class ReduceDescriptor2 * a2) -> void { return Tensor::std(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def_static("static_std", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *, bool)) &Tensor::std, "C++: Tensor::std(class Tensor *, class Tensor *, class ReduceDescriptor2 *, bool) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"), pybind11::arg("unbiased"));
		cl.def("var", [](Tensor &o) -> float { return o.var(); }, "");
		cl.def("var", (float (Tensor::*)(bool)) &Tensor::var, "Obtain the variance of all the values in the tensor.\n   \n\n Whether the variance is computed using the unbiased estimation or not.\n   \n\n The variance of all the elements in the tensor.\n\nC++: Tensor::var(bool) --> float", pybind11::arg("unbiased"));
		cl.def_static("static_var", [](class Tensor * a0) -> float { return Tensor::var(a0); }, "", pybind11::arg("A"));
		cl.def_static("static_var", (float (*)(class Tensor *, bool)) &Tensor::var, "Obtain the variance of all the values in a tensor.\n   \n\n Input tensor.\n   \n\n Whether the variance is computed using the unbiased estimation or not.\n   \n\n The variance of all the elements in the input tensor.\n\nC++: Tensor::var(class Tensor *, bool) --> float", pybind11::arg("A"), pybind11::arg("unbiased"));
		cl.def_static("static_var", [](class Tensor * a0, class Tensor * a1, class ReduceDescriptor2 * a2) -> void { return Tensor::var(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def_static("static_var", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *, bool)) &Tensor::var, "C++: Tensor::var(class Tensor *, class Tensor *, class ReduceDescriptor2 *, bool) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"), pybind11::arg("unbiased"));
		cl.def("mode", (int (Tensor::*)()) &Tensor::mode, "Obtain the mode of all the values in the tensor.\n   \n\n The mode of all the elements in the tensor.\n\nC++: Tensor::mode() --> int");
		cl.def_static("static_mode", (int (*)(class Tensor *)) &Tensor::mode, "Obtain the mode of all the values in a tensor.\n   \n\n Input tensor.\n   \n\n The mode of all the elements in the input tensor.\n\nC++: Tensor::mode(class Tensor *) --> int", pybind11::arg("A"));
		cl.def_static("static_mode", (void (*)(class Tensor *, class Tensor *, class ReduceDescriptor2 *)) &Tensor::mode, "C++: Tensor::mode(class Tensor *, class Tensor *, class ReduceDescriptor2 *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rd"));
		cl.def("abs_", (void (Tensor::*)()) &Tensor::abs_, "In-place element-wise abs operation\n\nC++: Tensor::abs_() --> void");
		cl.def("abs", (class Tensor * (Tensor::*)()) &Tensor::abs, "Element-wise abs operation\n   \n\n A new tensor with abs applied over A\n\nC++: Tensor::abs() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_abs", (void (*)(class Tensor *, class Tensor *)) &Tensor::abs, "Element-wise abs operation\n   \n\n The tensor where the operation is applied\n   \n\n A new tensor with abs applied over A\n\nC++: Tensor::abs(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("acos_", (void (Tensor::*)()) &Tensor::acos_, "In-place element-wise acos operation\n\nC++: Tensor::acos_() --> void");
		cl.def("acos", (class Tensor * (Tensor::*)()) &Tensor::acos, "In-place element-wise acos operation\n   \n\n A new tensor with the result of acos operation\n\nC++: Tensor::acos() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_acos", (void (*)(class Tensor *, class Tensor *)) &Tensor::acos, "Element-wise acos operation\n   \n\n The tensor where the operation is applied\n   \n\n A new tensor with acos applied\n\nC++: Tensor::acos(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("add_", (void (Tensor::*)(float)) &Tensor::add_, "In-place element-wise add operation of a tensor and a real value\n   \n\n The real number to add\n\nC++: Tensor::add_(float) --> void", pybind11::arg("v"));
		cl.def("add", (class Tensor * (Tensor::*)(float)) &Tensor::add, "Element-wise add operation of a tensor and a real value\n   \n\n The real number to add\n   \n\n A new tensor with the sum\n\nC++: Tensor::add(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def("add_", (void (Tensor::*)(class Tensor *)) &Tensor::add_, "In-place element-wise add operation of two tensors\n   \n\n The tensor to be added.\n\nC++: Tensor::add_(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("add", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::add, "Element-wise add operation of two tensors\n   \n\n The tensor to be added\n   \n\n a tensor with the element-wise sum\n\nC++: Tensor::add(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_add", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::add, "Element-wise add operation of a tensor and a real value\n   \n\n Input tensor\n   \n\n Output tensor. B = A + v\n   \n\n Real value to be added to A\n\nC++: Tensor::add(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("asin_", (void (Tensor::*)()) &Tensor::asin_, "In-place element-wise asin operation\n\nC++: Tensor::asin_() --> void");
		cl.def("asin", (class Tensor * (Tensor::*)()) &Tensor::asin, "Element-wise asin operation\n   \n\n A new tensor with the result\n\nC++: Tensor::asin() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_asin", (void (*)(class Tensor *, class Tensor *)) &Tensor::asin, "Element-wise asin operation\n   \n\n The tensor where the operation is applied\n   \n\n A new tensor with asin applied over A\n\nC++: Tensor::asin(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("atan_", (void (Tensor::*)()) &Tensor::atan_, "In-place element-wise atan operation\n\nC++: Tensor::atan_() --> void");
		cl.def("atan", (class Tensor * (Tensor::*)()) &Tensor::atan, "Element-wise atan operation\n   \n\n A new tensor with the result\n\nC++: Tensor::atan() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_atan", (void (*)(class Tensor *, class Tensor *)) &Tensor::atan, "Element-wise atan operation\n   \n\n The tensor where the operation is applied\n   \n\n A new tensor with atan applied over A\n\nC++: Tensor::atan(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("ceil_", (void (Tensor::*)()) &Tensor::ceil_, "In-place element-wise ceil operation\n\nC++: Tensor::ceil_() --> void");
		cl.def("ceil", (class Tensor * (Tensor::*)()) &Tensor::ceil, "Element-wise ceil operation\n   \n\n A new tensor with the result\n\nC++: Tensor::ceil() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_ceil", (void (*)(class Tensor *, class Tensor *)) &Tensor::ceil, "Element-wise ceil operation\n   \n\n The tensor where the operation is applied\n   \n\n A new tensor with ceil applied over A\n\nC++: Tensor::ceil(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("clamp_", (void (Tensor::*)(float, float)) &Tensor::clamp_, "In-place clamp all elements in the input tensor to the range [min, max].\n   \n\n The lower bound of the clamping range.\n   \n\n The upper bound of the clamping range.\n\nC++: Tensor::clamp_(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def("clamp", (class Tensor * (Tensor::*)(float, float)) &Tensor::clamp, "Clamp all elements in the input tensor to the range [min, max].\n   \n\n The lower bound of the clamping range.\n   \n\n The upper bound of the clamping range.\n   \n\n A new tensor with the clamped values in the input tensor.\n\nC++: Tensor::clamp(float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("static_clamp", (void (*)(class Tensor *, class Tensor *, float, float)) &Tensor::clamp, "Clamp all elements in the input tensor to the range [min, max].\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor with the result.\n   \n\n The lower bound of the clamping range.\n   \n\n The upper bound of the clamping range.\n\nC++: Tensor::clamp(class Tensor *, class Tensor *, float, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("min"), pybind11::arg("max"));
		cl.def("clampmax_", (void (Tensor::*)(float)) &Tensor::clampmax_, "In-place clamp all elements in the input tensor to the range [-infty, max].\n   \n\n The upper bound of the clamping range.\n\nC++: Tensor::clampmax_(float) --> void", pybind11::arg("max"));
		cl.def("clampmax", (class Tensor * (Tensor::*)(float)) &Tensor::clampmax, "Clamp all elements in the input tensor to the range [-infty, max].\n   \n\n The upper bound of the clamping range.\n   \n\n A new tensor with the clamped values.\n\nC++: Tensor::clampmax(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("max"));
		cl.def_static("static_clampmax", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::clampmax, "Clamp all elements in the input tensor to the range [-infty, max].\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n   \n\n The upper bound of the clamping range.\n\nC++: Tensor::clampmax(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("max"));
		cl.def("clampmin_", (void (Tensor::*)(float)) &Tensor::clampmin_, "In-place clamp all elements in the input tensor to the range [min, +infty].\n   \n\n The lower bound of the clamping range.\n\nC++: Tensor::clampmin_(float) --> void", pybind11::arg("min"));
		cl.def("clampmin", (class Tensor * (Tensor::*)(float)) &Tensor::clampmin, "Clamp all elements in the input tensor to the range [min, +infty].\n   \n\n The lower bound of the clamping range.\n   \n\n A new tensor with the clamped values.\n\nC++: Tensor::clampmin(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("min"));
		cl.def_static("static_clampmin", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::clampmin, "Clamp all elements in the input tensor to the range [min, +infty].\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n   \n\n The lower bound of the clamping range.\n\nC++: Tensor::clampmin(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("min"));
		cl.def("cos_", (void (Tensor::*)()) &Tensor::cos_, "In-place element-wise cos operation\n\nC++: Tensor::cos_() --> void");
		cl.def("cos", (class Tensor * (Tensor::*)()) &Tensor::cos, "Element-wise cos operation\n   \n\n A new tensor with cos applied.\n\nC++: Tensor::cos() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_cos", (void (*)(class Tensor *, class Tensor *)) &Tensor::cos, "Element-wise cos operation\n   \n\n The tensor where the operation is applied\n   \n\n The output tensor.\n\nC++: Tensor::cos(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("cosh_", (void (Tensor::*)()) &Tensor::cosh_, "In-place element-wise cosh operation\n\nC++: Tensor::cosh_() --> void");
		cl.def("cosh", (class Tensor * (Tensor::*)()) &Tensor::cosh, "Element-wise cosh operation\n   \n\n A new tensor with cosh applied.\n\nC++: Tensor::cosh() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_cosh", (void (*)(class Tensor *, class Tensor *)) &Tensor::cosh, "Element-wise cosh operation\n   \n\n The tensor where the operation is applied\n   \n\n The output tensor.\n\nC++: Tensor::cosh(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("div_", (void (Tensor::*)(float)) &Tensor::div_, "In-place element-wise division operation of a tensor and a real value\n   \n\n The real number to divide by\n\nC++: Tensor::div_(float) --> void", pybind11::arg("v"));
		cl.def("div", (class Tensor * (Tensor::*)(float)) &Tensor::div, "Element-wise division operation of a tensor and a real value\n   \n\n The real number to divide by.\n   \n\n A new tensor with the division.\n\nC++: Tensor::div(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def("div_", (void (Tensor::*)(class Tensor *)) &Tensor::div_, "In-place element-wise division operation of two tensors\n   \n\n The tensor to divide by\n\nC++: Tensor::div_(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("div", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::div, "In-place element-wise division operation of two tensors\n   \n\n The tensor to divide by\n   \n\n A new tensor with the division.\n\nC++: Tensor::div(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_div", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::div, "Element-wise division operation of a tensor and a real value.\n   \n\n The tensor where the operation is applied\n   \n\n The output tensor. B = A / v\n   \n\n The real number to divide by.\n\nC++: Tensor::div(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("exp_", (void (Tensor::*)()) &Tensor::exp_, "In-place element-wise exp operation of a tensor\n\nC++: Tensor::exp_() --> void");
		cl.def("exp", (class Tensor * (Tensor::*)()) &Tensor::exp, "Element-wise exp operation of a tensor\n   \n\n A new tensor with the exp operation applied\n\nC++: Tensor::exp() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_exp", (void (*)(class Tensor *, class Tensor *)) &Tensor::exp, "Element-wise exp operation of a tensor\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::exp(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("floor_", (void (Tensor::*)()) &Tensor::floor_, "In-place element-wise floor operation\n\nC++: Tensor::floor_() --> void");
		cl.def("floor", (class Tensor * (Tensor::*)()) &Tensor::floor, "Element-wise floor operation\n   \n\n A new tensor with the floor operation applied.\n\nC++: Tensor::floor() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_floor", (void (*)(class Tensor *, class Tensor *)) &Tensor::floor, "Element-wise floor operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::floor(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("inv_", [](Tensor &o) -> void { return o.inv_(); }, "");
		cl.def("inv_", (void (Tensor::*)(float)) &Tensor::inv_, "In-place element-wise 1/x operation\n   \n\n the value multiplying the inverse\n\nC++: Tensor::inv_(float) --> void", pybind11::arg("v"));
		cl.def("inv", [](Tensor &o) -> Tensor * { return o.inv(); }, "", pybind11::return_value_policy::automatic);
		cl.def("inv", (class Tensor * (Tensor::*)(float)) &Tensor::inv, "Element-wise 1/x operation\n   \n\n the value multiplying the inverse\n   \n\n A new tensor with the result.\n\nC++: Tensor::inv(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_inv", [](class Tensor * a0, class Tensor * a1) -> void { return Tensor::inv(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_inv", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::inv, "Element-wise 1/x operation\n   \n\n The input tensor.\n   \n\n The output tensor.\n   \n\n the value multiplying the inverse.\n\nC++: Tensor::inv(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("log_", (void (Tensor::*)()) &Tensor::log_, "In-place element-wise log operation\n\nC++: Tensor::log_() --> void");
		cl.def("log", (class Tensor * (Tensor::*)()) &Tensor::log, "Element-wise log operation\n   \n\n A new tensor with the log operation applied\n\nC++: Tensor::log() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_log", (void (*)(class Tensor *, class Tensor *)) &Tensor::log, "Element-wise log operation\n   \n\n The tensor where the operation is applied\n   \n\n The output tensor.\n\nC++: Tensor::log(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("log2_", (void (Tensor::*)()) &Tensor::log2_, "In-place element-wise log2 operation\n\nC++: Tensor::log2_() --> void");
		cl.def("log2", (class Tensor * (Tensor::*)()) &Tensor::log2, "Element-wise log2 operation\n   \n\n A new tensor with the log2 operation applied.\n\nC++: Tensor::log2() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_log2", (void (*)(class Tensor *, class Tensor *)) &Tensor::log2, "Element-wise log2 operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::log2(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("log10_", (void (Tensor::*)()) &Tensor::log10_, "In-place element-wise log10 operation\n\nC++: Tensor::log10_() --> void");
		cl.def("log10", (class Tensor * (Tensor::*)()) &Tensor::log10, "Element-wise log10 operation\n   \n\n A new tensor with the log10 operation applied.\n\nC++: Tensor::log10() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_log10", (void (*)(class Tensor *, class Tensor *)) &Tensor::log10, "Element-wise log10 operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::log10(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("logn_", (void (Tensor::*)(float)) &Tensor::logn_, "In-place element-wise logn operation.\n   \n\n The base of the logarithm.\n\nC++: Tensor::logn_(float) --> void", pybind11::arg("n"));
		cl.def("logn", (class Tensor * (Tensor::*)(float)) &Tensor::logn, "Element-wise logn operation.\n   \n\n The base of the logarithm.\n   \n\n A new tensor with the logn operation applied.\n\nC++: Tensor::logn(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("n"));
		cl.def_static("static_logn", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::logn, "Element-wise logn operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n   \n\n The base of the logarithm.\n\nC++: Tensor::logn(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("n"));
		cl.def("mod_", (void (Tensor::*)(float)) &Tensor::mod_, "In-place element-wise mod operation.\n   \n\n The mod operator\n\nC++: Tensor::mod_(float) --> void", pybind11::arg("v"));
		cl.def("mod", (class Tensor * (Tensor::*)(float)) &Tensor::mod, "Element-wise mod operation.\n   \n\n The mod operator\n   \n\n A new tensor with the operation applied.\n\nC++: Tensor::mod(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_mod", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::mod, "Element-wise mod operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n   \n\n The mod operator.\n\nC++: Tensor::mod(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("mult_", (void (Tensor::*)(float)) &Tensor::mult_, "In-place multiplication operation of a tensor by a scalar.\n   \n\n The value to multiply by\n\nC++: Tensor::mult_(float) --> void", pybind11::arg("v"));
		cl.def("mult", (class Tensor * (Tensor::*)(float)) &Tensor::mult, "Multiplication operation of a tensor by a scalar.\n   \n\n The value to multiply by\n   \n\n A tensor with the result\n\nC++: Tensor::mult(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def("mult_", (void (Tensor::*)(class Tensor *)) &Tensor::mult_, "In-place element-wise  multiplication operation of two 1D tensors.\n   \n\n The tensor to multiply by.\n\nC++: Tensor::mult_(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("mult", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::mult, "Element-wise multiplication operation of two 1D tensors.\n   \n\n The tensor to multiply by.\n   \n\n A tensor with the result.\n\nC++: Tensor::mult(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_mult", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::mult, "Element-wise multiplication operation of a tensor and a real value.\n   \n\n The input tensor.\n   \n\n The output tensor. B = A * v.\n   \n\n The value to multiply by.\n\nC++: Tensor::mult(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("neg_", (void (Tensor::*)()) &Tensor::neg_, "In-place element-wise change of sign operation.\n\nC++: Tensor::neg_() --> void");
		cl.def("neg", (class Tensor * (Tensor::*)()) &Tensor::neg, "Element-wise change of sign operation.\n   \n\n A tensor with the result.\n\nC++: Tensor::neg() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_neg", (void (*)(class Tensor *, class Tensor *)) &Tensor::neg, "Element-wise change of sign operation.\n   \n\n The tensor where the operation is applied.\n   \n\n A tensor with -A\n\nC++: Tensor::neg(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("normalize_", [](Tensor &o) -> void { return o.normalize_(); }, "");
		cl.def("normalize_", [](Tensor &o, float const & a0) -> void { return o.normalize_(a0); }, "", pybind11::arg("min"));
		cl.def("normalize_", (void (Tensor::*)(float, float)) &Tensor::normalize_, "In-place element-wise normalization of values in a given range.\n   \n\n The lower bound of the new range\n   \n\n The upper bound of the new range\n\nC++: Tensor::normalize_(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def("normalize", [](Tensor &o) -> Tensor * { return o.normalize(); }, "", pybind11::return_value_policy::automatic);
		cl.def("normalize", [](Tensor &o, float const & a0) -> Tensor * { return o.normalize(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("min"));
		cl.def("normalize", (class Tensor * (Tensor::*)(float, float)) &Tensor::normalize, "In-place element-wise normalization of values in a given range.\n   \n\n The lower bound of the new range.\n   \n\n The upper bound of the new range.\n   \n\n A tensor with the result.\n\nC++: Tensor::normalize(float, float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("min"), pybind11::arg("max"));
		cl.def_static("static_normalize", [](class Tensor * a0, class Tensor * a1) -> void { return Tensor::normalize(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_normalize", [](class Tensor * a0, class Tensor * a1, float const & a2) -> void { return Tensor::normalize(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("min"));
		cl.def_static("static_normalize", (void (*)(class Tensor *, class Tensor *, float, float)) &Tensor::normalize, "In-place element-wise normalization of values in a given range.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n   \n\n The lower bound of the new range\n   \n\n The upper bound of the new range\n\nC++: Tensor::normalize(class Tensor *, class Tensor *, float, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("min"), pybind11::arg("max"));
		cl.def("pow_", (void (Tensor::*)(float)) &Tensor::pow_, "In-place element-wise power operation with base e.\n   \n\n The exponent\n\nC++: Tensor::pow_(float) --> void", pybind11::arg("exp"));
		cl.def("pow", (class Tensor * (Tensor::*)(float)) &Tensor::pow, "Element-wise power operation with base e.\n   \n\n The exponent.\n   \n\n A tensor with the result.\n\nC++: Tensor::pow(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("exp"));
		cl.def_static("static_pow", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::pow, "Element-wise power operation with base e.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n   \n\n The exponent\n\nC++: Tensor::pow(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("exp"));
		cl.def("powb_", (void (Tensor::*)(float)) &Tensor::powb_, "In-place element-wise power operation.\n   \n\n The base of the power\n\nC++: Tensor::powb_(float) --> void", pybind11::arg("base"));
		cl.def("powb", (class Tensor * (Tensor::*)(float)) &Tensor::powb, "Element-wise power operation.\n   \n\n The base of the power.\n   \n\n A tensor with the result.\n\nC++: Tensor::powb(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("base"));
		cl.def_static("static_powb", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::powb, "Element-wise power operation.\n   \n\n the input tensor.\n   \n\n The output tensor.\n   \n\n The base of the power.\n\nC++: Tensor::powb(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("base"));
		cl.def("reciprocal_", (void (Tensor::*)()) &Tensor::reciprocal_, "In-place element-wise reciprocal operation.\n\nC++: Tensor::reciprocal_() --> void");
		cl.def("reciprocal", (class Tensor * (Tensor::*)()) &Tensor::reciprocal, "Element-wise reciprocal operation.\n   \n\n A tensor with the result\n\nC++: Tensor::reciprocal() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_reciprocal", (void (*)(class Tensor *, class Tensor *)) &Tensor::reciprocal, "Element-wise reciprocal operation.\n   \n\n The tensor where the operation is applied.\n   \n\n A tensor with reciprocal(A), element-wise\n\nC++: Tensor::reciprocal(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("remainder_", (void (Tensor::*)(float)) &Tensor::remainder_, "In-place element-wise reminder operation.\n   \n\n The real to divide A by\n\nC++: Tensor::remainder_(float) --> void", pybind11::arg("v"));
		cl.def("remainder", (class Tensor * (Tensor::*)(float)) &Tensor::remainder, "Element-wise reminder operation.\n   \n\n The real to divide A by\n   \n\n A tensor with A%v\n\nC++: Tensor::remainder(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_remainder", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::remainder, "Element-wise reminder operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n   \n\n The real to divide A by.\n\nC++: Tensor::remainder(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("round_", (void (Tensor::*)()) &Tensor::round_, "In-place element-wise round operation.\n\nC++: Tensor::round_() --> void");
		cl.def("round", (class Tensor * (Tensor::*)()) &Tensor::round, "Element-wise round operation.\n   \n\n A tensor with A rounded\n\nC++: Tensor::round() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_round", (void (*)(class Tensor *, class Tensor *)) &Tensor::round, "Element-wise round operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::round(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("rsqrt_", (void (Tensor::*)()) &Tensor::rsqrt_, "In-place element-wise inverse square root operation.\n\nC++: Tensor::rsqrt_() --> void");
		cl.def("rsqrt", (class Tensor * (Tensor::*)()) &Tensor::rsqrt, "Element-wise inverse square root operation.\n   \n\n A tensor with the result\n\nC++: Tensor::rsqrt() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_rsqrt", (void (*)(class Tensor *, class Tensor *)) &Tensor::rsqrt, "Element-wise inverse square root operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::rsqrt(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sigmoid_", (void (Tensor::*)()) &Tensor::sigmoid_, "In-place element-wise sigmoid operation.\n\nC++: Tensor::sigmoid_() --> void");
		cl.def("sigmoid", (class Tensor * (Tensor::*)()) &Tensor::sigmoid, "Element-wise sigmoid operation.\n   \n\n A tensor with sigmoid(A)\n\nC++: Tensor::sigmoid() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_sigmoid", (void (*)(class Tensor *, class Tensor *)) &Tensor::sigmoid, "Element-wise sigmoid operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::sigmoid(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sign_", [](Tensor &o) -> void { return o.sign_(); }, "");
		cl.def("sign_", (void (Tensor::*)(float)) &Tensor::sign_, "C++: Tensor::sign_(float) --> void", pybind11::arg("zero_sign"));
		cl.def("sign", [](Tensor &o) -> Tensor * { return o.sign(); }, "", pybind11::return_value_policy::automatic);
		cl.def("sign", (class Tensor * (Tensor::*)(float)) &Tensor::sign, "C++: Tensor::sign(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("zero_sign"));
		cl.def_static("static_sign", [](class Tensor * a0, class Tensor * a1) -> void { return Tensor::sign(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_sign", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::sign, "C++: Tensor::sign(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("zero_sign"));
		cl.def("sin_", (void (Tensor::*)()) &Tensor::sin_, "In-place element-wise sin operation.\n\nC++: Tensor::sin_() --> void");
		cl.def("sin", (class Tensor * (Tensor::*)()) &Tensor::sin, "Element-wise sin operation.\n   \n\n A tensor with the result.\n\nC++: Tensor::sin() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_sin", (void (*)(class Tensor *, class Tensor *)) &Tensor::sin, "Element-wise sin operation.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor.\n\nC++: Tensor::sin(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sinh_", (void (Tensor::*)()) &Tensor::sinh_, "In-place element-wise sinh operation.\n\nC++: Tensor::sinh_() --> void");
		cl.def("sinh", (class Tensor * (Tensor::*)()) &Tensor::sinh, "Element-wise sinh operation.\n   \n\n A tensor with the result.\n\nC++: Tensor::sinh() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_sinh", (void (*)(class Tensor *, class Tensor *)) &Tensor::sinh, "Element-wise sinh operation.\n   \n\n The tensor where the operation is applied.\n   \n\n Tensor with the result.\n\nC++: Tensor::sinh(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sqr_", (void (Tensor::*)()) &Tensor::sqr_, "In-place element-wise square operation. More efficient than using pow_(A, 2).\n\nC++: Tensor::sqr_() --> void");
		cl.def("sqr", (class Tensor * (Tensor::*)()) &Tensor::sqr, "Element-wise square operation. More efficient than using pow(A, 2).\n   \n\n A tensor with the result\n\nC++: Tensor::sqr() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_sqr", (void (*)(class Tensor *, class Tensor *)) &Tensor::sqr, "Element-wise square operation. More efficient than using pow(A, 2).\n   \n\n The tensor where the operation is applied.\n   \n\n tensor with the result.\n\nC++: Tensor::sqr(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sqrt_", (void (Tensor::*)()) &Tensor::sqrt_, "In-place element-wise square root operation.\n\nC++: Tensor::sqrt_() --> void");
		cl.def("sqrt", (class Tensor * (Tensor::*)()) &Tensor::sqrt, "Element-wise square operation.\n   \n\n A tensor with the result.\n\nC++: Tensor::sqrt() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_sqrt", (void (*)(class Tensor *, class Tensor *)) &Tensor::sqrt, "Element-wise square operation.\n   \n\n The tensor where the operation is applied.\n   \n\n tensor with the result.\n\nC++: Tensor::sqrt(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("sub_", (void (Tensor::*)(float)) &Tensor::sub_, "In-place element-wise substraction operation of a tensor and a scalar.\n   \n\n The value to substract to A.\n\nC++: Tensor::sub_(float) --> void", pybind11::arg("v"));
		cl.def("sub", (class Tensor * (Tensor::*)(float)) &Tensor::sub, "Element-wise substraction operation of a tensor and a scalar.\n   \n\n The value to substract to the input tensor.\n   \n\n A tensor with the result.\n\nC++: Tensor::sub(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def("sub_", (void (Tensor::*)(class Tensor *)) &Tensor::sub_, "In-place element-wise substraction operation of two tensors.\n   \n\n The tensor to substract.\n\nC++: Tensor::sub_(class Tensor *) --> void", pybind11::arg("A"));
		cl.def("sub", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::sub, "Element-wise substraction operation of two tensors.\n   \n\n The tensor to substract.\n   \n\n A tensor with the result.\n\nC++: Tensor::sub(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_sub", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::sub, "Element-wise substraction operation of a tensor and a real value.\n   \n\n The tensor where the operation is applied.\n   \n\n The output tensor. B = A - v.\n   \n\n The real value to substract.\n\nC++: Tensor::sub(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("tan_", (void (Tensor::*)()) &Tensor::tan_, "In-place element-wise tan operation.\n\nC++: Tensor::tan_() --> void");
		cl.def("tan", (class Tensor * (Tensor::*)()) &Tensor::tan, "Element-wise tan operation.\n   \n\n A tensor with the result.\n\nC++: Tensor::tan() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_tan", (void (*)(class Tensor *, class Tensor *)) &Tensor::tan, "Element-wise tan operation.\n   \n\n The tensor where the operation is applied.\n   \n\n A tensor with the result.\n\nC++: Tensor::tan(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("tanh_", (void (Tensor::*)()) &Tensor::tanh_, "In-place element-wise tanh operation.\n\nC++: Tensor::tanh_() --> void");
		cl.def("tanh", (class Tensor * (Tensor::*)()) &Tensor::tanh, "Element-wise tanh operation.\n   \n\n A tensor with the result.\n\nC++: Tensor::tanh() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_tanh", (void (*)(class Tensor *, class Tensor *)) &Tensor::tanh, "Element-wise tanh operation.\n   \n\n The tensor where the operation is applied.\n   \n\n A tensor with the result.\n\nC++: Tensor::tanh(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("trunc_", (void (Tensor::*)()) &Tensor::trunc_, "In-place element-wise truncate operation.\n\nC++: Tensor::trunc_() --> void");
		cl.def("trunc", (class Tensor * (Tensor::*)()) &Tensor::trunc, "Element-wise truncate operation.\n   \n\n A tensor with the result.\n\nC++: Tensor::trunc() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_trunc", (void (*)(class Tensor *, class Tensor *)) &Tensor::trunc, "Element-wise truncate operation.\n   \n\n The tensor where the operation is applied.\n   \n\n tensor with the result.\n\nC++: Tensor::trunc(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_add", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::add, "Element-wise add operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n A new tensor C = A + B.\n\nC++: Tensor::add(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_add", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::add, "Element-wise add operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n Output tensor. C = A + B.\n\nC++: Tensor::add(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("static_div", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::div, "Element-wise division operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n A new tensor C = A / B.\n\nC++: Tensor::div(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_div", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::div, "Element-wise division operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n Output tensor. C = A / B.\n\nC++: Tensor::div(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("static_mult", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::mult, "Element-wise multiplication operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n A new tensor C = A * B.\n\nC++: Tensor::mult(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_mult", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::mult, "Element-wise multiplication operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n Output tensor. C = A * B.\n\nC++: Tensor::mult(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("interpolate", (class Tensor * (*)(float, class Tensor *, float, class Tensor *)) &Tensor::interpolate, "Element-wise weighted sum (interpolation) operation of two tensors.\n   \n\n The weight for first member.\n   \n\n A tensor.\n   \n\n The weight for second member.\n   \n\n Another tensor.\n   \n\n A new tensor C = factor1*A + factor2*B.\n\nC++: Tensor::interpolate(float, class Tensor *, float, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("factor1"), pybind11::arg("A"), pybind11::arg("factor2"), pybind11::arg("B"));
		cl.def_static("interpolate", (void (*)(float, class Tensor *, float, class Tensor *, class Tensor *)) &Tensor::interpolate, "Element-wise weighted sum (interpolation) operation of two tensors.\n   \n\n The weight for first member.\n   \n\n A tensor.\n   \n\n The weight for second member.\n   \n\n Another tensor.\n   \n\n Output tensor. C = factor1*A + factor2*B.\n\nC++: Tensor::interpolate(float, class Tensor *, float, class Tensor *, class Tensor *) --> void", pybind11::arg("factor1"), pybind11::arg("A"), pybind11::arg("factor2"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("static_sub", (class Tensor * (*)(class Tensor *, class Tensor *)) &Tensor::sub, "Element-wise substraction operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n A new tensor C = A - B.\n\nC++: Tensor::sub(class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_sub", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sub, "Element-wise multiplication operation of two tensors.\n   \n\n A tensor.\n   \n\n Another tensor.\n   \n\n Output tensor. C = A - B.\n\nC++: Tensor::sub(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("fill_", (void (Tensor::*)(float)) &Tensor::fill_, "Fill tensor with a value\n   \n\n the value to fill the tensor with\n\nC++: Tensor::fill_(float) --> void", pybind11::arg("v"));
		cl.def("fill", (class Tensor * (Tensor::*)(float)) &Tensor::fill, "Fill tensor with a value\n   \n\n the value to fill the tensor with\n   \n\n A new tensor with the result\n\nC++: Tensor::fill(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_fill", (void (*)(class Tensor *, float)) &Tensor::fill, "Fill tensor with a value\n   \n\n The output tensor.\n   \n\n the value to fill the tensor with\n\nC++: Tensor::fill(class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("v"));
		cl.def("moveaxis_", (void (Tensor::*)(int, int)) &Tensor::moveaxis_, "C++: Tensor::moveaxis_(int, int) --> void", pybind11::arg("source"), pybind11::arg("destination"));
		cl.def("moveaxis", (class Tensor * (Tensor::*)(int, int)) &Tensor::moveaxis, "Move axes of an array to new positions.\n   \n\n Original position of the axis to move. These must be unique.\n   \n\n Destination position for the original axis. These must also be unique\n   \n\n A new tensor with the result\n\nC++: Tensor::moveaxis(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("source"), pybind11::arg("destination"));
		cl.def_static("static_moveaxis", (class Tensor * (*)(class Tensor *, int, int)) &Tensor::moveaxis, "C++: Tensor::moveaxis(class Tensor *, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("source"), pybind11::arg("destination"));
		cl.def("swapaxis_", (void (Tensor::*)(int, int)) &Tensor::swapaxis_, "C++: Tensor::swapaxis_(int, int) --> void", pybind11::arg("axis1"), pybind11::arg("axis2"));
		cl.def("swapaxis", (class Tensor * (Tensor::*)(int, int)) &Tensor::swapaxis, "Interchange two axes of an array.\n   \n\n First axis.\n   \n\n Destination position for the original axis. These must also be unique\n   \n\n A new tensor with the result\n\nC++: Tensor::swapaxis(int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("axis1"), pybind11::arg("axis2"));
		cl.def_static("static_swapaxis", (class Tensor * (*)(class Tensor *, int, int)) &Tensor::swapaxis, "C++: Tensor::swapaxis(class Tensor *, int, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("axis1"), pybind11::arg("axis2"));
		cl.def("flatten_", (void (Tensor::*)()) &Tensor::flatten_, "In-place conversion tensor to a 1D tensor.\n\nC++: Tensor::flatten_() --> void");
		cl.def("flatten", (class Tensor * (Tensor::*)()) &Tensor::flatten, "In-place conversion tensor to a 1D tensor.\n   \n\n A new tensor with the result\n\nC++: Tensor::flatten() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_flatten", (class Tensor * (*)(class Tensor *)) &Tensor::flatten, "Conversion tensor to a 1D tensor.\n   \n\n Output tensor where the flatten is stored.\n   \n\n A new tensor with the result\n\nC++: Tensor::flatten(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("squeeze_", [](Tensor &o) -> void { return o.squeeze_(); }, "");
		cl.def("squeeze_", (void (Tensor::*)(int)) &Tensor::squeeze_, "Returns a tensor with all the dimensions of input of size 1 removed.\n   \n\n if given, the input will be squeezed only in this dimension. Else (-1), squeezes all\n   dimensions of size 1\n\nC++: Tensor::squeeze_(int) --> void", pybind11::arg("axis"));
		cl.def("squeeze", [](Tensor &o) -> Tensor * { return o.squeeze(); }, "", pybind11::return_value_policy::automatic);
		cl.def("squeeze", (class Tensor * (Tensor::*)(int)) &Tensor::squeeze, "Remove all the dimensions of size 1 from the vector.\n   \n\n if given, the input will be squeezed only in this dimension. Else (-1), squeezes all\n   dimensions of size 1\n   \n\n A new tensor with the result\n\nC++: Tensor::squeeze(int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("axis"));
		cl.def_static("static_squeeze", [](class Tensor * a0) -> Tensor * { return Tensor::squeeze(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_squeeze", (class Tensor * (*)(class Tensor *, int)) &Tensor::squeeze, "Remove all the dimensions of size 1 from the vector.\n   \n\n Output tensor where the squeeze is stored.\n   \n\n if given, the input will be squeezed only in this dimension. Else (-1), squeezes all\n   dimensions of size 1\n   \n\n A new tensor with the result\n\nC++: Tensor::squeeze(class Tensor *, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("axis"));
		cl.def("unsqueeze_", [](Tensor &o) -> void { return o.unsqueeze_(); }, "");
		cl.def("unsqueeze_", (void (Tensor::*)(int)) &Tensor::unsqueeze_, "Sets a dimension of size one inserted at the specified position.\n   \n\n the index at which to insert the singleton dimension. Default: axis=0\n\nC++: Tensor::unsqueeze_(int) --> void", pybind11::arg("axis"));
		cl.def("unsqueeze", [](Tensor &o) -> Tensor * { return o.unsqueeze(); }, "", pybind11::return_value_policy::automatic);
		cl.def("unsqueeze", (class Tensor * (Tensor::*)(int)) &Tensor::unsqueeze, "Returns a new tensor with a dimension of size one inserted at the specified position.\n   \n\n the index at which to insert the singleton dimension. Default: axis=0\n   \n\n A new tensor with the result\n\nC++: Tensor::unsqueeze(int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("axis"));
		cl.def_static("static_unsqueeze", [](class Tensor * a0) -> Tensor * { return Tensor::unsqueeze(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_unsqueeze", (class Tensor * (*)(class Tensor *, int)) &Tensor::unsqueeze, "Returns a new tensor with a dimension of size one inserted at the specified position.\n   \n\n Output tensor where the unsqueeze is stored.\n   \n\n the index at which to insert the singleton dimension. Default: axis=0\n   \n\n A new tensor with the result\n\nC++: Tensor::unsqueeze(class Tensor *, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("axis"));
		cl.def("flip", [](Tensor &o) -> Tensor * { return o.flip(); }, "", pybind11::return_value_policy::automatic);
		cl.def("flip", (class Tensor * (Tensor::*)(int)) &Tensor::flip, "Flip the tensor.\n   \n\n The axis used to flip the tensor.\n\nC++: Tensor::flip(int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("axis"));
		cl.def_static("static_flip", [](class Tensor * a0, class Tensor * a1) -> void { return Tensor::flip(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_flip", (void (*)(class Tensor *, class Tensor *, int)) &Tensor::flip, "Flip the tensor.\n   \n\n Input tensor.\n   \n\n Output tensor.\n   \n\n The axis used to flip the tensor.\n\nC++: Tensor::flip(class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"));
		cl.def("flip_random", (class Tensor * (Tensor::*)(int)) &Tensor::flip_random, "Flip the tensor with some probability.\n   \n\n The axis used to flip the tensor.\n\nC++: Tensor::flip_random(int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("axis"));
		cl.def_static("static_flip_random", (void (*)(class Tensor *, class Tensor *, int)) &Tensor::flip_random, "Flip the tensor with some probability.\n   \n\n Input tensor.\n   \n\n Output tensor.\n   \n\n The axis used to flip the tensor.\n\nC++: Tensor::flip_random(class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"));
		cl.def("crop_random", [](Tensor &o, int const & a0, int const & a1) -> Tensor * { return o.crop_random(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("height"), pybind11::arg("width"));
		cl.def("crop_random", [](Tensor &o, int const & a0, int const & a1, float const & a2) -> Tensor * { return o.crop_random(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("height"), pybind11::arg("width"), pybind11::arg("cval"));
		cl.def("crop_random", (class Tensor * (Tensor::*)(int, int, float, bool)) &Tensor::crop_random, "Crop randomly the tensor.\n   \n\n Height of the crop (must be smaller than the original image)\n   \n\n Width of the crop (must be smaller than the original image)\n   \n\n\n   \n\n Keep original size\n\nC++: Tensor::crop_random(int, int, float, bool) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("height"), pybind11::arg("width"), pybind11::arg("cval"), pybind11::arg("keep_size"));
		cl.def_static("static_crop_random", (void (*)(class Tensor *, class Tensor *)) &Tensor::crop_random, "Crop randomly the tensor.\n   \n\n Input tensor.\n   \n\n Output tensor.\n\nC++: Tensor::crop_random(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("trace", [](Tensor &o) -> float { return o.trace(); }, "");
		cl.def("trace", (float (Tensor::*)(int)) &Tensor::trace, "Sum all the elements in a matrix diagonal.\n   \n\n Offset. Used to select the diagonal to be summed.\n   \n\n The sum of all the elements in the selected diagonal.\n\nC++: Tensor::trace(int) --> float", pybind11::arg("k"));
		cl.def_static("static_trace", [](class Tensor * a0) -> float { return Tensor::trace(a0); }, "", pybind11::arg("A"));
		cl.def_static("static_trace", (float (*)(class Tensor *, int)) &Tensor::trace, "Sum all the elements in a matrix diagonal.\n   \n\n Input tensor.\n   \n\n Offset. Used to select the diagonal to be summed.\n   \n\n The sum of all the elements in the selected diagonal.\n\nC++: Tensor::trace(class Tensor *, int) --> float", pybind11::arg("A"), pybind11::arg("k"));
		cl.def("nonzero", [](Tensor &o) -> Tensor * { return o.nonzero(); }, "", pybind11::return_value_policy::automatic);
		cl.def("nonzero", (class Tensor * (Tensor::*)(bool)) &Tensor::nonzero, "Returns a tensor containing the indices of nonzero elements.\n   \n\n Whether to sort the indices or not. (default: not sorted)\n\n   \n A tensor containing the indices of the nonzero elements.\n\nC++: Tensor::nonzero(bool) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("sort_indices"));
		cl.def_static("where", (class Tensor * (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::where, "Depending on ``condition``, returns a tensor whith elements from ``A`` or ``B``.\n   \n\n Tensor with the condition to be accomplished.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n A tensor with the same shape with elements from ``A`` if ``condition`` holds and from ``B`` otherwise..\n\nC++: Tensor::where(class Tensor *, class Tensor *, class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("condition"), pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("where", (void (*)(class Tensor *, class Tensor *, class Tensor *, class Tensor *)) &Tensor::where, "Depending on ``condition``, returns a tensor whith elements from ``A`` or ``B``.\n   \n\n Tensor with the condition to be accomplished.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n A tensor with elements from ``A`` if ``condition`` holds and from ``B`` otherwise..\n\nC++: Tensor::where(class Tensor *, class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("condition"), pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("all", (bool (Tensor::*)()) &Tensor::all, "Test whether all elements evaluate to True.\n\n  \n    bool\n\nC++: Tensor::all() --> bool");
		cl.def_static("static_all", (bool (*)(class Tensor *)) &Tensor::all, "C++: Tensor::all(class Tensor *) --> bool", pybind11::arg("A"));
		cl.def("any", (bool (Tensor::*)()) &Tensor::any, "Test whether any element evaluates to True.\n\n  \n    bool\n\nC++: Tensor::any() --> bool");
		cl.def_static("static_any", (bool (*)(class Tensor *)) &Tensor::any, "C++: Tensor::any(class Tensor *) --> bool", pybind11::arg("A"));
		cl.def("isfinite", (class Tensor * (Tensor::*)()) &Tensor::isfinite, "Test element-wise for finiteness (not infinity or not Not a Number).\n\n  \n    Tensor with the results of the test as booleans\n\nC++: Tensor::isfinite() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_isfinite", (void (*)(class Tensor *, class Tensor *)) &Tensor::isfinite, "C++: Tensor::isfinite(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("isinf", (class Tensor * (Tensor::*)()) &Tensor::isinf, "Test element-wise for positive or negative infinity.\n\n  \n    Tensor with the results of the test as booleans\n\nC++: Tensor::isinf() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_isinf", (void (*)(class Tensor *, class Tensor *)) &Tensor::isinf, "C++: Tensor::isinf(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("isnan", (class Tensor * (Tensor::*)()) &Tensor::isnan, "Test element-wise for Nan.\n\n  \n    Tensor with the results of the test as booleans\n\nC++: Tensor::isnan() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_isnan", (void (*)(class Tensor *, class Tensor *)) &Tensor::isnan, "C++: Tensor::isnan(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("anynan", (bool (Tensor::*)()) &Tensor::anynan, "C++: Tensor::anynan() --> bool");
		cl.def("isneginf", (class Tensor * (Tensor::*)()) &Tensor::isneginf, "Test element-wise for negative infinity.\n\n  \n    Tensor with the results of the test as booleans\n\nC++: Tensor::isneginf() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_isneginf", (void (*)(class Tensor *, class Tensor *)) &Tensor::isneginf, "C++: Tensor::isneginf(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("isposinf", (class Tensor * (Tensor::*)()) &Tensor::isposinf, "Test element-wise for positive infinity.\n\n  \n    Tensor with the results of the test as booleans\n\nC++: Tensor::isposinf() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_isposinf", (void (*)(class Tensor *, class Tensor *)) &Tensor::isposinf, "C++: Tensor::isposinf(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("logical_and", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::logical_and, "Compute the truth value of ``A and B`` element-wise.\n\n  \n   Tensor\n  \n\n    Tensor with the result of the operation\n\nC++: Tensor::logical_and(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_logical_and", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::logical_and, "C++: Tensor::logical_and(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("logical_or", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::logical_or, "Compute the truth value of ``A or B`` element-wise.\n\n  \n   Tensor\n  \n\n    Tensor with the result of the operation\n\nC++: Tensor::logical_or(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_logical_or", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::logical_or, "C++: Tensor::logical_or(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("logical_not", (class Tensor * (Tensor::*)()) &Tensor::logical_not, "Compute the truth value of ``not A`` element-wise.\n\n  \n   Tensor\n  \n\n    Tensor with the result of the operation\n\nC++: Tensor::logical_not() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def_static("static_logical_not", (void (*)(class Tensor *, class Tensor *)) &Tensor::logical_not, "C++: Tensor::logical_not(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def("logical_xor", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::logical_xor, "Compute the truth value of ``A xor B`` element-wise.\n\n  \n   Tensor\n  \n\n    Tensor with the result of the operation\n\nC++: Tensor::logical_xor(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_logical_xor", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::logical_xor, "C++: Tensor::logical_xor(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("allclose", [](Tensor &o, class Tensor * a0) -> bool { return o.allclose(a0); }, "", pybind11::arg("A"));
		cl.def("allclose", [](Tensor &o, class Tensor * a0, float const & a1) -> bool { return o.allclose(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("rtol"));
		cl.def("allclose", [](Tensor &o, class Tensor * a0, float const & a1, float const & a2) -> bool { return o.allclose(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("rtol"), pybind11::arg("atol"));
		cl.def("allclose", (bool (Tensor::*)(class Tensor *, float, float, bool)) &Tensor::allclose, "Returns True if two arrays accomplish, element-wise, the condition \n\n  \n   Input tensor.\n  \n\n relative tolerance.\n  \n\n absolute tolerance.\n  \n\n if ``True``, then two ``NaN``s will be considered equal.\n  \n\n    boolean indicating if all elements in tensor hold the condition\n\nC++: Tensor::allclose(class Tensor *, float, float, bool) --> bool", pybind11::arg("A"), pybind11::arg("rtol"), pybind11::arg("atol"), pybind11::arg("equal_nan"));
		cl.def_static("static_allclose", [](class Tensor * a0, class Tensor * a1) -> bool { return Tensor::allclose(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_allclose", [](class Tensor * a0, class Tensor * a1, float const & a2) -> bool { return Tensor::allclose(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rtol"));
		cl.def_static("static_allclose", [](class Tensor * a0, class Tensor * a1, float const & a2, float const & a3) -> bool { return Tensor::allclose(a0, a1, a2, a3); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rtol"), pybind11::arg("atol"));
		cl.def_static("static_allclose", (bool (*)(class Tensor *, class Tensor *, float, float, bool)) &Tensor::allclose, "C++: Tensor::allclose(class Tensor *, class Tensor *, float, float, bool) --> bool", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("rtol"), pybind11::arg("atol"), pybind11::arg("equal_nan"));
		cl.def("isclose", [](Tensor &o, class Tensor * a0) -> Tensor * { return o.isclose(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def("isclose", [](Tensor &o, class Tensor * a0, float const & a1) -> Tensor * { return o.isclose(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("rtol"));
		cl.def("isclose", [](Tensor &o, class Tensor * a0, float const & a1, float const & a2) -> Tensor * { return o.isclose(a0, a1, a2); }, "", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("rtol"), pybind11::arg("atol"));
		cl.def("isclose", (class Tensor * (Tensor::*)(class Tensor *, float, float, bool)) &Tensor::isclose, "Returns a boolean array where a position is true if elements in A and B accomplish \n\n  \n   Input tensor.\n  \n\n relative tolerance.\n  \n\n absolute tolerance.\n  \n\n if ``True``, then two ``NaN``s will be considered equal.\n  \n\n    boolean indicating if all elements in tensor hold the condition\n\nC++: Tensor::isclose(class Tensor *, float, float, bool) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"), pybind11::arg("rtol"), pybind11::arg("atol"), pybind11::arg("equal_nan"));
		cl.def_static("static_isclose", [](class Tensor * a0, class Tensor * a1, class Tensor * a2) -> void { return Tensor::isclose(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("static_isclose", [](class Tensor * a0, class Tensor * a1, class Tensor * a2, float const & a3) -> void { return Tensor::isclose(a0, a1, a2, a3); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("rtol"));
		cl.def_static("static_isclose", [](class Tensor * a0, class Tensor * a1, class Tensor * a2, float const & a3, float const & a4) -> void { return Tensor::isclose(a0, a1, a2, a3, a4); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("rtol"), pybind11::arg("atol"));
		cl.def_static("static_isclose", (void (*)(class Tensor *, class Tensor *, class Tensor *, float, float, bool)) &Tensor::isclose, "C++: Tensor::isclose(class Tensor *, class Tensor *, class Tensor *, float, float, bool) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("rtol"), pybind11::arg("atol"), pybind11::arg("equal_nan"));
		cl.def("greater_", (void (Tensor::*)(float)) &Tensor::greater_, "Return the truth value of the input elements > ``v`` element-wise. In-place operation.\n\n  \n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::greater_(float) --> void", pybind11::arg("v"));
		cl.def("greater", (class Tensor * (Tensor::*)(float)) &Tensor::greater, "Return the truth value of the input elements > ``v`` element-wise.\n\n  \n   Value to make the comparison with.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::greater(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_greater", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::greater, "Return the truth value of the input elements > ``v`` element-wise.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::greater(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("greater", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::greater, "Return the truth value of ``this > A`` element-wise.\n\n  \n   Input tensor.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::greater(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_greater", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::greater, "Return the truth value of ``A > B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n\nC++: Tensor::greater(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("greater_equal_", (void (Tensor::*)(float)) &Tensor::greater_equal_, "Return the truth value of the input elements >= ``v`` element-wise. In-place operation.\n\n  \n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::greater_equal_(float) --> void", pybind11::arg("v"));
		cl.def("greater_equal", (class Tensor * (Tensor::*)(float)) &Tensor::greater_equal, "Return the truth value of the input elements >= ``v`` element-wise.\n\n  \n   Value to make the comparison with.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::greater_equal(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_greater_equal", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::greater_equal, "Return the truth value of the input elements >= ``v`` element-wise.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::greater_equal(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("greater_equal", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::greater_equal, "Return the truth value of ``this >= A`` element-wise.\n\n  \n   Input tensor.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::greater_equal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_greater_equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::greater_equal, "Return the truth value of ``A >= B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::greater_equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("less_", (void (Tensor::*)(float)) &Tensor::less_, "Return the truth value of the input elements < ``v`` element-wise. In-place operation.\n\n  \n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::less_(float) --> void", pybind11::arg("v"));
		cl.def("less", (class Tensor * (Tensor::*)(float)) &Tensor::less, "Return the truth value of the input elements < ``v`` element-wise.\n\n  \n   Value to make the comparison with.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::less(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_less", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::less, "Return the truth value of the input elements < ``v`` element-wise.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::less(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("less", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::less, "Return the truth value of ``this < A`` element-wise.\n\n  \n   Input tensor.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::less(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_less", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::less, "Return the truth value of ``A < B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::less(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("less_equal_", (void (Tensor::*)(float)) &Tensor::less_equal_, "Return the truth value of the input elements <= ``v`` element-wise. In-place operation.\n\n  \n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::less_equal_(float) --> void", pybind11::arg("v"));
		cl.def("less_equal", (class Tensor * (Tensor::*)(float)) &Tensor::less_equal, "Return the truth value of the input elements <= ``v`` element-wise.\n\n  \n   Value to make the comparison with.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::less_equal(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_less_equal", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::less_equal, "Return the truth value of the input elements <= ``v`` element-wise.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::less_equal(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("less_equal", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::less_equal, "Return the truth value of ``this <= A`` element-wise.\n\n  \n   Input tensor.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::less_equal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_less_equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::less_equal, "Return the truth value of ``A <= B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::less_equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("equal_", (void (Tensor::*)(float)) &Tensor::equal_, "Return the truth value of the input elements == ``v`` element-wise. In-place operation.\n\n  \n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::equal_(float) --> void", pybind11::arg("v"));
		cl.def("equal", (class Tensor * (Tensor::*)(float)) &Tensor::equal, "Return the truth value of the input elements == ``v`` element-wise.\n\n  \n   Value to make the comparison with.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::equal(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_equal", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::equal, "Return the truth value of the input elements == ``v`` element-wise.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::equal(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("equal", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::equal, "Return the truth value of ``this == A`` element-wise.\n\n  \n   Input tensor.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::equal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::equal, "Return the truth value of ``A == B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("not_equal_", (void (Tensor::*)(float)) &Tensor::not_equal_, "Return the truth value of the input elements != ``v`` element-wise. In-place operation.\n\n  \n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::not_equal_(float) --> void", pybind11::arg("v"));
		cl.def("not_equal", (class Tensor * (Tensor::*)(float)) &Tensor::not_equal, "Return the truth value of the input elements != ``v`` element-wise.\n\n  \n   Value to make the comparison with.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::not_equal(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("static_not_equal", (void (*)(class Tensor *, class Tensor *, float)) &Tensor::not_equal, "Return the truth value of the input elements != ``v`` element-wise.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Value to make the comparison with.\n  \n\n    void\n\nC++: Tensor::not_equal(class Tensor *, class Tensor *, float) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("v"));
		cl.def("not_equal", (class Tensor * (Tensor::*)(class Tensor *)) &Tensor::not_equal, "Return the truth value of ``this != A`` element-wise.\n\n  \n   Input tensor.\n  \n\n    A tensor with the true values.\n\nC++: Tensor::not_equal(class Tensor *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("A"));
		cl.def_static("static_not_equal", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::not_equal, "Return the truth value of ``A != B`` element-wise.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n   Tensor store the results of the operation.\n  \n\n    void\n\nC++: Tensor::not_equal(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def("sort_", [](Tensor &o) -> void { return o.sort_(); }, "");
		cl.def("sort_", [](Tensor &o, bool const & a0) -> void { return o.sort_(a0); }, "", pybind11::arg("descending"));
		cl.def("sort_", (void (Tensor::*)(bool, bool)) &Tensor::sort_, "Sort a tensor in-place.\n\n  \n   Wether to sort the tensor descending or not.\n  \n\n   Wether to use stable sorting or not. Stable sorting keeps the order of equal elements.\n\nC++: Tensor::sort_(bool, bool) --> void", pybind11::arg("descending"), pybind11::arg("stable"));
		cl.def("sort", [](Tensor &o) -> Tensor * { return o.sort(); }, "", pybind11::return_value_policy::automatic);
		cl.def("sort", [](Tensor &o, bool const & a0) -> Tensor * { return o.sort(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("descending"));
		cl.def("sort", (class Tensor * (Tensor::*)(bool, bool)) &Tensor::sort, "Sort a tensor.\n\n  \n   Wether to sort the tensor descending or not.\n  \n\n   Wether to use stable sorting or not. Stable sorting keeps the order of equal elements.\n  \n\n    A tensor with the sorted elements.\n\nC++: Tensor::sort(bool, bool) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("descending"), pybind11::arg("stable"));
		cl.def_static("static_sort", [](class Tensor * a0, class Tensor * a1) -> void { return Tensor::sort(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_sort", [](class Tensor * a0, class Tensor * a1, bool const & a2) -> void { return Tensor::sort(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("descending"));
		cl.def_static("static_sort", (void (*)(class Tensor *, class Tensor *, bool, bool)) &Tensor::sort, "Sort a tensor.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Wether to sort the tensor descending or not.\n  \n\n   Wether to use stable sorting or not. Stable sorting keeps the order of equal elements.\n\nC++: Tensor::sort(class Tensor *, class Tensor *, bool, bool) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("descending"), pybind11::arg("stable"));
		cl.def("argsort", [](Tensor &o) -> Tensor * { return o.argsort(); }, "", pybind11::return_value_policy::automatic);
		cl.def("argsort", [](Tensor &o, bool const & a0) -> Tensor * { return o.argsort(a0); }, "", pybind11::return_value_policy::automatic, pybind11::arg("descending"));
		cl.def("argsort", (class Tensor * (Tensor::*)(bool, bool)) &Tensor::argsort, "Sort the indices of a tensor according to the elements in each position.\n\n  \n   Wether to sort the tensor descending or not.\n  \n\n   Wether to use stable sorting or not. Stable sorting keeps the order of equal elements.\n  \n\n    A tensor with the sorted indices.\n\nC++: Tensor::argsort(bool, bool) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("descending"), pybind11::arg("stable"));
		cl.def_static("static_argsort", [](class Tensor * a0, class Tensor * a1) -> void { return Tensor::argsort(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_argsort", [](class Tensor * a0, class Tensor * a1, bool const & a2) -> void { return Tensor::argsort(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("descending"));
		cl.def_static("static_argsort", (void (*)(class Tensor *, class Tensor *, bool, bool)) &Tensor::argsort, "Sort the indices of a tensor according to the elements in each position.\n\n  \n   Input tensor.\n  \n\n   Output tensor.\n  \n\n   Wether to sort the tensor descending or not.\n  \n\n   Wether to use stable sorting or not. Stable sorting keeps the order of equal elements.\n  \n\n    A tensor with the sorted indices.\n\nC++: Tensor::argsort(class Tensor *, class Tensor *, bool, bool) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("descending"), pybind11::arg("stable"));
		cl.def_static("select_back", (void (*)(class Tensor *, class Tensor *, class SelDescriptor *)) &Tensor::select_back, "C++: Tensor::select_back(class Tensor *, class Tensor *, class SelDescriptor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("sd"));
		cl.def_static("set_select_back", (void (*)(class Tensor *, class Tensor *, class SelDescriptor *)) &Tensor::set_select_back, "C++: Tensor::set_select_back(class Tensor *, class Tensor *, class SelDescriptor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("sd"));
		cl.def("clone", (class Tensor * (Tensor::*)()) &Tensor::clone, "Clone a tensor (same device). Similar to copy, but returning a new instance\n\n  \n    Tensor\n\nC++: Tensor::clone() --> class Tensor *", pybind11::return_value_policy::automatic);
		cl.def("reallocate", (void (Tensor::*)(class Tensor *)) &Tensor::reallocate, "Reallocates a tensor into this one.\n  Replaces the pointer of this tensor, with the pointer of a reference tensor.\n\n  \n Reference tensor\n  \n\n     void\n\nC++: Tensor::reallocate(class Tensor *) --> void", pybind11::arg("old_t"));
		cl.def("resize", [](Tensor &o, int const & a0) -> void { return o.resize(a0); }, "", pybind11::arg("b"));
		cl.def("resize", [](Tensor &o, int const & a0, float * a1) -> void { return o.resize(a0, a1); }, "", pybind11::arg("b"), pybind11::arg("fptr"));
		cl.def("resize", [](Tensor &o, int const & a0, float * a1, void * a2) -> void { return o.resize(a0, a1, a2); }, "", pybind11::arg("b"), pybind11::arg("fptr"), pybind11::arg("fptr2"));
		cl.def("resize", (void (Tensor::*)(int, float *, void *, bool)) &Tensor::resize, "Resizes a tensor ({2, 2, 2} => {10, 2, 2}).\n\n  \n\n\n\nC++: Tensor::resize(int, float *, void *, bool) --> void", pybind11::arg("b"), pybind11::arg("fptr"), pybind11::arg("fptr2"), pybind11::arg("delete_data"));
		cl.def("fill_rand_uniform_", (void (Tensor::*)(float)) &Tensor::fill_rand_uniform_, "Fills a tensor in-place, with values randomly sampled from a uniform distribution\n\n  \n  Scale factor of the values generated by the uniform distribution.\n\nC++: Tensor::fill_rand_uniform_(float) --> void", pybind11::arg("v"));
		cl.def("fill_rand_uniform", (class Tensor * (Tensor::*)(float)) &Tensor::fill_rand_uniform, "Fills a tensor in-place, with values randomly sampled from a uniform distribution\n\n  \n  Scale factor of the values generated by the uniform distribution.\n   \n\n A new tensor with the result\n\nC++: Tensor::fill_rand_uniform(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def("fill_rand_signed_uniform_", (void (Tensor::*)(float)) &Tensor::fill_rand_signed_uniform_, "Fills a tensor in-place, with values randomly sampled from a signed uniform distribution\n\n  \n  Scale factor of the values generated by the signed uniform distribution.\n\nC++: Tensor::fill_rand_signed_uniform_(float) --> void", pybind11::arg("v"));
		cl.def("fill_rand_signed_uniform", (class Tensor * (Tensor::*)(float)) &Tensor::fill_rand_signed_uniform, "Fills a tensor in-place, with values randomly sampled from a signed uniform distribution\n\n  \n  Scale factor of the values generated by the signed uniform distribution.\n   \n\n A new tensor with the result\n\nC++: Tensor::fill_rand_signed_uniform(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def("fill_rand_normal_", [](Tensor &o, float const & a0, float const & a1) -> void { return o.fill_rand_normal_(a0, a1); }, "", pybind11::arg("m"), pybind11::arg("s"));
		cl.def("fill_rand_normal_", (void (Tensor::*)(float, float, bool)) &Tensor::fill_rand_normal_, "Fills a tensor in-place, with values randomly sampled from a normal distribution\n\n  \n  Mean of the normal distribution.\n  \n\n  Standard deviation of the normal distribution.\n  \n\n  Whether to use or not the fast math mode.\n\nC++: Tensor::fill_rand_normal_(float, float, bool) --> void", pybind11::arg("m"), pybind11::arg("s"), pybind11::arg("fast_math"));
		cl.def("fill_rand_normal", [](Tensor &o, float const & a0, float const & a1) -> Tensor * { return o.fill_rand_normal(a0, a1); }, "", pybind11::return_value_policy::automatic, pybind11::arg("m"), pybind11::arg("s"));
		cl.def("fill_rand_normal", (class Tensor * (Tensor::*)(float, float, bool)) &Tensor::fill_rand_normal, "Fills a tensor in-place, with values randomly sampled from a normal distribution\n\n  \n  Mean of the normal distribution.\n  \n\n  Standard deviation of the normal distribution.\n  \n\n  Whether to use or not the fast math mode.\n   \n\n A new tensor with the result\n\nC++: Tensor::fill_rand_normal(float, float, bool) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("m"), pybind11::arg("s"), pybind11::arg("fast_math"));
		cl.def("fill_rand_binary_", (void (Tensor::*)(float)) &Tensor::fill_rand_binary_, "Fills a tensor in-place, with values randomly sampled from a binary distribution\n\n  \n Binarization threshold. 1 if rnd() >= t, 0 otherwise\n\nC++: Tensor::fill_rand_binary_(float) --> void", pybind11::arg("v"));
		cl.def("fill_rand_binary", (class Tensor * (Tensor::*)(float)) &Tensor::fill_rand_binary, "Fills a tensor in-place, with values randomly sampled from a binary distribution\n\n  \n Binarization threshold. 1 if rnd() >= t, 0 otherwise\n   \n\n A new tensor with the result\n\nC++: Tensor::fill_rand_binary(float) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("v"));
		cl.def_static("copy", (void (*)(class Tensor *, class Tensor *)) &Tensor::copy, "Copy data from tensor A to B.\n\n  \n   Tensor\n  \n\n   Tensor\n  \n\n    void\n\nC++: Tensor::copy(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_fill", (void (*)(class Tensor *, int, int, class Tensor *, int, int, int)) &Tensor::fill, "Fill tensor with values from another tensor\n   \n\n The tensor to take values from.\n   \n\n Initial position of A.\n   \n\n Final position of A.\n   \n\n The tensor to fill\n   \n\n Initial position of B\n   \n\n Final position of B\n   \n\n step to go from one position to the following one\n\nC++: Tensor::fill(class Tensor *, int, int, class Tensor *, int, int, int) --> void", pybind11::arg("A"), pybind11::arg("aini"), pybind11::arg("aend"), pybind11::arg("B"), pybind11::arg("bini"), pybind11::arg("bend"), pybind11::arg("inc"));
		cl.def_static("tile", (void (*)(class Tensor *, class Tensor *)) &Tensor::tile, "C++: Tensor::tile(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("static_add", (void (*)(float, class Tensor *, float, class Tensor *, class Tensor *, int)) &Tensor::add, "Weighted element-wise sum of two tensors.\n   \n\n Weight of tensor ``A``.\n   \n\n Input tensor.\n   \n\n Weight of tensor ``B``.\n   \n\n Input tensor.\n   \n\n Output tensor. C = sc*A + scB*B\n   \n\n if ``incC`` is 1, C += sc*A + scB*B\n\nC++: Tensor::add(float, class Tensor *, float, class Tensor *, class Tensor *, int) --> void", pybind11::arg("scA"), pybind11::arg("A"), pybind11::arg("scB"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("inc", (void (*)(class Tensor *, class Tensor *)) &Tensor::inc, "Increment element-wise one tensors with the values of another.\n   \n\n Input tensor.\n   \n\n Output tensor. The incremented tensor with values from ``A``.\n\nC++: Tensor::inc(class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("el_div", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_div, "Eelement-wise division of two tensors.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n Output tensor. C = A./B\n   \n\n if ``incC`` is 1, C += A./B\n\nC++: Tensor::el_div(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("el_mult", (void (*)(class Tensor *, class Tensor *, class Tensor *, int)) &Tensor::el_mult, "Eelement-wise multiplication of two tensors.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n Output tensor. C = A*B\n   \n\n if ``incC`` is 1, C += A*B\n\nC++: Tensor::el_mult(class Tensor *, class Tensor *, class Tensor *, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"), pybind11::arg("incC"));
		cl.def_static("sum2D_rowwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_rowwise, "Matrix sum row-wise of two 2D tensors.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n Output tensor. C = A+B, row-wise\n\nC++: Tensor::sum2D_rowwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("sum2D_colwise", (void (*)(class Tensor *, class Tensor *, class Tensor *)) &Tensor::sum2D_colwise, "Matrix sum column-wise of two 2D tensors.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n Output tensor. C = A+B, column-wise\n\nC++: Tensor::sum2D_colwise(class Tensor *, class Tensor *, class Tensor *) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
		cl.def_static("reduce_sum2D", (void (*)(class Tensor *, class Tensor *, int, int)) &Tensor::reduce_sum2D, "Reduction of a matrix to a 1-D tensor.\n   \n\n Input 2-D tensor.\n   \n\n Output 1-D tensor.\n   \n\n Dimension to be sumed.\n   \n\n if ``incB`` is 1, B += reduce(A)\n\nC++: Tensor::reduce_sum2D(class Tensor *, class Tensor *, int, int) --> void", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("axis"), pybind11::arg("incB"));
		cl.def_static("eqsize", (int (*)(class Tensor *, class Tensor *)) &Tensor::eqsize, "C++: Tensor::eqsize(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("sameDevice", (bool (*)(class Tensor *, class Tensor *)) &Tensor::sameDevice, "Check if two tensors are in the same device\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n 1 if they are equivalent, 0 otherwise.\n\nC++: Tensor::sameDevice(class Tensor *, class Tensor *) --> bool", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("sameSize", (bool (*)(class Tensor *, class Tensor *)) &Tensor::sameSize, "Check if two tensors have the same size. (Ignores shape)\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n 1 if they are equivalent, 0 otherwise.\n\nC++: Tensor::sameSize(class Tensor *, class Tensor *) --> bool", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("sameShape", (int (*)(class Tensor *, class Tensor *)) &Tensor::sameShape, "Check if two tensors have the same shape.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n 1 if they have the same shape, 0 otherwise.\n\nC++: Tensor::sameShape(class Tensor *, class Tensor *) --> int", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equivalent", [](class Tensor * a0, class Tensor * a1) -> int { return Tensor::equivalent(a0, a1); }, "", pybind11::arg("A"), pybind11::arg("B"));
		cl.def_static("equivalent", [](class Tensor * a0, class Tensor * a1, float const & a2) -> int { return Tensor::equivalent(a0, a1, a2); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("atol"));
		cl.def_static("equivalent", [](class Tensor * a0, class Tensor * a1, float const & a2, float const & a3) -> int { return Tensor::equivalent(a0, a1, a2, a3); }, "", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("atol"), pybind11::arg("rtol"));
		cl.def_static("equivalent", (int (*)(class Tensor *, class Tensor *, float, float, bool)) &Tensor::equivalent, "Check if two tensors have the same contents given a threshold.\n   \n\n Input tensor.\n   \n\n Input tensor.\n   \n\n Error threshold.\n   \n\n 1 if they are equivalent, 0 otherwise.\n\nC++: Tensor::equivalent(class Tensor *, class Tensor *, float, float, bool) --> int", pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("atol"), pybind11::arg("rtol"), pybind11::arg("equal_nan"));

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
#include <eddl/optimizers/optim.h>
#include <eddl/regularizers/regularizer.h>
#include <eddl/tensor/tensor.h>
#include <fstream>
#include <ios>
#include <iterator>
#include <layer_addons.hpp>
#include <loss_addons.hpp>
#include <memory>
#include <metric_addons.hpp>
#include <net_addons.hpp>
#include <sstream> // __str__
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <utils_addons.hpp>


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
	class Loss * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Loss *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Loss *>::value) {
				static pybind11::detail::overload_caster_t<class Loss *> caster;
				return pybind11::detail::cast_ref<class Loss *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Loss *>(std::move(o));
		}
		return Loss::clone();
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
	class Metric * clone() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Metric *>(this), "clone");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Metric *>::value) {
				static pybind11::detail::overload_caster_t<class Metric *> caster;
				return pybind11::detail::cast_ref<class Metric *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Metric *>(std::move(o));
		}
		return Metric::clone();
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
	void setTrainable(bool a0) override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Layer *>(this), "setTrainable");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>(a0);
			if (pybind11::detail::cast_is_temporary_value_reference<void>::value) {
				static pybind11::detail::overload_caster_t<void> caster;
				return pybind11::detail::cast_ref<void>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<void>(std::move(o));
		}
		return Layer::setTrainable(a0);
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
	class Optimizer * share() override { 
		pybind11::gil_scoped_acquire gil;
		pybind11::function overload = pybind11::get_overload(static_cast<const Optimizer *>(this), "share");
		if (overload) {
			auto o = overload.operator()<pybind11::return_value_policy::reference>();
			if (pybind11::detail::cast_is_temporary_value_reference<class Optimizer *>::value) {
				static pybind11::detail::overload_caster_t<class Optimizer *> caster;
				return pybind11::detail::cast_ref<class Optimizer *>(std::move(o), caster);
			}
			else return pybind11::detail::cast_safe<class Optimizer *>(std::move(o));
		}
		return Optimizer::share();
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
		cl.def("clone", (class Loss * (Loss::*)()) &Loss::clone, "C++: Loss::clone() --> class Loss *", pybind11::return_value_policy::automatic);
		cl.def("assign", (class Loss & (Loss::*)(const class Loss &)) &Loss::operator=, "C++: Loss::operator=(const class Loss &) --> class Loss &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		loss_addons(cl);
	}
	{ // CompServ file:eddl/net/compserv.h line:20
		pybind11::class_<CompServ, std::shared_ptr<CompServ>> cl(M(""), "CompServ", "");
		cl.def( pybind11::init( [](){ return new CompServ(); } ) );
		cl.def( pybind11::init( [](CompServ const &o){ return new CompServ(o); } ) );
		cl.def_readwrite("type", &CompServ::type);
		cl.def_readwrite("hw", &CompServ::hw);
		cl.def_readwrite("threads_arg", &CompServ::threads_arg);
		cl.def_readwrite("local_threads", &CompServ::local_threads);
		cl.def_readwrite("local_gpus", &CompServ::local_gpus);
		cl.def_readwrite("local_fpgas", &CompServ::local_fpgas);
		cl.def_readwrite("lsb", &CompServ::lsb);
		cl.def_readwrite("isshared", &CompServ::isshared);
		cl.def_readwrite("mem_level", &CompServ::mem_level);
		cl.def("share", (class CompServ * (CompServ::*)()) &CompServ::share, "C++: CompServ::share() --> class CompServ *", pybind11::return_value_policy::automatic);
		cl.def("clone", (class CompServ * (CompServ::*)()) &CompServ::clone, "C++: CompServ::clone() --> class CompServ *", pybind11::return_value_policy::automatic);
	}
	{ // Metric file:eddl/metrics/metric.h line:23
		pybind11::class_<Metric, std::unique_ptr<Metric, pybind11::nodelete>, PyCallBack_Metric> cl(M(""), "Metric", "");
		cl.def( pybind11::init( [](PyCallBack_Metric const &o){ return new PyCallBack_Metric(o); } ) );
		cl.def( pybind11::init( [](Metric const &o){ return new Metric(o); } ) );
		cl.def_readwrite("name", &Metric::name);
		cl.def("value", (float (Metric::*)(class Tensor *, class Tensor *)) &Metric::value, "C++: Metric::value(class Tensor *, class Tensor *) --> float", pybind11::arg("T"), pybind11::arg("Y"));
		cl.def("clone", (class Metric * (Metric::*)()) &Metric::clone, "C++: Metric::clone() --> class Metric *", pybind11::return_value_policy::automatic);
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
		cl.def_readwrite("isrecurrent", &Layer::isrecurrent);
		cl.def_readwrite("isshared", &Layer::isshared);
		cl.def_readwrite("iscloned", &Layer::iscloned);
		cl.def_readwrite("isnorm", &Layer::isnorm);
		cl.def_readwrite("isdecoder", &Layer::isdecoder);
		cl.def_readwrite("distributed_training", &Layer::distributed_training);
		cl.def_readwrite("params", &Layer::params);
		cl.def_readwrite("gradients", &Layer::gradients);
		cl.def_readwrite("states", &Layer::states);
		cl.def_readwrite("delta_states", &Layer::delta_states);
		cl.def_readwrite("acc_gradients", &Layer::acc_gradients);
		cl.def_readwrite("parent", &Layer::parent);
		cl.def_readwrite("child", &Layer::child);
		cl.def_readwrite("clones", &Layer::clones);
		cl.def_readwrite("mode", &Layer::mode);
		cl.def_readwrite("dev", &Layer::dev);
		cl.def_readwrite("lin", &Layer::lin);
		cl.def_readwrite("lout", &Layer::lout);
		cl.def_readwrite("delta_bp", &Layer::delta_bp);
		cl.def_readwrite("detached", &Layer::detached);
		cl.def_readwrite("do_deletes", &Layer::do_deletes);
		cl.def_readwrite("verbosity_level", &Layer::verbosity_level);
		cl.def("initialize", (void (Layer::*)()) &Layer::initialize, "C++: Layer::initialize() --> void");
		cl.def("info", (void (Layer::*)()) &Layer::info, "C++: Layer::info() --> void");
		cl.def("setmode", (void (Layer::*)(int)) &Layer::setmode, "C++: Layer::setmode(int) --> void", pybind11::arg("m"));
		cl.def("check_target", (void (Layer::*)()) &Layer::check_target, "C++: Layer::check_target() --> void");
		cl.def("detach", (void (Layer::*)(class Layer *)) &Layer::detach, "C++: Layer::detach(class Layer *) --> void", pybind11::arg("l"));
		cl.def("clamp", (void (Layer::*)(float, float)) &Layer::clamp, "C++: Layer::clamp(float, float) --> void", pybind11::arg("min"), pybind11::arg("max"));
		cl.def("set_detach", (void (Layer::*)()) &Layer::set_detach, "C++: Layer::set_detach() --> void");
		cl.def("set_mem_level", (void (Layer::*)(int)) &Layer::set_mem_level, "C++: Layer::set_mem_level(int) --> void", pybind11::arg("mem"));
		cl.def("mem_delta_parent", (void (Layer::*)()) &Layer::mem_delta_parent, "C++: Layer::mem_delta_parent() --> void");
		cl.def("mem_delta", (void (Layer::*)()) &Layer::mem_delta, "C++: Layer::mem_delta() --> void");
		cl.def("free_delta", (void (Layer::*)()) &Layer::free_delta, "C++: Layer::free_delta() --> void");
		cl.def("copy", (void (Layer::*)(class Layer *)) &Layer::copy, "C++: Layer::copy(class Layer *) --> void", pybind11::arg("l2"));
		cl.def("resize", (void (Layer::*)(int)) &Layer::resize, "C++: Layer::resize(int) --> void", pybind11::arg("batch"));
		cl.def("setTrainable", (void (Layer::*)(bool)) &Layer::setTrainable, "C++: Layer::setTrainable(bool) --> void", pybind11::arg("value"));
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
		cl.def("decrease_and_get_reference_counter", (int (Layer::*)()) &Layer::decrease_and_get_reference_counter, "C++: Layer::decrease_and_get_reference_counter() --> int");
		cl.def("increase_reference_counter", (void (Layer::*)()) &Layer::increase_reference_counter, "C++: Layer::increase_reference_counter() --> void");
		cl.def("assign", (class Layer & (Layer::*)(const class Layer &)) &Layer::operator=, "C++: Layer::operator=(const class Layer &) --> class Layer &", pybind11::return_value_policy::automatic, pybind11::arg(""));

		layer_addons(cl);
	}
	{ // Optimizer file:eddl/optimizers/optim.h line:27
		pybind11::class_<Optimizer, std::shared_ptr<Optimizer>, PyCallBack_Optimizer> cl(M(""), "Optimizer", "");
		cl.def( pybind11::init( [](){ return new Optimizer(); }, [](){ return new PyCallBack_Optimizer(); } ) );
		cl.def( pybind11::init( [](PyCallBack_Optimizer const &o){ return new PyCallBack_Optimizer(o); } ) );
		cl.def( pybind11::init( [](Optimizer const &o){ return new Optimizer(o); } ) );
		cl.def_readwrite("name", &Optimizer::name);
		cl.def_readwrite("layers", &Optimizer::layers);
		cl.def_readwrite("isshared", &Optimizer::isshared);
		cl.def_readwrite("clip_val", &Optimizer::clip_val);
		cl.def("set_clip_val", (void (Optimizer::*)(float)) &Optimizer::set_clip_val, "C++: Optimizer::set_clip_val(float) --> void", pybind11::arg("v"));
		cl.def("clip", (void (Optimizer::*)()) &Optimizer::clip, "C++: Optimizer::clip() --> void");
		cl.def("applygrads", (void (Optimizer::*)(int)) &Optimizer::applygrads, "C++: Optimizer::applygrads(int) --> void", pybind11::arg("batch"));
		cl.def("clone", (class Optimizer * (Optimizer::*)()) &Optimizer::clone, "C++: Optimizer::clone() --> class Optimizer *", pybind11::return_value_policy::automatic);
		cl.def("share", (class Optimizer * (Optimizer::*)()) &Optimizer::share, "C++: Optimizer::share() --> class Optimizer *", pybind11::return_value_policy::automatic);
		cl.def("assign", (class Optimizer & (Optimizer::*)(const class Optimizer &)) &Optimizer::operator=, "C++: Optimizer::operator=(const class Optimizer &) --> class Optimizer &", pybind11::return_value_policy::automatic, pybind11::arg(""));
	}
	{ // Net file: line:41
		pybind11::class_<Net, std::shared_ptr<Net>> cl(M(""), "Net", "");
		cl.def( pybind11::init( [](){ return new Net(); } ) );
		cl.def( pybind11::init( [](Net const &o){ return new Net(o); } ) );
		cl.def_readwrite("name", &Net::name);
		cl.def_readwrite("dev", &Net::dev);
		cl.def_readwrite("batch_size", &Net::batch_size);
		cl.def_readwrite("tr_batches", &Net::tr_batches);
		cl.def_readwrite("inferenced_samples", &Net::inferenced_samples);
		cl.def_readwrite("trmode", &Net::trmode);
		cl.def_readwrite("mem_level", &Net::mem_level);
		cl.def_readwrite("verbosity_level", &Net::verbosity_level);
		cl.def_readwrite("onnx_pretrained", &Net::onnx_pretrained);
		cl.def_readwrite("isrecurrent", &Net::isrecurrent);
		cl.def_readwrite("isbuild", &Net::isbuild);
		cl.def_readwrite("isdecoder", &Net::isdecoder);
		cl.def_readwrite("isencoder", &Net::isencoder);
		cl.def_readwrite("decsize", &Net::decsize);
		cl.def_readwrite("devsel", &Net::devsel);
		cl.def_readwrite("do_compserv_delete", &Net::do_compserv_delete);
		cl.def_readwrite("layers", &Net::layers);
		cl.def_readwrite("layersf", &Net::layersf);
		cl.def_readwrite("layersb", &Net::layersb);
		cl.def_readwrite("lin", &Net::lin);
		cl.def_readwrite("din", &Net::din);
		cl.def_readwrite("lout", &Net::lout);
		cl.def_readwrite("vfts", &Net::vfts);
		cl.def_readwrite("vbts", &Net::vbts);
		cl.def_readwrite("netinput", &Net::netinput);
		cl.def_readwrite("losses", &Net::losses);
		cl.def_readwrite("metrics", &Net::metrics);
		cl.def_readwrite("fiterr", &Net::fiterr);
		cl.def_readwrite("total_loss", &Net::total_loss);
		cl.def_readwrite("total_metric", &Net::total_metric);
		cl.def_readwrite("has_to_close_flog_tr", &Net::has_to_close_flog_tr);
		cl.def_readwrite("has_to_close_flog_ts", &Net::has_to_close_flog_ts);
		cl.def_readwrite("do_optimizer_delete", &Net::do_optimizer_delete);
		cl.def_readwrite("snets", &Net::snets);
		cl.def("toCPU", (void (Net::*)(int)) &Net::toCPU, "C++: Net::toCPU(int) --> void", pybind11::arg("t"));
		cl.def("fts", (void (Net::*)()) &Net::fts, "C++: Net::fts() --> void");
		cl.def("bts", (void (Net::*)()) &Net::bts, "C++: Net::bts() --> void");
		cl.def("split", (void (Net::*)(int, int)) &Net::split, "C++: Net::split(int, int) --> void", pybind11::arg("c"), pybind11::arg("todev"));
		cl.def("unroll_enc", (class Net * (Net::*)(int, int)) &Net::unroll_enc, "C++: Net::unroll_enc(int, int) --> class Net *", pybind11::return_value_policy::automatic, pybind11::arg("inl"), pybind11::arg("outl"));
		cl.def("unroll_enc_dec", (class Net * (Net::*)(int, int)) &Net::unroll_enc_dec, "C++: Net::unroll_enc_dec(int, int) --> class Net *", pybind11::return_value_policy::automatic, pybind11::arg("inl"), pybind11::arg("outl"));
		cl.def("unroll_dec", (class Net * (Net::*)(int, int)) &Net::unroll_dec, "C++: Net::unroll_dec(int, int) --> class Net *", pybind11::return_value_policy::automatic, pybind11::arg("inl"), pybind11::arg("outl"));
		cl.def("build_rnet", (void (Net::*)(int, int)) &Net::build_rnet, "C++: Net::build_rnet(int, int) --> void", pybind11::arg("inl"), pybind11::arg("outl"));
		cl.def("inNet", (int (Net::*)(class Layer *)) &Net::inNet, "C++: Net::inNet(class Layer *) --> int", pybind11::arg("l"));
		cl.def("inNetF", (int (Net::*)(class Layer *)) &Net::inNetF, "C++: Net::inNetF(class Layer *) --> int", pybind11::arg("l"));
		cl.def("inNetB", (int (Net::*)(class Layer *)) &Net::inNetB, "C++: Net::inNetB(class Layer *) --> int", pybind11::arg("l"));
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
		cl.def("collect_acc_grads", (void (Net::*)()) &Net::collect_acc_grads, "C++: Net::collect_acc_grads() --> void");
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
}


// File: eddl/net/netloss.cpp
#include <eddl/layers/layer.h>
#include <eddl/net/netloss.h>
#include <eddl/tensor/tensor.h>
#include <fstream>
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
#include <utils_addons.hpp>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_net_netloss(std::function< pybind11::module &(std::string const &namespace_) > &M)
{
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
#include <eddl/optimizers/optim.h>
#include <eddl/tensor/tensor.h>
#include <eddl_addons.hpp>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <functional>
#include <string>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <utils_addons.hpp>


#ifndef BINDER_PYBIND11_TYPE_CASTER
	#define BINDER_PYBIND11_TYPE_CASTER
	PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);
	PYBIND11_DECLARE_HOLDER_TYPE(T, T*);
	PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);
#endif

void bind_eddl_apis_eddl(std::function< pybind11::module &(std::string const &namespace_) > &M)
{

	eddl_addons(M("eddl"));
	// eddl::build(class Net *, class Optimizer *, class CompServ *, bool) file:eddl/apis/eddl.h line:74
	M("eddl").def("build", [](class Net * a0) -> void { return eddl::build(a0); }, "", pybind11::arg("net"));
	M("eddl").def("build", [](class Net * a0, class Optimizer * a1) -> void { return eddl::build(a0, a1); }, "", pybind11::arg("net"), pybind11::arg("o"));
	M("eddl").def("build", [](class Net * a0, class Optimizer * a1, class CompServ * a2) -> void { return eddl::build(a0, a1, a2); }, "", pybind11::arg("net"), pybind11::arg("o"), pybind11::arg("cs"));
	M("eddl").def("build", (void (*)(class Net *, class Optimizer *, class CompServ *, bool)) &eddl::build, "C++: eddl::build(class Net *, class Optimizer *, class CompServ *, bool) --> void", pybind11::arg("net"), pybind11::arg("o"), pybind11::arg("cs"), pybind11::arg("init_weigths"));

	// eddl::toGPU(class Net *) file:eddl/apis/eddl.h line:104
	M("eddl").def("toGPU", (void (*)(class Net *)) &eddl::toGPU, "C++: eddl::toGPU(class Net *) --> void", pybind11::arg("net"));

	// eddl::toCPU(class Net *, int) file:eddl/apis/eddl.h line:114
	M("eddl").def("toCPU", [](class Net * a0) -> void { return eddl::toCPU(a0); }, "", pybind11::arg("net"));
	M("eddl").def("toCPU", (void (*)(class Net *, int)) &eddl::toCPU, "Assign model operations to the CPU.\n\n  \n  Model\n  \n\n  CPU Threads\n  \n\n     (void)\n\nC++: eddl::toCPU(class Net *, int) --> void", pybind11::arg("net"), pybind11::arg("t"));

	// eddl::CS_CPU() file:eddl/apis/eddl.h line:123
	M("eddl").def("CS_CPU", (class CompServ * (*)()) &eddl::CS_CPU, "Executes the code in the CPU.\n\n  \n  Indicates the number of threads to use (-1 = all available threads)\n  \n\n  Indicates the memory consumption of the model. One of \"full_mem\" (default), \"mid_mem\" or \"low_mem\".\n  \n\n     The computer service itself.\n\nC++: eddl::CS_CPU() --> class CompServ *", pybind11::return_value_policy::automatic);

	// eddl::CS_CPU(int) file:eddl/apis/eddl.h line:132
	M("eddl").def("CS_CPU", (class CompServ * (*)(int)) &eddl::CS_CPU, "Executes the code in the CPU.\n\n  \n  Indicates the number of threads to use (-1 = all available threads)\n  \n\n     The computer service itself.\n\nC++: eddl::CS_CPU(int) --> class CompServ *", pybind11::return_value_policy::automatic, pybind11::arg("th"));

	// eddl::summary(class Net *) file:eddl/apis/eddl.h line:257
	M("eddl").def("summary", (void (*)(class Net *)) &eddl::summary, "Prints a summary representation of your model.\n\n  \n  Model to print\n  \n\n     (void) Prints the model\n\nC++: eddl::summary(class Net *) --> void", pybind11::arg("m"));

	// eddl::set_mode(class Net *, int) file:eddl/apis/eddl.h line:512
	M("eddl").def("set_mode", (void (*)(class Net *, int)) &eddl::set_mode, "Set model mode.\n\n  \n  Model\n  \n\n  Train 1, Test 0\n  \n\n     (void)\n\nC++: eddl::set_mode(class Net *, int) --> void", pybind11::arg("net"), pybind11::arg("mode"));

	// eddl::reset_loss(class Net *) file:eddl/apis/eddl.h line:519
	M("eddl").def("reset_loss", (void (*)(class Net *)) &eddl::reset_loss, "Resets model loss.\n\n  \n  Model\n  \n\n     (void)\n\nC++: eddl::reset_loss(class Net *) --> void", pybind11::arg("m"));

	// eddl::zeroGrads(class Net *) file:eddl/apis/eddl.h line:557
	M("eddl").def("zeroGrads", (void (*)(class Net *)) &eddl::zeroGrads, "Set model gradients to zero.\n\n  \n  Model\n  \n\n     (void)\n\nC++: eddl::zeroGrads(class Net *) --> void", pybind11::arg("m"));

	// eddl::backward(class Net *) file:eddl/apis/eddl.h line:572
	M("eddl").def("backward", (void (*)(class Net *)) &eddl::backward, "Computes the gradient of the model through the backward graph.\n\n  \n  Model\n  \n\n     (void)\n\nC++: eddl::backward(class Net *) --> void", pybind11::arg("net"));

	// eddl::backward(class NetLoss *) file:eddl/apis/eddl.h line:579
	M("eddl").def("backward", (void (*)(class NetLoss *)) &eddl::backward, "Computes the gradient of the model associated to the given loss object through the backward graph.\n\n  \n  Loss\n  \n\n     (void)\n\nC++: eddl::backward(class NetLoss *) --> void", pybind11::arg("l"));

	// eddl::optimize(class NetLoss *) file:eddl/apis/eddl.h line:580
	M("eddl").def("optimize", (void (*)(class NetLoss *)) &eddl::optimize, "C++: eddl::optimize(class NetLoss *) --> void", pybind11::arg("l"));

	// eddl::update(class Net *) file:eddl/apis/eddl.h line:588
	M("eddl").def("update", (void (*)(class Net *)) &eddl::update, "Updates the weights of the model\n\n  \n  Model\n  \n\n     (void)\n\nC++: eddl::update(class Net *) --> void", pybind11::arg("m"));

	// eddl::print_loss(class Net *, int) file:eddl/apis/eddl.h line:596
	M("eddl").def("print_loss", (void (*)(class Net *, int)) &eddl::print_loss, "Prints model loss at some batch.\n\n  \n  Model\n  \n\n  Batch number\n  \n\n     (void)\n\nC++: eddl::print_loss(class Net *, int) --> void", pybind11::arg("m"), pybind11::arg("batch"));

	// eddl::clamp(class Net *, float, float) file:eddl/apis/eddl.h line:623
	M("eddl").def("clamp", (void (*)(class Net *, float, float)) &eddl::clamp, "Model parameters values clipping.\n\n  \n  Model\n  \n\n  Minimum value\n  \n\n   Maximum value\n  \n\n     (void) Performs model clamp between min and max\n\nC++: eddl::clamp(class Net *, float, float) --> void", pybind11::arg("m"), pybind11::arg("min"), pybind11::arg("max"));

	// eddl::compute_loss(class NetLoss *) file:eddl/apis/eddl.h line:632
	M("eddl").def("compute_loss", (float (*)(class NetLoss *)) &eddl::compute_loss, "Computes loss of the associated model\n\n  \n  Loss\n  \n\n (float) Computed loss\n\nC++: eddl::compute_loss(class NetLoss *) --> float", pybind11::arg("L"));

	// eddl::compute_metric(class NetLoss *) file:eddl/apis/eddl.h line:639
	M("eddl").def("compute_metric", (float (*)(class NetLoss *)) &eddl::compute_metric, "Computes loss of the associated model (same as ``compute_loss``)\n\n  \n  Loss\n  \n\n (float) Computed loss\n\nC++: eddl::compute_metric(class NetLoss *) --> float", pybind11::arg("L"));

	// eddl::show_profile() file:eddl/apis/eddl.h line:711
	M("eddl").def("show_profile", (void (*)()) &eddl::show_profile, "Shows profile information.\n\nC++: eddl::show_profile() --> void");

	// eddl::GetStates(class Layer *) file:eddl/apis/eddl.h line:2041
	M("eddl").def("GetStates", (class Layer * (*)(class Layer *)) &eddl::GetStates, "C++: eddl::GetStates(class Layer *) --> class Layer *", pybind11::return_value_policy::automatic, pybind11::arg("parent"));

	// eddl::setDecoder(class Layer *) file:eddl/apis/eddl.h line:2043
	M("eddl").def("setDecoder", (void (*)(class Layer *)) &eddl::setDecoder, "C++: eddl::setDecoder(class Layer *) --> void", pybind11::arg("l"));

	// eddl::getOutput(class Layer *) file:eddl/apis/eddl.h line:2049
	M("eddl").def("getOutput", (class Tensor * (*)(class Layer *)) &eddl::getOutput, "C++: eddl::getOutput(class Layer *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("l1"));

	// eddl::getDelta(class Layer *) file:eddl/apis/eddl.h line:2050
	M("eddl").def("getDelta", (class Tensor * (*)(class Layer *)) &eddl::getDelta, "C++: eddl::getDelta(class Layer *) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("l1"));

	// eddl::getParam(class Layer *, int) file:eddl/apis/eddl.h line:2051
	M("eddl").def("getParam", (class Tensor * (*)(class Layer *, int)) &eddl::getParam, "C++: eddl::getParam(class Layer *, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("l1"), pybind11::arg("p"));

	// eddl::getGradient(class Layer *, int) file:eddl/apis/eddl.h line:2052
	M("eddl").def("getGradient", (class Tensor * (*)(class Layer *, int)) &eddl::getGradient, "C++: eddl::getGradient(class Layer *, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("l1"), pybind11::arg("p"));

	// eddl::getState(class Layer *, int) file:eddl/apis/eddl.h line:2053
	M("eddl").def("getState", (class Tensor * (*)(class Layer *, int)) &eddl::getState, "C++: eddl::getState(class Layer *, int) --> class Tensor *", pybind11::return_value_policy::automatic, pybind11::arg("l1"), pybind11::arg("p"));

	// eddl::copyOutput(class Layer *, class Layer *) file:eddl/apis/eddl.h line:2057
	M("eddl").def("copyOutput", (void (*)(class Layer *, class Layer *)) &eddl::copyOutput, "C++: eddl::copyOutput(class Layer *, class Layer *) --> void", pybind11::arg("l1"), pybind11::arg("l2"));

	// eddl::copyDelta(class Layer *, class Layer *) file:eddl/apis/eddl.h line:2058
	M("eddl").def("copyDelta", (void (*)(class Layer *, class Layer *)) &eddl::copyDelta, "C++: eddl::copyDelta(class Layer *, class Layer *) --> void", pybind11::arg("l1"), pybind11::arg("l2"));

	// eddl::copyParam(class Layer *, class Layer *, int) file:eddl/apis/eddl.h line:2059
	M("eddl").def("copyParam", [](class Layer * a0, class Layer * a1) -> void { return eddl::copyParam(a0, a1); }, "", pybind11::arg("l1"), pybind11::arg("l2"));
	M("eddl").def("copyParam", (void (*)(class Layer *, class Layer *, int)) &eddl::copyParam, "C++: eddl::copyParam(class Layer *, class Layer *, int) --> void", pybind11::arg("l1"), pybind11::arg("l2"), pybind11::arg("p"));

	// eddl::copyGradient(class Layer *, class Layer *, int) file:eddl/apis/eddl.h line:2060
	M("eddl").def("copyGradient", (void (*)(class Layer *, class Layer *, int)) &eddl::copyGradient, "C++: eddl::copyGradient(class Layer *, class Layer *, int) --> void", pybind11::arg("l1"), pybind11::arg("l2"), pybind11::arg("p"));

	// eddl::distributeParams(class Layer *) file:eddl/apis/eddl.h line:2061
	M("eddl").def("distributeParams", (void (*)(class Layer *)) &eddl::distributeParams, "C++: eddl::distributeParams(class Layer *) --> void", pybind11::arg("l"));

	// eddl::download_mnist() file:eddl/apis/eddl.h line:2201
	M("eddl").def("download_mnist", (void (*)()) &eddl::download_mnist, "Downloads MNIST Dataset.\n\n  \n   http://yann.lecun.com/exdb/mnist/\n\n  \n     (void) The binary files of MNIST\n\nC++: eddl::download_mnist() --> void");

	// eddl::download_cifar10() file:eddl/apis/eddl.h line:2209
	M("eddl").def("download_cifar10", (void (*)()) &eddl::download_cifar10, "Downloads CIFAR-10 Dataset.\n\n  \n   https://www.cs.toronto.edu/~kriz/cifar.html\n\n  \n     (void) The binary files of CIFAR-10\n\nC++: eddl::download_cifar10() --> void");

	// eddl::download_drive() file:eddl/apis/eddl.h line:2217
	M("eddl").def("download_drive", (void (*)()) &eddl::download_drive, "Downloads DRIVE Dataset.\n\n  \n   https://drive.grand-challenge.org/\n\n  \n     (void) The numpy files of DRIVE\n\nC++: eddl::download_drive() --> void");

	// eddl::download_imdb_2000() file:eddl/apis/eddl.h line:2226
	M("eddl").def("download_imdb_2000", (void (*)()) &eddl::download_imdb_2000, "Downloads IMDB Dataset. 2000 most frequent words\n\n  \n   https://ai.stanford.edu/~amaas/data/sentiment/\n\n  \n     (void) The binary files of IMDB\n\nC++: eddl::download_imdb_2000() --> void");

	// eddl::download_eutrans() file:eddl/apis/eddl.h line:2236
	M("eddl").def("download_eutrans", (void (*)()) &eddl::download_eutrans, "Downloads EuTrans Dataset.\n\n  \n\n\n\n  \n     (void) The binary files of EuTrans\n\nC++: eddl::download_eutrans() --> void");

	// eddl::download_flickr() file:eddl/apis/eddl.h line:2245
	M("eddl").def("download_flickr", (void (*)()) &eddl::download_flickr, "Downloads Flickr Dataset (small partition)\n\n  \n\n\n\n  \n     (void) The binary files of Flickr\n\nC++: eddl::download_flickr() --> void");

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
void bind_eddl_net_netloss(std::function< pybind11::module &(std::string const &namespace_) > &M);
void bind_eddl_apis_eddl(std::function< pybind11::module &(std::string const &namespace_) > &M);


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
	};
	for(auto &p : sub_modules ) modules[p.first.size() ? p.first+"::"+p.second : p.second] = modules[p.first].def_submodule(p.second.c_str(), ("Bindings for " + p.first + "::" + p.second + " namespace").c_str() );

	//pybind11::class_<std::shared_ptr<void>>(M(""), "_encapsulated_data_");

	bind_eddl_descriptors_tensor_descriptors(M);
	bind_eddl_utils(M);
	bind_eddl_tensor_tensor(M);
	bind_eddl_losses_loss(M);
	bind_eddl_net_netloss(M);
	bind_eddl_apis_eddl(M);

}

// Source list file: /pyeddl/codegen/bindings/_core.sources
// _core.cpp
// eddl/descriptors/tensor_descriptors.cpp
// eddl/tensor/tensor.cpp
// eddl/losses/loss.cpp
// eddl/net/netloss.cpp
// eddl/apis/eddl.cpp

// Modules list file: /pyeddl/codegen/bindings/_core.modules
// eddl 
