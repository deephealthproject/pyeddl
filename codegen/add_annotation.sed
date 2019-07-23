s/pybind11::class_<Tensor, std::shared_ptr<Tensor>> cl(M(""), "Tensor", "");/pybind11::class_<Tensor, std::shared_ptr<Tensor>> cl(M(""), "Tensor", pybind11::buffer_protocol());/
