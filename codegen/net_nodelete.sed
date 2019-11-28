s/pybind11::class_<Net, std::shared_ptr<Net>> cl(M(""), "Net", "");/pybind11::class_<Net, std::unique_ptr<Net, pybind11::nodelete>> cl(M(""), "Net", "");/
