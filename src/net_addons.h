#define net_addons(cl) \
cl.def(pybind11::init<vector<Layer*>, vector<Layer*>>(), pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>()); \
cl.def("summary", (string (Net::*)()) &Net::summary, "C++: Net::summary() --> string"); \
 cl.def("build", (void (Net::*)(Optimizer*, vloss, vmetrics, CompServ*)) &Net::build, "C++: Net::build(Optimizer*, vloss, vmetrics, CompServ*) --> void", pybind11::arg("opt"), pybind11::arg("lo"), pybind11::arg("me"), pybind11::arg("cs"), pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>(), pybind11::keep_alive<1, 4>(), pybind11::keep_alive<1, 5>()); \
 cl.def("fit", (void (Net::*)(vtensor, vtensor, int, int)) &Net::fit, "C++: Net::fit(vtensor, vtensor, int, int) --> void", pybind11::arg("tin"), pybind11::arg("tout"), pybind11::arg("batch_size"), pybind11::arg("epochs"), pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>()); \
  cl.def("evaluate", (void (Net::*)(vtensor, vtensor)) &Net::evaluate, "C++: Net::evaluate(vtensor, vtensor) --> void", pybind11::arg("tin"), pybind11::arg("tout"), pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>()); \
  cl.def("predict", (void (Net::*)(vtensor, vtensor)) &Net::predict, "C++: Net::predict(vtensor, vtensor) --> void", pybind11::arg("tin"), pybind11::arg("tout"), pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>()); \
  cl.def("save", [](Net &net, pybind11::object file_obj) { \
      int fd; \
      FILE *fp; \
      if ((fd = PyObject_AsFileDescriptor(file_obj.ptr())) == -1) { \
	throw std::runtime_error("can't convert object to file descriptor"); \
      } \
      if (!(fp = fdopen(fd, "w"))) { \
	throw std::runtime_error("failed to open file descriptor"); \
      } \
      net.save(fp); \
  }); \
  cl.def("load", [](Net &net, pybind11::object file_obj) { \
      int fd; \
      FILE *fp; \
      if ((fd = PyObject_AsFileDescriptor(file_obj.ptr())) == -1) { \
	throw std::runtime_error("can't convert object to file descriptor"); \
      } \
      if (!(fp = fdopen(fd, "r"))) { \
	throw std::runtime_error("failed to open file descriptor"); \
      } \
      net.load(fp); \
  });
