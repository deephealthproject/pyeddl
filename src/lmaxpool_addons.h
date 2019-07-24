#define lmaxpool_addons(cl) \
  cl.def(pybind11::init<Layer*, PoolDescriptor*, string, int>(), \
         pybind11::arg("parent"), pybind11::arg("D"), pybind11::arg("name"), \
         pybind11::arg("dev"), \
	 pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
