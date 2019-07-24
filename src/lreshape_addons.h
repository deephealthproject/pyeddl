#define lreshape_addons(cl) \
  cl.def(pybind11::init<Layer*, vector<int>, string, int>(), \
         pybind11::arg("parent"), pybind11::arg("shape"), \
	 pybind11::arg("name"), pybind11::arg("dev"), \
	 pybind11::keep_alive<1, 2>());
