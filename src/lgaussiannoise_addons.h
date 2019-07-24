#define lgaussiannoise_addons(cl) \
  cl.def(pybind11::init<Layer*, float, string, int>(), \
         pybind11::arg("parent"), pybind11::arg("stdev"), \
	 pybind11::arg("name"), pybind11::arg("dev"), \
	 pybind11::keep_alive<1, 2>());
