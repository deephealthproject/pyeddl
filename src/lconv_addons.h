#define lconv_addons(cl) \
  cl.def(pybind11::init<Layer*, int, const vector<int>&, const vector<int>&, \
	 string, int, const vector<int>&, bool, string, int>(), \
         pybind11::arg("parent"), pybind11::arg("filters"), \
	 pybind11::arg("kernel_size"), pybind11::arg("strides"), \
	 pybind11::arg("padding"), pybind11::arg("groups"), \
	 pybind11::arg("dilation_rate"), pybind11::arg("use_bias"), \
         pybind11::arg("name"), pybind11::arg("dev"), \
	 pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 4>(), \
	 pybind11::keep_alive<1, 5>(), pybind11::keep_alive<1, 8>());
