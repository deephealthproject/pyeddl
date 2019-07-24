#define pooldescriptor_addons(cl) \
  cl.def(pybind11::init<const vector<int>&, const vector<int>&, string>(), \
         pybind11::arg("ks"), pybind11::arg("st"), pybind11::arg("p"), \
	 pybind11::keep_alive<1, 2>(), pybind11::keep_alive<1, 3>());
