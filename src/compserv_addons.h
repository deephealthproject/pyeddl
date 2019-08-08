#define compserv_addons(cl) \
cl.def(pybind11::init<int, vector<int>, vector<int>, int>(), \
       pybind11::arg("threads"), pybind11::arg("g"), pybind11::arg("f"), \
       pybind11::arg("lsb") = 1);
