#define ldense_addons(cl) \
cl.def(pybind11::init<Layer*, int, bool, string, int>(), pybind11::keep_alive<1, 2>());
