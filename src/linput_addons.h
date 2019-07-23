#define linput_addons(cl) \
cl.def(pybind11::init<Tensor*, string, int>(), pybind11::keep_alive<1, 2>());
