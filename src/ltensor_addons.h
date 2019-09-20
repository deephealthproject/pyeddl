#define ltensor_addons(cl) \
cl.def(pybind11::init<string>()); \
cl.def(pybind11::init<vector<int>, int>());
