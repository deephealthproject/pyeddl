#define layer_addons(cl) \
  cl.def_readwrite("input", &Layer::input); \
  cl.def_readwrite("output", &Layer::output); \
  cl.def_readwrite("target", &Layer::target); \
  cl.def_readwrite("delta", &Layer::delta);
