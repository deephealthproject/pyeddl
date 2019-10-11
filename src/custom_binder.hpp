#pragma once

#include <pybind11/pybind11.h>
#include <eddl/metrics/metric.h>


class CustomMetric : public Metric {
public:
    pybind11::object pymetric;
    CustomMetric(pybind11::object pymetric, std::string name);
    float value(Tensor *T, Tensor *Y) override;
};

CustomMetric::CustomMetric(pybind11::object pymetric, std::string name) :
  Metric(name) {
    this->pymetric = pymetric;
}

float CustomMetric::value(Tensor *T, Tensor *Y) {
    pybind11::object pyvalue = pymetric(pybind11::cast(T), pybind11::cast(Y));
    return pyvalue.cast<float>();
}


template<typename Module>
void bind_custom_metric(Module &m) {
    pybind11::class_<CustomMetric, std::shared_ptr<CustomMetric>, Metric>
      cl(m, "CustomMetric");
    cl.def(pybind11::init<pybind11::object, std::string>());
    cl.def("value", &CustomMetric::value);
}
