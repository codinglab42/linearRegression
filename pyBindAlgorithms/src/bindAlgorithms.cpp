#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "linear_regression_one_var.h"

namespace py = pybind11;

PYBIND11_MODULE(pymlalgorithms, m) {
    m.doc() = "Machine Learning algorithms implemented in C++ with pybind11";
    
    // Linear Regression One Variable
    py::class_<LinearRegressionOneVar>(m, "LinearRegressionOneVar")
        .def(py::init<>())
        .def(py::init<double, int>(), 
             py::arg("learning_rate") = 0.01, 
             py::arg("iterations") = 1000)
        
        .def("fit", &LinearRegressionOneVar::fit, "Train the linear regression model with one variable")
        .def("predict", &LinearRegressionOneVar::predict, "Make predictions")
        .def("predict_single", &LinearRegressionOneVar::predict_single, "Predict single value")
        
        .def("compute_cost", &LinearRegressionOneVar::compute_cost, "Compute cost function J(theta0, theta1)")
        
        .def("get_theta0", &LinearRegressionOneVar::get_theta0, "Get theta0 parameter")
        .def("get_theta1", &LinearRegressionOneVar::get_theta1, "Get theta1 parameter")
        .def("get_cost_history", &LinearRegressionOneVar::get_cost_history, "Get cost history during training")
        .def("get_theta0_history", &LinearRegressionOneVar::get_theta0_history, "Get theta0 history during training")
        .def("get_theta1_history", &LinearRegressionOneVar::get_theta1_history, "Get theta1 history during training")
        
        .def("save_model", &LinearRegressionOneVar::save_model, "Save model to file")
        .def("load_model", &LinearRegressionOneVar::load_model, "Load model from file")
        
        .def_property_readonly("theta0", &LinearRegressionOneVar::get_theta0)
        .def_property_readonly("theta1", &LinearRegressionOneVar::get_theta1)
        .def_property_readonly("cost_history", &LinearRegressionOneVar::get_cost_history)
        .def_property_readonly("theta0_history", &LinearRegressionOneVar::get_theta0_history)
        .def_property_readonly("theta1_history", &LinearRegressionOneVar::get_theta1_history);
}