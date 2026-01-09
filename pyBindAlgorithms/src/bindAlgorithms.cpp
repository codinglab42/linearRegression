#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "linear_regression_one_var.h"
#include "linear_regression_multi_var.h"
#include "logistic_regression.h"

namespace py = pybind11;

// Helper per convertire liste/numpy
std::vector<std::vector<double>> convert_to_2d_vector(py::object obj) {
    std::vector<std::vector<double>> result;
    
    if (py::isinstance<py::list>(obj)) {
        py::list py_list = obj.cast<py::list>();
        for (auto& item : py_list) {
            if (py::isinstance<py::list>(item)) {
                py::list inner_list = item.cast<py::list>();
                std::vector<double> vec;
                for (auto& val : inner_list) {
                    vec.push_back(val.cast<double>());
                }
                result.push_back(vec);
            }
        }
    }
    
    return result;
}

PYBIND11_MODULE(pymlalgorithms, m) {
    m.doc() = "Machine Learning algorithms from Andrew Ng course";
    
    // ========== REGRESSIONE LINEARE UNA VARIABILE ==========
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


    // ========== REGRESSIONE LINEARE MULTI-VARIABILE ==========
    py::class_<LinearRegressionMultiVar>(m, "LinearRegressionMultiVar")
        .def(py::init<>())
        .def(py::init<double, int>(), 
             py::arg("learning_rate") = 0.01, 
             py::arg("iterations") = 1000)
        
        // Training
        .def("fit", [](LinearRegressionMultiVar& model,
                       const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y) {
            model.fit(X, y);
        }, py::arg("X"), py::arg("y"), 
           "Train multi-variable linear regression")
        
        // Predizioni
        .def("predict", &LinearRegressionMultiVar::predict, 
             "Predict single sample")
        
        .def("predict_batch", &LinearRegressionMultiVar::predict_batch, 
             "Predict multiple samples")
        
        // Getter
        .def("get_theta", &LinearRegressionMultiVar::get_theta, 
             "Get all theta parameters")
        
        .def("get_theta0", &LinearRegressionMultiVar::get_theta0, 
             "Get intercept (theta0)")
        
        .def("get_theta_i", &LinearRegressionMultiVar::get_theta_i, 
             py::arg("index"), "Get theta at specific index")
        
        .def("get_cost_history", &LinearRegressionMultiVar::get_cost_history)
        .def("get_theta_history", &LinearRegressionMultiVar::get_theta_history)
        .def("get_num_features", &LinearRegressionMultiVar::get_num_features)
        .def("get_learning_rate", &LinearRegressionMultiVar::get_learning_rate)
        .def("get_iterations", &LinearRegressionMultiVar::get_iterations)
        
        // Metriche
        .def("r2_score", &LinearRegressionMultiVar::r2_score,
             py::arg("X"), py::arg("y"), "Calculate R² score")
        
        .def("mse", &LinearRegressionMultiVar::mse,
             py::arg("X"), py::arg("y"), "Calculate Mean Squared Error")
        
        // Salva/Carica
        .def("save_model", &LinearRegressionMultiVar::save_model)
        .def("load_model", &LinearRegressionMultiVar::load_model)
        
        // Utility
        .def("print_model", &LinearRegressionMultiVar::print_model)
        .def("get_formula", &LinearRegressionMultiVar::get_formula)
        
        // Properties (accesso stile Python)
        .def_property_readonly("theta", &LinearRegressionMultiVar::get_theta)
        .def_property_readonly("theta0", &LinearRegressionMultiVar::get_theta0)
        .def_property_readonly("cost_history", &LinearRegressionMultiVar::get_cost_history)
        .def_property_readonly("num_features", &LinearRegressionMultiVar::get_num_features)
        
        .def("__repr__", [](const LinearRegressionMultiVar &model) {
            return "LinearRegressionMultiVar(n_features=" + 
                   std::to_string(model.get_num_features()) + ")";
        });


     // ========== LOGISTIC REGRESSION ==========
    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<>())
        .def(py::init<double, int, double>(), 
             py::arg("learning_rate") = 0.01, 
             py::arg("iterations") = 1000,
             py::arg("lambda") = 0.0,
             "Initialize Logistic Regression with optional regularization")
        
        // Training
        .def("fit", [](LogisticRegression& model,
                       const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y) {
            model.fit(X, y);
        }, py::arg("X"), py::arg("y"), 
           "Train logistic regression model (binary classification)")
        
        // Predizioni
        .def("predict_probability", &LogisticRegression::predict_probability, 
             py::arg("x"), "Predict probability of class 1")
        
        .def("predict_class", &LogisticRegression::predict_class, 
             py::arg("x"), py::arg("threshold") = 0.5,
             "Predict class (0 or 1) with given threshold")
        
        .def("predict_probabilities", &LogisticRegression::predict_probabilities, 
             py::arg("X"), "Predict probabilities for multiple samples")
        
        .def("predict_classes", &LogisticRegression::predict_classes, 
             py::arg("X"), py::arg("threshold") = 0.5,
             "Predict classes for multiple samples")
        
        // Getter
        .def("get_theta", &LogisticRegression::get_theta, 
             "Get all theta parameters")
        
        .def("get_theta0", &LogisticRegression::get_theta0, 
             "Get intercept (theta0)")
        
        .def("get_theta_i", &LogisticRegression::get_theta_i, 
             py::arg("index"), "Get theta at specific index")
        
        .def("get_cost_history", &LogisticRegression::get_cost_history)
        .def("get_theta_history", &LogisticRegression::get_theta_history)
        .def("get_num_features", &LogisticRegression::get_num_features)
        .def("get_learning_rate", &LogisticRegression::get_learning_rate)
        .def("get_iterations", &LogisticRegression::get_iterations)
        .def("get_lambda", &LogisticRegression::get_lambda)
        
        // Metriche
        .def("accuracy", &LogisticRegression::accuracy,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Calculate accuracy score")
        
        .def("precision_recall_f1", &LogisticRegression::precision_recall_f1,
             py::arg("X"), py::arg("y"), py::arg("threshold") = 0.5,
             "Calculate precision, recall and F1-score")
        
        // Decision boundary (per 2D)
        .def("get_decision_boundary_2d", &LogisticRegression::get_decision_boundary_2d,
             py::arg("threshold") = 0.5,
             "Get decision boundary line parameters for 2D data [intercept, slope]")
        
        // Salva/Carica
        .def("save_model", &LogisticRegression::save_model)
        .def("load_model", &LogisticRegression::load_model)
        
        // Utility
        .def("print_model", &LogisticRegression::print_model)
        .def("get_formula", &LogisticRegression::get_formula)
        
        // Properties (accesso stile Python)
        .def_property_readonly("theta", &LogisticRegression::get_theta)
        .def_property_readonly("theta0", &LogisticRegression::get_theta0)
        .def_property_readonly("cost_history", &LogisticRegression::get_cost_history)
        .def_property_readonly("num_features", &LogisticRegression::get_num_features)
        
        .def("__repr__", [](const LogisticRegression &model) {
            return "LogisticRegression(n_features=" + 
                   std::to_string(model.get_num_features()) + 
                   ", lambda=" + std::to_string(model.get_lambda()) + ")";
        });

}