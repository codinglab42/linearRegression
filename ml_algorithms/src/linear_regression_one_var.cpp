#include "linear_regression_one_var.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>

LinearRegressionOneVar::LinearRegressionOneVar() 
    : theta0(0.0), theta1(0.0), alpha(0.01), iterations(1000) {}

LinearRegressionOneVar::LinearRegressionOneVar(double learning_rate, int iter) 
    : theta0(0.0), theta1(0.0), alpha(learning_rate), iterations(iter) {}

double LinearRegressionOneVar::hypothesis(double x) const {
    return theta0 + theta1 * x;
}

double LinearRegressionOneVar::compute_cost(const std::vector<double>& X, 
                                          const std::vector<double>& y) const {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y must have same number of samples");
    }
    
    int m = X.size();
    double total_cost = 0.0;
    
    for (int i = 0; i < m; ++i) {
        double prediction = hypothesis(X[i]);
        double error = prediction - y[i];
        total_cost += error * error;
    }
    
    return total_cost / (2 * m);
}

void LinearRegressionOneVar::gradient_descent(const std::vector<double>& X, 
                                            const std::vector<double>& y) {
    int m = X.size();
    
    // Clear history
    cost_history.clear();
    theta0_history.clear();
    theta1_history.clear();
    
    for (int iter = 0; iter < iterations; ++iter) {
        double sum_error_theta0 = 0.0;
        double sum_error_theta1 = 0.0;
        
        // Compute gradients (partial derivatives)
        for (int i = 0; i < m; ++i) {
            double prediction = hypothesis(X[i]);
            double error = prediction - y[i];
            
            sum_error_theta0 += error;
            sum_error_theta1 += error * X[i];
        }
        
        // Update parameters simultaneously
        double temp_theta0 = theta0 - (alpha / m) * sum_error_theta0;
        double temp_theta1 = theta1 - (alpha / m) * sum_error_theta1;
        
        theta0 = temp_theta0;
        theta1 = temp_theta1;
        
        // Record history
        double cost = compute_cost(X, y);
        cost_history.push_back(cost);
        theta0_history.push_back(theta0);
        theta1_history.push_back(theta1);
        
        // Print progress every 100 iterations
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter 
                      << " | Cost: " << cost 
                      << " | theta0: " << theta0 
                      << " | theta1: " << theta1 << std::endl;
        }
    }
}

void LinearRegressionOneVar::fit(const std::vector<double>& X, 
                               const std::vector<double>& y) {
    if (X.empty() || y.empty()) {
        throw std::invalid_argument("X and y cannot be empty");
    }
    
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y must have same number of samples");
    }
    
    // Initialize parameters
    theta0 = 0.0;
    theta1 = 0.0;
    
    std::cout << "Starting Gradient Descent..." << std::endl;
    std::cout << "Initial cost: " << compute_cost(X, y) << std::endl;
    std::cout << "Initial parameters: theta0 = " << theta0 << ", theta1 = " << theta1 << std::endl;
    
    // Run gradient descent
    gradient_descent(X, y);
    
    std::cout << "Final cost: " << cost_history.back() << std::endl;
    std::cout << "Final parameters: theta0 = " << theta0 << ", theta1 = " << theta1 << std::endl;
}

std::vector<double> LinearRegressionOneVar::predict(const std::vector<double>& X) const {
    if (cost_history.empty()) {
        throw std::runtime_error("Model not trained. Call fit() first.");
    }
    
    std::vector<double> predictions;
    predictions.reserve(X.size());
    
    for (double x : X) {
        predictions.push_back(hypothesis(x));
    }
    
    return predictions;
}

double LinearRegressionOneVar::predict_single(double x) const {
    if (cost_history.empty()) {
        throw std::runtime_error("Model not trained. Call fit() first.");
    }
    
    return hypothesis(x);
}

// Getters
double LinearRegressionOneVar::get_theta0() const {
    return theta0;
}

double LinearRegressionOneVar::get_theta1() const {
    return theta1;
}

std::vector<double> LinearRegressionOneVar::get_cost_history() const {
    return cost_history;
}

std::vector<double> LinearRegressionOneVar::get_theta0_history() const {
    return theta0_history;
}

std::vector<double> LinearRegressionOneVar::get_theta1_history() const {
    return theta1_history;
}

void LinearRegressionOneVar::save_model(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving model");
    }
    
    file << theta0 << " " << theta1 << std::endl;
    file.close();
}

void LinearRegressionOneVar::load_model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for loading model");
    }
    
    file >> theta0 >> theta1;
    file.close();
}