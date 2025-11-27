#ifndef LINEAR_REGRESSION_ONE_VAR_H
#define LINEAR_REGRESSION_ONE_VAR_H

#include <vector>
#include <string>

class LinearRegressionOneVar {
private:
    double theta0;  // Intercept
    double theta1;  // Slope
    double alpha;   // Learning rate
    int iterations; // Number of iterations
    
    std::vector<double> cost_history;
    std::vector<double> theta0_history;
    std::vector<double> theta1_history;

public:
    // Constructors
    LinearRegressionOneVar();
    LinearRegressionOneVar(double learning_rate, int iter);
    
    // Core methods - specifically for one variable
    void fit(const std::vector<double>& X, const std::vector<double>& y);
    std::vector<double> predict(const std::vector<double>& X) const;
    double predict_single(double x) const;
    
    // Cost function for one variable (as in Andrew Ng videos)
    double compute_cost(const std::vector<double>& X, const std::vector<double>& y) const;
    
    // Gradient Descent for one variable
    void gradient_descent(const std::vector<double>& X, const std::vector<double>& y);
    
    // Getters
    double get_theta0() const;
    double get_theta1() const;
    std::vector<double> get_cost_history() const;
    std::vector<double> get_theta0_history() const;
    std::vector<double> get_theta1_history() const;
    
    // Utility methods
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

private:
    // Hypothesis function for one variable: h(x) = theta0 + theta1 * x
    double hypothesis(double x) const;
};

#endif