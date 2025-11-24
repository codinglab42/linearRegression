#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <vector>

namespace ml {
    
class LinearRegression {
private:
    std::vector<double> weights_;
    double bias_;
    
public:
    LinearRegression();
    
    // Metodi core
    void train(const std::vector<std::vector<double>>& X, 
               const std::vector<double>& y, 
               double learning_rate = 0.01, 
               int iterations = 1000);
    
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    
    // Getters
    const std::vector<double>& get_weights() const { return weights_; }
    double get_bias() const { return bias_; }
};

// Funzioni standalone
double mean_squared_error(const std::vector<double>& y_true, 
                                              const std::vector<double>& y_pred);

} // namespace ml

#endif