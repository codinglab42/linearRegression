#include <iostream>
#include <vector>
#include "ml_algorithms/algorithms.h"

int main() {
    ml::LinearRegression model;
    
    // Dati di test
    std::vector<std::vector<double>> X = {{1}, {2}, {3}, {4}};
    std::vector<double> y = {2, 4, 6, 8};
    
    model.train(X, y, 0.01, 1000);
    
    auto predictions = model.predict(X);
    std::cout << "Test C++ completato!" << std::endl;
    std::cout << "Bias: " << model.get_bias() << std::endl;
    
    return 0;
}