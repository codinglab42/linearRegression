#ifndef LINEAR_REGRESSION_MULTI_VAR_H
#define LINEAR_REGRESSION_MULTI_VAR_H

#include <vector>
#include <string>
#include <stdexcept>

class LinearRegressionMultiVar {
private:
    // Parametri del modello: θ₀ (intercept) + θ₁...θₙ per n features
    std::vector<double> theta;
    
    // Parametri di training
    double learning_rate;
    int iterations;
    int n_features;
    
    // Storia del training
    std::vector<double> cost_history;
    std::vector<std::vector<double>> theta_history;
    
    // Funzioni interne
    double compute_cost(const std::vector<std::vector<double>>& X, 
                       const std::vector<double>& y) const;
    
    std::vector<double> compute_gradients(const std::vector<std::vector<double>>& X,
                                         const std::vector<double>& y) const;
    
    // Normalizzazione
    std::vector<double> feature_means;
    std::vector<double> feature_stds;
    bool normalized;
    
    void normalize_features(std::vector<std::vector<double>>& X);
    std::vector<double> normalize_vector(const std::vector<double>& x) const;
    
public:
    // Costruttori
    LinearRegressionMultiVar();
    LinearRegressionMultiVar(double alpha, int iters);
    
    // Training
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<double>& y);
    
    // Predizioni
    double predict(const std::vector<double>& x) const;
    std::vector<double> predict_batch(const std::vector<std::vector<double>>& X) const;
    
    // Getter
    const std::vector<double>& get_theta() const { return theta; }
    double get_theta0() const { return theta.empty() ? 0.0 : theta[0]; }
    double get_theta_i(int i) const { 
        if (i < 0 || i >= theta.size()) throw std::out_of_range("Index out of range");
        return theta[i]; 
    }
    
    const std::vector<double>& get_cost_history() const { return cost_history; }
    const std::vector<std::vector<double>>& get_theta_history() const { return theta_history; }
    int get_num_features() const { return n_features; }
    double get_learning_rate() const { return learning_rate; }
    int get_iterations() const { return iterations; }
    
    // Salva/Carica modello
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Utility
    void print_model() const;
    std::string get_formula() const;
    
    // Normalizzazione
    const std::vector<double>& get_feature_means() const { return feature_means; }
    const std::vector<double>& get_feature_stds() const { return feature_stds; }
    bool is_normalized() const { return normalized; }
    
    // Valutazione
    double r2_score(const std::vector<std::vector<double>>& X, 
                    const std::vector<double>& y) const;
    double mse(const std::vector<std::vector<double>>& X, 
               const std::vector<double>& y) const;
};

#endif // LINEAR_REGRESSION_MULTI_VAR_H