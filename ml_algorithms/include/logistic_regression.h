#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

class LogisticRegression {
private:
    // Parametri del modello
    std::vector<double> theta;  // [θ₀, θ₁, θ₂, ..., θₙ]
    
    // Parametri di training
    double learning_rate;
    int iterations;
    int n_features;
    
    // Storia del training
    std::vector<double> cost_history;
    std::vector<std::vector<double>> theta_history;
    
    // Normalizzazione
    std::vector<double> feature_means;
    std::vector<double> feature_stds;
    bool normalized;
    
    // Funzioni interne
    double hypothesis(const std::vector<double>& x, 
                                const std::vector<double>& theta) const;

    double compute_cost(const std::vector<std::vector<double>>& X, 
                       const std::vector<double>& y) const;
    
    std::vector<double> compute_gradients(const std::vector<std::vector<double>>& X,
                                         const std::vector<double>& y) const;
    
    void normalize_features(std::vector<std::vector<double>>& X);
    std::vector<double> normalize_vector(const std::vector<double>& x) const;
    
    // Regularizzazione
    double lambda;  // Parametro di regolarizzazione
    
public:
    // Costruttori
    LogisticRegression();
    LogisticRegression(double alpha, int iters, double reg_lambda = 0.0);

    // Funzioni matematiche
    static double sigmoid(double z);
    
    // Training
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<double>& y);
    
    // Predizioni
    double predict_probability(const std::vector<double>& x) const;
    int predict_class(const std::vector<double>& x, double threshold = 0.5) const;
    std::vector<double> predict_probabilities(const std::vector<std::vector<double>>& X) const;
    std::vector<int> predict_classes(const std::vector<std::vector<double>>& X, 
                                    double threshold = 0.5) const;
    
    // Getter
    const std::vector<double>& get_theta() const { return theta; }
    double get_theta0() const { return theta.empty() ? 0.0 : theta[0]; }
    double get_theta_i(int i) const;
    
    const std::vector<double>& get_cost_history() const { return cost_history; }
    const std::vector<std::vector<double>>& get_theta_history() const { return theta_history; }
    int get_num_features() const { return n_features; }
    double get_learning_rate() const { return learning_rate; }
    int get_iterations() const { return iterations; }
    double get_lambda() const { return lambda; }
    
    // Metriche di valutazione
    double accuracy(const std::vector<std::vector<double>>& X, 
                   const std::vector<double>& y, 
                   double threshold = 0.5) const;
    
    std::vector<double> precision_recall_f1(const std::vector<std::vector<double>>& X,
                                          const std::vector<double>& y,
                                          double threshold = 0.5) const;
    
    // Salva/Carica modello
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Utility
    void print_model() const;
    std::string get_formula() const;
    
    // Decision boundary
    std::vector<double> get_decision_boundary_2d(double threshold = 0.5) const;
};

#endif // LOGISTIC_REGRESSION_H
