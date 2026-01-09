#include "logistic_regression.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>

using namespace std;

// Funzione sigmoide: g(z) = 1 / (1 + e^{-z})
double LogisticRegression::sigmoid(double z) {
    // Per evitare overflow
    if (z >= 0) {
        return 1.0 / (1.0 + exp(-z));
    } else {
        double exp_z = exp(z);
        return exp_z / (1.0 + exp_z);
    }
}

// Costruttori
LogisticRegression::LogisticRegression() 
    : learning_rate(0.01), iterations(1000), n_features(0), 
      normalized(false), lambda(0.0) {
    theta = {0.0}; // Solo intercept inizialmente
}

LogisticRegression::LogisticRegression(double alpha, int iters, double reg_lambda) 
    : learning_rate(alpha), iterations(iters), n_features(0), 
      normalized(false), lambda(reg_lambda) {
    theta = {0.0};
}

// Normalizzazione feature
void LogisticRegression::normalize_features(vector<vector<double>>& X) {
    if (X.empty()) return;
    
    n_features = X[0].size();
    feature_means.resize(n_features);
    feature_stds.resize(n_features);
    
    // Calcola media e deviazione standard per ogni feature
    for (int j = 0; j < n_features; ++j) {
        double sum = 0.0;
        double sq_sum = 0.0;
        
        for (size_t i = 0; i < X.size(); ++i) {
            sum += X[i][j];
            sq_sum += X[i][j] * X[i][j];
        }
        
        feature_means[j] = sum / X.size();
        double variance = (sq_sum / X.size()) - (feature_means[j] * feature_means[j]);
        feature_stds[j] = sqrt(variance + 1e-8); // Evita divisione per zero
        
        // Normalizza
        for (size_t i = 0; i < X.size(); ++i) {
            X[i][j] = (X[i][j] - feature_means[j]) / feature_stds[j];
        }
    }
    
    normalized = true;
}

// Normalizza un singolo vettore
vector<double> LogisticRegression::normalize_vector(const vector<double>& x) const {
    if (!normalized || x.size() != n_features) {
        return x;
    }
    
    vector<double> x_norm = x;
    for (size_t j = 0; j < x.size(); ++j) {
        x_norm[j] = (x[j] - feature_means[j]) / feature_stds[j];
    }
    return x_norm;
}

// Funzione ipotesi: h_θ(x) = g(θᵀx) = 1 / (1 + e^{-θᵀx})
double LogisticRegression::hypothesis(const std::vector<double> &x, const std::vector<double> &theta) const
{
    if (x.size() + 1 != theta.size()) {
        throw invalid_argument("Dimension mismatch between x and theta");
    }
    
    double z = theta[0]; // θ₀
    for (size_t i = 0; i < x.size(); ++i) {
        z += theta[i + 1] * x[i];
    }
    
    return LogisticRegression::sigmoid(z);
}

// Funzione costo Logistic Regression con regolarizzazione
// J(θ) = -1/m * Σ [yⁱ log(h_θ(xⁱ)) + (1 - yⁱ) log(1 - h_θ(xⁱ))] + λ/(2m) * Σ θⱼ² (j=1..n)
double LogisticRegression::compute_cost(const vector<vector<double>>& X, 
                                       const vector<double>& y) const {
    if (X.empty() || X.size() != y.size()) return 0.0;
    
    double cost = 0.0;
    int m = X.size();
    
    for (int i = 0; i < m; ++i) {
        double h = hypothesis(X[i], theta);
        
        // Evita log(0)
        double epsilon = 1e-15;
        h = max(epsilon, min(1.0 - epsilon, h));
        
        cost += y[i] * log(h) + (1 - y[i]) * log(1 - h);
    }
    
    cost = -cost / m;
    
    // Aggiungi regolarizzazione L2 (esclude θ₀)
    if (lambda > 0) {
        double reg_sum = 0.0;
        for (size_t j = 1; j < theta.size(); ++j) {
            reg_sum += theta[j] * theta[j];
        }
        cost += (lambda / (2.0 * m)) * reg_sum;
    }
    
    return cost;
}

// Calcola gradienti
vector<double> LogisticRegression::compute_gradients(const vector<vector<double>>& X,
                                                    const vector<double>& y) const {
    int m = X.size();
    int n = theta.size(); // n features + 1 (intercept)
    
    vector<double> gradients(n, 0.0);
    
    for (int i = 0; i < m; ++i) {
        double h = hypothesis(X[i], theta);
        double error = h - y[i];
        
        // Per θ₀ (senza regolarizzazione)
        gradients[0] += error;
        
        // Per θ₁...θₙ
        for (int j = 1; j < n; ++j) {
            gradients[j] += error * X[i][j - 1];
        }
    }
    
    // Divide per m e aggiungi regolarizzazione per j >= 1
    for (int j = 0; j < n; ++j) {
        gradients[j] /= m;
        if (j >= 1 && lambda > 0) {
            gradients[j] += (lambda / m) * theta[j];
        }
    }
    
    return gradients;
}

// Training con Gradient Descent
void LogisticRegression::fit(const vector<vector<double>>& X, 
                            const vector<double>& y) {
    if (X.empty() || X.size() != y.size()) {
        throw invalid_argument("X and y must have same size and non-empty");
    }
    
    // Verifica che y contenga solo 0 o 1
    for (double label : y) {
        if (label != 0.0 && label != 1.0) {
            throw invalid_argument("y must contain only 0 or 1 for binary classification");
        }
    }
    
    n_features = X[0].size();
    
    // Copia X per normalizzazione
    vector<vector<double>> X_copy = X;
    
    // Reset stato
    theta.clear();
    cost_history.clear();
    theta_history.clear();
    normalized = false;
    
    // Normalizza features
    normalize_features(X_copy);
    
    // Inizializza theta: [θ₀, θ₁, ..., θₙ]
    theta.resize(n_features + 1, 0.0);
    theta[0] = 0.0; // intercept
    
    // Inizializzazione random piccola
    srand(42);
    for (int i = 1; i <= n_features; ++i) {
        theta[i] = (rand() % 2000) / 10000.0 - 0.1; // [-0.1, 0.1]
    }
    
    // Gradient Descent
    for (int iter = 0; iter < iterations; ++iter) {
        // Calcola gradienti
        vector<double> gradients = compute_gradients(X_copy, y);
        
        // Aggiorna parametri
        for (size_t j = 0; j < theta.size(); ++j) {
            theta[j] -= learning_rate * gradients[j];
        }
        
        // Salva storia (ogni 100 iterazioni)
        if (iter % 100 == 0) {
            double cost = compute_cost(X_copy, y);
            cost_history.push_back(cost);
            theta_history.push_back(theta);
            
            // Debug output
            if (iter % 1000 == 0) {
                cout << "Iteration " << iter << ": Cost = " << cost 
                     << ", Theta[0] = " << theta[0] << endl;
            }
        }
    }
    
    // Denormalizza theta se abbiamo normalizzato
    if (normalized) {
        double new_theta0 = theta[0];
        for (int i = 1; i <= n_features; ++i) {
            if (feature_stds[i-1] > 1e-10) {
                new_theta0 -= theta[i] * feature_means[i-1] / feature_stds[i-1];
                theta[i] = theta[i] / feature_stds[i-1];
            }
        }
        theta[0] = new_theta0;
        
        normalized = false;
        feature_means.clear();
        feature_stds.clear();
    }
}

// Predice probabilità (0-1)
double LogisticRegression::predict_probability(const vector<double>& x) const {
    if (x.size() != n_features) {
        throw invalid_argument("Input x must have " + to_string(n_features) + " features");
    }
    
    return hypothesis(x, theta);
}

// Predice classe (0 o 1)
int LogisticRegression::predict_class(const vector<double>& x, double threshold) const {
    double prob = predict_probability(x);
    return (prob >= threshold) ? 1 : 0;
}

// Predizioni batch
vector<double> LogisticRegression::predict_probabilities(const vector<vector<double>>& X) const {
    vector<double> probs(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        probs[i] = predict_probability(X[i]);
    }
    return probs;
}

vector<int> LogisticRegression::predict_classes(const vector<vector<double>>& X, 
                                               double threshold) const {
    vector<int> classes(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        classes[i] = predict_class(X[i], threshold);
    }
    return classes;
}

// Getter per theta specifico
double LogisticRegression::get_theta_i(int i) const { 
    if (i < 0 || i >= (int)theta.size()) {
        throw out_of_range("Index out of range");
    }
    return theta[i]; 
}

// Accuracy
double LogisticRegression::accuracy(const vector<vector<double>>& X, 
                                  const vector<double>& y, 
                                  double threshold) const {
    if (X.empty() || X.size() != y.size()) return 0.0;
    
    int correct = 0;
    vector<int> predictions = predict_classes(X, threshold);
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == (int)y[i]) {
            ++correct;
        }
    }
    
    return static_cast<double>(correct) / X.size();
}

// Precision, Recall, F1-score
vector<double> LogisticRegression::precision_recall_f1(const vector<vector<double>>& X,
                                                     const vector<double>& y,
                                                     double threshold) const {
    if (X.empty() || X.size() != y.size()) {
        return {0.0, 0.0, 0.0};
    }
    
    vector<int> predictions = predict_classes(X, threshold);
    
    int true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        int pred = predictions[i];
        int actual = (int)y[i];
        
        if (pred == 1 && actual == 1) true_positive++;
        else if (pred == 1 && actual == 0) false_positive++;
        else if (pred == 0 && actual == 1) false_negative++;
        else if (pred == 0 && actual == 0) true_negative++;
    }
    
    double precision = 0.0, recall = 0.0, f1 = 0.0;
    
    if (true_positive + false_positive > 0) {
        precision = static_cast<double>(true_positive) / (true_positive + false_positive);
    }
    
    if (true_positive + false_negative > 0) {
        recall = static_cast<double>(true_positive) / (true_positive + false_negative);
    }
    
    if (precision + recall > 0) {
        f1 = 2.0 * precision * recall / (precision + recall);
    }
    
    return {precision, recall, f1};
}

// Decision boundary per 2 features: θ₀ + θ₁x₁ + θ₂x₂ = 0
vector<double> LogisticRegression::get_decision_boundary_2d(double threshold) const {
    if (n_features != 2) {
        throw runtime_error("Decision boundary only available for 2 features");
    }
    
    // Per threshold diverso da 0.5: θ₀ + θ₁x₁ + θ₂x₂ = log(threshold/(1-threshold))
    double log_odds = log(threshold / (1.0 - threshold));
    
    // x₂ = (-θ₀ - θ₁x₁ + log_odds) / θ₂
    // Restituisce [intercept, slope] per linea: x₂ = intercept + slope * x₁
    if (abs(theta[2]) < 1e-10) {
        throw runtime_error("theta[2] is zero, cannot compute decision boundary");
    }
    
    double intercept = (-theta[0] + log_odds) / theta[2];
    double slope = -theta[1] / theta[2];
    
    return {intercept, slope};
}

// Salva modello
void LogisticRegression::save_model(const string& filename) const {
    ofstream file(filename);
    if (file.is_open()) {
        file << "LogisticRegression v1.0" << endl;
        file << n_features << endl;
        file << learning_rate << endl;
        file << iterations << endl;
        file << lambda << endl;
        file << normalized << endl;
        
        file << theta.size() << endl;
        for (double t : theta) {
            file << t << endl;
        }
        
        if (normalized) {
            file << feature_means.size() << endl;
            for (double mean : feature_means) {
                file << mean << endl;
            }
            file << feature_stds.size() << endl;
            for (double std : feature_stds) {
                file << std << endl;
            }
        }
        
        file.close();
    }
}

// Carica modello
void LogisticRegression::load_model(const string& filename) {
    ifstream file(filename);
    if (file.is_open()) {
        string header;
        getline(file, header);
        
        file >> n_features;
        file >> learning_rate;
        file >> iterations;
        file >> lambda;
        file >> normalized;
        
        size_t theta_size;
        file >> theta_size;
        theta.resize(theta_size);
        for (size_t i = 0; i < theta_size; ++i) {
            file >> theta[i];
        }
        
        if (normalized) {
            size_t means_size;
            file >> means_size;
            feature_means.resize(means_size);
            for (size_t i = 0; i < means_size; ++i) {
                file >> feature_means[i];
            }
            
            size_t stds_size;
            file >> stds_size;
            feature_stds.resize(stds_size);
            for (size_t i = 0; i < stds_size; ++i) {
                file >> feature_stds[i];
            }
        }
        
        file.close();
    }
}

// Utility
void LogisticRegression::print_model() const {
    cout << "Logistic Regression Model" << endl;
    cout << "Number of features: " << n_features << endl;
    cout << "Formula: " << get_formula() << endl;
    cout << "Learning rate: " << learning_rate << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Regularization lambda: " << lambda << endl;
    cout << "Final cost: " << (cost_history.empty() ? 0.0 : cost_history.back()) << endl;
}

string LogisticRegression::get_formula() const {
    if (theta.empty()) return "Not trained";
    
    stringstream ss;
    ss << fixed << setprecision(4);
    ss << "P(y=1|x) = 1 / (1 + exp(-(" << theta[0];
    
    for (size_t i = 1; i < theta.size(); ++i) {
        if (theta[i] >= 0) {
            ss << " + " << theta[i] << "*x" << i;
        } else {
            ss << " - " << abs(theta[i]) << "*x" << i;
        }
    }
    
    ss << ")))";
    return ss.str();
}