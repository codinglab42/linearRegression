#include "linear_regression_multi_var.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>

using namespace std;

// Costruttori
LinearRegressionMultiVar::LinearRegressionMultiVar() 
    : learning_rate(0.01), iterations(1000), n_features(0), normalized(false) {
    theta = {0.0}; // Solo intercept inizialmente
}

LinearRegressionMultiVar::LinearRegressionMultiVar(double alpha, int iters) 
    : learning_rate(alpha), iterations(iters), n_features(0), normalized(false) {
    theta = {0.0};
}

// Funzione ipotesi: h_θ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
double hypothesis(const vector<double>& x, const vector<double>& theta) {
    if (x.size() + 1 != theta.size()) {
        throw invalid_argument("Dimension mismatch between x and theta");
    }
    
    double result = theta[0]; // θ₀
    for (size_t i = 0; i < x.size(); ++i) {
        result += theta[i + 1] * x[i];
    }
    return result;
}

// Normalizzazione feature (feature scaling)
void LinearRegressionMultiVar::normalize_features(vector<vector<double>>& X) {
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
        feature_stds[j] = sqrt(variance);
        
        // Normalizza (evita divisione per zero)
        if (feature_stds[j] > 1e-10) {
            for (size_t i = 0; i < X.size(); ++i) {
                X[i][j] = (X[i][j] - feature_means[j]) / feature_stds[j];
            }
        }
    }
    
    normalized = true;
}

// Normalizza un singolo vettore
vector<double> LinearRegressionMultiVar::normalize_vector(const vector<double>& x) const {
    if (!normalized || x.size() != n_features) {
        return x; // Ritorna originale se non normalizzato o dimensioni sbagliate
    }
    
    vector<double> x_norm = x;
    for (size_t j = 0; j < x.size(); ++j) {
        if (feature_stds[j] > 1e-10) {
            x_norm[j] = (x[j] - feature_means[j]) / feature_stds[j];
        }
    }
    return x_norm;
}

// Funzione costo: J(θ) = 1/(2m) * Σ(h_θ(xⁱ) - yⁱ)²
double LinearRegressionMultiVar::compute_cost(const vector<vector<double>>& X, 
                                             const vector<double>& y) const {
    if (X.empty() || X.size() != y.size()) return 0.0;
    
    double sum = 0.0;
    int m = X.size();
    
    for (int i = 0; i < m; ++i) {
        double prediction = hypothesis(X[i], theta);
        double error = prediction - y[i];
        sum += error * error;
    }
    
    return sum / (2.0 * m);
}

// Calcola gradienti: ∂J/∂θⱼ = 1/m * Σ(h_θ(xⁱ) - yⁱ) * xⱼⁱ
vector<double> LinearRegressionMultiVar::compute_gradients(const vector<vector<double>>& X,
                                                          const vector<double>& y) const {
    int m = X.size();
    int n = theta.size(); // n features + 1 (intercept)
    
    vector<double> gradients(n, 0.0);
    
    for (int i = 0; i < m; ++i) {
        double prediction = hypothesis(X[i], theta);
        double error = prediction - y[i];
        
        // Per θ₀ (intercept)
        gradients[0] += error;
        
        // Per θ₁...θₙ
        for (int j = 1; j < n; ++j) {
            gradients[j] += error * X[i][j - 1];
        }
    }
    
    // Divide per m
    for (double& grad : gradients) {
        grad /= m;
    }
    
    return gradients;
}

// Training con Gradient Descent
void LinearRegressionMultiVar::fit(const vector<vector<double>>& X, 
                                  const vector<double>& y) {
    if (X.empty() || X.size() != y.size()) {
        throw invalid_argument("X and y must have same size and non-empty");
    }
    
    n_features = X[0].size();
    
    // Copia X per normalizzazione (se necessaria)
    vector<vector<double>> X_copy = X;
    
    // Reset stato
    theta.clear();
    cost_history.clear();
    theta_history.clear();
    normalized = false;
    
    // Normalizza features (opzionale ma aiuta convergenza)
    normalize_features(X_copy);
    
    // Inizializza theta: [θ₀, θ₁, ..., θₙ]
    // θ₀ = 0, altri piccoli valori random
    theta.resize(n_features + 1, 0.0);
    theta[0] = 0.0; // intercept
    
    // Inizializzazione random piccola per altri theta
    srand(42);
    for (int i = 1; i <= n_features; ++i) {
        theta[i] = (rand() % 1000) / 10000.0 - 0.05; // [-0.05, 0.05]
    }
    
    // Gradient Descent
    for (int iter = 0; iter < iterations; ++iter) {
        // Calcola gradienti
        vector<double> gradients = compute_gradients(X_copy, y);
        
        // Aggiorna parametri
        for (size_t j = 0; j < theta.size(); ++j) {
            theta[j] -= learning_rate * gradients[j];
        }
        
        // Salva storia (ogni 100 iterazioni per risparmiare memoria)
        if (iter % 100 == 0) {
            double cost = compute_cost(X_copy, y);
            cost_history.push_back(cost);
            theta_history.push_back(theta);
            
            // Debug output
            if (iter % 1000 == 0) {
                cout << "Iteration " << iter << ": Cost = " << cost 
                     << ", Theta = [";
                for (size_t i = 0; i < min(theta.size(), size_t(3)); ++i) {
                    cout << theta[i];
                    if (i < min(theta.size(), size_t(3)) - 1) cout << ", ";
                }
                if (theta.size() > 3) cout << ", ...";
                cout << "]" << endl;
            }
        }
    }
    
    // Denormalizza theta se abbiamo normalizzato
    if (normalized) {
        // Per la normalizzazione: x_norm = (x - μ)/σ
        // Modello originale: y = θ₀ + Σ θᵢ * ((xᵢ - μᵢ)/σᵢ)
        //                   = (θ₀ - Σ θᵢ*μᵢ/σᵢ) + Σ (θᵢ/σᵢ) * xᵢ
        
        double new_theta0 = theta[0];
        for (int i = 1; i <= n_features; ++i) {
            if (feature_stds[i-1] > 1e-10) {
                new_theta0 -= theta[i] * feature_means[i-1] / feature_stds[i-1];
                theta[i] = theta[i] / feature_stds[i-1];
            }
        }
        theta[0] = new_theta0;
        
        // Ora theta è per i dati originali (non normalizzati)
        normalized = false;
        feature_means.clear();
        feature_stds.clear();
    }
}

// Predizione singola
double LinearRegressionMultiVar::predict(const vector<double>& x) const {
    if (x.size() != n_features) {
        throw invalid_argument("Input x must have " + to_string(n_features) + " features");
    }
    
    return hypothesis(x, theta);
}

// Predizione batch
vector<double> LinearRegressionMultiVar::predict_batch(const vector<vector<double>>& X) const {
    vector<double> predictions(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        predictions[i] = predict(X[i]);
    }
    return predictions;
}

// R² score (coefficiente di determinazione)
double LinearRegressionMultiVar::r2_score(const vector<vector<double>>& X, 
                                         const vector<double>& y) const {
    if (X.empty() || X.size() != y.size()) return 0.0;
    
    double ss_res = 0.0;  // Somma residui quadrati
    double ss_tot = 0.0;  // Somma totale quadrati
    double y_mean = accumulate(y.begin(), y.end(), 0.0) / y.size();
    
    for (size_t i = 0; i < X.size(); ++i) {
        double prediction = predict(X[i]);
        ss_res += (prediction - y[i]) * (prediction - y[i]);
        ss_tot += (y[i] - y_mean) * (y[i] - y_mean);
    }
    
    return 1.0 - (ss_res / ss_tot);
}

// Mean Squared Error
double LinearRegressionMultiVar::mse(const vector<vector<double>>& X, 
                                    const vector<double>& y) const {
    if (X.empty() || X.size() != y.size()) return 0.0;
    
    double sum = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        double prediction = predict(X[i]);
        double error = prediction - y[i];
        sum += error * error;
    }
    
    return sum / X.size();
}

// Salva modello su file
void LinearRegressionMultiVar::save_model(const string& filename) const {
    ofstream file(filename);
    if (file.is_open()) {
        // Salva metadati
        file << "LinearRegressionMultiVar v1.0" << endl;
        file << n_features << endl;
        file << learning_rate << endl;
        file << iterations << endl;
        file << normalized << endl;
        
        // Salta theta
        file << theta.size() << endl;
        for (double t : theta) {
            file << t << endl;
        }
        
        // Salva parametri normalizzazione se presenti
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

// Carica modello da file
void LinearRegressionMultiVar::load_model(const string& filename) {
    ifstream file(filename);
    if (file.is_open()) {
        string header;
        getline(file, header);
        
        file >> n_features;
        file >> learning_rate;
        file >> iterations;
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

// Utility: stampa modello
void LinearRegressionMultiVar::print_model() const {
    cout << "Linear Regression Model (Multi-Variable)" << endl;
    cout << "Number of features: " << n_features << endl;
    cout << "Formula: " << get_formula() << endl;
    cout << "Learning rate: " << learning_rate << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Final cost: " << (cost_history.empty() ? 0.0 : cost_history.back()) << endl;
}

// Genera formula stringa
string LinearRegressionMultiVar::get_formula() const {
    if (theta.empty()) return "Not trained";
    
    stringstream ss;
    ss << fixed << setprecision(4);
    ss << "y = " << theta[0];
    
    for (size_t i = 1; i < theta.size(); ++i) {
        if (theta[i] >= 0) {
            ss << " + " << theta[i] << "*x" << i;
        } else {
            ss << " - " << abs(theta[i]) << "*x" << i;
        }
    }
    
    return ss.str();
}