

#ifndef BatchNorm_hpp
#define BatchNorm_hpp
#include <stdio.h>
#include <cmath>
#include "Matrix_Operations.hpp"
#include "Layer.hpp"
using namespace std;
class BatchNorm : public Layer{
private:
    vector<vector<float>> input, normalized_input, output;
   
    vector<float> batch_means, batch_vars;
    
    vector<vector<float>> d_input;
    
    vector<vector<float>> dLRespectdNormX;
    
    const double epsilon = 1e-9;
    
    vector<float> d_gamma, m_gamma, v_gamma;
    vector<float> d_beta, m_beta, v_beta;
    
    
    size_t num_features;
    
    size_t batch_size;
    
public:
    
    vector<float> gamma;
    vector<float> beta;
    
    BatchNorm(int features){
        this->num_features = features;
        
        gamma.resize(features, 1.0f);
        
        d_gamma.resize(features, 0.0f);
        beta.resize(features, 0.0f);
        
        d_beta.resize(features, 0.0f);
        m_gamma.resize(features, 0.0f);
        
        v_gamma.resize(features, 0.0f);
        m_beta.resize(features, 0.0f);
        v_beta.resize(features, 0.0f);
        
    }
    
    void calculate_batch_mean();
    
    void calculate_batch_var();
    
    void calculate_norm_input();
    
    void calculate_output();
    
    void calculate_mean_correction(vector<float> & mean_correction);
    
    void calculate_var_correction(vector<float> & var_correction);
    
    vector<vector<float>> forward(const std::vector<std::vector<float>> & output_from_prev_layer) override;
    
    
    void calculate_batch_d_gamma(const std::vector<std::vector<float>> & gradient_from_next_layer);
    
    void calculate_batch_d_beta(const std::vector<std::vector<float>> & gradient_from_next_layer);
    
    void calculate_d_input(const std::vector<std::vector<float>> & gradient_from_next_layer);
    
    
    vector<vector<float>> backward(const std::vector<std::vector<float>> & gradient_from_next_layer) override;
    
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    void save_to_file(ofstream& file) override;
    
    void load_layer(std::ifstream& file) override;
    
    
    
    
    
};


#endif /* BatchNorm_hpp */
