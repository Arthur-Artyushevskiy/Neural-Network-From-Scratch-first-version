

#ifndef BatchNorm_hpp
#define BatchNorm_hpp
#include <stdio.h>
#include <cmath>
#include "Matrix_Operations.hpp"
#include "Layer.hpp"
#include "batchOperationsForward.hpp"
#include "batchOperationsBackward.hpp"
#include "batchOperationsForward.hpp"
using namespace std;

using matrix_f = std::vector<std::vector<float>>;

class BatchNorm : public Layer{
private:
    matrix_f input, normalized_input, output;
   
    vector<float> batch_means, batch_vars;
    
    matrix_f d_input;
    
    matrix_f dLRespectdNormX;
    
    batchOperationsBackward backwardBatchOperations;
    
    batchOperationsForward forwardBatchOperations;
    
    const double epsilon = 1e-9;
    
    vector<float> d_gamma, m_gamma, v_gamma;
    
    vector<float> d_beta, m_beta, v_beta;
    
    size_t num_features;
    
    size_t batch_size;
    
public:
    
    vector<float> gamma;
    vector<float> beta;
    
    BatchNorm(int features) : num_features(features){
        gamma.resize(features, 1.0f);
        
        d_gamma.resize(features, 0.0f);
        beta.resize(features, 0.0f);
        
        d_beta.resize(features, 0.0f);
        m_gamma.resize(features, 0.0f);
        
        v_gamma.resize(features, 0.0f);
        m_beta.resize(features, 0.0f);
        v_beta.resize(features, 0.0f);
        
    }
    
    
    matrix_f forward(const matrix_f& output_from_prev_layer) override;
    
    
    matrix_f backward(const matrix_f& gradient_from_next_layer) override;
    
    
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    void save_to_file(ofstream& file) override;
    
    void load_layer(std::ifstream& file) override;
    
    
    
    
    
};


#endif /* BatchNorm_hpp */
