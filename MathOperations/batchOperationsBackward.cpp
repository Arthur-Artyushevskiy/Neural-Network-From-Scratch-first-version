

#include "batchOperationsBackward.hpp"

void batchOperationsBackward::setParameters(const size_t batch_size, const size_t num_features, const matrix_f& input, std::vector<float>& batch_vars, const double epsilon){
    
    setOfParam = new parameters(batch_size, num_features, input, batch_vars, epsilon);
    
}



void batchOperationsBackward::calculate_batch_d_beta(const matrix_f& gradient_from_next_layer, std::vector<float>& d_beta){
    if(setOfParam->num_features != gradient_from_next_layer[0].size()){
        std::cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << std::endl;
        return;
    }
    d_beta.resize(setOfParam->num_features, 0.0f);
    
    
    for(size_t row{0}; row < setOfParam->batch_size; row++ ){
        for(size_t col{0}; col < setOfParam->num_features; col++){
            d_beta[col] += gradient_from_next_layer[row][col];
        }
    }
}

void batchOperationsBackward::calculate_batch_d_gamma(const matrix_f& gradient_from_next_layer, std::vector<float>& d_gamma, const matrix_f& normalized_input, std::vector<float>& gamma, matrix_f& dLRespectdNormX){
     if(setOfParam->num_features != gradient_from_next_layer[0].size()){
        std::cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << std::endl;
        return;
    }
    d_gamma.resize(setOfParam->num_features, 0.0f);
    dLRespectdNormX.resize(setOfParam->batch_size, std::vector<float>(setOfParam->num_features, 0.0f));
    
   
    
    for(size_t row{0}; row < setOfParam->batch_size; row++){
        for(size_t col{0}; col < setOfParam->num_features; col++){
            float grad = gradient_from_next_layer[row][col];
            d_gamma[col] += grad * normalized_input[row][col];
            
            dLRespectdNormX[row][col] = grad * gamma[col];
        }
        
    }
    
}

void batchOperationsBackward::calculate_mean_correction(std::vector<float> & mean_correction, matrix_f& dLRespectdNormX){
    // sums up the values across all of the batches in the gradient from the next layer
    std::vector<float> sum_dL_dNormX(setOfParam->num_features, 0.0f);
    
    for(size_t row{0}; row < setOfParam->batch_size; row++){
        for(size_t col{0}; col < setOfParam->num_features; col++){
            sum_dL_dNormX[col] += dLRespectdNormX[row][col];
        }
    }
    // multiplies the mean_correction matrix with 1/batch_size
    for(size_t col{0}; col < setOfParam->num_features; col++){
        float std_inv = 1.0f / pow(setOfParam->batch_vars[col] + setOfParam->epsilon, 0.5);
        mean_correction[col] = sum_dL_dNormX[col] * (-std_inv);
    }
    
    
    
}


void batchOperationsBackward::calculate_var_correction( std::vector<float> & var_correction, std::vector<float>& batch_means, matrix_f& dLRespectdNormX){
   
    var_correction.resize(setOfParam->num_features, 0.0f);
    
    for(size_t col{0}; col < setOfParam->num_features; col++){
        float sum_term = 0.0f;
        for(size_t row{0}; row < setOfParam->batch_size; row++){
            sum_term += dLRespectdNormX[row][col] * (setOfParam->input[row][col] - batch_means[col]);
        }
        float std_dev = pow(setOfParam->batch_vars[col] + setOfParam->epsilon, 0.5);
        var_correction[col] = sum_term * -0.5f * pow(std_dev, -3.0f);
    }
    
    
}

void batchOperationsBackward::calculate_d_input(const matrix_f& gradient_from_next_layer, std::vector<float>& batch_means, matrix_f& d_input, matrix_f& dLRespectdNormX){
    d_input.resize(setOfParam->batch_size, std::vector<float>(setOfParam->num_features));
    
    // We need dVar and dMean to combine into dInput
    std::vector<float> dL_dVar(setOfParam->num_features);
    std::vector<float> dL_dMean(setOfParam->num_features);
    
    calculate_var_correction(dL_dVar, batch_means, dLRespectdNormX);
    
    // For dInput calculation, we need the dMu term that includes the dVar influence
    // dL/dx = dL/dx_hat * (1/sigma) + dL/dVar * (2*(x-mu)/N) + dL/dMu * (1/N)
    
    for(size_t col{0}; col < setOfParam->num_features; col++){
        float sum_dL_dNormX = 0.0f;
        for(size_t row{0}; row < setOfParam->batch_size; row++){
            sum_dL_dNormX += dLRespectdNormX[row][col];
        }
        
        float std_inv = 1.0f / pow(setOfParam->batch_vars[col] + setOfParam->epsilon, 0.5);
        
        for(size_t row{0}; row < setOfParam->batch_size; row++){
            float x_minus_mu = setOfParam->input[row][col] - batch_means[col];
            
            float term1 = dLRespectdNormX[row][col] * std_inv;
            
            float term2 = dL_dVar[col] * (2.0f * x_minus_mu / setOfParam->batch_size);
            
            float term3 = (sum_dL_dNormX * -std_inv) / setOfParam->batch_size;
            
            d_input[row][col] = term1 + term2 + term3;
        }
        
    }
}
