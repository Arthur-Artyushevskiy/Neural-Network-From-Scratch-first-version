

#include "batchOperationsForward.hpp"


void batchOperationsForward::calculate_batch_mean(std::vector<float>& batch_means){
    batch_means.resize(setOfParam->num_features, 0.0f);
    
    for(size_t row{0}; row < setOfParam->batch_size; row++){
        for(size_t col{0}; col <setOfParam->num_features ; col++){
            batch_means[col] += setOfParam->input[row][col];
        }
    }
    
    for(size_t col{0}; col <setOfParam->num_features ; col++){
        batch_means[col] /= (float) setOfParam->batch_size;
    }
    //int row{0}; row < rowSize; row++ size_t col{0}; col <num_features ; col++
    //cout << "Batch Mean Calculated: " << endl;
    //print_matrix(batch_means);
}


void batchOperationsForward::calculate_batch_var(std::vector<float>& batch_vars, std::vector<float>& batch_means){
    batch_vars.resize(setOfParam->num_features, 0.0f);
    float sum{0.0f};
    for(size_t row{0}; row < setOfParam->batch_size; row++){
        sum = 0.0f;
        for(size_t col{0}; col <setOfParam->num_features ; col++){
            batch_vars[col] += pow(setOfParam->input[row][col] - batch_means[col], 2);
            
        }
    }
    for(size_t col{0}; col <setOfParam->num_features ; col++){
        batch_vars[col] /= (float) setOfParam->batch_size;
    }
   
    //cout << "Batch Variance Calculated: " << endl;
    //print_matrix(batch_vars);
}


void batchOperationsForward::calculate_norm_input(matrix_f& normalized_input, std::vector<float>& batch_means, std::vector<float>& batch_vars, const double epsilon){
    normalized_input.resize(setOfParam->batch_size, std::vector<float>(setOfParam->num_features));
    for(size_t col{0}; col <setOfParam->num_features ; col++){
        float std_dev = pow(batch_vars[col] + epsilon, 0.5);
        for(size_t row{0}; row < setOfParam->batch_size; row++){
            normalized_input[row][col] = (setOfParam->input[row][col] - batch_means[col]) / std_dev;
            
        }
    }
}

void batchOperationsForward::calculate_output(matrix_f& output, const matrix_f& normalized_input, std::vector<float>& beta,  std::vector<float>& gamma){
    output.resize(setOfParam->batch_size, std::vector<float>(setOfParam->num_features));
    
    for(size_t row{0}; row < setOfParam->batch_size; row++){
        for(size_t col{0}; col <setOfParam->num_features ; col++){
            output[row][col] = (gamma[col] * normalized_input[row][col]) + beta[col];
        }
    }
    
        
}
