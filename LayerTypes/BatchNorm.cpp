
#include "BatchNorm.hpp"


vector<vector<float>> BatchNorm::forward(const matrix_f& output_from_prev_layer){
    if(output_from_prev_layer.empty()) return {};
    
    if(num_features != output_from_prev_layer[0].size()){
        cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << endl;
        return output_from_prev_layer;
    }
    
    batch_size = output_from_prev_layer.size();
    
    input = output_from_prev_layer;
    
    forwardBatchOperations.setParameters(batch_size, num_features, input);
    
    forwardBatchOperations.calculate_batch_mean(batch_means);
    
    forwardBatchOperations.calculate_batch_var(batch_vars, batch_means);
    
    forwardBatchOperations.calculate_norm_input(normalized_input, batch_means, batch_vars, epsilon);
    
    forwardBatchOperations.calculate_output(output, normalized_input, beta, gamma);
    
    return output;
}




// Need to create an actual implementation of the backward method
matrix_f BatchNorm::backward(const matrix_f& gradient_from_next_layer){
    
    backwardBatchOperations.setParameters(batch_size, num_features, input, batch_vars, epsilon);
    
    backwardBatchOperations.calculate_batch_d_beta(gradient_from_next_layer, d_beta);
    
    backwardBatchOperations.calculate_batch_d_gamma(gradient_from_next_layer, d_gamma, normalized_input, gamma, dLRespectdNormX);
    
    backwardBatchOperations.calculate_d_input(gradient_from_next_layer, batch_means, d_input, dLRespectdNormX);
    
   
    return d_input;
    
}


void BatchNorm::update(float learning_rate,string OptimizationAlgorithm){
    if(OptimizationAlgorithm == "ADAM"){
        static int t = 0;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double mVector;
        double vVector;
        
        t++;
        
        for(int col{0}; col < num_features; col++){
            m_gamma[col] = beta1* m_gamma[col] + (1-beta1) * d_gamma[col];
            v_gamma[col] = beta2* v_gamma[col] + (1-beta2) * pow(d_gamma[col], 2);
            
            mVector = m_gamma[col] / (1 - pow(beta1, t));
            vVector = v_gamma[col] / (1 - pow(beta2, t));
            
            gamma[col] = gamma[col] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
            
            m_beta[col] = beta1* m_beta[col] + (1-beta1) * d_beta[col];
            v_beta[col] = beta2* v_beta[col] + (1-beta2) * pow(d_beta[col], 2);
            
            mVector = m_beta[col] / (1 - pow(beta1, t));
            vVector = v_beta[col] / (1 - pow(beta2, t));
            
            beta[col] = beta[col] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
        }
    }
    else {
            // SGD
            for (size_t j = 0; j < num_features; ++j) {
                gamma[j] -= learning_rate * d_gamma[j];
                beta[j]  -= learning_rate * d_beta[j];
            }
    }
}

void BatchNorm::save_to_file(ofstream& file){
    if(!file.is_open()){
        cerr << "Error: file not open for writing!" << endl;
        return ;
    }
    
    file << "BATCHNORM\n";
    
    file << num_features << "\n";
    
    for(float val : gamma){
        file << val << " ";
    
    }
    file << "\n";
    
    file << num_features<< "\n";
    
    for(float val : beta){
        file << val << " ";
    }
    file << "\n";
}

void BatchNorm::load_layer(std::ifstream& file){
    if(!file.is_open()){
        cerr << "ERROR: Could not open the file!" << endl;
        return;
    }
    
    string name;
    int size;
    
    file >> name;
    if(name != "BATCHNORM"){
        cerr << "ERROR: Could not find the DENSE Header! Found:" << name << endl;
        return;
    }
    
    file >> size;
    gamma.resize(size);
    num_features = size;
    for(int row{0}; row < size; row++){
        file >> gamma[row];
    }
    
    file >> size;
    beta.resize(size);
    for(int row{0}; row < size; row++){
        file >> beta[row];
    }
    
    d_gamma.resize(size, 0.0f); d_beta.resize(size, 0.0f);
    m_gamma.resize(size, 0.0f); v_gamma.resize(size, 0.0f);
    m_beta.resize(size, 0.0f); v_beta.resize(size, 0.0f);
    
}
