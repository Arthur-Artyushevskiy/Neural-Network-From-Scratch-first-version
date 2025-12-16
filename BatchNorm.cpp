
#include "BatchNorm.hpp"

void BatchNorm::calculate_batch_mean(){
    batch_means.resize(num_features, 0.0f);
    
    for(size_t row{0}; row < batch_size; row++){
        for(size_t col{0}; col <num_features ; col++){
            batch_means[col] += input[row][col];
        }
    }
    
    for(size_t col{0}; col <num_features ; col++){
        batch_means[col] /= (float) batch_size;
    }
    //int row{0}; row < rowSize; row++ size_t col{0}; col <num_features ; col++
    //cout << "Batch Mean Calculated: " << endl;
    //print_matrix(batch_means);
}

void BatchNorm::calculate_batch_var(){
    batch_vars.resize(num_features, 0.0f);
    float sum{0.0f};
    for(size_t row{0}; row < batch_size; row++){
        sum = 0.0f;
        for(size_t col{0}; col <num_features ; col++){
            batch_vars[col] += pow(input[row][col] - batch_means[col], 2);
            
        }
    }
    for(size_t col{0}; col <num_features ; col++){
        batch_vars[col] /= (float) batch_size;
    }
   
    //cout << "Batch Variance Calculated: " << endl;
    //print_matrix(batch_vars);
}
                                      
void BatchNorm::calculate_norm_input(){
    normalized_input.resize(batch_size, vector<float>(num_features));
    for(size_t col{0}; col <num_features ; col++){
        float std_dev = pow(batch_vars[col] + epsilon, 0.5);
        for(size_t row{0}; row < batch_size; row++){
            normalized_input[row][col] = (input[row][col] - batch_means[col]) / std_dev;
            
        }
    }
}

void BatchNorm::calculate_output(){
    output.resize(batch_size, vector<float>(num_features));
    
    for(size_t row{0}; row < batch_size; row++){
        for(size_t col{0}; col <num_features ; col++){
            output[row][col] = (gamma[col] * normalized_input[row][col]) + beta[col];
        }
    }
    
        
}

vector<vector<float>> BatchNorm::forward(const std::vector<std::vector<float>> & output_from_prev_layer){
    if(output_from_prev_layer.empty()) return {};
    
    if(num_features != output_from_prev_layer[0].size()){
        cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << endl;
        return output_from_prev_layer;
    }
    
    batch_size = output_from_prev_layer.size();
    
    input = output_from_prev_layer;
    
    //cout << "The number of rows for gamma: "<< gamma.size() << endl;
    //cout << "The number of rows for output: "<< output[0].size() << endl;
    
    calculate_batch_mean();
    calculate_batch_var();
    calculate_norm_input();
    calculate_output();
    
    return output;
}
// Need to create an actual implementation of the backward method
vector<vector<float>> BatchNorm::backward(const std::vector<std::vector<float>> & gradient_from_next_layer){
    
    calculate_batch_d_beta(gradient_from_next_layer);
    
    calculate_batch_d_gamma(gradient_from_next_layer);
    
    calculate_d_input(gradient_from_next_layer);
    
   
    return d_input;
    
}

void BatchNorm::calculate_batch_d_beta(const std::vector<std::vector<float>> & gradient_from_next_layer){
    if(num_features != gradient_from_next_layer[0].size()){
        cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << endl;
        return;
    }
    d_beta.resize(num_features, 0.0f);
    
    
    for(size_t row{0}; row < batch_size; row++ ){
        for(size_t col{0}; col < num_features; col++){
            d_beta[col] += gradient_from_next_layer[row][col];
        }
    }
}

void BatchNorm::calculate_batch_d_gamma(const std::vector<std::vector<float>> & gradient_from_next_layer){
     if(num_features != gradient_from_next_layer[0].size()){
        cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << endl;
        return;
    }
    d_gamma.resize(num_features, 0.0f);
    dLRespectdNormX.resize(batch_size, vector<float>(num_features, 0.0f));
    
   
    
    for(size_t row{0}; row < batch_size; row++){
        for(size_t col{0}; col < num_features; col++){
            float grad = gradient_from_next_layer[row][col];
            d_gamma[col] += grad * normalized_input[row][col];
            
            dLRespectdNormX[row][col] = grad * gamma[col];
        }
        
    }
    
}

void BatchNorm::calculate_mean_correction(vector<float> & mean_correction){
    // sums up the values across all of the batches in the gradient from the next layer
    vector<float> sum_dL_dNormX(num_features, 0.0f);
    
    for(size_t row{0}; row < batch_size; row++){
        for(size_t col{0}; col < num_features; col++){
            sum_dL_dNormX[col] += dLRespectdNormX[row][col];
        }
    }
    // multiplies the mean_correction matrix with 1/batch_size
    for(size_t col{0}; col < num_features; col++){
        float std_inv = 1.0f / pow(batch_vars[col] + epsilon, 0.5);
        mean_correction[col] = sum_dL_dNormX[col] * (-std_inv);
    }
    
    
    
}

void BatchNorm::calculate_var_correction(vector<float> & var_correction){
   
    var_correction.resize(num_features, 0.0f);
    
    for(size_t col{0}; col < num_features; col++){
        float sum_term = 0.0f;
        for(size_t row{0}; row < batch_size; row++){
            sum_term += dLRespectdNormX[row][col] * (input[row][col] - batch_means[col]);
        }
        float std_dev = pow(batch_vars[col] + epsilon, 0.5);
        var_correction[col] = sum_term * -0.5f * pow(std_dev, -3.0f);
    }
    
    
}


void BatchNorm::calculate_d_input(const std::vector<std::vector<float>> & gradient_from_next_layer){
    d_input.resize(batch_size, vector<float>(num_features));
    
    // We need dVar and dMean to combine into dInput
    vector<float> dL_dVar(num_features);
    vector<float> dL_dMean(num_features);
    
    calculate_var_correction(dL_dVar);
    
    // For dInput calculation, we need the dMu term that includes the dVar influence
    // dL/dx = dL/dx_hat * (1/sigma) + dL/dVar * (2*(x-mu)/N) + dL/dMu * (1/N)
    
    for(size_t col{0}; col < num_features; col++){
        float sum_dL_dNormX = 0.0f;
        for(size_t row{0}; row < batch_size; row++){
            sum_dL_dNormX += dLRespectdNormX[row][col];
        }
        
        float std_inv = 1.0f / pow(batch_vars[col] + epsilon, 0.5);
        
        for(size_t row{0}; row < batch_size; row++){
            float x_minus_mu = input[row][col] - batch_means[col];
            
            float term1 = dLRespectdNormX[row][col] * std_inv;
            
            float term2 = dL_dVar[col] * (2.0f * x_minus_mu / batch_size);
            
            float term3 = (sum_dL_dNormX * -std_inv) / batch_size;
            
            d_input[row][col] = term1 + term2 + term3;
        }
        
    }
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
