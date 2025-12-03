
#include "BatchNorm.hpp"

void BatchNorm::calculate_batch_mean(){
    batch_means.resize(rowSize, vector<float>(colSize, 0.0f));
    float sum{0.0f};
    for(int row{0}; row < rowSize; row++){
        sum = 0.0f;
        for(int depth{0}; depth < depthSize; depth++){
            sum+= input[depth][row][0];
        }
        batch_means[row][0] = sum / (float) depthSize;
    }
    //int row{0}; row < rowSize; row++
    //cout << "Batch Mean Calculated: " << endl;
    //print_matrix(batch_means);
}

void BatchNorm::calculate_batch_var(){
    batch_vars.resize(rowSize, vector<float>(colSize, 0.0f));
    float sum{0.0f};
    for(int row{0}; row < rowSize; row++){
        sum = 0.0f;
        float current_mean = batch_means[row][0];
        for(int depth{0}; depth < depthSize; depth++){
            sum+= pow(input[depth][row][0] - current_mean, 2);
            
        }
        batch_vars[row][0] = sum / (float) depthSize;
    }
    
   
    //cout << "Batch Variance Calculated: " << endl;
    //print_matrix(batch_vars);
}
                                      
void BatchNorm::calculate_norm_input(){
   
    for(int depth{0}; depth < depthSize; depth++){
        for(int row{0}; row < rowSize; row++){
            float current_mean = batch_means[row][0];
            float current_var = batch_vars[row][0];
            normalized_input[depth][row][0] = (input[depth][row][0] - current_mean) / (pow(current_var + epsilon, 0.5));
            
        }
    }
}

void BatchNorm::calculate_output(){
    for(int depth{0}; depth < depthSize; depth++){
        vector<vector<float>> temp_result(rowSize, vector<float>(1, 0));
        temp_result = element_wise_multiplication(gamma, normalized_input[depth]);
        temp_result = add_trasposed_vectors(beta, temp_result);
        output[depth] = temp_result;
        }
}

vector<vector<vector<float>>> BatchNorm::forward(const std::vector<std::vector<std::vector<float>>> & output_from_prev_layer){
    if(rowSize != output_from_prev_layer[0].size()){
        cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << endl;
        return output_from_prev_layer;
    }
    
    
    
    depthSize = output_from_prev_layer.size();
    
    input = output_from_prev_layer;
    
    normalized_input.resize(depthSize, vector<vector<float>>(rowSize, vector<float>(colSize, 0.0f)));
    
    output.resize(depthSize, vector<vector<float>>(rowSize, vector<float>(colSize, 0.0f)));
    
    //cout << "The number of rows for gamma: "<< gamma.size() << endl;
    //cout << "The number of rows for output: "<< output[0].size() << endl;
    
    calculate_batch_mean();
    calculate_batch_var();
    calculate_norm_input();
    calculate_output();
    return output;
}
// Need to create an actual implementation of the backward method
vector<vector<vector<float>>> BatchNorm::backward(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer){
    
    d_input.resize(depthSize, vector<vector<float>>(rowSize, vector<float>(colSize, 0.0f)));
    
    calculate_batch_d_beta(gradient_from_next_layer);
    
    calculate_batch_d_gamma(gradient_from_next_layer);
    
    calculate_d_input(gradient_from_next_layer);
    
   
    return d_input;
    
}

void BatchNorm::calculate_batch_d_beta(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer){
    if(rowSize != gradient_from_next_layer[0].size()){
        cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << endl;
        return;
    }
    
    vector<vector<float>> sum_matrix(rowSize, vector<float>(colSize, 0));
    for(int depth{0}; depth < depthSize; depth++){
        sum_matrix = add_trasposed_vectors(sum_matrix, gradient_from_next_layer[depth]);
    }
    d_beta = sum_matrix;
    
}

void BatchNorm::calculate_batch_d_gamma(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer){
     if(rowSize != gradient_from_next_layer[0].size()){
        cerr << "ERROR: The number of rows in the BatchNorm object is not the same to input_tensor!" << endl;
        return;
    }
    vector<vector<float>> temp_result(rowSize, vector<float>(colSize, 0));
    vector<vector<float>> sum(rowSize, vector<float>(colSize, 0));
    for(int depth{0}; depth < depthSize; depth++){
       temp_result = element_wise_multiplication(gradient_from_next_layer[depth], normalized_input[depth]);
        
        sum = add_trasposed_vectors(temp_result, sum);
    }
    d_gamma = sum;
    
}

void BatchNorm::calculate_mean_correction(vector<vector<vector<float>>> & mean_correction){
    // sums up the values across all of the batches in the gradient from the next layer
    vector<vector<float>> sum(gamma.size(), vector<float>(1, 0));
    for(int depth{0}; depth < input.size(); depth++){
        sum = add_trasposed_vectors(dLRespectdNormX[depth],sum);
    }
    // multiplies the mean_correction matrix with 1/batch_size
    for(int row{0}; row <rowSize; row++){
       sum[row][0] *=  (double) 1/depthSize;
    }
    
    for(int depth{0}; depth < depthSize; depth++){
        mean_correction[depth] = sum;
    }
    
}

void BatchNorm::calculate_var_correction(vector<vector<vector<float>>> & var_correction){
    /*
    vector<vector<vector<float>>> mean_correction(
            input.size(),vector<vector<float>>(gamma.size(),vector<float>(1, 0)));
    */
    vector<vector<float>> sum(rowSize, vector<float>(colSize, 0));
    for(int depth{0}; depth < depthSize; depth++){
        vector<vector<float>> temp_result = element_wise_multiplication( dLRespectdNormX[depth], normalized_input[depth]);
        
        sum = add_trasposed_vectors(sum, temp_result);
    }
    // multiplies the mean_correction matrix with 1/batch_size
    for(int row{0}; row < rowSize; row++){
        sum[row][0] *= (double) 1/depthSize;
    }
    for(int depth{0}; depth < depthSize; depth++){
        var_correction[depth] = sum;
        var_correction[depth] = element_wise_multiplication(normalized_input[depth], var_correction[depth]);
    }
       
    
}


void BatchNorm::calculate_d_input(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer){
    

    dLRespectdNormX.resize(input.size(), vector<vector<float>>(rowSize, vector<float>(colSize, 0)));
    // calculate the resulting normalized input gradient that will be necessary to calculate the mean and variance correction term
    for(int depth{0}; depth < depthSize; depth++){
        dLRespectdNormX[depth] = element_wise_multiplication(gradient_from_next_layer[depth], gamma);
    }
    
    vector<vector<float>> dL_dmu(rowSize, vector<float>(1,0.0f));
    vector<vector<float>> dL_dvar(rowSize, vector<float>(1,0.0f));
    
    vector<vector<float>> total_sum_dL_dNormX(rowSize, vector<float>(1, 0.0f));
    vector<vector<float>> total_sum_dL_dNormX_times_x_minus_mu(rowSize, vector<float>(1, 0.0f));
    
    for(int row{0}; row < rowSize; row++){
        
        float sum_dL_dNormX = 0.0f;
        float sum_dL_dNormX_times_x_minus_mu = 0.0f;
        
        
        float mean = batch_means[row][0];
        float var = batch_vars[row][0];
        
        for(int depth{0}; depth < depthSize; depth++){
            float dL_dNormX = dLRespectdNormX[depth][row][0];
            float x_minus_mu = input[depth][row][0] - mean;
            
            sum_dL_dNormX += dL_dNormX;
            sum_dL_dNormX_times_x_minus_mu += dL_dNormX * x_minus_mu;
        }
        total_sum_dL_dNormX[row][0] = sum_dL_dNormX;
        total_sum_dL_dNormX_times_x_minus_mu[row][0] = sum_dL_dNormX_times_x_minus_mu;
    }
    
    d_input.resize(input.size(), vector<vector<float>>(rowSize, vector<float>(colSize, 0)));
    for(int depth{0}; depth < depthSize; depth++){
        for(int row{0}; row < rowSize; row++){
            
            float dL_dNormX = dLRespectdNormX[depth][row][0];
            float mean = batch_means[row][0];
            float var = batch_vars[row][0];
            float inv_std_dev = 1.0f / pow(var + epsilon, 0.5);
            
            float current_sum_dL_dNormX = total_sum_dL_dNormX[row][0];
            float current_sum_dL_dNormX_times_x_minus_mu = total_sum_dL_dNormX_times_x_minus_mu[row][0];
            
            float mean_corr = current_sum_dL_dNormX / (float) depthSize;
            
            float var_corr = (input[depth][row][0] - mean) / (var + epsilon) * (current_sum_dL_dNormX_times_x_minus_mu / (float) depthSize);
            
            d_input[depth][row][0] = inv_std_dev * (dL_dNormX - mean_corr - var_corr);
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
        
        for(int row{0}; row < rowSize; row++){
            m_gamma[row][0] = beta1* m_gamma[row][0] + (1-beta1) * d_gamma[row][0];
            v_gamma[row][0] = beta2* v_gamma[row][0] + (1-beta2) * pow(d_gamma[row][0], 2);
            
            mVector = m_gamma[row][0] / (1 - pow(beta1, t));
            vVector = v_gamma[row][0] / (1 - pow(beta2, t));
            
            gamma[row][0] = gamma[row][0] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
            
            m_beta[row][0] = beta1* m_beta[row][0] + (1-beta1) * d_beta[row][0];
            v_beta[row][0] = beta2* v_beta[row][0] + (1-beta2) * pow(d_beta[row][0], 2);
            
            mVector = m_beta[row][0] / (1 - pow(beta1, t));
            vVector = v_beta[row][0] / (1 - pow(beta2, t));
            
            beta[row][0] = beta[row][0] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
        }
    }
}

void BatchNorm::save_to_file(ofstream& file){
    if(!file.is_open()){
        cerr << "Error: file not open for writing!" << endl;
        return ;
    }
    
    file << "BATCHNORM\n";
    
    file << gamma.size() << " " << gamma[0].size() << "\n";
    
    for(const auto& row : gamma){
        for(float val : row){
            file << val << " ";
        }
    }
    file << "\n";
    
    file << beta.size() << " " << beta[0].size() << "\n";
    
    for(const auto& row : beta){
        for(float val : row){
            file << val << " ";
        }
    }
    file << "\n";
}

void BatchNorm::load_layer(std::ifstream& file){
    if(!file.is_open()){
        cerr << "ERROR: Could not open the file!" << endl;
        return;
    }
    
    string name;
    int rows, cols;
    
    file >> name;
    if(name != "BATCHNORM"){
        cerr << "ERROR: Could not find the DENSE Header! Found:" << name << endl;
        return;
    }
    
    file >> rows >> cols;
    cout << "Number of rows:" << rows << endl;
    cout << "Number of cols:" << cols << endl;
    gamma.resize(rows);
    for(int row{0}; row < rows; row++){
        gamma[row].resize(cols);
        
        for (int col{0}; col < cols; col++) {
            file >> gamma[row][col];
        }
    }
    
    file >> rows >> cols;
    beta.resize(rows);
    for(int row{0}; row < rows; row++){
      beta[row].resize(cols);
        
        for (int col{0}; col < cols; col++) {
            file >> beta[row][col];
        }
    }

    
}
