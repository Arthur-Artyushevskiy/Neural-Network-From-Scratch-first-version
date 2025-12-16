
#include "Dense_Layer.hpp"
#include <iostream>

void Dense_Layer::he_init(){
    random_device rd;
        mt19937 gen(rd());
        double stddev = sqrt(2.0 / (double)input_features);
        normal_distribution<> d(0.0, stddev);

        for(int i = 0; i < input_features; ++i) {
            for(int j = 0; j < output_features; ++j) {
                weights[i][j] = d(gen);
            }
        }
        // Biases initialized to 0 usually, or small constant
        fill(biases.begin(), biases.end(), 0.0f);
}


// there is an error where the gradient from the next layer does not change and stays the same initial gradient
vector<vector<float>> Dense_Layer::backward(const vector<vector<float>> & gradient_from_next_layer){
    
    vector<vector<float>> input_T = transpose(input);
    
    d_weights = multiply(input_T, gradient_from_next_layer);
    
    d_biases = sum_dim0(gradient_from_next_layer);
    
    vector<vector<float>> weights_T = transpose(weights);
    return multiply(gradient_from_next_layer, weights_T);
}

void Dense_Layer::SGD(float learning_rate){
    for(int row{0}; row < weights.size(); row++){
        for(int col{0}; col < weights[0].size(); col++){
            // updates the value of weights using the calculated gradient from the next layer
            weights[row][col] = weights[row][col] - learning_rate * d_weights[row][col];
        }
    }
    for(int row{0}; row < biases.size(); row++){
        biases[row] = biases[row] - learning_rate * d_biases[row];
    }
}

// this method updates the values of weights and biases for each dense layer
void Dense_Layer::update(float learning_rate,string OptimizationAlgorithm){
    
    if(d_weights.size() != weights.size() || d_weights[0].size() != weights[0].size()) {
            cerr << "ERROR: d_weights dimensions mismatch in update(). Skipping update." << endl;
            return;
        }
        if(d_biases.size() != biases.size()) {
            cerr << "ERROR: d_biases dimensions mismatch in update(). Skipping update." << endl;
            return;
        }
    
    if(OptimizationAlgorithm == "ADAM"){
        static int t = 0;
        
        const double epsilon = 1e-9;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double mVector;
        double vVector;
        
        t++;
       
        for(int row{0}; row < weights.size(); row++){
           
            for(int col{0}; col < weights[0].size(); col++){
                m_weights[row][col] = beta1* m_weights[row][col] + (1-beta1) * d_weights[row][col];
                v_weights[row][col] = beta2* v_weights[row][col] + (1-beta2) * pow(d_weights[row][col], 2);
                
                mVector = m_weights[row][col] / (1 - pow(beta1, t));
                vVector = v_weights[row][col] / (1 - pow(beta2, t));
                
                weights[row][col] = weights[row][col] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
            }
        }
       
        
        for(int row{0}; row < biases.size(); row++){
           
            m_biases[row] = beta1* m_biases[row] + (1-beta1) * d_biases[row];
            v_biases[row] = beta2* v_biases[row] + (1-beta2) * pow(d_biases[row], 2);
            
            mVector = m_biases[row] / (1 - pow(beta1, t));
            vVector = v_biases[row] / (1 - pow(beta2, t));
            
           biases[row] = biases[row] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
        }
    }
    else if(OptimizationAlgorithm == "SGD"){
        SGD(learning_rate);
        
    }
    else{
        cout << "You didn't Enter 'ADAM' or 'SGD'. Therefore the model will use SGD for optimization." << endl;
        SGD(learning_rate);
    }
    
}

vector<vector<float>> Dense_Layer::forward(const vector<vector<float>> & output_from_prev_layer){
    this->input = output_from_prev_layer;
    
    // Matrix Multiply: [Batch x In] * [In x Out] = [Batch x Out]
    vector<vector<float>> output = multiply(output_from_prev_layer, weights);
    
    output = add_bias_to_batch(output, biases);
   
    return output;

}

void Dense_Layer::save_to_file(ofstream& file){
    
    if(!file.is_open()){
        cerr << "Error: file not open for writing!" << endl;
        return ;
    }
    
    file << "DENSE\n";
    
    file << input_features << " " << output_features << "\n";
    
    for(const auto& row : weights){
        for(float val : row){
            file << val << " ";
        }
    }
    file << "\n";
    
    file << biases.size() << " " << 1 << "\n";
    
    for(const auto& row : biases){
        file << row << " ";
    }
    file << "\n";
}

void Dense_Layer::load_layer(ifstream& file){
    
    if(!file.is_open()){
        cerr << "ERROR: Could not open the file!" << endl;
        return;
    }
    
    string name;
    int rows, cols;
    
    file >> name;
    if(name != "DENSE"){
        cerr << "ERROR: Could not find the DENSE Header! Found:" << name << endl;
        return;
    }
    
    file >> rows >> cols;
    cout << "Number of rows:" << rows << endl;
    cout << "Number of cols:" << cols << endl;
    weights.resize(rows);
    for(int row{0}; row < rows; row++){
        weights[row].resize(cols);
        
        for (int col{0}; col < cols; col++) {
            file >> weights[row][col];
        }
    }
    
    file >> rows >> cols;
    biases.resize(rows);
    for(int row{0}; row < rows; row++){
       file >> biases[row];
        
    }

    
}


