#include "Activation_Layer.hpp"
#include <iostream>

vector<vector<float>> Activation_Layer::ReLu(float k){
    k_factor = k;
    for(int row{0}; row < batch_size; row++){
        for(int col{0}; col < features; col++){
            if(input[row][col] < 0){
                output[row][col] = k * input[row][col];
            }
            else{
                output[row][col] = input[row][col];
            }
        }
    }
   
    return output;
}

vector<vector<float>> Activation_Layer::Sigmoid(){
    for(int row{0}; row < batch_size; row++){
        for(int col{0}; col < features; col++){
            output[row][col]= Sigmoid(input[row][col]);
        }
    }
    return output;
}

/* unnecessary code that I probably won't use it
float Activation_Layer::SoftMax(int depth, int row){
    float sum = SoftMaxSum();
    return (pow(2.71828182845904523, input[depth][row][0]))/sum;
}
*/
vector<float> Activation_Layer::SoftMaxSum(){
    vector<float> sums(input.size(),0);
    float max{input[0][0]};
    float sum{0.0f};
    input_copy = input;
    // finds the maximum among the numbers in the matrix
    
    for(int row{0}; row < batch_size; row++){
        max = input[row][0];
        sum = 0.0f;
        for(int col{0}; col < features; col++){
            if(max < input[row][col]){
                max = input[row][col];
            }
        }
        
        for(int col{0}; col < features; col++){
            input_copy[row][col] -= max;
        }
        
        
        for(int col{0}; col < features; col++){
            sum += pow(2.71828182845904523, input_copy[row][col]);
        }
        sums[row] = sum;
    }
return sums;
}



vector<vector<float>> Activation_Layer::SoftMax() {
    vector<float> sums = SoftMaxSum();
    //vector<vector<float>> result  = output;
    for(int row{0}; row < batch_size; row++){
        for(int col{0}; col < features; col++){
            output[row][col] = (pow(2.71828182845904523, input_copy[row][col]))/sums[row];
        }
    }
return output;
}

// I decided to use one-hot label instead of only one label, because I want this code to have more flexibility for other projects that may require a whole one-hot label
vector<vector<float>> Activation_Layer::backward(const vector<vector<float>> & gradient_from_next_layer){
    
    vector<vector<float>> d_input = gradient_from_next_layer;
    
    if (this->activation_function == "softmax") {
            // This is the combined gradient from the loss function, just pass it through.
            return gradient_from_next_layer;
        }
    
    if(this->activation_function!="relu" && this->activation_function!="ReLU"){
        for(size_t row{0}; row < batch_size; row++){
            for(size_t col{0}; col < features; col++){
                if(input[row][col] < 0){
                    d_input[row][col] = 0;
                }
            }
        }
    }
    
    if(this->activation_function!="leak_relu" && this->activation_function!="leak_ReLU"){
        for(size_t row{0}; row < batch_size; row++){
            for(size_t col{0}; col < features; col++){
                if(input[row][col] < 0){
                    d_input[row][col] = k_factor;
                }
            }
        }
    }
    
    return d_input;
}

vector<vector<float>> Activation_Layer::forward(const vector<vector<float>> & output_from_prev_layer){
    
    input = output_from_prev_layer;
    output = output_from_prev_layer;
    
    batch_size = input.size();
    features = input[0].size();
    
    if(activation_function == "ReLU" || activation_function == "relu"){
        return ReLu(0);
    }
    if(activation_function == "Leak_ReLU" || activation_function == "leak_relu"){
        return ReLu(0.01);
    }
    if(activation_function == "Sigmoid" || activation_function == "sigmoid"){
        return Sigmoid();
    }
    return SoftMax();
}



void Activation_Layer::update(float learning_rate,string OptimizationAlgorithm){
   // cout << "A DUMMY METHOD!!!" << endl;
}

void Activation_Layer::save_to_file(ofstream& file){};

void Activation_Layer::load_layer(ifstream& file){};

// There will be a possibility to use the d_z_loss
 
float Activation_Layer::Sigmoid_Prime(float input){
    return Sigmoid(input)/ (1-Sigmoid(input));
}

float Activation_Layer::Sigmoid(float input){
    return 1/(1+ pow(2.71828182845904523,input));
}
