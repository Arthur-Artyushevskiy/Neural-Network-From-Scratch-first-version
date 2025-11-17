#include "Activation_Layer.hpp"
#include <iostream>
float Activation_Layer::Sigmoid(float input){
    return 1/(1+ pow(2.71828182845904523,input));
}

 vector<vector<vector<float>>> Activation_Layer::ReLu(float k){
    for(int depth{0}; depth < input.size(); depth++){
        for(int row{0}; row < input[0].size(); row++){
            if(input[depth][row][0] < 0){
                output[depth][row][0] = k * input[depth][row][0];
            }
            else{
                output[depth][row][0] = input[depth][row][0];
            }
        }
    }
   
    return output;
}

vector<vector<vector<float>>> Activation_Layer::Sigmoid(){
    for(int depth{0}; depth < input.size(); depth++){
        for(int row{0}; row < input[0].size(); row++){
            output[depth][row][0]= Sigmoid(input[depth][row][0]);
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
    float max{input[0][0][0]};
    float sum{0.0f};
    input_copy = input;
    // finds the maximum among the numbers in the matrix
    
    for(int depth{0}; depth < input.size(); depth++){
        max = input[depth][0][0];
        sum = 0.0f;
        for(int row{0}; row < input[0].size(); row++){
            if(max < input[depth][row][0]){
                max = input[depth][row][0];
            }
        }
        
        for(int row{0}; row < input_copy[0].size(); row++){
            input_copy[depth][row][0] -= max;
        }
        
        
        for(int row{0}; row < input_copy[0].size(); row++){
            sum += pow(2.71828182845904523, input_copy[depth][row][0]);
        }
        sums[depth] = sum;
    }
return sums;
}



vector<vector<vector<float>>> Activation_Layer::SoftMax() {
    vector<float> sums = SoftMaxSum();
    //vector<vector<float>> result  = output;
    for(int depth{0}; depth < input_copy.size(); depth++){
        for(int row{0}; row < input_copy[0].size(); row++){
            output[depth][row][0] = (pow(2.71828182845904523, input_copy[depth][row][0]))/sums[depth];
        }
    }
return output;
}

// There will be a possibility to use the d_z_loss
 
float Activation_Layer::Sigmoid_Prime(float input){
    return Sigmoid(input)/ (1-Sigmoid(input));
}

float Activation_Layer::ReLu_Prime(float input, float k){
    if(input < 0){
        return k;
    }
    else{
        return 1;
    }
}
// I decided to use one-hot label instead of only one label, because I want this code to have more flexibility for other projects that may require a whole one-hot label
vector<vector<vector<float>>> Activation_Layer::backward(const vector<vector<vector<float>>> & gradient_from_next_layer){
    
    if (this->activation_function == "softmax") {
            // This is the combined gradient from the loss function, just pass it through.
            return gradient_from_next_layer;
        }
    
    if(this->activation_function!="relu" && this->activation_function!="ReLU"){
        return gradient_from_next_layer;
    }
    
    double k_factor = 0.01;
    vector<vector<vector<float>>> gradient_for_previous_layer = gradient_from_next_layer;
    for(int depth{0}; depth < gradient_from_next_layer.size(); depth++){
        for(int row{0}; row<gradient_from_next_layer[0].size(); row++){
            if(this->input[depth][row][0] <= 0){
                
                gradient_for_previous_layer[depth][row][0] = gradient_from_next_layer[depth][row][0] * k_factor;
            }
        }
    }
return gradient_for_previous_layer;
}

vector<vector<vector<float>>> Activation_Layer::forward(const  vector<vector<vector<float>>> & output_from_prev_layer){
    input = output_from_prev_layer;
    output = output_from_prev_layer;
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

vector<vector<vector<float>>> Activation_Layer::get_output(){
    return output;
}

vector<vector<vector<float>>> Activation_Layer::getInput(){
    return input;
}


void Activation_Layer::update(float learning_rate,string OptimizationAlgorithm){
   // cout << "A DUMMY METHOD!!!" << endl;
}

void Activation_Layer::save_to_file(ofstream& file){};

void Activation_Layer::load_layer(ifstream& file){};
