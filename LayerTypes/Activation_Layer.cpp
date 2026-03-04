#include "Activation_Layer.hpp"
#include <iostream>

// There will be a possibility to use the d_z_loss



// I decided to use one-hot label instead of only one label, because I want this code to have more flexibility for other projects that may require a whole one-hot label
matrix_f Activation_Layer::backward(const matrix_f & gradient_from_next_layer){
    
    matrix_f d_input = gradient_from_next_layer;
    
    
    if (this->activation_function == "softmax") {
            // This is the combined gradient from the loss function, just pass it through.
            return gradient_from_next_layer;
        }
    
    if(this->activation_function!="relu" && this->activation_function!="ReLU"){
        this->reluFunc.backward(batch_size, features, input, d_input, 0.0f);
    }
    
    if(this->activation_function!="leak_relu" && this->activation_function!="leak_ReLU"){
        this->reluFunc.backward(batch_size, features, input, d_input, k_factor);
    }
    
    return d_input;
}

matrix_f Activation_Layer::forward(const matrix_f & output_from_prev_layer){
    
    input = output_from_prev_layer;
    output = output_from_prev_layer;
    
   
    
    batch_size = input.size();
    features = input[0].size();
    
    if(activation_function == "ReLU" || activation_function == "relu"){
        return this->reluFunc.ReLU(0, batch_size, features, output, input);
    }
    if(activation_function == "Leak_ReLU" || activation_function == "leak_relu"){
        return this->reluFunc.ReLU(0, batch_size, features, output, input);
    }
    if(activation_function == "Sigmoid" || activation_function == "sigmoid"){
        return sigmoidFunc.sigmoid(batch_size, features, input, output);
    }
    
    softMaxFunc.softMax(batch_size, features, input, input_copy, output);
    
    return output;
}



void Activation_Layer::update(float learning_rate,string OptimizationAlgorithm){
   // cout << "A DUMMY METHOD!!!" << endl;
}

void Activation_Layer::save_to_file(ofstream& file){};

void Activation_Layer::load_layer(ifstream& file){};




