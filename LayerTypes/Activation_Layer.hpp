
#ifndef Activation_Layer_hpp
#define Activation_Layer_hpp
#include <stdio.h>
#include "Layer.hpp"
#include "LossFunctions.hpp"
#include "ReLUFunction.hpp"
#include "SigmoidFunction.hpp"
#include "SoftMaxFunction.hpp"

using matrix_f = std::vector<std::vector<float>>;


class Activation_Layer : public Layer{
private:
    
    matrix_f input; // the input batch for the current layer
    
    matrix_f output; // the output batch that is returned as a prediction

    matrix_f input_copy; // takes the copy of the input batch
    
    ReLUFunction reluFunc; // separate relu object
    
    SigmoidFunction sigmoidFunc; // separate sigmoid object
    
    SoftMaxFunction softMaxFunc;
    
    string activation_function; // decides which activation function to use for this Activatoin Layer
    
    size_t batch_size;
    
    size_t features;
    
    double k_factor;

    
public:
    
    // the constuctor that sets the input matrix and the output matrix
    Activation_Layer(string act_func) : activation_function(act_func){
    }
    
    
    matrix_f backward(const  matrix_f & gradient_from_next_layer) override;
    
    matrix_f forward(const  matrix_f & output_from_prev_layer) override;
    
    void save_to_file(ofstream& file) override;
    
    void load_layer(ifstream& file) override;
    
    float SoftMax(int depth, int row);
    
   
    
    // a dummy method for this class
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    
};
#endif /* Activation_Layer_hpp */

