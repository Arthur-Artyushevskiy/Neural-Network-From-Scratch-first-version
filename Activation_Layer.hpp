
#ifndef Activation_Layer_hpp
#define Activation_Layer_hpp
#include <stdio.h>
#include "Layer.hpp"
#include "LossFunctions.hpp"
using namespace std;
class Activation_Layer : public Layer{
private:
    
    vector<vector<vector<float>>> input; // the input batch for the current layer
    
    vector<vector<vector<float>>> output; // the output batch that is returned as a prediction

    vector<vector<vector<float>>> input_copy; // takes the copy of the input batch
    
    string activation_function; // decides which activation function to use for this Activatoin Layer

public:
    
    // the constuctor that sets the input matrix and the output matrix
    Activation_Layer(string activation_function){
        this->activation_function = activation_function;
    }
    
    // the sigmoid function for a single value
    float Sigmoid(float input);
    
    // returns the sum of all of the values in the input matrix to use for the softmax function
    vector<float> SoftMaxSum();
    
    // the ReLu method for the whole matrix
    vector<vector<vector<float>>> ReLu(float k);
    
    // the Sigmoid method for the whole matrix
    vector<vector<vector<float>>> Sigmoid();
    
    // the SoftMax method for the whole matrix
    vector<vector<vector<float>>> SoftMax();
    
    vector<vector<vector<float>>> backward(const  vector<vector<vector<float>>> & gradient_from_next_layer) override;
    
    vector<vector<vector<float>>> forward(const  vector<vector<vector<float>>> & output_from_prev_layer) override;
    
    void save_to_file(ofstream& file) override;
    
    void load_layer(ifstream& file) override;
    
    float SoftMax(int depth, int row);
    
    vector<vector<float>> SoftMaxPrime();
    
    float Sigmoid_Prime(float input);
    
    // the gradient of the ReLu function
    float ReLu_Prime(float input, float k);
    
    // a dummy method for this class
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    vector<vector<vector<float>>> get_output();
    
    vector<vector<vector<float>>> getInput();
};
#endif /* Activation_Layer_hpp */

