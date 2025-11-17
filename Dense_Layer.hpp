
#ifndef Dense_Layer_hpp
#define Dense_Layer_hpp
#include <stdio.h>
#include "Matrix_Operations.hpp"
#include "Layer.hpp"
class Dense_Layer : public Layer{
private:
    // 2D matrix with 1 col and n rows
    std::vector<std::vector<std::vector<float>>> input;
    // 2D matrix with z col and n rows
    std::vector<std::vector<float>> weights;
    // 2D matrix with 1 col and n rows
    std::vector<std::vector<float>> biases;
    // A matrix for the weight gradient
    std::vector<std::vector<float>> d_weights;
    // A matrix for the bias gradient
    std::vector<std::vector<float>> d_biases;
    
    std::vector<std::vector<double>> m_weights;
    
    std::vector<std::vector<double>> v_weights;
    
    std::vector<std::vector<double>> m_biases;
    
    std::vector<std::vector<double>> v_biases;
    // the number of rows for the current matrix from the current layer that is set by the constructor
    int row;
    // the number of col for the current matrix from the previous layer is set by the constructor
    int col;
    
    int batch_num;
public:
   
    //UPDATE: I will change the old version of the constructor to allow the creation of an empty model. Then I will create a forward pass that will allow the input to be defined through the output of the previous layer
    // is that the weights and biases matrixes are not printing, basically not initilazing
    Dense_Layer(int input_size,int row){
        this-> row = row;
        col = input_size;
        weights = he_init_weights();
        biases =  he_init_biases();
        d_weights = weights;
        d_biases  = biases;
    }
   
    int getRow();
    
    float generate_random_float(float min, float max);
    
    // One of the  most important functions that calculates the gradient and adjusts the weights and biases
    vector<vector<vector<float>>> backward(const vector<vector<vector<float>>> & gradient_from_next_layer) override;
    // updates the weights and biases using the hyperparameter
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    void SGD(float learning_rate);
    // this method is important that will go through the model the pass the output of the previous layer to this current layer as an input
    vector<vector<vector<float>>> forward(const  vector<vector<vector<float>>> & output_from_prev_layer) override;
    
    void save_to_file(ofstream& file) override;
   
    void load_layer(ifstream& file) override;
    
    
    
    
    
    // sets the weight matrix with random numbers
    std::vector<std::vector<float>> he_init_weights();
    // sets the bias matrix with random numbers
    std::vector<std::vector<float>> he_init_biases();
    // sets the input matrix
    void set_inputs( std::vector<std::vector<std::vector<float>>> inputs);
    // sets the weights matrix
    void set_weights( std::vector<std::vector<float>> weights);
    // sets the biases matrix
    void set_biases( std::vector<std::vector<float>> biases);
    // returns the input matrix
    std::vector<std::vector<std::vector<float>>> get_inputs();
    // returns the weights matrix
    std::vector<std::vector<float>> get_weights() const;
    // returns the biases matrix
    std::vector<std::vector<float>> get_biases() const;
    // returns the prediction using input, weights, and biases matrix
    std::vector<std::vector<std::vector<float>>> get_prediction() const;
    
    std::vector<std::vector<std::vector<float>>> getInput();
    // A Dummy override method that does nothing for Dense Layer class
    
};
#endif // !Dense_Layer_hpp

