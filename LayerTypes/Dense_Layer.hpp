
#ifndef Dense_Layer_hpp
#define Dense_Layer_hpp
#include <stdio.h>
#include "Matrix_Operations.hpp"
#include "Layer.hpp"

using matrix_f = std::vector<std::vector<float>>;
using matrix_d = std::vector<std::vector<double>>;
class Dense_Layer : public Layer{
private:
    
    int input_features;
    int output_features;
    
    matrix_f input; // [Batch_Size x Input_Features]
    
    matrix_f weights; // [Input_Features x Ouput_Features]
    
    vector<float> biases; // the biases matrix R
    // A matrix for the weight gradient
    matrix_f d_weights; // the gradient of weights R x C
    // A matrix for the bias gradient
    vector<float> d_biases; // the gradien of biases R
    
    matrix_d m_weights,v_weights; // the mvector and vvector (used for ADAM) for weights R x C
    
    vector<double> m_biases,v_biases; // the mvector and vvector (used for ADAM) for biases R x 1
    
public:
   
   Dense_Layer(int input_size, int output_size) : input_features(input_size), output_features(output_size){
        weights.resize(input_features, vector<float>(output_features));
        biases.resize(output_features, 0.0f);
        
        d_weights = weights;
        d_biases  = biases;
        
        m_weights.resize(input_features, vector<double>(output_features, 0.0));
        v_weights.resize(input_features, vector<double>(output_features, 0.0));
        
        m_biases.resize(output_features, 0.0);
        v_biases.resize(output_features, 0.0);
        
        he_init();
    }
   
    matrix_f forward(const  matrix_f& output_from_prev_layer) override;
    // One of the  most important functions that calculates the gradient and adjusts the weights and biases
    matrix_f backward(const matrix_f& gradient_from_next_layer) override;
   
    // updates the weights and biases using the hyperparameter
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    void SGD(float learning_rate);
    // this method is important that will go through the model the pass the output of the previous layer to this current layer as an input
    
    // saves the weights and biases for the current dense layer object
    void save_to_file(ofstream& file) override;
   
    // loads the parameters for the current dense layer object
    void load_layer(ifstream& file) override;
    
    
    void he_init();
    
    
    // returns the prediction using input, weights, and biases matrix
    matrix_f get_prediction() const;
    
};
#endif // !Dense_Layer_hpp

