
#ifndef Dense_Layer_hpp
#define Dense_Layer_hpp
#include <stdio.h>
#include "Matrix_Operations.hpp"
#include "Layer.hpp"
class Dense_Layer : public Layer{
private:
    
    vector<vector<vector<float>>> input; // the input batch B x R x 1
    
    vector<vector<float>> weights; // the weights matrix R x C
    
    vector<vector<float>> biases; // the biases matrix R x 1
    // A matrix for the weight gradient
    vector<vector<float>> d_weights; // the gradient of weights R x C
    // A matrix for the bias gradient
    vector<vector<float>> d_biases; // the gradien of biases R x 1
    
    vector<vector<double>> m_weights; // the mvector (used for ADAM) for weights R x C
    
    vector<vector<double>> v_weights; // the v vector (used for ADAM) for weights R x C
    
    vector<vector<double>> m_biases; // the mvector (used for ADAM) for biases R x 1
    
    vector<vector<double>> v_biases; // the v vector (used for ADAM) for biases R x 1
    
    
    int row; // the number of rows for the current matrix from the current layer that is set by the constructor
    
    int col; // the number of col for the current matrix from the previous layer is set by the constructor
    
    int batch_num; // the size of the current batch
    
public:
   
   
    Dense_Layer(int input_size,int row){
        this-> row = row;
        col = input_size;
        
        weights = he_init_weights();
        biases =  he_init_biases();
        d_weights = weights;
        d_biases  = biases;
        m_weights.resize(row, vector<double>(col, 0.0));
        v_weights.resize(row, vector<double>(col, 0.0));
        m_biases.resize(row, vector<double>(1, 0.0));
        v_biases.resize(row, vector<double>(1, 0.0));
    }
   
    int getRow();
    
    // One of the  most important functions that calculates the gradient and adjusts the weights and biases
    vector<vector<vector<float>>> backward(const vector<vector<vector<float>>> & gradient_from_next_layer) override;
    // updates the weights and biases using the hyperparameter
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    void SGD(float learning_rate);
    // this method is important that will go through the model the pass the output of the previous layer to this current layer as an input
    vector<vector<vector<float>>> forward(const  vector<vector<vector<float>>> & output_from_prev_layer) override;
    
    // saves the weights and biases for the current dense layer object
    void save_to_file(ofstream& file) override;
   
    // loads the parameters for the current dense layer object
    void load_layer(ifstream& file) override;
    
    
    
    
    
    // sets the weight matrix with random numbers
    vector<vector<float>> he_init_weights();
    // sets the bias matrix with random numbers
    vector<vector<float>> he_init_biases();
    // sets the input matrix
    void set_inputs(vector<vector<vector<float>>> inputs);
    // sets the weights matrix
    void set_weights(vector<vector<float>> weights);
    // sets the biases matrix
    void set_biases(vector<vector<float>> biases);
    // returns the weights matrix
    vector<vector<float>> get_weights() const;
    // returns the biases matrix
    vector<vector<float>> get_biases() const;
    // returns the prediction using input, weights, and biases matrix
    vector<vector<vector<float>>> get_prediction() const;
    
    vector<vector<vector<float>>> getInput();
    // A Dummy override method that does nothing for Dense Layer class
    
};
#endif // !Dense_Layer_hpp

