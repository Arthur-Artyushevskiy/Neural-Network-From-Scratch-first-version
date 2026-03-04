
#ifndef Model_hpp
#define Model_hpp
#include <stdio.h>// Include for std::unique_ptr and std::make_unique
#include "Layer.hpp"
#include "Load_Data.hpp"
#include "Dense_Layer.hpp"
#include "Activation_Layer.hpp"
#include "Matrix_Operations.hpp"
#include "LossFunctions.hpp"
#include "BatchNorm.hpp"
using namespace std;

class NeuralNetwork{
private:
    vector<unique_ptr<Layer>> model;
    string OptimizationAlgorithm;
    vector<float> learning_rates;
    float current_learning_rate;
public:
    
    NeuralNetwork(int input_size, int FirstNumOfNeurons, int finalNumOfNeurons, int k_factor, string algorithm, bool autoCreate, vector<float> learning_rates){
        
        if(autoCreate) model = createModel(input_size, FirstNumOfNeurons, finalNumOfNeurons, k_factor);
        OptimizationAlgorithm = algorithm;
        this->learning_rates = learning_rates;
        if(learning_rates.size() > 0) current_learning_rate = learning_rates[0];
        else cout << "ERROR: Your list of learning rates is EMPTY. Put a list of size greater than 0." << endl;
    }
    
    void addDense(int numNeuronsInput, int numNeuronsOuput);
    
    void addBatchNorm(int numNeurons);
    
    void addActivation(string activation_function);
    
    vector<unique_ptr<Layer>> createModel(int input_size, int num, int finalNum, int k_factor);
    
    void back_propagation(vector<vector<int>> one_hot_labels, vector<vector<float>>& output);
        
   vector<vector<float>> forward_pass(vector<vector<float>>& images);
        
    pair<double, double> start_batch_training(vector<vector<float>>& batch_images, vector<vector<int>>& batch_one_hot_labels);
       
    void start_training(MNISTData& train, MNISTData& test, int numberOfImages);

    string print_progress_bar(double num, double batch_size);
     //vector<vector<float>> getResult( vector<unique_ptr<Layer>> model);

    void save_model(ofstream& file);

    void load_model(ifstream& file);
        
    double evaluate_model(MNISTData& test, int batch_size);
};



    


#endif // !Model_hpp
