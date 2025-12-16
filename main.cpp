
#include "Dense_Layer.hpp"
#include "Matrix_Operations.hpp"
#include "Layer.hpp"
#include "Activation_Layer.hpp"
#include "Load_Data.hpp"
#include "LossFunctions.hpp"
#include "Model.hpp"
#include "BatchNorm.hpp"

using namespace std;
/*
 Currently 11/03/25: I was able to make a simple version of a neural network, but I need to find a way to save the model parameters to save the progress insted of training the model repeatedly. Also, It would be helpful to make modle stop based on a certain value of the average loss and be able to test the model using the test data set.
 */

int main(int argc, const char * argv[]) {
    
    MNISTData test = load_data("mnist_test.csv");
    //MNISTData train = load_data("mnist_train.csv");
    int input_layer_size = 784;
    int first_dense_layer_size = 64;
    int output_layer_size = 10;
    int k_factor_ReLu = 0.01;
    string optimization_algorithm = "ADAM";
    size_t training_size = test.images.size();
    bool autoCreate = 1;
    
    NeuralNetwork model = NeuralNetwork(input_layer_size, first_dense_layer_size, output_layer_size, k_factor_ReLu, optimization_algorithm, autoCreate);
    //model.start_training(train, training_size);
    /*
    model.addDense(input_layer_size, first_dense_layer_size);
    model.addBatchNorm(first_dense_layer_size);
    model.addActivation("leak_relu");
    model.addDense(first_dense_layer_size, 32);
    model.addBatchNorm(32);
    model.addActivation("leak_relu");
    model.addDense(32, 16);
    model.addBatchNorm(16);
    model.addActivation("leak_relu");
    model.addDense(16, 10);
    model.addActivation("leak_relu");
    
    model.start_training(train, 2048);
    */
   
    
    ifstream load("saved_parameters_experiment.txt");
    model.load_model(load);
    /*
    vector<vector<vector<float>>> batch_images;
    for(int ind{0}; ind < 32; ind++){
        batch_images.push_back(transpose(test.images[ind]));
    }
    batch_images = model.forward_pass(batch_images);
    
    for(int ind{0}; ind < batch_images.size(); ind++){
        print_matrix(batch_images[ind]);
        print_example(test, ind);
        
    }
    */
    
    double accuracy = model.evaluate_model(test, 32);
    cout << "The overall accuracy of the model is: " << accuracy << "%" << endl;
    
        
    return EXIT_SUCCESS;
}
