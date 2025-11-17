
#include "Dense_Layer.hpp"
#include "Matrix_Operations.hpp"
#include "Layer.hpp"
#include "Activation_Layer.hpp"
#include "Load_Data.hpp"
#include "LossFunctions.hpp"
#include "Model.hpp"


using namespace std;
/*
 Currently 11/03/25: I was able to make a simple version of a neural network, but I need to find a way to save the model parameters to save the progress insted of training the model repeatedly. Also, It would be helpful to make modle stop based on a certain value of the average loss and be able to test the model using the test data set.
 */

int main(int argc, const char * argv[]) {
    
    //MNISTData test = load_data("mnist_test.csv");
    MNISTData train = load_data("mnist_train.csv");
    NeuralNetwork model = NeuralNetwork(784, 64, 10, 0.01, "ADAM");
    //model.start_training(test, test.images.size());
    
    ifstream load("saved_parameters_experiment.txt");
    model.load_model(load);
    double accuracy = model.evaluate_model(train, 32);
    cout << "The overall accuracy of the model is: " << accuracy << "%" << endl;
    
    return EXIT_SUCCESS;
}
