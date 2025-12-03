#include "Model.hpp"
/* Need to have a concrete idea on how can i add dense layers and activation layers. The main issue is that I have no idea on how to create a design that allows me to only use one parameter to add the dense layer and somehow get the number of nuerons from the previous layer.
void NeuralNetwork::addDense(int numNeurons){
    model.push_back(make_unique<Dense_Layer>(model[model.size()- 1]))
}
*/



// this method creates a neural network with the initial number of neurons
vector<unique_ptr<Layer>> NeuralNetwork::createModel(int input_size, int num, int finalNum, int k_factor ){
    vector<unique_ptr<Layer>> model;
    vector<int> num_of_neurons{};
    while(num > finalNum){
        num_of_neurons.push_back(num);
        num = num/2;
    }
   
    model.push_back(make_unique<Dense_Layer>(input_size, num_of_neurons[0]));//0
    model.push_back(make_unique<BatchNorm>(num_of_neurons[0]));
    model.push_back(make_unique<Activation_Layer>("leak_relu")); //1
    for(int i{1}; i < num_of_neurons.size(); i++){
        
        int prev_layer_size = num_of_neurons[i-1];
        model.push_back(make_unique<Dense_Layer>(prev_layer_size, num_of_neurons[i])); // 2
        model.push_back(make_unique<BatchNorm>(num_of_neurons[i]));
        model.push_back(make_unique<Activation_Layer>("leak_relu")); // 3
        
        
    }
    int prev_layer_size =num_of_neurons[num_of_neurons.size()-1];
    model.push_back(make_unique<Dense_Layer>(prev_layer_size, finalNum)); // 8
    model.push_back(make_unique<Activation_Layer>("softmax"));//9
    return model;
}

void NeuralNetwork::addDense(int numNeuronsInput, int numNeuronsOuput){
    model.push_back(make_unique<Dense_Layer>(numNeuronsInput, numNeuronsOuput));//0
}

void NeuralNetwork::addBatchNorm(int numNeurons){
    model.push_back(make_unique<BatchNorm>(numNeurons));
}

void NeuralNetwork::addActivation(string activation_function){
    model.push_back(make_unique<Activation_Layer>(activation_function));
}

// A method that allows to train the model
void  NeuralNetwork::back_propagation(vector<vector<int>> one_hot_labels, float learning_rate, vector<vector<vector<float>>> output){
   
        // takes the prediction after going through the softmax function
    vector<vector<vector<float>>> prediction = output;
        // starts the gradient chain by calculating the gradient of the loss function and softmax
    vector<vector<vector<float>>> gradient_chain;
    // takes each image and calcualtes the initial gradient for the model
    for(int depth{0}; depth < prediction.size(); depth++){
       gradient_chain.push_back(get_loss_gradient(one_hot_labels[depth], prediction[depth]));
    }
    // does a backward step for each layer and updates the parameters of the model
    for(int i = model.size()-2; i >= 0; i--){
        gradient_chain = model[i]->backward(gradient_chain);
    }
    
    // updates the parameters
    for(int i{0}; i < model.size(); i++){
        model[i]->update(learning_rate, OptimizationAlgorithm);
    }
}

// this method is designed to traverse through the model to get the prediction and pass it to the back_propagation method
vector<vector<vector<float>>>  NeuralNetwork::forward_pass(vector<vector<vector<float>>> images ){
    vector<vector<vector<float>>> current_output = images;
   for(int layer{0}; layer < model.size(); layer++){
            current_output = model[layer]->forward(current_output);
        }
    return current_output;
}

// this method is designed to start the btach training by taking a btach of a certain size
pair<double, double>  NeuralNetwork::start_batch_training (vector<vector<vector<float>>> batch_images, vector<vector<int>>batch_one_hot_labels){
    vector<vector<vector<float>>> output;
    // just to be safe to prevent possible error that could happen
    if(batch_images.size() != batch_one_hot_labels.size()){
        //cout << "ERROR: The size of your images batch is not equal to the size of the batch of labels!!!";
        return {0.0, 0.0};
    }
    int correct_prediction = 0;
    float total_loss{0.0f};
    
    // passes through the model to get the result as a softmax matrix or in other words logit
    output = forward_pass(batch_images);
    // calculates the total loss for the current batch
    for(int depth{0}; depth < batch_images.size(); depth++){
        total_loss += categorical_cross_softmax(output[depth], batch_one_hot_labels[depth]);
       
        int predicted_label = get_predicted_label(output[depth]);
        int true_label = get_true_label(batch_one_hot_labels[depth]);
        
        if(predicted_label == true_label) correct_prediction++;

    }
    
    back_propagation(batch_one_hot_labels, 0.001, output);
    
    // FIX: Return TOTALS, not averages
    return {total_loss, (double)correct_prediction};
}

void NeuralNetwork::start_training(MNISTData& train, int numberOfImages){
    vector<vector<vector<float>>> batch_images;
    vector<vector<int>>batch_one_hot_labels;
    pair<double, double> performance;
    double total_loss{0.0};
    double total_accuracy{0.0};
    string progress_bpar;
    int batch_limit = numberOfImages; // the final version should be equal to train.images.size()
    cout << "WELCOME! This is my first neural network!" << endl;
 
    
    cout << "The number of layers in the model" << endl;
    cout << model.size() << endl;
    int batch_size;
    
    cout << "Input the batch size:";
    cin >> batch_size;
    int num_epochs;
    int images_processed_in_epoch = 0;
    cout << "Input the number of epochs:";
    cin >> num_epochs;
    
    for(int epoch{1}; epoch <= num_epochs; epoch++){
        shuffleImages(train);
        total_loss = 0.0;
        total_accuracy = 0.0;
        int batch_count = 0;
        images_processed_in_epoch = 0;
        cout << "Epoch: " << epoch;
        int total_batches = batch_limit / batch_size;
        
        // Improved loop to handle batches correctly, even if not perfectly divisible
        for(int batch_start = 0; batch_start < batch_limit; batch_start += batch_size){
            
            cout << "\rProgress Bar: " << print_progress_bar(batch_count + 1, total_batches) << flush;
            // this_thread::sleep_for(chrono::milliseconds(50)); // Optional sleep
            
            int batch_end = min(batch_start + batch_size, batch_limit);

            for(int ind = batch_start; ind < batch_end; ind++){
                 batch_images.push_back(transpose(train.images[ind]));
                 batch_one_hot_labels.push_back(train.one_hot_labels[ind]);
            }

            if (!batch_images.empty()) {
                 performance = start_batch_training(batch_images, batch_one_hot_labels);
                 total_loss += performance.first;
                 total_accuracy += performance.second;
                 images_processed_in_epoch += batch_images.size();
            }

            batch_images.clear();
            batch_one_hot_labels.clear();
            batch_count++;
        }
        
        cout << endl;
        // FIX: Calculate final epoch stats correctly by dividing by total images processed
        if (images_processed_in_epoch > 0) {
            cout << "Average Loss: " << total_loss / images_processed_in_epoch
                 << ", Average Accuracy: " << (total_accuracy / images_processed_in_epoch) * 100.0 << "%" << endl;
        } else {
            cout << "No images processed in this epoch." << endl;
        }
        cout << endl;
    }
    string save_parameters = "saved_parameters_experiment.txt";
    ofstream save(save_parameters);

    save_model(save);
}


string NeuralNetwork::print_progress_bar(double num, double batch_size){
    string output;
    // Prevent division by zero if batch_size is 0
    if (batch_size == 0) return "[ERROR] Batch size is 0";
    
    double percentage = min((num/batch_size) * 100.0, 100.0); // Clamp to 100%
    ostringstream oss;
        oss << fixed << setprecision(2) << percentage; // 1 decimal place
        string percentage_str = oss.str();
    output += "[";
    for(int count{0}; count < 10; count++){
        if(count < (int)(percentage / 10)){
            output += "|";
        }
        else{
            output += " ";
        }
     
    }
    output += "] Percentage Done: ";
    output += percentage_str + "%";
    return output;
}

void  NeuralNetwork::save_model(ofstream& file){
    if(!file.is_open()){
        cerr << "ERROR: Could not open the file to save the parameters." << endl;
        return;
    }

    for(auto& layer : model){
        if(file.eof()){ // Check EOF before writing might not be necessary here depending on stream state, but good practice in loops reading
             // Usually EOF is checked AFTER a failed read. For writing, check 'fail()' or 'bad()'
             if (file.fail()) {
                 cerr << "Error writing to file." << endl;
                 break;
             }
        }
        layer->save_to_file(file);
    }
    file.close();
    cout << "Paramters are saved successfully!" << endl;
}

void  NeuralNetwork::load_model(ifstream& file){
    if(!file.is_open()){
        cerr << "ERROR: Could not open the file to load the parameters." << endl;
        return;
    }
    
    for(auto& layer: model){
        // Better EOF check for reading loop
        if(file.peek() == EOF){
             break;
        }
        layer->load_layer(file);
    }
    
    file.close();
    cout << "Paramters are loaded successfully!" << endl;
    
}

double  NeuralNetwork::evaluate_model(MNISTData& train, int batch_size){
    cout << "Evaluation starts right now." << endl;
    int correct_prediction = 0;
    int size = train.images.size(); // Use actual size of training data passed
    if (size == 0) return 0.0;

    vector<vector<vector<float>>> batch_images;
    // vector<vector<int>> batch_one_hot_labels; // Not strictly needed if you just use train.one_hot_labels directly by index, but cleaner to keep consistent with training
    
    for(int batch_start = 0; batch_start < size; batch_start += batch_size){
        int batch_end = min(batch_start + batch_size, size);
        
        for(int ind = batch_start; ind < batch_end; ind++){
            batch_images.push_back(transpose(train.images[ind]));
            // batch_one_hot_labels.push_back(train.one_hot_labels[ind]);
        }
        
        vector<vector<vector<float>>> current_output = forward_pass(batch_images);
        
        for(int depth{0}; depth < current_output.size(); depth++){
            int predicted_label = get_predicted_label(current_output[depth]);
            int true_label_index = batch_start + depth;
            int true_label = get_true_label(train.one_hot_labels[true_label_index]);

            if(predicted_label == true_label){
                correct_prediction++;
            }
        }
        batch_images.clear();
        // batch_one_hot_labels.clear();
    }
    return ((double)correct_prediction / (double)size) * 100.0;
}
