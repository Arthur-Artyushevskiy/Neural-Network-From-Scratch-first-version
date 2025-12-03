
#include "Dense_Layer.hpp"
#include <iostream>

// sets the random weights using the he initialization
vector<vector<float>> Dense_Layer::he_init_weights(){
    vector<vector<float>> weights(row, vector<float>(col, 0));
    // creates a standart deviation for the normal distribution to create random weights
    double stddev = sqrt(2.0 / (double)col);
    
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0.0, stddev);
    
    for(int row{0}; row < weights.size(); row++){
        for(int col{0}; col < weights[0].size(); col++){
            weights[row][col] = d(gen);
        }
    }
    return weights;
}

// sets the random biases using the he initialization
vector<vector<float>> Dense_Layer::he_init_biases(){
    vector<vector<float>> biases(row, vector<float>(1, 0));
    double stddev = sqrt(2.0 / (double) col);
    
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0.0, stddev);
    
    for(int row{0}; row < biases.size(); row++){
        for(int col{0}; col < biases[0].size(); col++){
            biases[row][col] = d(gen);
        }
    }
    return biases;
}

void Dense_Layer::set_inputs(vector<vector<vector<float>>>inputs){
    this->input = inputs;
}

void Dense_Layer::set_weights(vector<vector<float>> weights){
    this->weights = weights;
}

void Dense_Layer::set_biases(vector<vector<float>> biases){
    this->biases = biases;
}



vector<vector<float>> Dense_Layer::get_weights() const {
    return weights;
}

vector<vector<float>> Dense_Layer::get_biases() const {
    return biases;
}

vector<vector<vector<float>>> Dense_Layer::get_prediction() const{
    vector<vector<vector<float>>> output;
    for(int depth{0}; depth < input.size(); depth++){
        output.push_back(add_trasposed_vectors(multiply(weights, input[depth]), biases));
    }
    return output;
}

vector<vector<vector<float>>> Dense_Layer::getInput(){
    return input;
}
// there is an error where the gradient from the next layer does not change and stays the same initial gradient
vector<vector<vector<float>>> Dense_Layer::backward(const vector<vector<vector<float>>> & gradient_from_next_layer){
    
    vector<vector<vector<float>>>batch_d_weights(input.size(),vector<vector<float>>(d_weights.size(),vector<float>(d_weights[0].size(), 0)));
    
    vector<vector<vector<float>>> batch_d_biases(input.size(), vector<vector<float>>(d_biases.size(),vector<float>(1, 0)));
    vector<vector<vector<float>>> gradient_for_prev_layer = gradient_from_next_layer;
   
    for(int depth{0}; depth < input.size(); depth++){
        vector<vector<float>> transposed_input = transpose(input[depth]);
        // finds the gradient for weights knowing the transposed values from the next activation layer and multiplies with the gradient from the next layer. Important to remember that transposed_input is a n*1 matrix and gradient from next layer is also a n*1 matrix, because my multiply methods can only use a n*1 vector as a first parameter and a 2d  matrix as a second one. If not used carefully, this could lead to errors
        batch_d_weights[depth] = multiply(gradient_from_next_layer[depth], transposed_input);
        // creates a new gradient for the previous layer
        // creates a copy of weights and transposes it
        vector<vector<float>> transposed_weights = transpose(weights);
        // finds the gradient for the previous layer using a n*1 gradient matric and a 2D transposed weight matrix
        gradient_for_prev_layer[depth] = multiply(transposed_weights, gradient_from_next_layer[depth]);
    }
    batch_d_biases = gradient_from_next_layer;
   
    double sum{0.0};
    for(int row{0}; row < batch_d_weights[0].size(); row++){
        for(int col{0}; col < batch_d_weights[0][0].size(); col++){
            sum = 0.0;
            for(int depth{0}; depth < batch_d_weights.size(); depth++){
                sum += batch_d_weights[depth][row][col];
            }
            d_weights[row][col] = sum / (double) batch_d_weights.size();
        }
        
    }
    
    for(int row{0}; row < batch_d_biases[0].size(); row++){
        for(int col{0}; col < batch_d_biases[0][0].size(); col++){
            sum = 0.0;
            for(int depth{0}; depth < batch_d_biases.size(); depth++){
                sum += batch_d_biases[depth][row][col];
            }
            d_biases[row][col] = sum / (double) batch_d_biases.size();
        }
        
    }
    
    return gradient_for_prev_layer;
}

void Dense_Layer::SGD(float learning_rate){
    for(int row{0}; row < weights.size(); row++){
        for(int col{0}; col < weights[0].size(); col++){
            // updates the value of weights using the calculated gradient from the next layer
            weights[row][col] = weights[row][col] - learning_rate * d_weights[row][col];
        }
    }
    for(int row{0}; row < biases.size(); row++){
        biases[row][0] = biases[row][0] - learning_rate * d_biases[row][0];
    }
}

// this method updates the values of weights and biases for each dense layer
void Dense_Layer::update(float learning_rate,string OptimizationAlgorithm){
    if(OptimizationAlgorithm == "ADAM"){
        static int t = 0;
        
        const double epsilon = 1e-9;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double mVector;
        double vVector;
        
        t++;
       
        for(int row{0}; row < weights.size(); row++){
           
            for(int col{0}; col < weights[0].size(); col++){
                m_weights[row][col] = beta1* m_weights[row][col] + (1-beta1) * d_weights[row][col];
                v_weights[row][col] = beta2* v_weights[row][col] + (1-beta2) * pow(d_weights[row][col], 2);
                
                mVector = m_weights[row][col] / (1 - pow(beta1, t));
                vVector = v_weights[row][col] / (1 - pow(beta2, t));
                
                weights[row][col] = weights[row][col] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
            }
        }
       
        
        for(int row{0}; row < biases.size(); row++){
           
            m_biases[row][0] = beta1* m_biases[row][0] + (1-beta1) * d_biases[row][0];
            v_biases[row][0] = beta2* v_biases[row][0] + (1-beta2) * pow(d_biases[row][0], 2);
            
            mVector = m_biases[row][0] / (1 - pow(beta1, t));
            vVector = v_biases[row][0] / (1 - pow(beta2, t));
            
           biases[row][0] = biases[row][0] - (learning_rate / (epsilon + pow(vVector, 0.5))) * mVector;
        }
    }
    else if(OptimizationAlgorithm == "SGD"){
        SGD(learning_rate);
        
    }
    else{
        cout << "You didn't Enter 'ADAM' or 'SGD'. Therefore the model will use SGD for optimization." << endl;
        SGD(learning_rate);
    }
    
}

vector<vector<vector<float>>> Dense_Layer::forward(const  vector<vector<vector<float>>> & output_from_prev_layer){
    batch_num = input.size();
    input = output_from_prev_layer;
    return get_prediction();
}

void Dense_Layer::save_to_file(ofstream& file){
    
    if(!file.is_open()){
        cerr << "Error: file not open for writing!" << endl;
        return ;
    }
    
    file << "DENSE\n";
    
    file << weights.size() << " " << weights[0].size() << "\n";
    
    for(const auto& row : weights){
        for(float val : row){
            file << val << " ";
        }
    }
    file << "\n";
    
    file << biases.size() << " " << biases[0].size() << "\n";
    
    for(const auto& row : biases){
        for(float val : row){
            file << val << " ";
        }
    }
    file << "\n";
}

void Dense_Layer::load_layer(ifstream& file){
    
    if(!file.is_open()){
        cerr << "ERROR: Could not open the file!" << endl;
        return;
    }
    
    string name;
    int rows, cols;
    
    file >> name;
    if(name != "DENSE"){
        cerr << "ERROR: Could not find the DENSE Header! Found:" << name << endl;
        return;
    }
    
    file >> rows >> cols;
    cout << "Number of rows:" << rows << endl;
    cout << "Number of cols:" << cols << endl;
    weights.resize(rows);
    for(int row{0}; row < rows; row++){
        weights[row].resize(cols);
        
        for (int col{0}; col < cols; col++) {
            file >> weights[row][col];
        }
    }
    
    file >> rows >> cols;
    biases.resize(rows);
    for(int row{0}; row < rows; row++){
      biases[row].resize(cols);
        
        for (int col{0}; col < cols; col++) {
            file >> biases[row][col];
        }
    }

    
}


int Dense_Layer::getRow(){
    return row;
}
