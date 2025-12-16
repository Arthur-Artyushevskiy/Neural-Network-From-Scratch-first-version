
#include "LossFunctions.hpp"
// a mean square error that could be usefull for a regression problem
float mean_square_error(vector<float> prediction, vector<float> desired){
    float error =0;
    for(int row{0}; row < prediction.size(); row++){
        error += pow((desired[row] - prediction[row]), 2);
        
    }
    return (error/prediction.size());
}

// a softmax method that is designed for this classification problem
float categorical_cross_softmax(vector<float> prediction, vector<int> one_hot_label){
    float error = 0.00;
    int label{};
    const float epsilon = 1e-9f;
    for(int row{0}; row < one_hot_label.size(); row++){
        if(one_hot_label[row] != 0){
            label = row;
            break;
        }
    }
    //takes the true class and computes the true class value  -log(true_class_value)
    error -= log(max(prediction[label], epsilon));
    //error -= log(prediction[label][0]);
    return error;
}

// a prime of the mean square error loss function
vector<float> mean_square_error_prime(vector<float> prediction, vector<float> desired){
    vector<float> result(prediction.size(), 0.0f);
     for(int row{0}; row < prediction.size(); row++){
         result[row] = (2/prediction.size())*(prediction[row] - desired[row]);
         
     }
     return result;
}

// returns a loss gradient for categorical cross softmax
vector<float> get_loss_gradient(vector<int> one_hot_label, vector<float> prediction){
    if(one_hot_label.size() != prediction.size()){
        cerr << "Error Different Dimensions" << endl;
        return {};
    }
    
    vector<float> d_z_Loss(prediction.size(), 0.0f);
    for(int row{0}; row <one_hot_label.size(); row++){
        d_z_Loss[row] = prediction[row] - one_hot_label[row];
    }
    return d_z_Loss;
}

int get_predicted_label(vector<float>& prediction){
    float max = -numeric_limits<float>::infinity();
    int label{0};
    for(int row{0}; row < prediction.size(); row++){
        if(prediction[row] > max){
            max =prediction[row];
            label = row;
        }
    }
    return label;
}
int get_true_label(const vector<int>& desired){
    for(int row{0}; row < desired.size(); row++){
        if(desired[row] == 1){
            return row;
        }
    }
    return -1;
}


