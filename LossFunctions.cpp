
#include "LossFunctions.hpp"
// a mean square error that could be usefull for a regression problem
float mean_square_error(vector<vector<float>> prediction, vector<vector<float>> desired){
    float error =0;
    for(int row{0}; row < prediction.size(); row++){
        error += pow((desired[row][0] - prediction[row][0]), 2);
        
    }
    return (error/prediction.size());
}

// a softmax method that is designed for this classification problem
float categorical_cross_softmax(vector<vector<float>> prediction, vector<int> one_hot_label){
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
    error -= log(max(prediction[label][0], epsilon));
    //error -= log(prediction[label][0]);
    return error;
}

// a prime of the mean square error loss function
vector<vector<float>> mean_square_error_prime(vector<vector<float>> prediction, vector<vector<float>> desired){
    vector<vector<float>> result(prediction.size(), vector<float>(1,0));
     for(int row{0}; row < prediction.size(); row++){
         result[row][0] = (2/prediction.size())*(prediction[row][0] - desired[row][0]);
         
     }
     return result;
}

// returns a loss gradient for categorical cross softmax
vector<vector<float>> get_loss_gradient(vector<int> one_hot_label, vector<vector<float>> prediction){
    vector<vector<float>> d_z_Loss(prediction.size(), vector<float>(1,0));
    for(int row{0}; row <one_hot_label.size(); row++){
        d_z_Loss[row][0] = prediction[row][0] - one_hot_label[row];
    }
    return d_z_Loss;
}

int get_predicted_label(vector<vector<float>>& prediction){
    float max = -numeric_limits<float>::infinity();
    int label{0};
    for(int row{0}; row < prediction.size(); row++){
        if(prediction[row][0] > max){
            max =prediction[row][0];
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


