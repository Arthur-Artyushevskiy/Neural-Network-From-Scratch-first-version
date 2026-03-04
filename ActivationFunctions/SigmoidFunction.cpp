

#include "SigmoidFunction.hpp"

float SigmoidFunction::singleSigmoid(float input){
    return 1/(1+ pow(2.71828182845904523,input));
}

float SigmoidFunction::sigmoid_Prime(float input){
    return singleSigmoid(input)/ (1-singleSigmoid(input));
}

matrix_f SigmoidFunction::sigmoid(const size_t& batch_size, const size_t& features, const matrix_f& input, matrix_f& output){
    for(int row{0}; row < batch_size; row++){
        for(int col{0}; col < features; col++){
            output[row][col]= singleSigmoid(input[row][col]);
        }
    }
    return output;
}
