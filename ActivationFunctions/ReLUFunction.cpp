

#include "ReLUFunction.hpp"


matrix_f ReLUFunction::ReLU(const float& k_factor, const size_t& batch_size, const size_t& features, matrix_f& output, const matrix_f& input){
    for(int row{0}; row < batch_size; row++){
        for(int col{0}; col < features; col++){
            if(input[row][col] < 0){
                output[row][col] = k_factor * input[row][col];
            }
            else{
                output[row][col] = input[row][col];
            }
        }
    }
   
    return output;
}

void ReLUFunction::backward(const size_t& batch_size, const size_t& features, const matrix_f& input, matrix_f& d_input, const float k_factor){
    for(size_t row{0}; row < batch_size; row++){
        for(size_t col{0}; col < features; col++){
            if(input[row][col] < 0){
                d_input[row][col] = k_factor;
            }
        }
    }
}
