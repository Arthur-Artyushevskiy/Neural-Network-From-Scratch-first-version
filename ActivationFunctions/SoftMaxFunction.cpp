

#include "SoftMaxFunction.hpp"

void SoftMaxFunction::softMax(const size_t& batch_size, const size_t& features, const matrix_f& input, matrix_f& input_copy, matrix_f& output) {
    std::vector<float> sums = SoftMaxSum(batch_size, features, input, input_copy, output);
    
    for(int row{0}; row < batch_size; row++){
        for(int col{0}; col < features; col++){
            output[row][col] = (pow(2.71828182845904523, input_copy[row][col]))/sums[row];
        }
    }

}


std::vector<float> SoftMaxFunction::SoftMaxSum(const size_t& batch_size, const size_t& features, const matrix_f& input, matrix_f& input_copy, matrix_f& output){
    std::vector<float> sums(input.size(),0);
    float max{input[0][0]};
    float sum{0.0f};
    input_copy = input;
    // finds the maximum among the numbers in the matrix
    
    for(int row{0}; row < batch_size; row++){
        max = input[row][0];
        sum = 0.0f;
        for(int col{0}; col < features; col++){
            if(max < input[row][col]){
                max = input[row][col];
            }
        }
        
        for(int col{0}; col < features; col++){
            input_copy[row][col] -= max;
        }
        
        
        for(int col{0}; col < features; col++){
            sum += pow(2.71828182845904523, input_copy[row][col]);
        }
        sums[row] = sum;
    }
return sums;
}
