//
//  SoftMaxFunction.hpp
//  Neural Network
//
//  Created by Arthur on 01/03/26.
//

#ifndef SoftMaxFunction_hpp
#define SoftMaxFunction_hpp

#include <stdio.h>
#include <vector>

using matrix_f = std::vector<std::vector<float>>;

class SoftMaxFunction{

    
public:
    
    void softMax(const size_t& batch_size, const size_t& features, const matrix_f& input,  matrix_f& input_copy, matrix_f& output);
    
    std::vector<float> SoftMaxSum(const size_t& batch_size, const size_t& features, const matrix_f& input,matrix_f& input_copy, matrix_f& output);
};

#endif /* SoftMaxFunction_hpp */
