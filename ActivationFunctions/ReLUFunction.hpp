

#ifndef ReLUFunction_hpp
#define ReLUFunction_hpp

#include <stdio.h>
#include <vector>
using matrix_f = std::vector<std::vector<float>>;

class ReLUFunction{
public:
    // the ReLu method for the whole matrix
    matrix_f ReLU(const float& k_factor, const size_t& batch_size, const size_t& features, matrix_f& output, const matrix_f& input);
    
    void backward(const size_t& batch_size, const size_t& features, const matrix_f& input, matrix_f& d_input, const float k_factor);
};

#endif /* ReLUFunction_hpp */
