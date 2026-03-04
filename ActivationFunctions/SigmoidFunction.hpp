

#ifndef SigmoidFunction_hpp
#define SigmoidFunction_hpp

#include <stdio.h>
#include <vector>
using matrix_f = std::vector<std::vector<float>>;

class SigmoidFunction{
private:
    float singleSigmoid(float input);
    
public:
    matrix_f sigmoid(const size_t& batch_size, const size_t& features, const matrix_f& input, matrix_f& output);
    float sigmoid_Prime(float input);
};

#endif /* SigmoidFunction_hpp */
