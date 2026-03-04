
#ifndef batchOperationsBackward_hpp
#define batchOperationsBackward_hpp

#include <stdio.h>
#include <vector>
#include <iostream>

using matrix_f = std::vector<std::vector<float>>;

class batchOperationsBackward{
private:
    struct parameters{
        const size_t batch_size;
        const size_t num_features;
        const matrix_f& input;
        std::vector<float>& batch_vars;
        const double epsilon;
    };

    parameters* setOfParam;
    
    

public:
    
    void setParameters(const size_t batch_size, const size_t num_features, const matrix_f& input, std::vector<float>& batch_vars, const double epsilon);
    
    void calculate_mean_correction(std::vector<float> & mean_correction, matrix_f& dLRespectdNormX);
    
    void calculate_var_correction(std::vector<float> & var_correction, std::vector<float>& batch_means, matrix_f& dLRespectdNormX);
    
    
    void calculate_batch_d_gamma(const matrix_f& gradient_from_next_layer, std::vector<float>& d_gamma, const matrix_f& normalized_input, std::vector<float>& gamma, matrix_f& dLRespectdNormX);
    
    void calculate_batch_d_beta(const matrix_f& gradient_from_next_layer, std::vector<float>& d_beta);
    
    void calculate_d_input(const matrix_f& gradient_from_next_layer, std::vector<float>& batch_means, matrix_f& d_input, matrix_f& dLRespectdNormX);
    
};


#endif /* batchOperationsBackward_hpp */
