
#ifndef batchOperationsForward_hpp
#define batchOperationsForward_hpp

#include <stdio.h>
#include <vector>

using matrix_f = std::vector<std::vector<float>>;

class batchOperationsForward{
private:
    struct parameters{
        const size_t batch_size;
        const size_t num_features;
        matrix_f& input;
    };
     
    parameters* setOfParam;
    
public:
    
    void setParameters(const size_t batch_size, const size_t num_feature, matrix_f& input){
        setOfParam = new parameters(batch_size, num_feature, input);
    }
    
    void calculate_batch_mean(std::vector<float>& batch_means);
    
    void calculate_batch_var(std::vector<float>& batch_vars, std::vector<float>& batch_means);
    
    void calculate_norm_input(matrix_f& normalized_input, std::vector<float>& batch_means, std::vector<float>& batch_vars, const double epsilon);
    
    void calculate_output(matrix_f& output, const matrix_f& normalized_input, std::vector<float>& beta,  std::vector<float>& gamma);
    
    
    
};

#endif /* batchOperationsForward_hpp */
