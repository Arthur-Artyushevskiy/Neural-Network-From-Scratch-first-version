

#ifndef BatchNorm_hpp
#define BatchNorm_hpp
#include <stdio.h>
#include <cmath>
#include "Matrix_Operations.hpp"
#include "Layer.hpp"
using namespace std;
class BatchNorm : public Layer{
private:
    vector<vector<vector<float>>> input;
    vector<vector<vector<float>>> normalized_input;
    vector<vector<vector<float>>> output;
    
    vector<vector<float>> batch_means;
    vector<vector<float>> batch_vars;
   
    
    const double epsilon = 1e-9;
    
    vector<vector<float>> d_gamma;
    vector<vector<float>> d_beta;
    
    vector<vector<float>> m_gamma;
    vector<vector<float>> m_beta;
   
    vector<vector<float>> v_gamma;
    vector<vector<float>> v_beta;
    
    vector<vector<vector<float>>> d_input;
    
    vector<vector<vector<float>>> dLRespectdNormX;
    
    size_t rowSize;
    
    size_t colSize;
    
    size_t depthSize;
public:
    
    vector<vector<float>> gamma;
    vector<vector<float>> beta;
    
    BatchNorm(int rows){
        gamma.resize(rows, vector<float>(1, 1.0f));
        rowSize = rows;
        colSize = 1;
        d_gamma.resize(rows, vector<float>(1, 0.0f));
        beta.resize(rows, vector<float>(1, 0.0f));
        d_beta.resize(rows, vector<float>(1, 0.0f));
        m_gamma.resize(rowSize, vector<float>(colSize, 0.0f));
        v_gamma.resize(rowSize, vector<float>(colSize, 0.0f));
        m_beta.resize(rowSize, vector<float>(colSize, 0.0f));
        v_beta.resize(rowSize, vector<float>(colSize, 0.0f));
        
    }
    
    void calculate_batch_mean();
    
    void calculate_batch_var();
    
    void calculate_norm_input();
    
    void calculate_output();
    
    void calculate_mean_correction(vector<vector<vector<float>>> & mean_correction);
    
    void calculate_var_correction(vector<vector<vector<float>>> & var_correction);
    
    vector<vector<vector<float>>> forward(const std::vector<std::vector<std::vector<float>>> & output_from_prev_layer) override;
    
    
    void calculate_batch_d_gamma(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer);
    
    void calculate_batch_d_beta(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer);
    
    void calculate_d_input(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer);
    
    
    vector<vector<vector<float>>> backward(const std::vector<std::vector<std::vector<float>>> & gradient_from_next_layer) override;
    
    void update(float learning_rate,string OptimizationAlgorithm) override;
    
    void save_to_file(ofstream& file) override;
    
    void load_layer(std::ifstream& file) override;
    
    
    
    
    
};


#endif /* BatchNorm_hpp */
