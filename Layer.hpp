//
//  Layer.hpp
//  Neural Network
//
//  Created by Arthur on 05/10/25.
//

#ifndef Layer_hpp
#define Layer_hpp
#include <stdio.h>
#include <vector>
#include "Load_Data.hpp"
class Layer{
public:
    // IDEA: I need to delete most of the virtual functions because they are not that useful to use since my model only uses forward and backward virual methods
    virtual ~Layer()= default;
    // allows
    
    virtual vector<vector<vector<float>>> backward(const vector<vector<vector<float>>> & gradient_from_next_layer) = 0;
    
    virtual void update(float learning_rate,string OptimizationAlgorithm) = 0;
    
    virtual vector<vector<vector<float>>> forward(const vector<vector<vector<float>>> & output_from_prev_layer) = 0;
 
    virtual void save_to_file(ofstream& file) = 0;
    
    virtual void load_layer(std::ifstream& file) = 0;
};

#endif /* Layer_hpp */
