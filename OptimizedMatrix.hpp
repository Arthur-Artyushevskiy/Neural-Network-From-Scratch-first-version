//
//  OptimizedMatrix.hpp
//  Neural Network
//
//  Created by Arthur on 14/12/25.
//

#ifndef OptimizedMatrix_hpp
#define OptimizedMatrix_hpp
#include <vector>
#include <stdio.h>
using namespace std;
template<typename T>
class OptimizedMatrix{
public:
    vector<T> oneDVector;
    vector<vector<T>> twoDVector;
    size_t depths,rows, cols;
    
    OptimizedMatrix(size_t rows, size_t cols){
        this->rows = rows;
        this->cols = cols;
    }
    
    OptimizedMatrix(size_t rows, size_t cols, size_t depths){
        this->rows = rows;
        this->cols = cols;
        this->depths = depths;
    }
    
    vector<T> twoDToOneD(vector<vector<T>>& input){
        if(input.empty()) return{};
        vector<T> output(rows*cols, 0);
        for(size_t row{0}; row < rows; row++){
            for(size_t col{0}; col < cols; col++){
                output[row * cols + col] = input[row][col];
            }
        }
        return output;
    }
    
    vector<vector<T>> threeDToTwoD(vector<vector<vector<T>>>& input){
        if(input.empty()) return{};
        vector<vector<T>> output;
        for(size_t depth{0}; depth < depths; depth++){
            for(size_t row{0}; row < rows; row++){
                
            }
        }
    }
};


#endif /* OptimizedMatrix_hpp */
