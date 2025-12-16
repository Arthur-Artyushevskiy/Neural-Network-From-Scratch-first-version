
#ifndef Matrix_Operations_hpp
#define Matrix_Operations_hpp
#include <stdio.h>
#include "Layer.hpp"
using namespace std;

    

vector<vector<float>> multiply(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B);
vector<vector<float>> multiplyWith1D(const vector<vector<float>>& matrix_A, const vector<float>& matrix_B);
vector<vector<float>> add_bias_to_batch(const vector<vector<float>>& matrix, const vector<float>& bias);
vector<float> sum_dim0(const vector<vector<float>>& matrix);
vector<vector<float>> subtract_matrices(const vector<vector<float>>& A, const vector<vector<float>>& B);
vector<vector<float>> add_matrices(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B);
vector<vector<float>> element_wise_multiplication(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B);
vector<vector<float>> scalar_multiply(const vector<vector<float>>& matrix, float scalar);

// switches the dimension of a 2D matrix

vector<vector<float>> transpose(vector<vector<float>>& input);
// switches the dimensions of a vector

vector<vector<float>> transpose(vector<float>& input);

template<typename T>
// prints out a 2D matrix
const void print_matrix(vector<vector<T>>& matrix){
    int count = 0;
    for(vector<T> row : matrix){
        for(T num : row){
            cout << count <<": "<<num << ", ";
            count++;
        }
        cout << endl;
    }
    cout << endl;
}

#endif /* Matrix_Operations_hpp */
