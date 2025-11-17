
#ifndef Matrix_Operations_hpp
#define Matrix_Operations_hpp
#include <stdio.h>
#include "Layer.hpp"
using namespace std;

    

vector<vector<float>> multiply(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B);
vector<vector<float>> add_bias_vector( vector<vector<float>> Transposed_Vector, vector<vector<float>> TwoDMatrix);
vector<vector<float>> element_wise_multiplication(vector<vector<float>> Transposed_Vector, vector<vector<float>> Transposed_Vector_2);
vector<vector<float>> transpose(vector<vector<float>> input);
vector<vector<float>> transpose(vector<float>input);
const void  print_matrix(vector<vector<float>> matrix);
#endif /* Matrix_Operations_hpp */
