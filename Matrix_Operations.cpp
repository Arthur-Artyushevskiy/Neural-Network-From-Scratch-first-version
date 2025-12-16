
#include "Matrix_Operations.hpp"
#include <iostream>
using namespace std;
// This method multiplies two matrices based on certain rule
vector<vector<float>> multiply(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B){
    if(matrix_A.empty() || matrix_B.empty()) return{};
    
    
        size_t A_rows = matrix_A.size();
        
        size_t A_cols = matrix_A[0].size();
        
        size_t B_rows = matrix_B.size();
        
        size_t B_cols = matrix_B[0].size();
   
    // this is the rule
    if (A_cols != B_rows) {
            cout << "ERROR: Matrix dimensions are incompatible for multiplication!" << endl;
            cout << "A_cols (" << A_cols << ") != B_rows (" << B_rows << ")" << endl;
            return {};
        }
    
    // the mulitplication portion
    vector<vector<float>> result(A_rows, vector<float>(B_cols, 0.0f));
    // need to learn more on how the math works for cache effiecincy
    vector<vector<float>> B_T(B_cols, vector<float>(B_rows));
    for(size_t B_row{0}; B_row < B_rows; B_row++){
        for(size_t B_col{0}; B_col < B_cols; B_col++){
            B_T[B_col][B_row] = matrix_B[B_row][B_col];
        }
    }
    
    
    for (size_t rowA{0}; rowA < A_rows; rowA++) {
        for (size_t colB{0}; colB < B_cols; colB++) {
                float sum = 0.0f;
                for (size_t k{0}; k < A_cols; ++k) { // Loop over cols of A / rows of B
                    sum += matrix_A[rowA][k] * B_T[colB][k];
                }
                result[rowA][colB] = sum;
            }
        }
    return result;
}



// this method adds a bias vector because it is designed for only n*1 matrix
vector<vector<float>> add_bias_to_batch(const vector<vector<float>>& matrix, const vector<float>& bias){
    if (matrix.empty()) return {};
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    if(bias.size() != cols){
        cerr << "Error: Bias size (" << bias.size() << ") must match matrix columns (" << cols << ")" << endl;
        return matrix;
    }
    
    
    vector<vector<float>> result = matrix;
    
    for(size_t row{0}; row < rows ; row++){
      for(size_t col{0}; col < cols; col++){
          result[row][col] += bias[col];
        }
    }
    return result;
}

vector<float> sum_dim0(const vector<vector<float>>& matrix) {
    if (matrix.empty()) return {};
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    vector<float> sums(cols, 0.0f);
    
    for (size_t row{0}; row < rows; row++) {
        for (size_t col{0}; col < cols; col++) {
            sums[col] += matrix[row][col];
        }
    }
    return sums;
}

vector<vector<float>> subtract_matrices(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B) {
    size_t rows = matrix_A.size();
    size_t cols = matrix_A[0].size();
    
    if(rows != matrix_B.size() || cols != matrix_B[0].size()){
        cout << "ERROR: the number of rows in your first vector and second vector are not equal!" << endl;
        return matrix_A;
    }
    
    vector<vector<float>> result(rows, vector<float>(cols));
    for(size_t row{0}; row<rows; row++)
        for(size_t col{0}; col<cols; col++)
            result[row][col] = matrix_A[row][col] - matrix_B[row][col];
    return result;
}

// an element_wise_multiplication that I didn't use because of simplifications in the back propagation process
vector<vector<float>> element_wise_multiplication(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B){
    size_t rows = matrix_A.size();
    size_t cols = matrix_A[0].size();

    if(rows != matrix_B.size() || cols != matrix_B[0].size()){
        cout << "ERROR: the number of rows in your first vector and second vector are not equal!" << endl;
        return matrix_A;
    }
    

    vector<vector<float>> result(rows, vector<float>(cols, 0));
    
    for(size_t row{0}; row < rows ; row++){
        for(size_t col{0}; col < cols; col++){
            result[row][col] = matrix_A[row][col] * matrix_B[row][col];
        }
    }
    return result;
}

vector<vector<float>> add_matrices(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B) {
    size_t rows = matrix_A.size();
    size_t cols = matrix_A[0].size();
    
    if(rows != matrix_B.size() || cols != matrix_B[0].size()){
        cout << "ERROR: the number of rows in your first vector and second vector are not equal!" << endl;
        return matrix_A;
    }
    
    vector<vector<float>> result(rows, vector<float>(cols));
    for(size_t row{0}; row<rows; row++)
        for(size_t col{0}; col<cols; col++)
            result[row][col] = matrix_A[row][col] + matrix_B[row][col];
    return result;
}

vector<vector<float>> scalar_multiply(const vector<vector<float>>& matrix, float scalar) {
    vector<vector<float>> result = matrix;
    for(auto& row : result) {
        for(auto& val : row) val *= scalar;
    }
    return result;
}

vector<vector<float>> transpose(vector<vector<float>>& input){
    if(input.empty()) return{};
    size_t rows = input.size();
    size_t cols = input[0].size();
    vector<vector<float>> output(cols, vector<float>(rows, 0));
    for(size_t row{0}; row < rows ; row++){
        for(size_t col{0}; col < cols; col++){
            output[col][row] = input[row][col];
        }
    }
    return output;
}

vector<vector<float>> transpose(vector<float>& input){
    if(input.empty()) return{};
    size_t rows = input.size();
    vector<vector<float>> output(input.size(), vector<float>(1, 0));
    for(size_t row{0}; row < rows; row++){
        output[0][row] = input[row];
        
    }
    return output;
}
