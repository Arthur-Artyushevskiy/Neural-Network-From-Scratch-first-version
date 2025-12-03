
#include "Matrix_Operations.hpp"
#include <iostream>
using namespace std;
// This method multiplies two matrices based on certain rule
vector<vector<float>> multiply(const vector<vector<float>>& matrix_A, const vector<vector<float>>& matrix_B){
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
    for (int rowA = 0; rowA < A_rows;rowA++) {
            for (int colB = 0; colB < B_cols; colB++) {
                float sum = 0.0f;
                
                for (int k = 0; k < A_cols; ++k) { // Loop over cols of A / rows of B
                    sum += matrix_A[rowA][k] * matrix_B[k][colB];
                }
                result[rowA][colB] = sum;
            }
        }
    return result;
}
// this method adds a bias vector because it is designed for only n*1 matrix
vector<vector<float>>add_trasposed_vectors(vector<vector<float>> Transposed_Vector, vector<vector<float>> TwoDMatrix){
    long num_row_vector = Transposed_Vector.size();
    long num_row_matrix = TwoDMatrix.size();
    long num_col_matrix = TwoDMatrix[0].size();
    
    if(num_row_vector != num_row_matrix){
        cout << "ERROR: the number of rows in your vector and matrix are not equal!" << endl;
        return Transposed_Vector;
    }
    vector<vector<float>> result(num_row_matrix, vector<float>(num_col_matrix, 0));
    
    for(int row{0}; row < num_row_matrix ; row++){

        for(int col{0}; col < num_col_matrix; col++){
            result[row][col] = TwoDMatrix[row][col] + Transposed_Vector[row][0];
        }
    }
    return result;
}

vector<vector<float>> subtract_trasposed_vectors( vector<vector<float>> Transposed_Vector, vector<vector<float>> TwoDMatrix){
    long num_row_vector = Transposed_Vector.size();
    long num_row_matrix = TwoDMatrix.size();
    long num_col_matrix = TwoDMatrix[0].size();
    
    if(num_row_vector != num_row_matrix){
        cout << "ERROR: the number of rows in your vector and matrix are not equal!" << endl;
        return Transposed_Vector;
    }
    vector<vector<float>> result(num_row_matrix, vector<float>(num_col_matrix, 0));
    
    for(int row{0}; row < num_row_matrix ; row++){

        for(int col{0}; col < num_col_matrix; col++){
            result[row][col] = TwoDMatrix[row][col] - Transposed_Vector[row][0];
        }
    }
    return result;

}

// an element_wise_multiplication that I didn't use because of simplifications in the back propagation process
vector<vector<float>> element_wise_multiplication(vector<vector<float>> Transposed_Vector, vector<vector<float>> Transposed_Vector_2){
    size_t num_row_vector_1 = Transposed_Vector.size();
    
    size_t num_row_vector_2 = Transposed_Vector_2.size();
   
    if(num_row_vector_1 != num_row_vector_2){
        cout << "ERROR: the number of rows in your first vector and second vector are not equal!" << endl;
        return Transposed_Vector;
    }
    
    vector<vector<float>> result(num_row_vector_1, vector<float>(1, 0));
    
    for(int row{0}; row < num_row_vector_1 ; row++){
        result[row][0] = Transposed_Vector[row][0] * Transposed_Vector_2[row][0];
        
    }
    return result;
}
// switches the dimension of a 2D matrix
vector<vector<float>> transpose(vector<vector<float>> input){
    vector<vector<float>> output(input[0].size(), vector<float>(input.size(), 0));
    for(int row{0}; row < output.size() ; row++){
        for(int col{0}; col < output[0].size(); col++){
            output[row][col] = input[col][row];
        }
    }
    return output;
}
// switches the dimensions of a vector
vector<vector<float>> transpose(vector<float>input){
    vector<vector<float>> output(input.size(), vector<float>(1, 0));
    for(int row{0}; row < output.size(); row++){
        output[row][0] = input[row];
        
    }
    return output;
}
// prints out a 2D matrix 
const void print_matrix(vector<vector<float>> matrix){
    int count = 0;
    for(vector<float> row : matrix){
        for(float num : row){
            cout << count <<": "<<num << ", ";
            count++;
        }
        cout << endl;
    }
    cout << endl;
}
