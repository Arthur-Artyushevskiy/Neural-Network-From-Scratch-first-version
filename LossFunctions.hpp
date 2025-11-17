

#ifndef LossFunctions_hpp
#define LossFunctions_hpp
#include <stdio.h>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;

float mean_square_error(vector<vector<float>> prediction, vector<vector<float>> desired);

float categorical_cross_softmax(vector<vector<float>> prediction, vector<int> one_hot_label);

vector<vector<float>> mean_square_error_prime(vector<vector<float>> prediction, vector<vector<float>> desired);

vector<vector<float>> get_loss_gradient(vector<int> one_hot_labels, vector<vector<float>> prediction);

int get_predicted_label(vector<vector<float>>& prediction);

int get_true_label(const vector<int>& desired);
#endif // !LossFunctions_hpp
