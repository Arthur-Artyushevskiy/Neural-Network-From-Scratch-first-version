

#ifndef LossFunctions_hpp
#define LossFunctions_hpp
#include <stdio.h>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
using namespace std;

float mean_square_error(vector<float> prediction, vector<float> desired);

float categorical_cross_softmax(vector<float> prediction, vector<int> one_hot_label);

vector<float> mean_square_error_prime(vector<float> prediction, vector<float> desired);

vector<float> get_loss_gradient(vector<int> one_hot_labels, vector<float> prediction);

int get_predicted_label(vector<float>& prediction);

int get_true_label(const vector<int>& desired);
#endif // !LossFunctions_hpp
