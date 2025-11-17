//


#ifndef Load_Data_hpp
#define Load_Data_hpp
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
//#include <algorithm>
#include <random>
//#include <chrono>
//#include <numeric>
using namespace std;

// Separate object that holds the images and labels
struct MNISTData{
    vector<vector<float>> images;
    vector<vector<int>> int_images;
    vector<int> labels;
    vector<vector<int>> one_hot_labels;
};



MNISTData load_data(string filename);

void shuffleImages(MNISTData& data);

void print_example(MNISTData data, int ind);
#endif /* Load_Data_hpp */
