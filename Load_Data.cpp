
#include "Load_Data.hpp"



MNISTData load_data(string filename){
    MNISTData data;
    string line;
    ifstream file(filename);
    
    // checks if the file was able to open
    if(!file.is_open()){
        cerr << "Error opening file!" << endl;
    }
    
    while(getline(file, line)){
        stringstream ss(line);
        string value;
        // get the label for each line (basically image)
        getline(ss, value, ',');
        try{
            // if successfully then the int if the label is added to the label vector
            data.labels.push_back(stoi(value));
            // based on the value of the label, the program adds the one-hot encoded vector
            switch(stoi(value)){
                case 0:
                    data.one_hot_labels.push_back({1,0,0,0,0,0,0,0,0,0});
                    break;
                case 1:
                    data.one_hot_labels.push_back({0,1,0,0,0,0,0,0,0,0});
                    break;
                case 2:
                    data.one_hot_labels.push_back({0,0,1,0,0,0,0,0,0,0});
                    break;
                case 3:
                    data.one_hot_labels.push_back({0,0,0,1,0,0,0,0,0,0});
                    break;
                case 4:
                    data.one_hot_labels.push_back({0,0,0,0,1,0,0,0,0,0});
                    break;
                case 5:
                    data.one_hot_labels.push_back({0,0,0,0,0,1,0,0,0,0});
                    break;
                case 6:
                    data.one_hot_labels.push_back({0,0,0,0,0,0,1,0,0,0});
                    break;
                case 7:
                    data.one_hot_labels.push_back({0,0,0,0,0,0,0,1,0,0});
                    break;
                case 8:
                    data.one_hot_labels.push_back({0,0,0,0,0,0,0,0,1,0});
                    break;
                case 9:
                    data.one_hot_labels.push_back({0,0,0,0,0,0,0,0,0,1});
                    break;
            }
        }
        catch(const invalid_argument& e){
            cerr << "Skipping missing labels" << endl;
            continue;
        }
        
        vector<float> current_image;
        vector<int> current_int_image;
        current_image.reserve(784);
        // take each pixel of the current image
        while(getline(ss, value, ',')){
            try{
                // if successfully then the int value of the pixel is addeed to the vector current image
                current_image.push_back(static_cast<float>(stoi(value)) / 255.0f);
                current_int_image.push_back(stoi(value));
            }
            catch(const invalid_argument&(e)){
                cerr << "Invalid pixel value" << endl;
            }
            
        }
        // checks if there are no missing pixels in the image
        if(current_image.size() == 784){
            data.images.push_back(current_image);
        }
        if(current_int_image.size() == 784){
            data.int_images.push_back(current_int_image);
        }
        else{
            cerr << "Skipping row with incorrent pixel count" << endl;
        }
    }
    // closes the file after finsishing
    file.close();
    return data;
    
}
// this method just prints examples of the data set
void print_example(MNISTData data, int ind){
   
    if(ind > data.images.size()-1){
        cerr << "ERROR: Can not use the index due to out of bounds!" << endl;
    }
    
    
    cout << "Succesfully loaded " << data.int_images.size() << " images" << endl;
    cout << "Number of labels: " << data.labels.size() << endl;
    
    cout << "\n --- First Image ---" << endl;
   
    
        cout << "Label: " << data.labels[ind] << endl;
        cout << "One-Hot Label: ";
        for(int value : data.one_hot_labels[ind]){
            cout << value << ", ";
        }
        cout << endl;
        cout << "Pixel count: " << data.int_images[ind].size() << endl;
        
        cout << "ASCII Art: " << endl;
        for(int row{0}; row < 28; row++){
         for(int col{0}; col < 28; ++col){
           
            int pixel_value = data.int_images[ind][row*28 + col];
            if(pixel_value > 200){
                cout << "##";
            }
            else if(pixel_value > 50){
                cout << "::";
            }
            else{
                cout << "  ";
            }
            
        }
        cout << endl;
    }
   
}


void shuffleImages(MNISTData& data){
    vector<size_t> indices(data.images.size());
    iota(indices.begin(), indices.end(), 0);
    
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
       std::mt19937 g(seed);
    shuffle(indices.begin(), indices.end(), g);
    
    vector<vector<float>> shuffled_images = data.images;
    vector<vector<int>> shuffled_labels = data.one_hot_labels;
    
    for(int i{0}; i < indices.size(); i++ ){
        shuffled_images[i] = data.images[indices[i]];
        shuffled_labels[i] = data.one_hot_labels[indices[i]];
    }
    
    data.images = shuffled_images;
    data.one_hot_labels = shuffled_labels;
    
}


