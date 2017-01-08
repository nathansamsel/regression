#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <assert.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <time.h>

//EXAMPLE and THRESHOLD used for printing a given example
#define EXAMPLE 41
#define THRESHOLD 200
#define LEARNING_RATE .1
#define INPUT_COUNT 784

// training set of 60k images of the numbers 0-9
// test set of 10k images of the numbers 0-9
// source:  http://yann.lecun.com/exdb/mnist/

//must be modified for proper location of data
std::string basePath = "C:\\Users\\Nate\\Documents\\Visual Studio 2015\\Projects\\regression\\regression\\";

//////////////////////
//					//
//  Image Data		//
//					//
//////////////////////

struct image {
	uint8_t get(int x, int y) const
	{
		return 255 - pixel[x + y * 28];
	}
	uint8_t pixel[28 * 28];
	uint8_t value;
};

//////////////////////
//					//
//  Loading Data	//
//					//
//////////////////////

void SwapEndian(uint32_t &val){
	val = ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) | ((val >> 8) & 0xFF00) |
		((val >> 24) & 0xFF);
}

void LoadData(std::vector<image> &images, const char *imageFile, const char *labelFile){
	uint32_t dataCount;
	std::string path = basePath;
	path += imageFile;
	FILE *f = fopen(path.c_str(), "rb");
	uint32_t tmp;

	fread(&tmp, sizeof(tmp), 1, f);
	SwapEndian(tmp);
	assert(tmp == 0x00000803); // magic number
	printf("Read: 0x%X\n", tmp);

	fread(&dataCount, sizeof(dataCount), 1, f);
	SwapEndian(dataCount);
	printf("Read: 0x%X\n", tmp); // num entries

	fread(&tmp, sizeof(tmp), 1, f);
	SwapEndian(tmp);
	assert(tmp == 28);
	printf("Read: 0x%X\n", tmp); // image size

	fread(&tmp, sizeof(tmp), 1, f);
	SwapEndian(tmp);
	assert(tmp == 28);
	printf("Read: 0x%X\n", tmp); // image size

	image i;
	for (int x = 0; x < dataCount; x++) // read image data
	{
		size_t read = fread(i.pixel, sizeof(uint8_t), 28 * 28, f);
		assert(read == 28 * 28);
		images.push_back(i);
	}
	fclose(f);

	path = basePath;
	if (path.back() != '/' && path.size() > 1)
		path.push_back('/');
	path += labelFile;
	fopen(path.c_str(), "rb");

	fread(&tmp, sizeof(tmp), 1, f);
	SwapEndian(tmp);
	assert(tmp == 0x00000801); // magic number
	printf("Read: 0x%X\n", tmp);

	fread(&tmp, sizeof(tmp), 1, f);
	SwapEndian(tmp);
	assert(tmp == dataCount); // num instances
	printf("Read: 0x%X\n", tmp);

	for (int x = 0; x < dataCount; x++) // labels
	{
		size_t read = fread(&images[x].value, sizeof(uint8_t), 1, f);
		assert(read == 1);
	}
	fclose(f);
}

////////////////////////
//					  //
// Logistic Regression//
//					  //
////////////////////////

class LogisticRegression {
public:
	LogisticRegression(int input_count, float learning_rate, uint8_t binary_target);
	float test(std::vector<uint8_t> features);
	void train(std::vector<uint8_t> features, uint8_t target);
	float hypothesis(std::vector<uint8_t> features);
	std::vector<float> weights;
	int input_count;
	uint8_t binary_target;
	float learning_rate;
private:
};

//MNIST dataset training should use 784 inputs
LogisticRegression::LogisticRegression(int input_count, float learning_rate, uint8_t binary_target) {
	//initialize
	this->input_count = input_count;
	this->learning_rate = learning_rate;
	this->binary_target = binary_target;
	//make sure weights is empty
	weights.clear();

	//input_count + 1 so there are weights for each input plus an intercept W0 (bias)
	for (int i = 0; i < input_count + 1; i++) {
		//initialize weight vector to any starting point in weight space
		//works because convex L2 loss function implies no local minima
		weights.push_back(.1f);
	}
}

//takes features x and returns a hypothesis h(x)
float LogisticRegression::test(std::vector<uint8_t> features) {
	return hypothesis(features);
}

//take features x and returns (1 / 1 + e ^ (-w * x)) where w is the vector of weights
float LogisticRegression::hypothesis(std::vector<uint8_t> features) {
	//(1 / 1 + e ^ (-w * x))
	//accumulater for sumnation of (w * x)
	float sum = 0;

	//loop through all features and multiply each feature by each corresponding weight
	//note: added a dummy feature X0 as 1 so only one update rule is needed
	//and let the bias be accounted for correctly
	for (int i = 0; i < features.size(); i++) {
		//accumulate sum of (Wi * Xi) for all features
		sum += (weights.at(i) * features.at(i));
	}
	//returns (1 / 1 + e ^ (-w * x))
	return (1.0f / (1.0f + exp((-1 * sum))));
}

//take features and the target output of the image
//modifies the weights of this logistic regression instance
void LogisticRegression::train(std::vector<uint8_t> features, uint8_t target) {
	//features here is the vector of pixel values from the image

	//target is the image.value of the image struct that contains what the image is supposed to be

	//binary target here is used as the type of binary classifier this istance of linear regression is used as
	uint8_t target_output;
	if (target == binary_target) {
		//set to 1 if actual value of the image is the type this classifier is trying to classify
		target_output = 1;
	}
	else {
		//set to 0 send the update rule term negative
		target_output = 0;
	}

	//hard threshold decision boundary
	uint8_t result;
//	if(std::inner_product(begin(features), end(features), begin(weights), 0.0) >= 0.0){
	if(hypothesis(features) >= 0.0f){
		//y
		result = 1;
	}
	else {
		result = 0;
	}

	//loop through all the features and update the weights
	for (int i = 0; i < features.size(); i++) {

		//apply learning rule
		//only one learning rule because added dummy input X0 = 1 to be multiplied by bias weight
		weights.at(i) = weights.at(i) + learning_rate * (target_output - result) * result * (1 - result) * features.at(i);
	}
}

////////////////////////
//					  //
//  Linear Regression //
//					  //
////////////////////////

class LinearRegression {
public:
	LinearRegression(int input_count, float learning_rate, uint8_t binary_target);
	int test(std::vector<uint8_t> features);
	void train(std::vector<uint8_t> features, uint8_t target);
	int hypothesis(std::vector<uint8_t> features);
	std::vector<float> weights;
	int input_count;
	uint8_t binary_target;
	float learning_rate;
private:
};

//MNIST data set training should use 784 inputs
LinearRegression::LinearRegression(int input_count, float learning_rate, uint8_t binary_target) {
	//initialize
	this->input_count = input_count;
	this->learning_rate = learning_rate;
	this->binary_target = binary_target;
	//make sure weights is empty
	weights.clear();

	//input_count + 1 so there are weights for each input plus an intercept W0 (bias)
	for (int i = 0; i < input_count + 1; i++) {
		//initialize weight vector to any starting point in weight space
		//works because convex L2 loss function implies no local minima
		weights.push_back(1.0f/(1.0f + (float) input_count));
	}
}

//takes features x and returns a hypothesis h(x)
int LinearRegression::test(std::vector<uint8_t> features) {
	return hypothesis(features);
}

//take features and returns the sumnation of (Wi * Xi)
int LinearRegression::hypothesis(std::vector<uint8_t> features) {
	//accumulater for sumnation
	float sum = 0;
	
	//dot product of w * x

	//loop through all features and multiply each feature by each corresponding weight
	//note: added a dummy feature X0 as 1 so only one update rule is needed
	//and let the bias be accounted for correctly
	for (int i = 0; i < features.size(); i++) {
		//accumulate sum of (Wi * Xi)
		sum += (weights.at(i) * features.at(i));
	}

	//decision boundary / threshold function
	if (sum >= 0) {
		return 1;
	}else {
		return 0;
	}
}

//take features and the target output of the image
//modifies the weights of this linear regression instance
void LinearRegression::train(std::vector<uint8_t> features, uint8_t target) {
	//features here is the vector of pixel values from the image
	//target is the image.value of the image struct that contains what the image is supposed to be
	
	//binary target here is used as the type of binary classifier this istance of linear regression is used as
	uint8_t target_output;
	if (target == binary_target) {
		//set to 1 if actual value of the image is the type this classifier is trying to classify
		target_output = 1;
	}else {
		//set to 0 to send update rule negative
		target_output = 0;
	}
	
	//loop through all the features
	for (int i = 0; i < features.size(); i++) {

		//apply learning rule
		//only one learning rule because added dummy input of X0 = 1 to be multiplied by bias weight
		weights.at(i) = weights.at(i) + learning_rate * (target_output - hypothesis(features)) * features.at(i);
	}
}

//////////////////////
//					//
//  Main Function	//
//					//
//////////////////////

int main(int argc, char* argv[]) {
	
	//load datasets
	std::cout << "Starting...\n" << std::endl;
	std::vector<image> images;
	std::cout << "Loading Training Data...\n" << std::endl;
	LoadData(images, "train-images.idx3-ubyte", "train-labels.idx1-ubyte");
	std::cout << "\nLoading Training Data... Complete\n" << std::endl;
	std::cout << "Loading Test Data...\n" << std::endl;
	std::vector<image> test_images;
	LoadData(test_images, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
	std::cout << "\nLoading Test Data... Complete\n" << std::endl;
	srand(time(NULL));

	/*
	//show a training example in cmd prompt
	std::cout << "Training Example: " << EXAMPLE << std::endl;
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (images[EXAMPLE].get(j, i) > THRESHOLD) {
				std::cout << ' ' << " ";
			}
			else {
				std::cout << std::to_string(images[EXAMPLE].get(i, j)) << " ";
			}
		}
		std::cout << std::endl;
	}
	*/

	//Linear Regression

	//create and train 10 binary classifiers for 0-9 
	std::vector<LinearRegression> binary_classifiers;
	for (int i = 0; i < 10; i++) {
		//create linear regression classifier for ith character/number
		LinearRegression classifier(INPUT_COUNT, LEARNING_RATE, i);
		std::vector<uint8_t> features;
		//loop through all the images in the training data
		for (image example : images) {
			//clear features vector and add X0 = 1 to front to use against bias term to allow for 1 learning rule
			features.clear();
			features.push_back(1);
			//train is expecting vector, so throw the pixel values into a vector
			for (int i = 0; i < INPUT_COUNT; i++) {
				features.push_back(example.pixel[i]);
			}
			//train the classifier weights given the features and its expected output
			classifier.train(features, example.value);
		}
		//add trained classifier to vector of classifiers 
		//(1 for each of the possible expected outputs for MINST dataset 0-9)
		binary_classifiers.push_back(classifier);
	}

	//init tallies of correct answers and total
	int correct = 0, total = 0;
	
	//output vector holds hypothesis output for each of the 10 classifiers
	std::vector<int> output;
	//features vector to use to load pixels into vector and test against classifier
	std::vector<uint8_t> features;
	//loop through all the images in the test data
	for (image example : test_images) {
		//clear output and feature vectors for new image
		output.clear();
		features.clear();
		//add X0 = 1 to front of features to use against bias term to allow for 1 learning rule
		features.push_back(1);
		//throw pixel values into vector for test
		for (int i = 0; i < INPUT_COUNT; i++) {
			features.push_back(example.pixel[i]);
		}
		//test each binary classifier against the features of the test image
		for (int i = 0; i < binary_classifiers.size(); i++) {
			output.push_back(binary_classifiers.at(i).test(features));
		}
		//use max to find the index of the max value from the vector of hypothesiss of each classifier
		int classification = -1;
		for (int i = 0; i < output.size(); i++) {
			if (output[classification] == 1) {
				classification = i;
			}
		}
		if (classification == -1) {
			classification = rand() % 10;
		}
		//if the answer value of max is the same as the expected output for that image then we win
		if (example.value == classification){
			correct++;
		}
		//regardless of whether we were correct, increment total
		total++;
	}

	//display results of testing against test data in %
	//past results have lead me to around 82%
	std::cout << "Linear Regression % Correct: " << std::to_string(((float)correct/(float)total) * 100) << "%" << std::endl;
	
	// Logistic Regression
	// used in the same way as above but with different hypothesis of: 
	// (1 / 1 + e ^ (-w * x))

	//with enough training I have gotten logistic regression up to 86%-88%
	
	std::vector<LogisticRegression> log_binary_classifiers;
	for (int i = 0; i < 10; i++) {
		LogisticRegression classifier(INPUT_COUNT, .00001f, i);
		std::vector<uint8_t> features;
		for (int x = 0; x < 20; x++) {
			for (image example : images) {
				features.clear();
				features.push_back(1);
				for (int i = 0; i < INPUT_COUNT; i++) {
					features.push_back(example.pixel[i]);
				}
				classifier.train(features, example.value);
			}
		}
		log_binary_classifiers.push_back(classifier);
	}

	correct = 0, total = 0;

	std::vector<float> log_output;
	std::vector<uint8_t> log_features;
	for (image example : test_images) {
		log_output.clear();
		log_features.clear();
		log_features.push_back(1);
		for (int i = 0; i < INPUT_COUNT; i++) {
			log_features.push_back(example.pixel[i]);
		}
		for (int i = 0; i < log_binary_classifiers.size(); i++) {
			log_output.push_back(log_binary_classifiers.at(i).test(log_features));
		}
		int max = 0;
		for (int i = 1; i < log_output.size(); i++) {
			if (log_output[i] > log_output[max]) {
				max = i;
			}
		}
		if (example.value == max) {
			correct++;
		}
		total++;
	}

	//display correct answers as a %
	std::cout << "Logistic Regression % Correct: " << std::to_string(((float)correct / (float)total) * 100) << "%" << std::endl;
	
	//hold output open
	getchar();
}

