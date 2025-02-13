#ifndef READ_MNIST_H
#define READ_MNIST_H

#include <stdio.h>
#include <stdlib.h>

// Define dataset parameters
#define NUM_TRAINING_IMAGES 60000
#define NUM_TESTING_IMAGES 10000
#define IMAGE_SIZE 28 * 28

// Define the file paths
#define TRAINING_IMAGES "dataset/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte"
#define TRAINING_LABELS "dataset/mnist/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
#define TESTING_IMAGES "dataset/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
#define TESTING_LABELS "dataset/mnist/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"


// Function to read the MNIST .ubyte files
unsigned char* read_ubyte_image(const char* filename, int* num_images, int* num_rows, int* num_cols);

// Function to read the labels
unsigned char* read_labels(const char* filename, int* num_labels);

// function to flattern the image
void flattern_image(unsigned char* images, int num_images, int num_rows, int num_cols, unsigned char* flattern_images);

#endif // READ_MNIST_H