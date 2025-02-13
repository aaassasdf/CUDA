#ifndef READ_MNIST_H
#define READ_MNIST_H

#include <stdio.h>
#include <stdlib.h>

// Function to read the MNIST .ubyte files
unsigned char* read_ubyte_image(const char* filename, int* num_images, int* num_rows, int* num_cols);

unsigned char* read_labels(const char* filename, int* num_labels);
#endif // READ_MNIST_H