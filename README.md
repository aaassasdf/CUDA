## Summary

### Objective
The objective of this project is to implement a neural network using CUDA to leverage the parallel processing capabilities of GPUs. This implementation aims to speed up the training process of the neural network by performing computations in parallel. The project includes functions for initializing the network, forward propagation, backpropagation, and updating weights. Additionally, it includes functionality to read and preprocess the MNIST dataset, which is used for training and testing the neural network.

### Program Structure

#### neural_network.cu
This file contains the CUDA implementation of a neural network. It includes the following key components:
- **Initialization**: Functions to initialize the neural network, including setting up weights and biases.
- **Forward Propagation**: Functions to perform forward propagation through the network layers, using CUDA kernels for parallel computation.
- **Backpropagation**: Functions to perform backpropagation, calculating gradients and updating weights and biases.
- **Weight Updates**: Functions to update the weights and biases based on the calculated gradients.

#### read_MNIST.h
This header file declares functions for reading the MNIST dataset. It includes function prototypes for:
- **Loading Images**: Functions to load images from the MNIST dataset files.
- **Loading Labels**: Functions to load labels from the MNIST dataset files.

#### read_MNIST.cpp
This source file implements the functions declared in `read_MNIST.h`. It contains the logic for:
- **Reading and Parsing**: Reading and parsing the MNIST dataset files, including handling the file format and byte order.
- **Data Conversion**: Converting the data into a format suitable for use in the neural network, such as flattening images and normalizing pixel values.

### Usage
1. **Compile the Code**: Use a CUDA-compatible compiler to compile the `neural_network.cu` and `read_MNIST.cpp` files.
2. **Run the Program**: Execute the compiled program to train the neural network on the MNIST dataset.
3. **Evaluate the Results**: The program will output the training progress and evaluation metrics, such as accuracy on the test set.

### Dependencies
- **CUDA Toolkit**: Ensure that the CUDA Toolkit is installed and configured on your system.
- **MNIST Dataset**: Download the MNIST dataset files and place them in the appropriate directory as specified in the code.