#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <read_MNIST.h>
#include <read_MNIST.cpp>

//define model parameters
#define INPUT_SIZE IMAGE_SIZE
#define HIDDEN_SIZE 512
#define OUTPUT_SIZE 10
#define DROPOUT_RATE 0.5
#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 128

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Leaky ReLU activation function
__device__ float leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

// Forward hidden layer
__global__ void forward_hidden_layer(float *input, float *weights, float *bias, float *output, int input_size, int output_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = sigmoid(sum + bias[idx]);
    }
}

// Forward output layer
__global__ void forward_output_layer(float *input, float *weights, float *bias, float *output, int input_size, int output_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = sum + bias[idx];
    }
}

//Backward hidden layer
__global__ void backward_hidden_layer(float *input, float *weights, float *bias, float *output, int input_size, int output_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = leaky_relu(sum + bias[idx]);
    }
}

//Backward output layer
__global__ void backward_output_layer(float *input, float *weights, float *bias, float *output, int input_size, int output_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        output[idx] = sum + bias[idx];
    }
}

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int input_size = INPUT_SIZE;
    int hidden_size = HIDDEN_SIZE;
    int output_size = OUTPUT_SIZE;

    float h_input[] = {1.0f, 2.0f, 3.0f};
    float h_hidden_weights[] = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
        1.0f, 1.1f, 1.2f
    };
    float h_hidden_bias[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float h_output_weights[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f
    };
    float h_output_bias[] = {0.1f, 0.2f};

    float *d_input, *d_hidden_weights, *d_hidden_bias, *d_hidden_output;
    float *d_output_weights, *d_output_bias, *d_output;

    cudaMalloc((void **)&d_input, input_size * sizeof(float));
    cudaMalloc((void **)&d_hidden_weights, hidden_size * input_size * sizeof(float));
    cudaMalloc((void **)&d_hidden_bias, hidden_size * sizeof(float));
    cudaMalloc((void **)&d_hidden_output, hidden_size * sizeof(float));
    cudaMalloc((void **)&d_output_weights, output_size * hidden_size * sizeof(float));
    cudaMalloc((void **)&d_output_bias, output_size * sizeof(float));
    cudaMalloc((void **)&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_weights, h_hidden_weights, hidden_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden_bias, h_hidden_bias, hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, h_output_weights, output_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_bias, h_output_bias, output_size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int hiddenBlocksPerGrid = (hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    int outputBlocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    forward_hidden_layer<<<hiddenBlocksPerGrid, threadsPerBlock>>>(d_input, d_hidden_weights, d_hidden_bias, d_hidden_output, input_size, hidden_size);
    forward_output_layer<<<outputBlocksPerGrid, threadsPerBlock>>>(d_hidden_output, d_output_weights, d_output_bias, d_output, hidden_size, output_size);

    float h_output[output_size];
    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%.2f\n", h_output[i]);
    }

    int N = 1000;
    size_t size = N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("%.2f\n", h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_input);
    cudaFree(d_hidden_weights);
    cudaFree(d_hidden_bias);
    cudaFree(d_hidden_output);
    cudaFree(d_output_weights);
    cudaFree(d_output_bias);
    cudaFree(d_output);

    return 0;
}