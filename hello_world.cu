#include <stdio.h>

__global__ void hello_cuda() {
    printf("Hello, Cuda!\n");
    printf("blockIdx.x: %d, blockIdx.y: %d, Thread Index X: %d\n, Thread Index Y: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(int argc, char **argv) {
    hello_cuda<<<2,1>>>();
    cudaDeviceSynchronize();

    return 0;
}