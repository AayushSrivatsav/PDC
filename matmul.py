#include <iostream>

#define N 3

__global__ void matrixMul(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main() {
    int a[N][N] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int b[N][N] = {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}};
    int c[N][N];
    int *dev_a, *dev_b, *dev_c;

    // Allocate device memory
    cudaMalloc((void **)&dev_a, N * N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * N * sizeof(int));

    // Copy host matrices to device
    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);
    matrixMul<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    // Copy result back to host
    cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Print result
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << c[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}