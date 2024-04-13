#include <iostream>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int n = 10;
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    // Allocate host memory
    a = new int[n];
    b = new int[n];
    c = new int[n];

    // Initialize host arrays a and b
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Allocate device memory
    cudaMalloc((void **)&dev_a, n * sizeof(int));
    cudaMalloc((void **)&dev_b, n * sizeof(int));
    cudaMalloc((void **)&dev_c, n * sizeof(int));

    // Copy host arrays to device
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    vectorAdd<<<1, n>>>(dev_a, dev_b, dev_c, n);

    // Copy result back to host
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // Print result
    for (int i = 0; i < n; i++) {
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
    }

    // Free host memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}