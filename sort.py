#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void merge(int *data, int left, int mid, int right, int *temp) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (data[i] <= data[j]) {
            temp[k++] = data[i++];
        } else {
            temp[k++] = data[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = data[i++];
    }

    while (j <= right) {
        temp[k++] = data[j++];
    }

    for (i = left; i <= right; i++) {
        data[i] = temp[i];
    }
}

__global__ void mergeSortKernel(int *data, int *temp, int n) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    for (int size = 2; size <= n; size <<= 1) {
        for (int left = idx * size; left < n; left += stride * size) {
            int mid = left + size / 2 - 1;
            int right = min(left + size - 1, n - 1);
            if (mid < right)
                merge(data, left, mid, right, temp);
        }
        __syncthreads();
    }
}

void initializeArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100; // random numbers between 0 and 99
    }
}

int main() {
    const int n = 1024; // example array size
    int *h_data = (int *)malloc(n * sizeof(int));
    int *d_data, *d_temp;

    initializeArray(h_data, n);

    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));

    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    mergeSortKernel<<<numBlocks, blockSize>>>(d_data, d_temp, n);

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the sorted array
    for (int i = 0; i < n; i++) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_temp);
    free(h_data);

    return 0;
}
