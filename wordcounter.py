#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ int is_delimiter(char c, const char *delimiters) {
    int i = 0;
    while (delimiters[i] != '\0') {
        if (c == delimiters[i]) return 1;
        ++i;
    }
    return 0;
}

__global__ void countWords(char *data, int size, int *count, const char *delimiters) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int local_count = 0;

    for (int i = index; i < size; i += stride) {
        if (is_delimiter(data[i], delimiters) == 0 && (i == 0 || is_delimiter(data[i-1], delimiters) == 1)) {
            local_count++;
        }
    }

    atomicAdd(count, local_count);
}

int main(int argc, char **argv) {
    FILE *file = fopen("book.txt", "r");
    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *data = (char *)malloc(size + 1);
    fread(data, 1, size, file);
    fclose(file);
    data[size] = '\0';

    char *d_data;
    int *d_count;
    cudaMalloc(&d_data, size);
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    countWords<<<numBlocks, blockSize>>>(d_data, size, d_count, " \t\n.,;:!?");

    int count;
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Total words: %d\n", count);

    cudaFree(d_data);
    cudaFree(d_count);
    free(data);

    return 0;
}
