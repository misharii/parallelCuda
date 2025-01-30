
#include <stdio.h>
#include <stdlib.h>

// #define SIZE 8
#define BASE 10


// get max element in array (host)
int getMaxCPU(int arr[], int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > mx)
            mx = arr[i];
    }
    return mx;
}

// kernel to count occurrences of each digit
__global__ void countDigitsKernel(const int *d_arr, int *d_count, int n, int exp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int digit = (d_arr[idx] / exp) % BASE;
        atomicAdd(&d_count[digit], 1);
    }
}

// kernel to perform an inclusive prefix sum on the digit count array
__global__ void prefixSumKernel(int *d_count) {
    __shared__ int temp[BASE];
    int tid = threadIdx.x;

    temp[tid] = d_count[tid];
    __syncthreads();

    // inclusive scan
    for (int offset = 1; offset < BASE; offset <<= 1) {
        int val = 0;
        if (tid >= offset) val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    d_count[tid] = temp[tid];
}

// reorder inclusive kernel
__global__ void reorderKernel(const int *d_in, int *d_out, int *d_count, int n, int exp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int revIdx = n - 1 - idx;
    if (revIdx < 0) return;

    int val = d_in[revIdx];
    int digit = (val / exp) % BASE;

    int pos = atomicSub(&d_count[digit], 1) - 1;
    d_out[pos] = val;
}



// Radix sort function using GPU kernels
void radixSortGPU(int arr[], int n) {
    int *d_arrIn, *d_arrOut, *d_count;
    cudaMalloc((void**)&d_arrIn, n * sizeof(int));
    cudaMalloc((void**)&d_arrOut, n * sizeof(int));
    cudaMalloc((void**)&d_count, BASE * sizeof(int));

    cudaMemcpy(d_arrIn, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int mx = getMaxCPU(arr, n);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (int exp = 1; mx/exp > 0; exp *= 10) {

        // Step 1: count digits
        cudaMemset(d_count, 0, BASE * sizeof(int));
        countDigitsKernel<<<blocks, threadsPerBlock>>>(d_arrIn, d_count, n, exp);
        cudaDeviceSynchronize();

        // Step 2: prefix sum (inclusive)
        prefixSumKernel<<<1, BASE>>>(d_count);
        cudaDeviceSynchronize();

        // Step 3: reorder using counts (stable)
        reorderKernel<<<blocks, threadsPerBlock>>>(d_arrIn, d_arrOut, d_count, n, exp);
        cudaDeviceSynchronize();

        // Prepare for next digit iteration
        cudaMemcpy(d_arrIn, d_arrOut, n*sizeof(int), cudaMemcpyDeviceToDevice);
    }

    // copy final result back to host
    cudaMemcpy(arr, d_arrIn, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arrIn);
    cudaFree(d_arrOut);
    cudaFree(d_count);
}

// print array on host
void printArr(int arr[], int n) {
    printf("{");
    for (int i = 0; i < n; i++)
        printf("%s%d", i ? "," : "", arr[i]);
    printf("}\n");
}





int main() {
    int arr[] = {436, 7, 3, 44, 8392, 27, 362, 61};
    int size = sizeof(arr) / sizeof(arr[0]); // Automatically calculate the array size

    printf("Original array:\n");
    printArr(arr, size);

    radixSortGPU(arr, size);

    printf("Sorted array:\n");
    printArr(arr, size);

    return 0;
}
