// #include <stdio.h>
// #include <stdlib.h>
// #define BASE 10
//
//
// // Kernel to extract a specific digit of each number
// __global__ void extractDigitKernel(const int *d_in, int *d_digits, int n, int exp) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         int value = d_in[idx];
//         int digit = (value / exp) % BASE;
//         d_digits[idx] = digit;
//     }
// }
//
// // Kernel to count the occurrences of each digit
// __global__ void countDigitsKernel(const int *d_digits, int *d_count, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         int digit = d_digits[idx];
//         atomicAdd(&d_count[digit], 1);
//     }
// }
//
// // Prefix sum (exclusive) for BASE=10 counts using a single-block scan.
// // For large bases, a more scalable parallel prefix sum would be needed.
// __global__ void prefixSumKernel(int *d_count) {
//     __shared__ int temp[BASE];
//     int tid = threadIdx.x;
//
//     // Load counts into shared memory
//     temp[tid] = d_count[tid];
//     __syncthreads();
//
//     // Perform inclusive prefix sum (scan)
//     for (int offset = 1; offset < BASE; offset <<= 1) {
//         int val = 0;
//         if (tid >= offset)
//             val = temp[tid - offset];
//         __syncthreads();
//         temp[tid] += val;
//         __syncthreads();
//     }
//
//     // Convert inclusive prefix sums to exclusive prefix sums by shifting
//     // After inclusive scan: temp[0] = c0, temp[1] = c0+c1, ...
//     // We want exclusive: temp[0] = 0, temp[1] = c0, ...
//     __syncthreads();
//     int val = (tid == 0) ? 0 : temp[tid - 1];
//     __syncthreads();
//     temp[tid] = val;
//     __syncthreads();
//
//     // Write back
//     d_count[tid] = temp[tid];
// }
//
// // Kernel to scatter elements into output array based on prefix sums
// __global__ void reorderKernel(const int *d_in, int *d_out, const int *d_digits, int *d_prefix, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         int digit = d_digits[idx];
//         // Use atomicAdd to get the correct position for this digit
//         int pos = atomicAdd(&d_prefix[digit], 1);
//         d_out[pos] = d_in[idx];
//     }
// }
//
// // Function to perform one digit iteration of radix sort
// void radixSortDigit(int *d_in, int *d_out, int *d_auxDigits, int *d_count, int *d_prefix, int n, int exp, int threadsPerBlock) {
//     int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
//
//     // Extract the digit
//     extractDigitKernel<<<blocks, threadsPerBlock>>>(d_in, d_auxDigits, n, exp);
//     cudaDeviceSynchronize());
//
//     // Initialize counts
//     cudaMemset(d_count, 0, BASE * sizeof(int)));
//
//     // Count digits
//     countDigitsKernel<<<blocks, threadsPerBlock>>>(d_auxDigits, d_count, n);
//     cudaDeviceSynchronize());
//
//     // Compute prefix sums (exclusive) on the counts
//     prefixSumKernel<<<1, BASE>>>(d_count);
//     cudaDeviceSynchronize());
//
//     // Copy prefix sums into a prefix array for reordering
//     cudaMemcpy(d_prefix, d_count, BASE * sizeof(int), cudaMemcpyDeviceToDevice));
//
//     // Reorder elements according to the digit positions
//     reorderKernel<<<blocks, threadsPerBlock>>>(d_in, d_out, d_auxDigits, d_prefix, n);
//     cudaDeviceSynchronize());
// }
//
// // Host function to find maximum element
// int findMaxCPU(int *arr, int n) {
//     int mx = arr[0];
//     for (int i = 1; i < n; i++)
//         if (arr[i] > mx) mx = arr[i];
//     return mx;
// }
//
// int main() {
//     // Original array
//     int h_in[] = {436, 7, 3, 44, 8392, 27, 362, 61};
//     int n = sizeof(h_in)/sizeof(h_in[0]);
//
//     // Find max number to determine number of digits
//     int maxVal = findMaxCPU(h_in, n);
//     int digits = 0;
//     {
//         int temp = maxVal;
//         do {
//             temp /= 10;
//             digits++;
//         } while (temp > 0);
//     }
//
//     // Allocate device memory
//     int *d_in, *d_out, *d_auxDigits, *d_count, *d_prefix;
//     cudaMalloc((void**)&d_in, n*sizeof(int)));
//     cudaMalloc((void**)&d_out, n*sizeof(int)));
//     cudaMalloc((void**)&d_auxDigits, n*sizeof(int)));
//     cudaMalloc((void**)&d_count, BASE*sizeof(int)));
//     cudaMalloc((void**)&d_prefix, BASE*sizeof(int)));
//
//     // Copy input to device
//     cudaMemcpy(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice));
//
//     // Determine block and grid sizes
//     int threadsPerBlock = 256;
//
//     // LSD radix sort - go through each digit
//     int exp = 1;
//     for (int i = 0; i < digits; i++) {
//         // Sort by digit at exp
//         radixSortDigit(d_in, d_out, d_auxDigits, d_count, d_prefix, n, exp, threadsPerBlock);
//
//         // Swap d_in and d_out for next iteration
//         int *temp = d_in;
//         d_in = d_out;
//         d_out = temp;
//
//         exp *= 10;
//     }
//
//     // After the final iteration, d_in holds the sorted array
//     cudaMemcpy(h_in, d_in, n*sizeof(int), cudaMemcpyDeviceToHost));
//
//     // Print the result
//     printf("Sorted array:\n");
//     for (int i = 0; i < n; i++) {
//         printf("%d ", h_in[i]);
//     }
//     printf("\n");
//
//     // Free device memory
//     cudaFree(d_in));
//     cudaFree(d_out));
//     cudaFree(d_auxDigits));
//     cudaFree(d_count));
//     cudaFree(d_prefix));
//
//     return 0;
// }


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
