# GPU-Based Radix Sort with CUDA

This repository contains a **CUDA-accelerated Radix Sort** implementation for sorting integer arrays efficiently using **parallel computing on GPUs**.

##  Overview

Radix Sort is a non-comparative sorting algorithm that sorts numbers digit by digit. This implementation leverages CUDA to parallelize the process, achieving faster sorting for large datasets.

##  Features

- **GPU Acceleration**: Uses CUDA kernels for digit counting, prefix sum, and reordering.
- **Efficient Parallelization**: Implements atomic operations and shared memory to optimize performance.
- **Stable Sorting**: Maintains the relative order of elements with equal keys.
- **Host (CPU) and Device (GPU) Operations**: CPU handles array management, while GPU processes sorting.

## How It Works

1. **Digit Counting (GPU Kernel)** - Counts occurrences of each digit in the current position.
2. **Prefix Sum (GPU Kernel)** - Computes an inclusive prefix sum for positioning elements.
3. **Reordering (GPU Kernel)** - Places numbers in the correct order based on the prefix sum.
4. **Iterative Sorting** - Repeats the process for each digit until the entire array is sorted.

## Getting Started

### Prerequisites

- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** installed.
- **C/C++ Compiler** (e.g., GCC, Clang, MSVC).

### Compilation

Use `nvcc` (NVIDIA CUDA Compiler) to compile the program:

```sh
nvcc radix.cu -o executable
or
nvcc -allow-unsupported-compiler radix.cu -o executable
