"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        a[i] = a[i] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n) { a[i] = expf(a[i]); }
}

# Step 4 - row_max
#include <cmath>
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float max =-INFINITY;
    for (int c = 0; c < cols; c++){
        float val = matrix[r*cols+c];
            if (max < val){
                max = val;
            }
        } 
        out[r] = max;
}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float sum = 0;
    for (int c = 0; c < cols; c++){
        sum += matrix[r*cols+c];
        } 
        out[r] = sum;
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    float dot = 0;
    for (int i=0; i<n; i++){
        dot += a[i] * b[i];
    }
    return dot;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  
    int j = blockIdx.x * blockDim.x + threadIdx.x;  

    if (i >= m || j >= n) return;
    float sum = 0;
    for (int l = 0; l < k; l++){
                float a_il = a[i*k + l];
                float b_lj = b[l*n + j];
                sum += a_il * b_lj;
        }
    c[i * n + j] = sum;
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    out[col * rows + row] = in[row * cols + col];
}

# Step 9 - qk_scores (not yet solved)
# TODO: implement

# Step 10 - softmax_rows (not yet solved)
# TODO: implement

# Step 11 - pv_matmul (not yet solved)
# TODO: implement

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

