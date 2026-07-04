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

# Step 9 - qk_scores
#include <cmath>
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (i >= seq_len || j >= seq_len) return;

    scores[i * seq_len + j] = dot_product(q + i * head_dim, k + j * head_dim, head_dim) * (1/sqrtf((float)head_dim));
    
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float max =-INFINITY;
    for (int c = 0; c < cols; c++){
        float val = matrix[r*cols+c];
            if (max < val){
                max = val;
            }
    }
    __syncthreads(); 
    
    float sum = 0;
    for (int c = 0; c < cols; c++){
        sum += expf(matrix[r*cols+c] - max);
    }
    __syncthreads(); 
    for (int c = 0; c < cols; c++){
        matrix[r*cols+c] = (expf(matrix[r*cols+c] - max))/sum;
    } 
    __syncthreads(); 
}

# Step 11 - pv_matmul
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  
    int j = blockIdx.x * blockDim.x + threadIdx.x;  

    if (i >= seq_len || j >= head_dim) return;
    float sum = 0;
    for (int l = 0; l < seq_len; l++){
                float p_il = p[i*seq_len + l];
                float v_lj = v[l*head_dim + j];
                sum += p_il * v_lj;
        }
    out[i * head_dim + j] = sum;
}

# Step 12 - naive_attention
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {   

float* d_scores;
cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));

dim3 threadsPerBlock(16, 16);
dim3 numBlocksScores((seq_len + 15) / 16, (seq_len + 15) / 16);
qk_scores<<<numBlocksScores, threadsPerBlock>>>(d_q, d_k, d_scores, seq_len, head_dim);

int softmaxThreads = 128;
int softmaxBlocks = (seq_len + softmaxThreads - 1) / softmaxThreads;
softmax_rows<<<softmaxBlocks, softmaxThreads>>>(d_scores, seq_len, seq_len);

dim3 numBlocksOut((head_dim + 15) / 16, (seq_len + 15) / 16);
pv_matmul<<<numBlocksOut, threadsPerBlock>>>(d_scores, d_v, d_out, seq_len, head_dim);

cudaFree(d_scores);
}

# Step 13 - online_max
__device__ float online_max(float old_max, float new_val) {
    if (old_max < new_val){
        return new_val;
    }
    else {
        return old_max;
    }
}

# Step 14 - correction_factor
#include<cmath>
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# Step 15 - update_running_sum
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return correction * old_sum + block_sum;
}

# Step 16 - rescale_output
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int i = 0; i < head_dim; i++){
        out_row[i] *= correction;
    }
}

# Step 17 - load_tile
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    
    int tile_size = tile_rows * tile_cols;
    for (int id = thread_id; id < tile_size; id +=num_threads){
        int l_row = id / tile_cols;
        int l_col = id % tile_cols;
        int g_row = src_row_start + l_row;
        int g_col = src_col_start + l_col;
        if(g_row < src_rows && g_col < src_cols){
            shared_dst[l_row * tile_cols + l_col]= src[g_row * src_cols + g_col];
        }
        else {
            shared_dst[l_row * tile_cols + l_col] = 0.0f;
        }
    }


}

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

