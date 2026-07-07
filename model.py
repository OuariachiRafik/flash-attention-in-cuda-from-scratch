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

# Step 18 - tile_scores
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {

    int tile_size = tile_q * tile_k;

    for (int id = thread_id; id < tile_size; id += num_threads) {
        int row = id / tile_k;
        int col = id % tile_k;

        float sum = 0.0f;

        for (int d = 0; d < head_dim; d++) {
            sum += q_tile[row * head_dim + d] *
                   k_tile[col * head_dim + d];
        }

        s_tile[row * tile_k + col] = sum * scale;
    }
}

# Step 19 - tile_rowmax
#include <cmath>
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    for (int row = thread_id; row < tile_q; row += num_threads) {
        float max_val = -INFINITY;

        for (int col = 0; col < tile_k; ++col) {
            float val = s_tile[row * tile_k + col];
            if (val > max_val)
                max_val = val;
        }

        row_max_out[row] = max_val;
    }
}

# Step 20 - tile_exp
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int tile_size = tile_q * tile_k;

    for (int id = thread_id; id < tile_size; id += num_threads) {
        int row = id / tile_k;
        s_tile[id] = expf(s_tile[id] - row_max[row]);
    }
}

# Step 21 - tile_rowsum
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int row = thread_id; row < tile_q; row += num_threads) {
        float sum = 0.0f;

        for (int col = 0; col < tile_k; ++col) {
            sum += p_tile[row * tile_k + col];
        }

        row_sum_out[row] = sum;
    }
}

# Step 22 - accumulate_pv
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    int total = tile_q * head_dim;

    for (int id = thread_id; id < total; id += num_threads) {
        int row = id / head_dim;
        int d   = id % head_dim;

        float sum = 0.0f;

        for (int k = 0; k < tile_k; ++k) {
            sum += p_tile[row * tile_k + k] *
                   v_tile[k * head_dim + d];
        }

        out_acc[row * head_dim + d] += sum;
    }
}

# Step 23 - flash_attention_kernel
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = q_tile + tile_q * head_dim;
    float* v_tile = k_tile + tile_k * head_dim;
    float* s_tile = v_tile + tile_k * head_dim;
    float* acc    = s_tile + tile_q * tile_k;
    float* m      = acc + tile_q * head_dim;
    float* l      = m + tile_q;
    float* tmp    = l + tile_q;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int q_row_start = blockIdx.x * tile_q;

    load_tile(q, q_tile, q_row_start, 0, seq_len, head_dim, tile_q, head_dim, tid, num_threads);
    for (int row = tid; row < tile_q; row += num_threads) {
        m[row] = -INFINITY;
        l[row] = 0.0f;
    }
    for (int idx = tid; idx < tile_q * head_dim; idx += num_threads) {
        acc[idx] = 0.0f;
    }
    __syncthreads();

    for (int k_start = 0; k_start < seq_len; k_start += tile_k) {
        load_tile(k, k_tile, k_start, 0, seq_len, head_dim, tile_k, head_dim, tid, num_threads);
        load_tile(v, v_tile, k_start, 0, seq_len, head_dim, tile_k, head_dim, tid, num_threads);
        __syncthreads();

        tile_scores(q_tile, k_tile, s_tile, tile_q, tile_k, head_dim, scale, tid, num_threads);
        __syncthreads();

        tile_rowmax(s_tile, tmp, tile_q, tile_k, tid, num_threads);
        __syncthreads();

        for (int row = tid; row < tile_q; row += num_threads) {
            tmp[row] = online_max(m[row], tmp[row]);
        }
        __syncthreads();

        tile_exp(s_tile, tmp, tile_q, tile_k, tid, num_threads);
        __syncthreads();

        for (int row = tid; row < tile_q; row += num_threads) {
            float correction = correction_factor(m[row], tmp[row]);
            rescale_output(&acc[row * head_dim], head_dim, correction);
            l[row] *= correction;
            m[row] = tmp[row];
        }
        __syncthreads();

        tile_rowsum(s_tile, tmp, tile_q, tile_k, tid, num_threads);
        __syncthreads();

        for (int row = tid; row < tile_q; row += num_threads) {
            l[row] += tmp[row];
        }
        __syncthreads();

        accumulate_pv(s_tile, v_tile, acc, tile_q, tile_k, head_dim, tid, num_threads);
        __syncthreads();
    }

    for (int idx = tid; idx < tile_q * head_dim; idx += num_threads) {
        int row = idx / head_dim;
        int d = idx % head_dim;
        int qi = q_row_start + row;
        if (qi < seq_len) {
            out[qi * head_dim + d] = acc[row * head_dim + d] / l[row];
        }
    }
}

# Step 24 - flash_attention_launcher
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int threadsPerBlock = 128;
    int numBlocks = (seq_len + tile_q - 1) / tile_q;

    size_t shmem_bytes = (tile_q * head_dim
                        + tile_k * head_dim
                        + tile_k * head_dim
                        + tile_q * tile_k
                        + tile_q * head_dim
                        + 3 * tile_q) * sizeof(float);

    flash_attention_kernel<<<numBlocks, threadsPerBlock, shmem_bytes>>>(
        d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale);
}

# Step 25 - causal_mask
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    int total = tile_q * tile_k;
    for (int idx = thread_id; idx < total; idx += num_threads) {
        int row = idx / tile_k;
        int col = idx % tile_k;
        int global_q = q_row_start + row;
        int global_k = k_col_start + col;
        if (global_k > global_q) {
            s_tile[idx] = -INFINITY;
        }
    }
}

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

