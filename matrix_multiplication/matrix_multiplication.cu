#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#define BLOCK_SIZE 16

//this function is for filling the matrices with cos and sin values randomly
//I transform the matrices to square matrix in order to perform better multiplication
__host__ int fill(float **Lmatrix, float **Rmatrix, int LdimX, int LdimY, int RdimX, int RdimY) {

    int sqr_dim_X, sqr_dim_Y, size;

    sqr_dim_X = RdimX;
    if (LdimX > RdimX) {
        sqr_dim_X = LdimX;
    }

    sqr_dim_Y = RdimY;
    if (LdimY > RdimY) {
        sqr_dim_Y = LdimY;
    }

    size = sqr_dim_Y;
    if (sqr_dim_X > sqr_dim_Y) {
        size = sqr_dim_X;
    }

    int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
    size = temp * BLOCK_SIZE;

    size_t pt_size = size * size * sizeof(float);

    *Lmatrix = (float *) malloc(pt_size);
    *Rmatrix = (float *) malloc(pt_size);

    memset(*Lmatrix, 0, pt_size);
    memset(*Rmatrix, 0, pt_size);

    for (int i = 0; i < LdimX; i++) {
        for (int j = 0; j < LdimY; j++) {
            int dummy = size * i + j;
            (*Lmatrix)[dummy] = sinf(dummy);
        }
    }
    for (int i = 0; i < RdimX; i++) {
        for (int j = 0; j < RdimY; j++) {
            int dummy = size * i + j;
            (*Rmatrix)[dummy] = cosf(dummy);
        }
    }
    return size;
}

// Kernel that executes on the CUDA device
/* left: left operand
 * right: right operand
 * res : result array
 * dim: M dimension of MxM matrix
 * Blok_size: defines block size
 *
 * this function divides the matrices to tiles and load those tiles to shared memory
 * After loading to shared memory it function multiplies with the corresponding tile of other matrix
 * After finishing multiplication of 1 row and 1 column by collecting results of different tiles
 * it stores the result in global memory
 * Function has coalesced access to the global memory and prevent bank conflict
 */
__global__ void multiply(float *left, float *right, float *res, int dim) {

    int i,j;
    float temp = 0;

    __shared__ float Left_shared_t [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Right_shared_t[BLOCK_SIZE][BLOCK_SIZE];

    // Row i of matrix left
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++) {

        // Column j of matrix left
        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;
        // Load left[i][j] to shared mem

        Left_shared_t[threadIdx.y][threadIdx.x] = left[row * dim + j];// Coalesced access
        // Load right[i][j] to shared mem

        Right_shared_t[threadIdx.y][threadIdx.x] = right[i * dim + col]; // Coalesced access
        // Synchronize before computation
        __syncthreads();

        // Accumulate one tile of res from tiles of left and right in shared mem
        for (int k = 0; k < BLOCK_SIZE; k++) {

            temp += Left_shared_t[threadIdx.y][k] * Right_shared_t[k][threadIdx.x]; //no shared memory bank conflict
        }
        // Synchronize
        __syncthreads();
    }
    // Store accumulated value to res
    res[row * dim + col] = temp;
}

// main routine that executes on the host
int main(void)
{
    //size of the vectors to be processed  and matrix dimensions
    int Left_matrix_x = 8192;
    int Left_matrix_y = 8192;
    int Right_matrix_x = 8192;
    int Right_matrix_y = 8192;

    float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_d;  // Pointer to host & device arrays

    int dim = fill(&Left_Vector_h, &Right_Vector_h, Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y); //fills the matrices with random values


    size_t vector_size;
    vector_size = dim*dim * sizeof(float);

    cudaMalloc((void **) &Left_Vector_d, vector_size);     // Allocate array on device for LHS operand
    cudaMalloc((void **) &Right_Vector_d, vector_size);   // Allocate array on device for RHS operand but this is vector 1xN
    cudaMalloc((void **) &Res_d, vector_size);     // Allocate array on device for result

    cudaMemcpy(Left_Vector_d, Left_Vector_h, vector_size, cudaMemcpyHostToDevice);      // copy values to device
    cudaMemcpy(Right_Vector_d, Right_Vector_h, vector_size, cudaMemcpyHostToDevice);   // copy values to device

    //Block dimension is directly from block_size
    dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
    //Grid dimension is found by dividing matrix dimension to block_size
    dim3 Grid_dim(dim / BLOCK_SIZE, dim / BLOCK_SIZE);

    //commented out the functions which helps to calculate time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //kernel call
    multiply << < Grid_dim, Block_dim >> > (Left_Vector_d, Right_Vector_d, Res_d, dim);

    //commented out the functions which helps to calculate time
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //commented out the functions which helps to calculate time
    printf("GPU time= %f ms\n", et);
    printf("\n");

    // Cleanup
    free(Left_Vector_h);
    free(Right_Vector_h);
    cudaFree(Left_Vector_d);
    cudaFree(Right_Vector_d);
    cudaFree(Res_d);
}