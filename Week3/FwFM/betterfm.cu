#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define N_FEATURE 1024
#define N_FIELD 16
#define DIM_FEAT 4
#define INPUT_CNT 1024
#define TILE_SIZE 16

__global__ void betterFM(float *result, float *feature, int *input_data, float *interaction_weight)
{
    __shared__ int s_A[INPUT_CNT];
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    s_A[data_idx] = input_data[data_idx * N_FIELD + 15];
    // __syncthreads();
    float res = 0;
    int didx=0;
    int djdx =0;
    for (int i = 0; i < TILE_SIZE; i++)
    {
        for (int j = i + 1; j < TILE_SIZE; j++)
        {
            didx = input_data[data_idx * TILE_SIZE + i];
            if (j == TILE_SIZE - 1)
            {
                djdx = s_A[data_idx];
            }
            else{
                djdx = input_data[data_idx * TILE_SIZE + j];
            }
            float sum = 0;
            for (int k = 0; k < DIM_FEAT; k++)
            {
                sum += feature[didx * DIM_FEAT + k] * feature[djdx * DIM_FEAT + k];
            }
            res += interaction_weight[i * N_FIELD + j] * sum;
        }
    }
    result[data_idx] = res;
}
float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}

int main()
{

    // set random seed
    srand(1);

    float *feature;
    float *interaction_weight;

    int *input_data;
    float *result;

    // set feature

    cudaMallocManaged(&feature, N_FEATURE * DIM_FEAT * sizeof(float));
    cudaMallocManaged(&interaction_weight, N_FIELD * N_FIELD * sizeof(float));

    // randomly set feature
    for (int i = 0; i < N_FEATURE; i++)
    {
        for (int j = 0; j < DIM_FEAT; j++)
        {
            feature[i * DIM_FEAT + j] = rand_float();
        }
    }

    // randomly set interaction weight
    for (int i = 0; i < N_FIELD; i++)
    {
        for (int j = i + 1; j < N_FIELD; j++)
        {
            interaction_weight[i * N_FIELD + j] = rand_float();
        }
    }

    // set input data

    cudaMallocManaged(&input_data, INPUT_CNT * N_FIELD * sizeof(int));

    // randomly set input data.
    for (int i = 0; i < INPUT_CNT * N_FIELD; i++)
    {
        input_data[i] = rand() % N_FEATURE;
    }

    cudaMallocManaged(&result, INPUT_CNT * sizeof(float));
    dim3 block(16);
    dim3 grid(ceil(N_FEATURE / block.x));
    // run, measure time
    float start = clock();
    for (int i = 0; i < 10000; i++)
    {
        betterFM<<<grid, block>>>(result, feature, input_data, interaction_weight);
        cudaDeviceSynchronize();
    }
    float end = clock();
    float time = (end - start) / CLOCKS_PER_SEC;
    // dim3 block(16, 1024);
    // dim3 grid(ceil(INPUT_CNT / block.x));
    // // run, measure time
    // float start = clock();
    // for (int i = 0; i < 1; i++)
    // {
    //     betterFM<<<grid, block>>>(result, feature, input_data, interaction_weight);
    //     cudaDeviceSynchronize();
    // }
    // float end = clock();
    // float time = (end - start) / CLOCKS_PER_SEC;
    printf("time: %f\n", time);

    // check result
    printf("Output:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("result %d, %f\n", i, result[i]);
    }
    printf("result %d, %f\n", INPUT_CNT - 1, result[INPUT_CNT - 1]);

    cudaFree(feature);
    cudaFree(interaction_weight);
    cudaFree(input_data);
    cudaFree(result);
    return 0;
}
