/*
# Before compile, install opencv2 with:

sudo apt-get install libopencv-dev

* Compile : nvcc -o fractal fractal.cu -I.. -lcuda $(pkg-config opencv4 --libs --cflags)
* Run : ./fractal < image file path>
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#define MANDELBROT_ITERATIONS 256
#define JULIA_ITERATIONS 256

// Input image has 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__ void fractal(unsigned char *out, int width, int height)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < height && Col < width)
    {
        for (int i = 0; i < MANDELBROT_ITERATIONS; i++)
        {
            float x = Col / (float)width * 4.0 - 2.0;
            float y = Row / (float)height * 4.0 - 2.0;
            float x0 = x;
            float y0 = y;
            int iteration = 0;
            while (x * x + y * y < 4 && iteration < MANDELBROT_ITERATIONS)
            {
                float xtemp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = xtemp;
                iteration++;
            }
            out[Row * width + Col] = iteration;
        }
    }
}

__global__ void julia(unsigned char *out, int width, int height, int time)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int TimeIdx = blockIdx.z * blockDim.z + threadIdx.z;
    if (Row < height && Col < width && TimeIdx < time)
    {
        float x = Col / (float)width * 4.0 - 2.0;
        float y = Row / (float)height * 4.0 - 2.0;
        float ti = 2 * 3.141592 * TimeIdx / (float)time;
        float c_real = sin(ti) * 0.99;
        float c_imag = cos(ti) * 0.99;
        int iteration = 0;
        while (x * x + y * y < 4 && iteration < JULIA_ITERATIONS)
        {
            float xtemp = x * x - y * y + c_real;
            y = 2 * x * y + c_imag;
            x = xtemp;
            iteration++;
        }
        out[(TimeIdx * width * height + (Row * width + Col)) * 3] = 0;
        out[(TimeIdx * width * height + (Row * width + Col)) * 3 + 1] = 0;
        out[(TimeIdx * width * height + (Row * width + Col)) * 3 + 2] = iteration;
    }
}

void Usage(char prog_name[])
{
    fprintf(stderr, "Usage: %s <image output path>\n", prog_name);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        Usage(argv[0]);
    }

    const char *file_name = argv[1];
    int width = 512, height = 512, time = 360, channels = 3;

    // unsigned char *h_resultVideo;
    unsigned char *d_resultVideo;

    // h_resultVideo = (unsigned char *)malloc(width * height * time * channels * sizeof(unsigned char));
    cudaMalloc((void **)&d_resultVideo, width * height * time * channels * sizeof(unsigned char));

    // Launch the Kernel
    const int block_size = 16;
    dim3 threads(block_size, block_size, block_size);
    dim3 grid(ceil(width / (double)threads.x), ceil(height / (double)threads.y), ceil(time / (double)threads.z));
    // fractal << <grid, threads >> > (d_resultImg, width, height);
    julia<<<grid, threads>>>(d_resultVideo, width, height, time);

    // Copy the device result in device memory to the host result in host memory
    // cudaMemcpy(h_resultVideo, d_resultVideo, width * height * channels * time * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cv::VideoWriter videoWriter;
    float videoFPS = 30.0f;
    videoWriter.open(file_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), videoFPS, cv::Size(width, height), true);

    for (int i = 0; i < time; i++)
    {
        cv::Mat frame(height, width, CV_8UC3);
        cudaMemcpy(frame.data, d_resultVideo + (time * i * width * height * channels) * sizeof(unsigned char), width * height * channels, cudaMemcpyDeviceToHost);

        videoWriter<<frame;
    }
    cudaDeviceSynchronize();
    // Free device global memory
    cudaFree(d_resultVideo);

    return 0;
}
