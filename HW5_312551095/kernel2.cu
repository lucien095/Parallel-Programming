#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int *device_image, size_t pitch, float lowerX, float lowerY, float stepX, float stepY, int resX, int maxIteration) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = threadIdx.x + blockIdx.x * blockDim.x;
    int thisY = threadIdx.y + blockIdx.y * blockDim.y;

    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;

    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIteration; ++i){
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    int *row = (int *)((char*)device_image + thisY * pitch);
    row[thisX] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *host_image;
    int *device_image;

    size_t pitch;
    cudaHostAlloc((void **)&host_image, resX * resY * sizeof(int), cudaHostAllocDefault);
    cudaMallocPitch((void **)&device_image, &pitch, resX * sizeof(int), resY);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);

    mandelKernel<<<numBlocks, threadsPerBlock>>>(device_image, pitch, lowerX, lowerY, stepX, stepY, resX, maxIterations);

    cudaMemcpy2D(host_image, resX * sizeof(int), device_image, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, host_image, resX * resY * sizeof(int));

    cudaFreeHost(host_image);
    cudaFree(device_image);
}
