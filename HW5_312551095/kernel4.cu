#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int *device_image, float lowerX, float lowerY, float stepX, float stepY, int resX, int maxIteration) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = threadIdx.x + blockIdx.x * blockDim.x;
    int thisY = threadIdx.y + blockIdx.y * blockDim.y;
    int assign_index = thisX + thisY * resX;

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
    device_image[assign_index] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int *device_image;

    cudaHostRegister(img, resX * resY * sizeof(int), cudaHostRegisterMapped);
    cudaHostGetDevicePointer(&device_image, img, 0);

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(device_image,lowerX, lowerY,stepX, stepY,  resX, maxIterations);


    cudaDeviceSynchronize();
    cudaHostUnregister(img);
}
