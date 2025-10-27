__kernel void convolution(__global float *inputImage, __global float *outputImage, __constant float *filter, int imageHeight, int imageWidth, int filterWidth) 
{
    int index = get_global_id(0);
    int row = index /imageWidth;
    int column = index % imageWidth;
    int halffilterSize = filterWidth >> 1;
    int k, l;
    float sum = 0.0f;

    // Apply the filter to the neighborhood
    for (k = -halffilterSize; k <= halffilterSize; ++k)
    {
        for (l = -halffilterSize; l <= halffilterSize; ++l)
        {
            if(filter[(k + halffilterSize) * filterWidth + l + halffilterSize] != 0)
            {
                if (row + k >= 0 && row + k < imageHeight &&
                    column + l >= 0 && column + l < imageWidth)
                {
                    sum += inputImage[(row + k) * imageWidth + column + l] *
                            filter[(k + halffilterSize) * filterWidth +
                                    l + halffilterSize];
                }
            }
        }
    }
    outputImage[row * imageWidth + column] = sum;
}
