#include <wb.h>
// System Includes
#include <stdio.h>
// For cuda runtime apis
#include <cuda_runtime.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define Mask_width 5
#define Mask_radius Mask_width / 2
#define OUT_TILE_WIDTH 12 //Output Tile Width, set such that input tile width = 16
#define INPUT_TILE_WIDTH  (OUT_TILE_WIDTH + Mask_width - 1)
#define w (OUT_TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))
#define CHANNELS 3 //Number of channels (fixed)

//Convolution kernel
__global__
void convolution(float * dIn, const float * __restrict__ dM, float * dOut,
            int imageChannels, int imageWidth, int imageHeight)
{
  //Input tile
  __shared__ float dInTile[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH * CHANNELS];

  //1D Row and Column Indices for output tile
  int outRowIdx = threadIdx.y + blockIdx.y*OUT_TILE_WIDTH;
  int outColIdx = threadIdx.x + blockIdx.x*OUT_TILE_WIDTH;

  //1D Row and Column Indices for input tile
  int inRowIdx = outRowIdx - Mask_radius;
  int inColIdx = outColIdx - Mask_radius;

  //Copy data to shared memory tites
  //Every thread copies one element from dIn
  if((inRowIdx >= 0) && (inRowIdx < imageHeight) && (inColIdx >= 0) &&\
     (inColIdx < imageWidth))
  {
    //Copy each channel data
    for(int i=0; i<imageChannels; i++)
    {
      dInTile[threadIdx.y][(threadIdx.x*CHANNELS)+i] = \
                            dIn[(inRowIdx*imageWidth + inColIdx)*CHANNELS + i];
    }
  }
  else
  {
    dInTile[threadIdx.y][(threadIdx.x*CHANNELS)] = 0.0; //Red
    dInTile[threadIdx.y][(threadIdx.x*CHANNELS)+1] = 0.0;//Green
    dInTile[threadIdx.y][(threadIdx.x*CHANNELS)+2] = 0.0;//Blue
  }
  //synchronization
  __syncthreads();

  //compute convolution
  float rPixVal; //channel0 of each pixel
  float gPixVal; //channel1 of each pixel
  float bPixVal; //channel2 of each pixel
  //some threads are not required for output value computation
  if((threadIdx.x<OUT_TILE_WIDTH) && (threadIdx.y<OUT_TILE_WIDTH))
  {
    for(int j=0; j < Mask_width; j++)//iterate over each row
    {
      for(int k=0; k<Mask_width; k++)//iterate over each Column
      {
        rPixVal += dM[j][k] * dInTile[j+threadIdx.y][(k*CHANNELS)+(threadIdx.x*CHANNELS)];
        gPixVal += dM[j][k] * dInTile[j+threadIdx.y][(k*CHANNELS)+(threadIdx.x*CHANNELS + 1)];
        bPixVal += dM[j][k] * dInTile[j+threadIdx.y][(k*CHANNELS)+(threadIdx.x*CHANNELS + 2)];
      }
    }
    //write values to global output
    if((outRowIdx < imageHeight) && (outColIdx < imageWidth))
    {
      dIn[(outRowIdx*imageWidth + outColIdx)*CHANNELS] = clamp(rPixVal);
      dIn[(outRowIdx*imageWidth + outColIdx)*CHANNELS + 1] = clamp(gPixVal);
      dIn[(outRowIdx*imageWidth + outColIdx)*CHANNELS + 2] = clamp(bPixVal);
    }
  }

}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;
  int imageDataLenInBytes; // size of input and output image
  int maskDataLenInBytes; // size of input Mask
  cudaError_t cudaApiErrVal; // CUDA Error Check
  cudaError_t cudaKernelErrVal; // CUDA Error Check

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  // Calculate Data lengths in bytes
  imageDataLenInBytes = imageWidth * imageHeight * imageChannels * sizeof(float);
  maskDataLenInBytes = maskColumns * maskRows * sizeof(float);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //Memory allocation in GPU
  cudaApiErrVal = cudaMalloc(&deviceMaskData,maskDataLenInBytes);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceMaskData returned error %s (code %d),\
    line(%d)\n", cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&deviceInputImageData,imageDataLenInBytes);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceInputImageData returned error %s (code %d),\
    line(%d)\n", cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&deviceOutputImageData,imageDataLenInBytes);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceOutputImageData returned error %s (code %d),\
    line(%d)\n", cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //Copy data to GPU memory
  cudaApiErrVal = cudaMemcpy(deviceMaskData, hostMaskData, maskDataLenInBytes,
                             cudaMemcpyHostToDevice);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy deviceMaskData returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMemcpy(deviceInputImageData, hostInputImageData,
                             imageDataLenInBytes, cudaMemcpyHostToDevice);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy deviceInputImageData returned error %s (code %d),\
    line(%d)\n", cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //Thread block size same as Output Tile width
  dim3 dimBlock(INPUT_TILE_WIDTH, INPUT_TILE_WIDTH);
  dim3 dimGrid(ceil(imageWidth/(float)dimBlock.x),
               ceil(imageHeight/(float)dimBlock.y));
  convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData,
                                     deviceOutputImageData, imageChannels,
                                     imageWidth, imageHeight);
  cudaKernelErrVal = cudaGetLastError();
  if(cudaSuccess != cudaKernelErrVal)
  {
    printf("Failed to launch the cuda kernel %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaKernelErrVal), cudaKernelErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageDataLenInBytes,
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  //Update outputImage data
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(arg, outputImage);

  //Free the GPU memory here
  cudaFree(deviceMaskData);
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  free(hostMaskData);
  free(hostInputImageData);
  free(hostOutputImageData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
