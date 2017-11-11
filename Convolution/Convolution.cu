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
#define TILE_WIDTH 16 //Output Tile Width
#define INPUT_TILE_WIDTH  (TILE_WIDTH + Mask_width - 1)
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//Convolution kernel
__global__
convolution(float * dIn, const float __restrict__ * dM, float * dOut,
            int imageChannels, int imageWidth, int imageHeight)
{
  //Input and Output tiles
  __shared__ float dOutTile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float dInTile[INPUT_TILE_WIDTH][INPUT_TILE_WIDTH];
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
    printf("cudaMalloc deviceMaskData returned error %s (code %d),
    line(%d)\n", cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&deviceInputImageData,imageDataLenInBytes);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceInputImageData returned error %s (code %d),
    line(%d)\n", cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&deviceOutputImageData,imageDataLenInBytes);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceOutputImageData returned error %s (code %d),
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
    printf("cudaMemcpy deviceInputImageData returned error %s (code %d),
    line(%d)\n", cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //Thread block size same as Output Tile width
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
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

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ Insert code here

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
