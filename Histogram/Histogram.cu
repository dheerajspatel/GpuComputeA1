#include <wb.h>
// System Includes
#include <stdio.h>
// For cuda runtime apis
#include <cuda_runtime.h>

// Number of Bins - Fixed for this assignment
#define D_NUM_BINS 4096

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

// Histogram Kernel
__global__
void computeHistogram(const unsigned int * dInput, unsigned int dBins,
  int dInLen)
{
  /* code */
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  int dInLenInBytes; // Input length in bytes
  cudaError_t cudaApiErrVal; // CUDA Error Check
  cudaError_t cudaKernelErrVal; // CUDA Error Check

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  //Calculate input lentgh in bytes for allocating memory on device
  dInLenInBytes = inputLength * sizeof(int);

  wbTime_start(GPU, "Allocating GPU memory.");
  // Allocate GPU memory - START
  cudaApiErrVal = cudaMalloc(&deviceInput,dInLenInBytes);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceInput returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&deviceBins,D_NUM_BINS);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceBins returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  // Allocate GPU memory - END
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  // Copy memory to the GPU - START

  cudaApiErrVal = cudaMemcpy(deviceInput, hostInput, dInLenInBytes,
                             cudaMemcpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy deviceInput returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  // Copy memory to the GPU - END
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Initialize the grid and block dimensions
  dim3 dimBlock(BLOCK_WIDTH);
  dim3 dimGrid((inputLength/dimBlock.x)+1);

  // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  // Perform kernel computation here
  computeHistogram<<<dimGrid, dimBlock>>>(deviceInput, deviceBins,
    inputLength);

  cudaKernelErrVal = cudaGetLastError();
  if(cudaSuccess != cudaKernelErrVal)
  {
    printf("Failed to launch the cuda kernel %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaKernelErrVal), cudaKernelErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //Copy the GPU memory back to the CPU here
  cudaApiErrVal = cudaMemcpy(hostBins, deviceBins, D_NUM_BINS,
                             cudaMemcpyDeviceToHost);

  if (cudaSuccess != cudaApiErrVal)
  {
	  printf("cudaMemcpy hostBins returned error %s (code %d), line(%d)\n",
		  cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
	  exit(EXIT_FAILURE);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);
  wbTime_stop(GPU, "Freeing GPU Memory");

  // Verify correctness
  // -----------------------------------------------------
  wbSolution(args, hostBins, D_NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
