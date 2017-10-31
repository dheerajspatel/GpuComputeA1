#include <wb.h>
// System Includes
#include <stdio.h>
// For cuda runtime apis
#include <cuda_runtime.h>
// Math apis
#include <math.h>

// Number of Bins - Fixed for this assignment
#define D_NUM_BINS 	4096
#define D_BLOCK_WIDTH	256

// Defines for the two kernel implementations
#define D_KERNEL_WITH_BIN_SAT	1
#define D_KERNEL_WO_BIN_SAT	2
#define D_KERNEL_TYPE		(D_KERNEL_WITH_BIN_SAT)

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

#if(D_KERNEL_TYPE == D_KERNEL_WO_BIN_SAT)
// Histogram Kernel with Privatization Method and No Saturation
__global__
void computeHistogram_privatized_nosat(unsigned int * dInput,
  unsigned int * dBins, int dInLen)
{
  /* Private copy of the Bins */
  __shared__ unsigned int sm_Bins[D_NUM_BINS];
  /* Bin index */
  unsigned int binIdx;
  /* Thread index */
  int tIdx = threadIdx.x;
  /* Input index */
  int inIdx = threadIdx.x + blockIdx.x * blockDim.x;
  /* Stride */
  int stride = blockDim.x * gridDim.x;
  /* Initialize the Private Bins Copy */
  for(int i=tIdx; i < D_NUM_BINS; i+=blockDim.x)
  {
    sm_Bins[i] = 0;
  }
  /* synchronization */
  __syncthreads();

  /* Compute Histogram */
  while(inIdx<dInLen)
  {
    binIdx = dInput[inIdx];
    if(binIdx < D_NUM_BINS)
    {
      atomicAdd(&(sm_Bins[binIdx]), 1);
    }
    inIdx+=stride;
  }
  /* synchronization */
  __syncthreads();

  /* Update to Global copy */
  for(int j=tIdx; j < D_NUM_BINS; j+=blockDim.x)
  {
    atomicAdd(&(dBins[j]), sm_Bins[j]);
  }
}

// Kernel to Saturate the Bin Values
__global__
void saturateHistoBins(unsigned int * dBins, int dBinLen)
{
  int Idx = threadIdx.x + blockIdx.x * blockDim.x;

  if(Idx < dBinLen)
  {
    dBins[Idx] = min(dBins[Idx], 127);
  }
}
#else
// Histogram Kernel with Privatization Method and No Saturation
__global__
void computeHistogram_privatized_withsat(unsigned int * dInput,
  unsigned int * dBins, int dInLen)
{
  /* Private copy of the Bins */
  __shared__ unsigned int sm_Bins[D_NUM_BINS];
  /* Bin index */
  unsigned int binIdx;
  /* Thread index */
  int tIdx = threadIdx.x;
  /* Input index */
  int inIdx = threadIdx.x + blockIdx.x * blockDim.x;
  /* Stride */
  int stride = blockDim.x * gridDim.x;
  /* Initialize the Private Bins Copy */
  for(int i=tIdx; i < D_NUM_BINS; i+=blockDim.x)
  {
    sm_Bins[i] = 0;
  }
  /* synchronization */
  __syncthreads();

  /* Compute Histogram */
  while(inIdx<dInLen)
  {
    binIdx = dInput[inIdx];
    if(binIdx < D_NUM_BINS)
    {
      atomicAdd(&(sm_Bins[binIdx]), 1);
    }
    inIdx+=stride;
  }
  /* synchronization */
  __syncthreads();

  /* Update to Global copy */
  for(int j=tIdx; j < D_NUM_BINS; j+=blockDim.x)
  {
    atomicAdd(&(dBins[j]), sm_Bins[j]);
  }

  /* Saturate the results to maximum 127 */
  // Though the following saturation will be performed by
  // every thread block, it won't affect the overall
  // Bins result.
  __syncthreads();

  // Use atomicMin for updating the saturated value in 
  // GlobalMem directly
  for(int k=tIdx; k < D_NUM_BINS; k+=blockDim.x)
  {
    atomicMin(&(dBins[k]), 127);
  } 
}
#endif

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;
  int inputLenInBytes; // Input length in bytes
  int binsLenInBytes;  // Bins length in bytes
  cudaError_t cudaApiErrVal; // CUDA Error Check
  cudaError_t cudaKernelErrVal; // CUDA Error Check

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(D_NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", D_NUM_BINS);

  //Initialize HostBins array
  for(int i=0; i<D_NUM_BINS; i++)
  {
    hostBins[i] = 0;
  }
  //Calculate input lentgh in bytes for allocating memory on device
  inputLenInBytes = inputLength * sizeof(unsigned int);
  binsLenInBytes = D_NUM_BINS * sizeof(unsigned int);

  wbTime_start(GPU, "Allocating GPU memory.");
  // Allocate GPU memory - START
  cudaApiErrVal = cudaMalloc((void **)&deviceInput,inputLenInBytes);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceInput returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc((void **)&deviceBins,binsLenInBytes);

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

  cudaApiErrVal = cudaMemcpy(deviceInput, hostInput, inputLenInBytes,
                             cudaMemcpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy deviceInput returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMemcpy(deviceBins, hostBins, binsLenInBytes,
                             cudaMemcpyHostToDevice);
  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy deviceBins returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  // Copy memory to the GPU - END
  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Initialize the grid and block dimensions
  dim3 dimBlock(D_BLOCK_WIDTH);
  dim3 dimGrid((ceil(D_NUM_BINS/float(dimBlock.x))));

  // Launch kernel
  // ----------------------------------------------------------
  wbLog(TRACE, "Launching kernel");

  #if(D_KERNEL_TYPE == D_KERNEL_WO_BIN_SAT)
	  wbTime_start(Compute, "Performing CUDA computation");
	  // Perform kernel computation here
	  computeHistogram_privatized_nosat<<<dimGrid, dimBlock>>>(deviceInput,
	    deviceBins, inputLength);

	  cudaKernelErrVal = cudaGetLastError();
	  if(cudaSuccess != cudaKernelErrVal)
	  {
	    printf("Failed to launch the cuda kernel %s (code %d), line(%d)\n",
	    cudaGetErrorString(cudaKernelErrVal), cudaKernelErrVal, __LINE__);
	    exit(EXIT_FAILURE);
	  }
	  CUDA_CHECK(cudaDeviceSynchronize());

	  wbTime_stop(Compute, "Performing CUDA computation");

	  //  Call Saturation kernel
	  // Block and Grid dimension is same as before

	  wbTime_start(Compute, "Performing computation for Saturation");
	  // Perform kernel computation here
	  saturateHistoBins<<<dimGrid, dimBlock>>>(deviceBins, D_NUM_BINS);

	  cudaKernelErrVal = cudaGetLastError();
	  if(cudaSuccess != cudaKernelErrVal)
	  {
	    printf("Failed to launch the cuda kernel %s (code %d), line(%d)\n",
	    cudaGetErrorString(cudaKernelErrVal), cudaKernelErrVal, __LINE__);
	    exit(EXIT_FAILURE);
	  }
	  CUDA_CHECK(cudaDeviceSynchronize());
  #else
          wbTime_start(Compute, "Performing CUDA computation");
	  // Perform kernel computation here
	  computeHistogram_privatized_withsat<<<dimGrid, dimBlock>>>(deviceInput,
	    deviceBins, inputLength);

	  cudaKernelErrVal = cudaGetLastError();
	  if(cudaSuccess != cudaKernelErrVal)
	  {
	    printf("Failed to launch the cuda kernel %s (code %d), line(%d)\n",
	    cudaGetErrorString(cudaKernelErrVal), cudaKernelErrVal, __LINE__);
	    exit(EXIT_FAILURE);
	  }
	  CUDA_CHECK(cudaDeviceSynchronize());

	  wbTime_stop(Compute, "Performing CUDA computation");
  #endif

  wbTime_stop(Compute, "Performing computation for Saturation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //Copy the GPU memory back to the CPU here
  cudaApiErrVal = cudaMemcpy(hostBins, deviceBins, binsLenInBytes,
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
