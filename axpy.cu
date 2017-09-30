#include <wb.h>
// System Includes
#include <stdio.h>

// the kernel code
__global__
void kernel_axpy (float * gpu_vecX, float * gpu_vecY, float gpu_scalar, int gpu_vecLen)
{
  int Idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(Idx < gpu_vecLen)
  {
    gpu_vecY[Idx] += gpu_scalar * gpu_vecX[Idx];
  }
  return;
}

// support function on the host
void sf_axpy(const float * h_x, float * h_y, float a, int len)
{
  // Data definition - START
  // 1. Size of Vectors
  int vecLenInBytes = len * sizeof(float);
  // 2. X Vector Data on device
  float *gpu_vecX;
  // 3. Y Vector Data on device
  float *gpu_vecY;
  // 4. CUDA Error Check
  cudaError_t cudaApiErrVal;
  // Data definition - END

  // Allocate Memory in CUDA Global Memory - START
  cudaApiErrVal = cudaMalloc(&gpu_vecX,vecLenInBytes);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc gpu_vecX returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&gpu_vecY,vecLenInBytes);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc gpu_vecY returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  // Allocate Memory in CUDA Global Memory - END

  // Copy Values from Host Memory to Device Global Memory - START
  cudaApiErrVal = cudaMemcpy(gpu_vecX, h_x, vecLenInBytes, cudaMemcpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy gpu_vecX returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMemcpy(gpu_vecY, h_y, vecLenInBytes, cudaMemcpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy gpu_vecY returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  // END of Copy Values from Host Memory to Device Global Memory

  // Launch the CUDA Kernel
  dim3 blockDim(256, 1, 1);
  dim3 gridDim((ceil(len/blockDim.x)), 1, 1);

  kernel_axpy<<<gridDim, blockDim>>>(gpu_vecX, gpu_vecY, a, len);

  // START of Copy Data From CUDA Memory to Host Memory
  cudaApiErrVal = cudaMemcpy(h_y, gpu_vecY, vecLenInBytes, cudaMemcpyDeviceToHost);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy h_y returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  // END of Copy Data From CUDA Memory to Host Memory

  // Free Device Memory
  cudaFree(gpu_vecX);
  cudaFree(gpu_vecY);
}

//Sequential Implementation
void h_axpy(const float * x, float * y, float a, int len) {

    for (int i = 0; i < len; i++) {
        y[i] += a * x[i];
    }

}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *h_x;
  float *h_y;
  float a;

  // Initialization of Data on the Host memory from the input Data

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  h_x =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  h_y =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  {
      int aLength;
      float * pA = (float *)wbImport(wbArg_getInputFile(args, 2), &aLength);
      a = *pA;

      free(pA);
  }

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
  sf_axpy(h_x,h_y,a,inputLength);
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Verify correctness of the results
   wbSolution(args, h_y, inputLength);

  //Free the host memory
  free(h_x);
  free(h_y);

  return 0;
}
