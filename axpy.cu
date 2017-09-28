#include <wb.h>
// System Includes
#include <stdio.h>

//@@ Complete this function
void d_axpy(const float * h_x, float * h_y, float a, int len)
{
  // Data definition - START
  // 1. Size of Vectors
  int vecLen = len * sizeof(float);
  // 2. X Vector Data on device
  float *gpu_vecX;
  // 3. Y Vector Data on device
  float *gpu_vecY;
  // 4. CUDA Error Check
  cudaError_t cudaApiErrVal;
  // Data definition - END

  // Allocate Memory in CUDA Global Memory - START
  cudaApiErrVal = cudaMalloc(&gpu_vecX,vecLen);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc gpu_vecX returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&gpu_vecY,vecLen);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc gpu_vecY returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  // Allocate Memory in CUDA Global Memory - END

  // Copy Values from Host Memory to Device Global Memory - START
  cudaApiErrVal = cudaMemCpy(gpu_vecX, h_x, vecLen, cudaMemCpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemCpy gpu_vecX returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMemCpy(gpu_vecY, h_y, vecLen, cudaMemCpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemCpy gpu_vecY returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  // Copy Values from Host Memory to Device Global Memory - END

  // Launch the CUDA Kernel
  

  // Copy Data From CUDA Memory to Host Memory

  // Free Device Memory

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
  d_axpy(h_x,h_y,a,inputLength);
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");


  // Verify correctness of the results
   wbSolution(args, h_y, inputLength);


  //Free the host memory
  free(h_x);
  free(h_y);

  return 0;
}
