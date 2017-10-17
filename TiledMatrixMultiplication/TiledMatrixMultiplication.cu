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

#define TILE_WIDTH	16U
#define BLOCK_WIDTH 16U

__global__
void matrixMultiplyTiled(const float * A, const float * B, float * C,
  int numARows, int numAColumns, int numBColumns)
{

  // Declaration of shared memory matrices for storing a tile from A and B
	__shared__ float sm_TileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sm_TileB[TILE_WIDTH][TILE_WIDTH];

	// Block Indices
	int bIdx = blockIdx.x;
	int bIdy = blockIdx.y;

	// Thread Indices
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;

	// Variable for storing the value in matric C computed by each thread
	float tempC = 0.0;

	// Phase iteration counter
	int phCnt = 0;
  // Compute iteration counter
  int ciCnt = 0;

	// Compute 1-D Row and Column Indices for accessing A and B from Global
	// memory
	int RowIdx = tIdy + bIdy * blockDim.y;
	int ColIdx = tIdx + bIdx * blockDim.x;

	// Loop over all the tiles of A and B to compute the element in C
  for(phCnt = 0; phCnt < (ceil(numAColumns/(float)TILE_WIDTH)); ++phCnt)
  {
    // Each thread loading one element from A and B into shared memory
    // Boundary check to handle rectangular matrices
    if((RowIdx < numARows) && ((phCnt*TILE_WIDTH + tIdx) < numAColumns))
    {
      sm_TileA[tIdy][tIdx] = A[RowIdx*numAColumns + (phCnt*TILE_WIDTH + tIdx)];
    }
    else
    {
      sm_TileA[tIdy][tIdx] = 0.0;
    }

    if(((phCnt*TILE_WIDTH + tIdy) < numAColumns) && (ColIdx < numBColumns))
    {
      sm_TileB[tIdy][tIdx] = B[(phCnt*TILE_WIDTH + tIdy)*numBColumns + ColIdx];
    }
    else
    {
      sm_TileB[tIdy][tIdx] = 0.0;
    }
    // Barrier synchronization
    __syncthreads();

    // Compute element in C
    for(ciCnt = 0; ciCnt < TILE_WIDTH; ++ciCnt)
    {
      tempC += sm_TileA[tIdy][ciCnt] * sm_TileB[ciCnt][tIdx];
    }
    // Barrier synchronization
    __syncthreads();
  }

  //Boundary check and assignment to global memory
  if((RowIdx < numARows) && (ColIdx < numBColumns))
  {
    C[(RowIdx*numBColumns) + ColIdx] = tempC;
  }
}

__global__
void matrixMultiplyMultiTile(const float * A, const float * B, float * C,
                             int numARows, int numAColumns, int numBColumns)
{
  // Declaration of shared memory matrices for storing a tile from A and B
	__shared__ float sm_TileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sm_TileB[TILE_WIDTH*4U][TILE_WIDTH*4U];

	// Block Indices
	int bIdx = blockIdx.x;
	int bIdy = blockIdx.y;

	// Thread Indices
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;

	// Variable for storing the value in matric C computed by each thread
	float tempC = 0.0;
  float tempC1 = 0.0;
  float tempC2 = 0.0;
  float tempC3 = 0.0;

	// Phase iteration counter
	int phCnt = 0;
  // Compute iteration counter
  int ciCnt = 0;

	// Compute 1-D Row and Column Indices for accessing A and B from Global
	// memory
	int RowIdx = tIdy + bIdy * blockDim.y;
	int ColIdx = tIdx + bIdx * blockDim.x;

	// Loop over all the tiles of A and B to compute the element in C
  for(phCnt = 0; phCnt < (ceil(numAColumns/(float)TILE_WIDTH)); ++phCnt)
  {
    // Each thread loading one element from A and B into shared memory
    // Boundary check to handle rectangular matrices
    if((RowIdx < numARows) && ((phCnt*TILE_WIDTH + tIdx) < numAColumns))
    {
      sm_TileA[tIdy][tIdx] = A[RowIdx*numAColumns + (phCnt*TILE_WIDTH + tIdx)];
    }
    else
    {
      sm_TileA[tIdy][tIdx] = 0.0;
    }

    if(((phCnt*TILE_WIDTH + tIdy) < numAColumns) && (ColIdx < numBColumns))
    {
      sm_TileB[tIdy][tIdx] = B[(phCnt*TILE_WIDTH + tIdy)*numBColumns + ColIdx];
    }
    else
    {
      sm_TileB[tIdy][tIdx] = 0.0;
    }

    if(((phCnt*TILE_WIDTH + tIdy) < numAColumns) && ((ColIdx+1) < numBColumns))
    {
      sm_TileB[tIdy][tIdx+1] = B[(phCnt*TILE_WIDTH + tIdy)*numBColumns + (ColIdx+1)];
    }
    else
    {
      sm_TileB[tIdy][tIdx+1] = 0.0;
    }

    if(((phCnt*TILE_WIDTH + tIdy) < numAColumns) && ((ColIdx+2) < numBColumns))
    {
      sm_TileB[tIdy][tIdx+2] = B[(phCnt*TILE_WIDTH + tIdy)*numBColumns + (ColIdx+2)];
    }
    else
    {
      sm_TileB[tIdy][tIdx+2] = 0.0;
    }

    if(((phCnt*TILE_WIDTH + tIdy) < numAColumns) && ((ColIdx+3) < numBColumns))
    {
      sm_TileB[tIdy][tIdx+3] = B[(phCnt*TILE_WIDTH + tIdy)*numBColumns + (ColIdx+3)];
    }
    else
    {
      sm_TileB[tIdy][tIdx+3] = 0.0;
    }
    // Barrier synchronization
    __syncthreads();

    // Compute element in C
    for(ciCnt = 0; ciCnt < TILE_WIDTH; ++ciCnt)
    {
      tempC += sm_TileA[tIdy][ciCnt] * sm_TileB[ciCnt][tIdx];
      tempC1 += sm_TileA[tIdy][ciCnt] * sm_TileB[ciCnt][tIdx+1];
      tempC2 += sm_TileA[tIdy][ciCnt] * sm_TileB[ciCnt][tIdx+2];
      tempC3 += sm_TileA[tIdy][ciCnt] * sm_TileB[ciCnt][tIdx+3];
    }
    // Barrier synchronization
    __syncthreads();
  }

  //Boundary check and assignment to global memory
  if(RowIdx < numARows)
  {
    if(ColIdx < numBColumns)
    {
      C[(RowIdx*numBColumns) + ColIdx] = tempC;
    }
    if((ColIdx+1) < numBColumns)
    {
      C[(RowIdx*numBColumns) + (ColIdx+1)] = tempC1;
    }
    if((ColIdx+2) < numBColumns)
    {
      C[(RowIdx*numBColumns) + (ColIdx+2)] = tempC2;
    }
    if((ColIdx+3) < numBColumns)
    {
      C[(RowIdx*numBColumns) + (ColIdx+3)] = tempC3;
    }
  }
}


void matrixMultiplyHost(const float * A, const float * B, float * C,
                        int numARows, int numAColumns, int numBColumns)
{

    for (int i = 0; i < numARows; ++i) {

        for (int j = 0; j < numBColumns; ++j) {

            float value = 0;

            for (int k = 0; k < numAColumns; ++k) {

                value += A[i * numAColumns + k] * B[k * numBColumns + j];

            }

            C[i * numBColumns + j] = value;

        }

    }

}

int main(int argc, char **argv)
{
  // Data definition - START
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  int dALenInBytes; // Size of matrix A on GPU device
  int dBLenInBytes; // Size of matrix B on GPU device
  int dCLenInBytes; // Size of matrix C on GPU device
  cudaError_t cudaApiErrVal; // CUDA Error Check
  cudaError_t cudaKernelErrVal; // CUDA Error Check
  // Data definition - END

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);

  //Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;

  //Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  dALenInBytes = numARows * numAColumns * sizeof(float);
  dBLenInBytes = numBRows * numBColumns * sizeof(float);
  dCLenInBytes = numCRows * numCColumns * sizeof(float);

  wbTime_start(GPU, "Allocating GPU memory.");

  // Allocate GPU memory - START
  cudaApiErrVal = cudaMalloc(&deviceA,dALenInBytes);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceA returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&deviceB,dBLenInBytes);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceB returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMalloc(&deviceC,dCLenInBytes);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMalloc deviceC returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  // Allocate GPU memory - END
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");

  // Copy memory to the GPU - START

  cudaApiErrVal = cudaMemcpy(deviceA, hostA, dALenInBytes, cudaMemcpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy deviceA returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaApiErrVal = cudaMemcpy(deviceB, hostB, dBLenInBytes, cudaMemcpyHostToDevice);

  if(cudaSuccess != cudaApiErrVal)
  {
    printf("cudaMemcpy deviceB returned error %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }
  // Copy memory to the GPU - END
  cudaDeviceSynchronize();
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // Initialize the grid and block dimensions
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
  dim3 dimGrid((ceil(numCColumns/ dimBlock.x)), (ceil(numCRows/ dimBlock.y)));

  wbTime_start(Compute, "Performing basic tiled computation");

  // Launch the basic tiled GPU Kernel
  matrixMultiplyTiled<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
	  numARows, numAColumns, numBColumns);

  cudaKernelErrVal = cudaGetLastError();

  if(cudaSuccess != cudaKernelErrVal)
  {
    printf("Failed to launch the cuda kernel %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaKernelErrVal), cudaKernelErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing basic tiled computation");

  wbTime_start(Copy, "Copying output memory to the CPU");

  // Copy the GPU memory back to the CPU
  cudaApiErrVal = cudaMemcpy(hostC, deviceC, dCLenInBytes, cudaMemcpyDeviceToHost);

  if (cudaSuccess != cudaApiErrVal)
  {
	  printf("cudaMemcpy hostC returned error %s (code %d), line(%d)\n",
		  cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
	  exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
  wbTime_stop(Copy, "Copying output memory to the CPU");

  // chack the basic tiled solution
  wbSolution(args, hostC, numCRows, numCColumns);


  wbTime_start(Compute, "Performing multi-tiled computation");

  // Launch the multi-tiled GPU Kernel here
  matrixMultiplyMultiTile<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
	  numARows, numAColumns, numBColumns);

  cudaKernelErrVal = cudaGetLastError();

  if(cudaSuccess != cudaKernelErrVal)
  {
    printf("Failed to launch the cuda kernel %s (code %d), line(%d)\n",
    cudaGetErrorString(cudaKernelErrVal), cudaKernelErrVal, __LINE__);
    exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing multi-tiled computation");

  wbTime_start(Copy, "Copying output memory to the CPU 2");

  // Copy the GPU memory back to the CPU
  cudaApiErrVal = cudaMemcpy(hostC, deviceC, dCLenInBytes, cudaMemcpyDeviceToHost);

  if (cudaSuccess != cudaApiErrVal)
  {
	  printf("cudaMemcpy hostC returned error %s (code %d), line(%d)\n",
		  cudaGetErrorString(cudaApiErrVal), cudaApiErrVal, __LINE__);
	  exit(EXIT_FAILURE);
  }

  cudaDeviceSynchronize();
  wbTime_stop(Copy, "Copying output memory to the CPU 2");

  // Check the multi-tiled solution
  wbSolution(args, hostC, numCRows, numCColumns);

  wbTime_start(GPU, "Freeing GPU Memory");

  // Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
