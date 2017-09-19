#include <wb.h>

//@@ Complete this function
void d_axpy(const float * h_x, float * h_y, float a, int len) {



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
