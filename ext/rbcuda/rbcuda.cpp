#include <ruby.h>
#include <algorithm> // std::min
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "cublasXt.h"
#include <cuda_profiler_api.h>
#include "nmatrix.h"


/*
 * Project Includes
 */

#include "rbcuda.h"


namespace rb_cu {

}

extern "C" {
  #include "ruby_rbcuda.c"
}
