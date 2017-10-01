extern "C" {
  __global__ void Vector_Addition(int *a, int *b, int *c)
  {
    int tid = blockIdx.x;
    if (tid < 100)
        c[tid] = a[tid] + b[tid];
  }
}
