require_relative '../lib/rbcuda.so'

arr = Array.new(100) { 2  }
gpu_ary1    = RbCUDA::Runtime.cudaMalloc([10,10]);
arr2 = Array.new(100) { 2  }
RbCUDA::Runtime.cudaMemcpy(gpu_ary1, arr, 100, :cudaMemcpyHostToDevice)
gpu_ary2    = RbCUDA::Runtime.cudaMalloc([10,10]);
RbCUDA::Runtime.cudaMemcpy(gpu_ary2, arr2, 100, :cudaMemcpyHostToDevice)
gpu_ary3    = RbCUDA::Runtime.cudaMalloc([10,10]);
RbCUDA::Arithmetic.add(gpu_ary1, gpu_ary2, 100)
temp = []
# d = RbCUDA::Runtime.cudaMemcpy(temp, c, 100, :cudaMemcpyDeviceToHost)
# puts c
