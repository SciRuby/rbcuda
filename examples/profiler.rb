require_relative '../lib/rbcuda.so'
require 'benchmark'

RbCUDA::Profiler.cudaProfilerInitialize("", "/tmp/cuda_profile_data.csv", :cudaCSV);

RbCUDA::Profiler.cudaProfilerStart()

iters = 5

shape = [100, 100]

puts "shape "+shape.to_s
cpu_ary1 = Array.new(shape[0]*shape[1]) { 2 }
cpu_ary2 = Array.new(shape[0]*shape[1]) { 2 }
cpu_ary_res = []


gpu_ary1    = RbCUDA::Runtime.cudaMalloc(shape);
gpu_ary2    = RbCUDA::Runtime.cudaMalloc(shape);
gpu_ary_res = RbCUDA::Runtime.cudaMalloc(shape);

RbCUDA::Runtime.cudaMemcpy(gpu_ary1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice);
RbCUDA::Runtime.cudaMemcpy(gpu_ary2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice);

lda= m = shape[0]
ldb= n = shape[0]
ldc= k = shape[0]
alf = 1
bet = 0
alpha = alf
beta = bet
handle = RbCUDA::CuBLAS_v2.cublasCreate_v2()
iters.times do
  puts Benchmark.measure{
    RbCUDA::CuBLAS_v2.cublasDgemm_v2(handle, :CUBLAS_OP_N, :CUBLAS_OP_N, m, n, k, alpha, gpu_ary1, lda, gpu_ary2, ldb, beta, gpu_ary_res, ldc)
  }
end
# puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res, shape[0]*shape[1], :cudaMemcpyDeviceToHost);
RbCUDA::CuBLAS_v2.cublasDestroy_v2(handle)
RbCUDA::Runtime.cudaFree(gpu_ary1)
RbCUDA::Runtime.cudaFree(gpu_ary2)
RbCUDA::Runtime.cudaFree(gpu_ary_res)

RbCUDA::Profiler.cudaProfilerStop()

puts "Profiler output in /tmp/cuda_profile_data.csv"
