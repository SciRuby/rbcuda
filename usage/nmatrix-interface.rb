require_relative '../lib/rbcuda.rb'
require 'nmatrix'

# Note:  Works in pry; Else initialization is wrong

shape = [2,2]

nm = NMatrix.new shape, [1,2,3,4],  dtype: :float64
nm2 = NMatrix.new shape, [1,2,3,4],  dtype: :float64

gpu_ary_1 = nm.to_dev_array
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_1, 4, :cudaMemcpyDeviceToHost)

gpu_ary_2 = nm2.to_dev_array
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_2, 4, :cudaMemcpyDeviceToHost)

gpu_ary_res = RbCUDA::Runtime.cudaMalloc([2,2])

lda = m = shape[0]
ldb = n = shape[1]
ldc = k = shape[0]
alpha = 1
beta = 0

handle = RbCUDA::CuBLAS_v2.cublasCreate_v2()

RbCUDA::CuBLAS_v2.cublasDgemm_v2(handle, :CUBLAS_OP_N, :CUBLAS_OP_N, m, n, k, alpha, gpu_ary_1, lda, gpu_ary_2, ldb, beta, gpu_ary_res, ldc)

puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res, 4, :cudaMemcpyDeviceToHost)
