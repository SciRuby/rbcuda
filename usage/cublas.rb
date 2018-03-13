require_relative '../lib/rbcuda.rb'

shape = [4, 1]
cpu_ary1 = [1, 3, 5, 7]
cpu_ary2 = [4, 1, 0, 7]

gpu_ary1 = RbCUDA::Runtime.cudaMalloc(shape)
gpu_ary2 = RbCUDA::Runtime.cudaMalloc(shape)

handle = RbCUDA::CuBLAS_v2.cublasCreate_v2

RbCUDA::Runtime.cudaMemcpy(gpu_ary1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice)
RbCUDA::Runtime.cudaMemcpy(gpu_ary2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice)

puts RbCUDA::CuBLAS_v2.cublasDnrm2_v2(handle, shape[0], gpu_ary1, 1)

puts RbCUDA::CuBLAS_v2.cublasDdot_v2(handle, shape[0], gpu_ary1, 1, gpu_ary2, 1)

puts "==========================="

alpha = 2
shape = [4,1]
gpu_ary_res = RbCUDA::Runtime.cudaMalloc(shape)
cpu_ary_res = [1, 2, 3, 4]
RbCUDA::Runtime.cudaMemcpy(gpu_ary_res, cpu_ary_res, shape[0]*shape[1], :cudaMemcpyHostToDevice)
RbCUDA::CuBLAS_v2.cublasDscal_v2(handle, shape[0], alpha, gpu_ary_res, 1)
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
RbCUDA::Runtime.cudaFree(gpu_ary_res)


puts "==========================="
shape = [4,1]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)
gpu_ary_res2 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice)
RbCUDA::Runtime.cudaMemcpy(gpu_ary_res2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice)

RbCUDA::CuBLAS_v2.cublasDaxpy_v2(handle, shape[0], 2, gpu_ary_res1, 1, gpu_ary_res2, 1)
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res2, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
RbCUDA::Runtime.cudaFree(gpu_ary_res1)
RbCUDA::Runtime.cudaFree(gpu_ary_res2)

puts "============================"

shape = [4,1]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)
gpu_ary_res2 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice)
RbCUDA::Runtime.cudaMemcpy(gpu_ary_res2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice)

RbCUDA::CuBLAS_v2.cublasDcopy_v2(handle, shape[0], gpu_ary_res1, 1, gpu_ary_res2, 1)
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res2, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
RbCUDA::Runtime.cudaFree(gpu_ary_res1)
RbCUDA::Runtime.cudaFree(gpu_ary_res2)

puts "============================"

shape = [4,1]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)
gpu_ary_res2 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice)
RbCUDA::Runtime.cudaMemcpy(gpu_ary_res2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice)

RbCUDA::CuBLAS_v2.cublasDcopy_v2(handle, shape[0], gpu_ary_res1, 1, gpu_ary_res2, 1)
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res2, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
RbCUDA::Runtime.cudaFree(gpu_ary_res1)
RbCUDA::Runtime.cudaFree(gpu_ary_res2)

puts "============================"

shape = [4,1]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)
gpu_ary_res2 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice)
RbCUDA::Runtime.cudaMemcpy(gpu_ary_res2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice)

RbCUDA::CuBLAS_v2.cublasDswap_v2(handle, shape[0], gpu_ary_res1, 1, gpu_ary_res2, 1)

puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res1, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res2, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
RbCUDA::Runtime.cudaFree(gpu_ary_res1)
RbCUDA::Runtime.cudaFree(gpu_ary_res2)

puts "============================"

shape = [4,1]
arr = [-3, -12, -46, 5]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, arr, shape[0]*shape[1], :cudaMemcpyHostToDevice)

puts RbCUDA::CuBLAS_v2.cublasIdamax_v2(handle, shape[0], gpu_ary_res1, 1)

RbCUDA::Runtime.cudaFree(gpu_ary_res1)

puts "============================"

shape = [4,1]
arr = [-3, -12, -46, 5]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, arr, shape[0]*shape[1], :cudaMemcpyHostToDevice)

puts RbCUDA::CuBLAS_v2.cublasIdamin_v2(handle, shape[0], gpu_ary_res1, 1)

RbCUDA::Runtime.cudaFree(gpu_ary_res1)

puts "============================"

shape = [4,1]
arr = [-3, -12, -46, 5]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, arr, shape[0]*shape[1], :cudaMemcpyHostToDevice)

puts RbCUDA::CuBLAS_v2.cublasDasum_v2(handle, shape[0], gpu_ary_res1, 1)

RbCUDA::Runtime.cudaFree(gpu_ary_res1)

puts "============================"

shape = [4,1]
gpu_ary_res1 = RbCUDA::Runtime.cudaMalloc(shape)
gpu_ary_res2 = RbCUDA::Runtime.cudaMalloc(shape)

RbCUDA::Runtime.cudaMemcpy(gpu_ary_res1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice)
RbCUDA::Runtime.cudaMemcpy(gpu_ary_res2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice)

RbCUDA::CuBLAS_v2.cublasDrot_v2(handle, shape[0], gpu_ary_res1, 1, gpu_ary_res2, 1, 2, 1)

puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res1, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res2, shape[0]*shape[1], :cudaMemcpyDeviceToHost)
RbCUDA::Runtime.cudaFree(gpu_ary_res1)
RbCUDA::Runtime.cudaFree(gpu_ary_res2)

puts "============================"
a = 4
b = 3
c = 2
d = 5
puts RbCUDA::CuBLAS_v2.cublasDrotg_v2(handle, a, b, c, d)

puts "============================"