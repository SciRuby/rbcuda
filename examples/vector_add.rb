require_relative '../lib/rbcuda.so'


require "test/unit/assertions"
include Test::Unit::Assertions

require 'benchmark'
require 'json'

class ResultCollect


  def self.generate

    iters = 5
    result = {}

    result[:mat_mult] = []

    shapeArray = [[100, 1], [1000, 1], [10000, 1]]

    shapeArray.each do |shape|
      puts "shape"+shape.to_s
      cpu_ary1 = Array.new(shape[0]) { |index| index }
      cpu_ary2 = Array.new(shape[0]) { |index| index * 2 }
      cpu_ary_res = []


      gpu_ary1    = RbCUDA::Runtime.cudaMalloc(shape);
      gpu_ary2    = RbCUDA::Runtime.cudaMalloc(shape);
      gpu_ary_res = RbCUDA::Runtime.cudaMalloc(shape);
        
      handle = RbCUDA::CuBLAS_v2.cublasCreate_v2()

      RbCUDA::Runtime.cudaMemcpy(gpu_ary1, cpu_ary1, shape[0], :cudaMemcpyHostToDevice);
      RbCUDA::Runtime.cudaMemcpy(gpu_ary2, cpu_ary2, shape[0], :cudaMemcpyHostToDevice);
        
      puts "GPU:"
      iters.times do
        puts Benchmark.measure{
          RbCUDA::CuBLAS_v2.cublasDaxpy_v2(handle, shape[0], 1, gpu_ary1, 1, gpu_ary2, 1)
        }
        
        # Check if the resulting vector has each element of three times its index
        r = RbCUDA::Runtime.cudaMemcpy([], gpu_ary2, shape[0], :cudaMemcpyDeviceToHost);
        for i in 0..shape[0] - 1
            assert (r[i] == i * 3)
        end
          
        RbCUDA::Runtime.cudaMemcpy(gpu_ary2, cpu_ary2, shape[0], :cudaMemcpyHostToDevice);
      end
      
      # CPU results, you can see that the CPU times increase linearly 
      #   with the array length, but GPU time stays mostly constant
#       puts "CPU:"
#       iters.times do
#         puts Benchmark.measure{
#           cpu_ary1.each_index { |i| cpu_ary_res[i] = cpu_ary1[i] + cpu_ary2[i] }
#         }
#       end
        
      RbCUDA::CuBLAS_v2.cublasDestroy_v2(handle)
      RbCUDA::Runtime.cudaFree(gpu_ary1)
      RbCUDA::Runtime.cudaFree(gpu_ary2)
      RbCUDA::Runtime.cudaFree(gpu_ary_res)
    end

  end

end

ResultCollect.generate
