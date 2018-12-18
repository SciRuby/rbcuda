require_relative '../lib/rbcuda.so'


require 'benchmark'
require 'json'

class ResultCollect


  def self.generate

    iters = 5
    result = {}

    result[:mat_mult] = []

    shapeArray = [
                  [10,10],[50,50],[100,100],[500,500],
                  [1000,1000],[2000,2000],[3000,3000],
                  [4000,4000],[5000,5000]
                ]

    shapeArray.each do |shape|
      puts "shape"+shape.to_s
      cpu_ary1 = Array.new(shape[0]*shape[1]) { |index| index }
      cpu_ary2 = Array.new(shape[0]*shape[1]) { |index| index * 2 }
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
      # prints the result of the [10 x 10] [10 x 10] matrix product
#       if shape == shapeArray[0]
#           puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res, shape[0]*shape[1], :cudaMemcpyDeviceToHost);
#       end
      RbCUDA::CuBLAS_v2.cublasDestroy_v2(handle)
      RbCUDA::Runtime.cudaFree(gpu_ary1)
      RbCUDA::Runtime.cudaFree(gpu_ary2)
      RbCUDA::Runtime.cudaFree(gpu_ary_res)
    end

  end

end

ResultCollect.generate
