require_relative '../lib/rbcuda.so'


require 'benchmark'
require 'json'

class ResultCollect


  def self.generate

    iters = 0
    result = {}

    result[:mat_mult] = []

    shapeArray = [
                  [10,10],[50,50],[100,100],[500,500],
                  [1000,1000],[2000,2000],[3000,3000],
                  [4000,4000],[5000,5000]
                ]

    shapeArray.each do |shape|
      cpu_ary1 = Array.new(shape[0]*shape[1]) { rand(1...999999) }
      cpu_ary2 = Array.new(shape[0]*shape[1]) { rand(1...999999) }


      gpu_ary1    = cudaMalloc(shape[0] * shape[1]);
      gpu_ary2    = cudaMalloc(shape[0] * shape[1]);
      gpu_ary_res = cudaMalloc(shape[0] * shape[1]);

      cudaMemcpy(gpu_ary1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice);
      cudaMemcpy(gpu_ary2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice);

      gpu_blas_mmul(gpu_ary1, gpu_ary2, gpu_ary_res, shape[0], shape[1], shape[1]);

      cudaMemcpy(cpu_ary_res, gpu_ary_res, shape[0]*shape[1], :cudaMemcpyDeviceToHost);

      cudaFree(gpu_ary1)
      cudaFree(gpu_ary2)
      cudaFree(gpu_ary_res)
    end

  end
end

ResultCollect.generate