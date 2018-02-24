require_relative '../lib/rbcuda.so'
require 'tempfile'

def self.compile(src_str)
  src_file = Tempfile.new(["kernel_src", ".cu"])
  src_file.write(src_str)
  src_file.close

  out_file = Tempfile.new(["kernel", ".ptx"])
  out_file.close

  case RbConfig::CONFIG['target_os']
    when /darwin/    # Build universal binary for i386 and x86_64 platforms.
      f32 = Tempfile.new(["kernel32", ".so"])
      f64 = Tempfile.new(["kernel64", ".so"])
      f32.close
      f64.close
      system %{nvcc -shared -m32 -Xcompiler -fPIC #{src_file.path} -o #{f32.path}}
      system %{nvcc -shared -m64 -Xcompiler -fPIC #{src_file.path} -o #{f64.path}}
      system %{lipo -arch i386 #{f32.path} -arch x86_64 #{f64.path} -create -output #{out_file.path}}
    else    # Build default platform binary.
      system %{nvcc -ptx -Xcompiler -fPIC #{src_file.path} -o #{out_file.path}}
  end

  out_file
end

vadd_kernel_src = <<-EOS
  extern "C" {
    __global__ void matSum(double *a, double *b, double *c)
    {
      int tid = blockIdx.x;
      if (tid < 100)
          c[tid] = a[tid] + b[tid];
    }


  }
  EOS

module_file = compile(vadd_kernel_src)
puts module_file.path


RbCUDA::CUDA.cuInit(0);

device = RbCUDA::CUDA.cuDeviceGet(0);
puts device

puts RbCUDA::CUDA.cuDeviceGetCount;

puts RbCUDA::CUDA.cuDeviceGetName(100, device);
puts (RbCUDA::CUDA.cuDeviceTotalMem_v2(device)/(1024**2)).to_s + " MB"

ctx = RbCUDA::CUDA.cuCtxCreate_v2(0, device)

puts ctx

mod = RbCUDA::CUDA.cuModuleLoad(module_file.path);
puts mod;

func =  RbCUDA::CUDA.cuModuleGetFunction(mod, "matSum")
puts func

shape = [10000, 1]
N = 10000

cpu_ary1 = Array.new(shape[0]*shape[1]) { 2 }
cpu_ary2 = Array.new(shape[0]*shape[1]) { 2 }
cpu_ary_res = []


gpu_ary1    = RbCUDA::Runtime.cudaMalloc(shape);
gpu_ary2    = RbCUDA::Runtime.cudaMalloc(shape);
gpu_ary_res = RbCUDA::Runtime.cudaMalloc(shape);

RbCUDA::Runtime.cudaMemcpy(gpu_ary1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice);
RbCUDA::Runtime.cudaMemcpy(gpu_ary2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice);

puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary1, shape[0]*shape[1], :cudaMemcpyDeviceToHost)[2];

args = [gpu_ary1, gpu_ary2, gpu_ary_res]
RbCUDA::CUDA.cuLaunchKernel(func, N, 1, 1,
                                  1, 1, 1,
                                  0, 0, args, 0 );

puts RbCUDA::Runtime.cudaMemcpy([], args[2], shape[0]*shape[1], :cudaMemcpyDeviceToHost)[2];

puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res, shape[0]*shape[1], :cudaMemcpyDeviceToHost)[2];