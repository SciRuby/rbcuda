require_relative '../lib/rbcuda.so'
require 'tempfile'

module Add

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

  def self.code

    vadd_kernel_src = <<-EOS
    extern "C" {
      __global__ void matSum(int *a, int *b, int *c)
      {
        int tid = blockIdx.x;
        if (tid < 100)
            c[tid] = a[tid] + b[tid];
      }


    }
    EOS

    f = compile(vadd_kernel_src)
    puts f.path

    RbCUDA::Driver.test_kernel(f.path)
  end
end

Add.code()