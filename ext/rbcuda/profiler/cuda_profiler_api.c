static VALUE rb_cudaProfilerInitialize(VALUE self, VALUE config_file_val, VALUE output_file_val, VALUE output_mode_val){
  char* config_file = StringValueCStr(config_file_val);
  char* output_file = StringValueCStr(output_file_val);
  cudaProfilerInitialize(config_file, output_file, rb_cuda_output_from_rbsymbol(output_mode_val));
  return Qtrue;
}

static VALUE rb_cudaProfilerStart(VALUE self){
  cudaProfilerStart();
  return Qtrue;
}

static VALUE rb_cudaProfilerStop(VALUE self){
  cudaProfilerStop();
  return Qtrue;
}
