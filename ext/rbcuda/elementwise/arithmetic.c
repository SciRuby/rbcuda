static VALUE rb_elementwise_addition(VALUE self, VALUE array1_val, VALUE array2_val, VALUE size){

  char* module_file = "/home/prasun/dev/rbcuda/ext/rbcuda/elementwise/kernel.ptx";
  char* kernel = "Vector_Addition";
  dev_ptr* ptr_a;
  dev_ptr* ptr_b;
  dev_ptr* ptr_c = ALLOC(dev_ptr);
  Data_Get_Struct(array1_val, dev_ptr, ptr_a);
  Data_Get_Struct(array2_val, dev_ptr, ptr_b);

  cudaMalloc((void **)&ptr_c->carray, sizeof(double) * NUM2INT(size));
  initCUDA(module_file, kernel);

  printf("# Running the kernel...\n");

  // initialize
  printf("- Initializing...\n");
  initCUDA(module_file, kernel);

  // allocate memory
  runKernel2(ptr_a->carray, ptr_b->carray, ptr_c->carray);

  printf("# Kernel complete.\n");
  finalizeCUDA();

  double elements[100];

  cudaMemcpy((void*)elements, (void*)ptr_c->carray, sizeof(double)* NUM2UINT(size), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 100; ++i)
  {
    printf("%d\n", elements[i]);
  }

  return Data_Wrap_Struct(Dev_Array, NULL, rbcu_free, ptr_c);
}

