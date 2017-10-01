static VALUE rb_dev_ary_to_nmatrix(VALUE self, VALUE shape) {
  dev_ptr* ptr;
  Data_Get_Struct(self, dev_ptr, ptr);

  size_t* shape_ary = ALLOC_N(size_t, 2);
  size_t count = 1;
  for (size_t index = 0; index < 2; index++) {
    shape_ary[index] = (size_t)NUM2LONG(RARRAY_AREF(shape, index));
    count *= (size_t)NUM2LONG(RARRAY_AREF(shape, index));
  }
  size_t size = sizeof(double)*count;

  double* elements = ALLOC_N(double, count);

  cudaMemcpy((void*)elements, (void*)ptr->carray, sizeof(double)*count, cudaMemcpyDeviceToHost);

  return rb_nmatrix_dense_create(nm::FLOAT64, shape_ary, 2, elements, (int)count);
}

extern VALUE rb_nmatrix_to_gpu_ary_method(VALUE nmatrix) {
  if (NM_DIM(nmatrix) > 3) {
    rb_raise(rb_eStandardError,
      "NMatrix must not have greater than 4 dimensions.");
  }

  if (NM_DTYPE(nmatrix) == nm::FLOAT64) {
    return Data_Wrap_Struct(Dev_Array, NULL, rbcu_free, rb_nmatrix_to_dev_ary(nmatrix));
  }
  else {
    rb_raise(rb_eStandardError,
      "NMatrix should be either :complex64, :complex128, :int32 or :float64 type.");
  }
  return Qnil;
}


dev_ptr* rb_nmatrix_to_dev_ary(VALUE nm) {
  DENSE_STORAGE* nmat = NM_STORAGE_DENSE(nm);
  dev_ptr* ptr = ALLOC(dev_ptr);

  if (nmat->dtype != nm::FLOAT64) {
    rb_raise(rb_eStandardError, "requires dtype of :float64 to convert to an Af_Array");
  }

  cudaMemcpy((void*)ptr->carray, (void*)nmat->elements, sizeof(double) * nmat->count, cudaMemcpyHostToDevice);

  return ptr;
}
