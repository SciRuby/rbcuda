const char* const Cuda_MemcpyKind[5] = {
  "cudaMemcpyHostToHost",
  "cudaMemcpyHostToDevice",
  "cudaMemcpyDeviceToHost",
  "cudaMemcpyDeviceToDevice",
  "cudaMemcpyDefault"
};

cudaMemcpyKind rbcu_memcopy_kind(VALUE sym) {
  ID sym_id = SYM2ID(sym);

  for (size_t index = 0; index < 5; ++index) {
    if (sym_id == rb_intern(Cuda_MemcpyKind[index])) {
      return static_cast<cudaMemcpyKind>(index);
    }
  }

  VALUE str = rb_any_to_s(sym);
  rb_raise(rb_eArgError, "invalid data type symbol (:%s) specified", RSTRING_PTR(str));
}

