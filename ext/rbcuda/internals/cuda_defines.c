typedef unsigned long CUdeviceptr;

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

std::map<char*, size_t> Cuda_OutputMode = {
  {"cudaKeyValuePair", 0x00},
  {"cudaCSV",          0x01},
};

cudaOutputMode_t rb_cuda_output_from_rbsymbol(VALUE sym) {
  ID sym_id = SYM2ID(sym);

  for(std::map<char*, size_t>::value_type& entry : Cuda_OutputMode) {
    if (sym_id == rb_intern(entry.first)) {
      return static_cast<cudaOutputMode_t>(entry.second);
    }
  }

  VALUE str = rb_any_to_s(sym);
  rb_raise(rb_eArgError, "invalid matrix type symbol (:%s) specified", RSTRING_PTR(str));
}

std::map<char*, size_t> CUresult_enum = {
  {"CUDA_SUCCESS", 0},
  {"CUDA_ERROR_INVALID_VALUE", 1},
  {"CUDA_ERROR_OUT_OF_MEMORY", 2},
  {"CUDA_ERROR_NOT_INITIALIZED", 3},
  {"CUDA_ERROR_DEINITIALIZED", 4},
  {"CUDA_ERROR_PROFILER_DISABLED", 5},
  {"CUDA_ERROR_PROFILER_NOT_INITIALIZED", 6},
  {"CUDA_ERROR_PROFILER_ALREADY_STARTED", 7},
  {"CUDA_ERROR_PROFILER_ALREADY_STOPPED", 8},
  {"CUDA_ERROR_NO_DEVICE", 100},
  {"CUDA_ERROR_INVALID_DEVICE", 101},
  {"CUDA_ERROR_INVALID_IMAGE", 200},
  {"CUDA_ERROR_INVALID_CONTEXT", 201},
  {"CUDA_ERROR_CONTEXT_ALREADY_CURRENT", 202},
  {"CUDA_ERROR_MAP_FAILED", 205},
  {"CUDA_ERROR_UNMAP_FAILED", 206},
  {"CUDA_ERROR_ARRAY_IS_MAPPED", 207},
  {"CUDA_ERROR_ALREADY_MAPPED", 208},
  {"CUDA_ERROR_NO_BINARY_FOR_GPU", 209},
  {"CUDA_ERROR_ALREADY_ACQUIRED", 210},
  {"CUDA_ERROR_NOT_MAPPED", 211},
  {"CUDA_ERROR_NOT_MAPPED_AS_ARRAY", 212},
  {"CUDA_ERROR_NOT_MAPPED_AS_POINTER", 213},
  {"CUDA_ERROR_ECC_UNCORRECTABLE", 214},
  {"CUDA_ERROR_UNSUPPORTED_LIMIT", 215},
  {"CUDA_ERROR_CONTEXT_ALREADY_IN_USE", 216},
  {"CUDA_ERROR_PEER_ACCESS_UNSUPPORTED", 217},
  {"CUDA_ERROR_INVALID_PTX", 218},
  {"CUDA_ERROR_INVALID_GRAPHICS_CONTEXT", 219},
  {"CUDA_ERROR_INVALID_SOURCE", 300},
  {"CUDA_ERROR_FILE_NOT_FOUND", 301},
  {"CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", 302},
  {"CUDA_ERROR_SHARED_OBJECT_INIT_FAILED", 303},
  {"CUDA_ERROR_OPERATING_SYSTEM", 304},
  {"CUDA_ERROR_INVALID_HANDLE", 400},
  {"CUDA_ERROR_NOT_FOUND", 500},
  {"CUDA_ERROR_NOT_READY", 600},
  {"CUDA_ERROR_ILLEGAL_ADDRESS", 700},
  {"CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 701},
  {"CUDA_ERROR_LAUNCH_TIMEOUT", 702},
  {"CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", 703},
  {"CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", 704},
  {"CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", 705},
  {"CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", 708},
  {"CUDA_ERROR_CONTEXT_IS_DESTROYED", 709},
  {"CUDA_ERROR_ASSERT", 710},
  {"CUDA_ERROR_TOO_MANY_PEERS", 711},
  {"CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED", 712},
  {"CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED", 713},
  {"CUDA_ERROR_HARDWARE_STACK_ERROR", 714},
  {"CUDA_ERROR_ILLEGAL_INSTRUCTION", 715},
  {"CUDA_ERROR_MISALIGNED_ADDRESS", 716},
  {"CUDA_ERROR_INVALID_ADDRESS_SPACE", 717},
  {"CUDA_ERROR_INVALID_PC", 718},
  {"CUDA_ERROR_LAUNCH_FAILED", 719},
  {"CUDA_ERROR_NOT_PERMITTED", 800},
  {"CUDA_ERROR_NOT_SUPPORTED", 801},
  {"CUDA_ERROR_UNKNOWN", 999}
};

CUresult rb_cuda_cu_result_from_rbsymbol(VALUE sym) {
  ID sym_id = SYM2ID(sym);

  for(std::map<char*, size_t>::value_type& entry : CUresult_enum) {
    if (sym_id == rb_intern(entry.first)) {
      return static_cast<CUresult>(entry.second);
    }
  }

  VALUE str = rb_any_to_s(sym);
  rb_raise(rb_eArgError, "invalid CUresult type symbol (:%s) specified", RSTRING_PTR(str));
}

