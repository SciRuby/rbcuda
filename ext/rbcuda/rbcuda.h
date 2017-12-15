#ifndef RUBY_CUDA_H
  #define RUBY_CUDA_H
#endif


typedef struct DEV_PTR
{
  double* carray;
}dev_ptr;

typedef struct CONTEXT_PTR
{
  CUcontext ctx;
}ctx_ptr;

typedef struct MODULE_PTR
{
  CUmodule module;
}mod_ptr;

typedef struct FUNCTION_PTR
{
  CUfunction function;
}function_ptr;

typedef struct TEXTURE_PTR{
  CUtexref texture;
}texture_ptr;

typedef struct SURFACE_PTR{
  CUsurfref surface;
}surface_ptr;

typedef struct RB_CUBLAS_HANDLE
{
  cublasHandle_t handle;
}rb_cublas_handle;


#ifndef HAVE_RB_ARRAY_CONST_PTR
static inline const VALUE *
rb_array_const_ptr(VALUE a)
{
  return FIX_CONST_VALUE_PTR((RBASIC(a)->flags & RARRAY_EMBED_FLAG) ?
    RARRAY(a)->as.ary : RARRAY(a)->as.heap.ptr);
}
#endif

#ifndef RARRAY_CONST_PTR
# define RARRAY_CONST_PTR(a) rb_array_const_ptr(a)
#endif

#ifndef RARRAY_AREF
# define RARRAY_AREF(a, i) (RARRAY_CONST_PTR(a)[i])
#endif

/*
 * Functions
 */

#ifdef __cplusplus
typedef VALUE (*METHOD)(...);
//}; // end of namespace nm

// Interfaces

#endif


#ifdef __cplusplus
extern "C" {
#endif

  void Init_rbcuda();

#ifdef __cplusplus
}
#endif
