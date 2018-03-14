// cusolverStatus_t CUDENSEAPI cusolverDnCreate(cusolverDnHandle_t *handle);
static VALUE rb_cusolverDnCreate(VALUE self){
  rb_cusolver_handle* handler = ALLOC(rb_cusolver_handle);
  cusolverStatus_t status = cusolverDnCreate(&handler->handle);

  return Data_Wrap_Struct(CuSolverHandler, NULL, rbcu_free, handler);
}

// cusolverStatus_t CUDENSEAPI cusolverDnDestroy(cusolverDnHandle_t handle);
static VALUE rb_cusolverDnDestroy(VALUE self, VALUE handler_val){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  cusolverStatus_t status = cusolverDnDestroy(handler->handle);
  return Qtrue;
}

// cusolverStatus_t CUDENSEAPI cusolverDnSetStream (cusolverDnHandle_t handle, cudaStream_t streamId);
static VALUE rb_cusolverDnSetStream(VALUE self, VALUE handler_val, VALUE stream_id){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  cudastream_ptr* stream_ptr;
  Data_Get_Struct(stream_id, cudastream_ptr, stream_ptr);

  cusolverStatus_t status = cusolverDnSetStream(handler->handle, stream_ptr->stream);

  return Qtrue;
}

// cusolverStatus_t CUDENSEAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId);
static VALUE rb_cusolverDnGetStream(VALUE self, VALUE handler_val){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  cudastream_ptr* stream_ptr = ALLOC(cudastream_ptr);

  cusolverStatus_t status = cusolverDnGetStream(handler->handle, &stream_ptr->stream);

  return Data_Wrap_Struct(RbCuCUDAStream, NULL, rbcu_free, stream_ptr);
}

/* Cholesky factorization and its solver */
// cusolverStatus_t CUDENSEAPI cusolverDnSpotrf_bufferSize( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, int *Lwork );
static VALUE rb_cusolverDnSpotrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDpotrf_bufferSize( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork );
static VALUE rb_cusolverDnDpotrf_bufferSize(VALUE self, VALUE handler_val, VALUE uplo, VALUE n, VALUE A, VALUE lda, VALUE Lwork){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  lwork_size = NUM2INT(Lwork);

  cusolverStatus_t status = cusolverDnDpotrf_bufferSize(handler->handle, rbcu_cublasFillMode_t(uplo), NUM2INT(n),
                                                          ptr_A->carray, NUM2INT(lda), &lwork_size );

  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCpotrf_bufferSize( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, int *Lwork );
static VALUE rb_cusolverDnCpotrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZpotrf_bufferSize( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, int *Lwork);
static VALUE rb_cusolverDnZpotrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnSpotrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnSpotrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDpotrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnDpotrf(VALUE self, VALUE handler_val, VALUE uplo, VALUE n, VALUE A, VALUE lda, VALUE Workspace, VALUE Lwork, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  cusolverStatus_t status = cusolverDnDpotrf(handler->handle, rbcu_cublasFillMode_t(uplo), NUM2INT(n), ptr_A->carray,
                                              NUM2INT(lda), double *Workspace, NUM2INT(Lwork), int *devInfo );

  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCpotrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, cuComplex *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnCpotrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZpotrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnZpotrf(VALUE self){
  return Qnil;
}


// cusolverStatus_t CUDENSEAPI cusolverDnSpotrs( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float *A, int lda, float *B, int ldb, int *devInfo);
static VALUE rb_cusolverDnSpotrs(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDpotrs( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo);
static VALUE rb_cusolverDnDpotrs(VALUE self, VALUE handler_val, VALUE uplo, VALUE n, VALUE nrhs, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  dev_ptr* ptr_B;
  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(B, dev_ptr, ptr_B);

  cusolverStatus_t status = cusolverDnDpotrs(handler->handle, rbcu_cublasFillMode_t(uplo), NUM2INT(n), NUM2INT(nrhs),
                                              ptr_A->carray, NUM2INT(lda), ptr_B->carray, NUM2INT(ldb), int *devInfo);

  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCpotrs( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuComplex *A, int lda, cuComplex *B, int ldb, int *devInfo);
static VALUE rb_cusolverDnCpotrs(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZpotrs( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const cuDoubleComplex *A, int lda, cuDoubleComplex *B, int ldb, int *devInfo);
static VALUE rb_cusolverDnZpotrs(VALUE self){
  return Qnil;
}


/* LU Factorization */
// cusolverStatus_t CUDENSEAPI cusolverDnSgetrf_bufferSize( cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork );
static VALUE rb_cusolverDnSgetrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgetrf_bufferSize( cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork );
static VALUE rb_cusolverDnDgetrf_bufferSize(VALUE self, VALUE handler_val, VALUE m, VALUE n, VALUE A, VALUE lda, VALUE Lwork){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  lwork_size = NUM2INT(Lwork);

  cusolverStatus_t status = cusolverDnDgetrf_bufferSize(handler->handle, NUM2INT(m), NUM2INT(n), ptr_A->carray, NUM2INT(lda), &lwork_size );

  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgetrf_bufferSize( cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, int *Lwork );
static VALUE rb_cusolverDnCgetrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgetrf_bufferSize( cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, int *Lwork );
static VALUE rb_cusolverDnZgetrf_bufferSize(VALUE self){
  return Qnil;
}


// cusolverStatus_t CUDENSEAPI cusolverDnSgetrf( cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo );
static VALUE rb_cusolverDnSgetrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgetrf( cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo );
static VALUE rb_cusolverDnDgetrf(VALUE self, VALUE handler_val, VALUE m, VALUE n, VALUE A, VALUE lda, VALUE Workspace, VALUE devIpiv, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  cusolverStatus_t status = cusolverDnDgetrf(handler->handle, NUM2INT(m), NUM2INT(n), ptr_A->carray, NUM2INT(lda), double *Workspace,
                                              int *devIpiv, int *devInfo );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgetrf( cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, cuComplex *Workspace, int *devIpiv, int *devInfo );
static VALUE rb_cusolverDnCgetrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgetrf( cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *Workspace, int *devIpiv, int *devInfo );
static VALUE rb_cusolverDnZgetrf(VALUE self){
  return Qnil;
}

/* Row pivoting */
// cusolverStatus_t CUDENSEAPI cusolverDnSlaswp( cusolverDnHandle_t handle, int n, float *A, int lda, int k1, int k2, const int *devIpiv, int incx);
static VALUE rb_cusolverDnSlaswp(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDlaswp( cusolverDnHandle_t handle, int n, double *A, int lda, int k1, int k2, const int *devIpiv, int incx);
static VALUE rb_cusolverDnDlaswp(VALUE self, VALUE handler_val, VALUE n, VALUE A, VALUE lda, VALUE k1, VALUE k2, VALUE devIpiv, VALUE incx){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  cusolverStatus_t status = cusolverDnDlaswp(handler->handle, NUM2INT(n), ptr_A->carray, NUM2INT(lda), NUM2INT(k1), NUM2INT(k2),
                                              const int *devIpiv, NUM2INT(incx));
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnClaswp( cusolverDnHandle_t handle, int n, cuComplex *A, int lda, int k1, int k2, const int *devIpiv, int incx);
static VALUE rb_cusolverDnClaswp(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZlaswp( cusolverDnHandle_t handle, int n, cuDoubleComplex *A, int lda, int k1, int k2, const int *devIpiv, int incx);
static VALUE rb_cusolverDnZlaswp(VALUE self){
  return Qnil;
}

/* LU solve */
// cusolverStatus_t CUDENSEAPI cusolverDnSgetrs( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, int lda, const int *devIpiv, float *B, int ldb, int *devInfo );
static VALUE rb_cusolverDnSgetrs(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgetrs( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo );
static VALUE rb_cusolverDnDgetrs(VALUE self, VALUE handler_val, VALUE trans, VALUE n, VALUE nrhs, VALUE A, VALUE lda, VALUE devIpiv, VALUE B, VALUE ldb, VALUE devInfo ){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  dev_ptr* ptr_B;
  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(B, dev_ptr, ptr_B);

  cusolverStatus_t status = cusolverDnDgetrs(handler->handle, rbcu_cublasOperation_t(trans), NUM2INT(n), NUM2INT(nrhs),
                                              ptr_A->carray, NUM2INT(lda), const int *devIpiv, double *B, int ldb, int *devInfo );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgetrs( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex *A, int lda, const int *devIpiv, cuComplex *B, int ldb, int *devInfo );
static VALUE rb_cusolverDnCgetrs(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgetrs( cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex *A, int lda, const int *devIpiv, cuDoubleComplex *B, int ldb, int *devInfo );
static VALUE rb_cusolverDnZgetrs(VALUE self){
  return Qnil;
}

/* QR factorization */
// cusolverStatus_t CUDENSEAPI cusolverDnSgeqrf( cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *TAU, float *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnSgeqrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgeqrf( cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *TAU, double *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnDgeqrf(VALUE self, VALUE handler_val, VALUE m, VALUE n, VALUE A, VALUE lda, VALUE TAU, VALUE Workspace, VALUE Lwork, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  cusolverStatus_t status = cusolverDnDgeqrf(handler->handle, NUM2INT(m), NUM2INT(n), ptr_A->carray, NUM2INT(lda),
                                              double *TAU, double *Workspace, NUM2INT(Lwork), int *devInfo );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgeqrf( cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, cuComplex *TAU, cuComplex *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnCgeqrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgeqrf( cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, cuDoubleComplex *TAU, cuDoubleComplex *Workspace, int Lwork, int *devInfo );
static VALUE rb_cusolverDnZgeqrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnSormqr( cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float *A, int lda, const float *tau, float *C, int ldc, float *work, int lwork, int *devInfo);
static VALUE rb_cusolverDnSormqr(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDormqr( cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double *A, int lda, const double *tau, double *C, int ldc, double *work, int lwork, int *devInfo);
static VALUE rb_cusolverDnDormqr(VALUE self, VALUE handler_val, VALUE side, VALUE trans, VALUE m, VALUE n, VALUE k, VALUE A, VALUE lda, VALUE tau, VALUE C, VALUE ldc, VALUE work, VALUE lwork, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  dev_ptr* ptr_C;
  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(C, dev_ptr, ptr_C);

  cusolverStatus_t status = cusolverDnDormqr(handler->handle, rbcu_cublasSideMode_t(side), rbcu_cublasOperation_t(trans),
                                              NUM2INT(m), NUM2INT(n), NUM2INT(k), ptr_A->carray, NUM2INT(lda), const double *tau,
                                              ptr_C->carray, NUM2INT(ldc), double *work, NUM2INT(lwork), int *devInfo);
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCunmqr( cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuComplex *A, int lda, const cuComplex *tau, cuComplex *C, int ldc, cuComplex *work, int lwork, int *devInfo);
static VALUE rb_cusolverDnCunmqr(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZunmqr( cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const cuDoubleComplex *A, int lda, const cuDoubleComplex *tau, cuDoubleComplex *C, int ldc, cuDoubleComplex *work, int lwork, int *devInfo);
static VALUE rb_cusolverDnZunmqr(VALUE self){
  return Qnil;
}


/* QR factorization workspace query */
// cusolverStatus_t CUDENSEAPI cusolverDnSgeqrf_bufferSize( cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork );
static VALUE rb_cusolverDnSgeqrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgeqrf_bufferSize( cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork );
static VALUE rb_cusolverDnDgeqrf_bufferSize(VALUE self, VALUE handler_val, VALUE m, VALUE n, VALUE A, VALUE lda, VALUE Lwork){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  lwork_size = NUM2INT(Lwork);

  cusolverStatus_t status = cusolverDnDgeqrf_bufferSize(handler->handle, NUM2INT(m), NUM2INT(n), ptr_A->carray, NUM2INT(lda), &lwork_size);
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgeqrf_bufferSize( cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, int *Lwork );
static VALUE rb_cusolverDnCgeqrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgeqrf_bufferSize( cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, int *Lwork );
static VALUE rb_cusolverDnZgeqrf_bufferSize(VALUE self){
  return Qnil;
}


/* bidiagonal */
// cusolverStatus_t CUDENSEAPI cusolverDnSgebrd( cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *D, float *E, float *TAUQ, float *TAUP, float *Work, int Lwork, int *devInfo );
static VALUE rb_cusolverDnSgebrd(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgebrd( cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *D, double *E, double *TAUQ, double *TAUP, double *Work, int Lwork, int *devInfo );
static VALUE rb_cusolverDnDgebrd(VALUE self, VALUE handler_val, VALUE m, VALUE n, VALUE A, VALUE lda, VALUE D, VALUE E, VALUE TAUQ, VALUE TAUP, VALUE Work, VALUE Lwork, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  cusolverStatus_t status = cusolverDnDgebrd(handler->handle, NUM2INT(m), NUM2INT(n), ptr_A->carray, NUM2INT(lda), double *D,
                                              double *E, double *TAUQ, double *TAUP, double *Work, NUM2INT(Lwork), int *devInfo );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgebrd( cusolverDnHandle_t handle, int m, int n, cuComplex *A, int lda, float *D, float *E, cuComplex *TAUQ, cuComplex *TAUP, cuComplex *Work, int Lwork, int *devInfo );
static VALUE rb_cusolverDnCgebrd(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgebrd( cusolverDnHandle_t handle, int m, int n, cuDoubleComplex *A, int lda, double *D, double *E, cuDoubleComplex *TAUQ, cuDoubleComplex *TAUP, cuDoubleComplex *Work, int Lwork, int *devInfo );
static VALUE rb_cusolverDnZgebrd(VALUE self){
  return Qnil;
}


// cusolverStatus_t CUDENSEAPI cusolverDnSsytrd( cusolverDnHandle_t handle, signed char uplo, int n, float *A, int lda, float *D, float *E, float *tau, float *Work, int Lwork, int *info);
static VALUE rb_cusolverDnSsytrd(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDsytrd( cusolverDnHandle_t handle, signed char uplo, int n, double *A, int lda, double *D, double *E, double *tau, double *Work, int Lwork, int *info);
static VALUE rb_cusolverDnDsytrd(VALUE self, VALUE handler_val, VALUE uplo, VALUE n, VALUE A, VALUE lda, VALUE D, VALUE E, VALUE tau, VALUE Work, VALUE Lwork, VALUE info){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  cusolverStatus_t status = cusolverDnDsytrd(handler->handle, signed char uplo, NUM2INT(n), ptr_A->carray, NUM2INT(lda),
                                              double *D, double *E, double *tau, double *Work, NUM2INT(Lwork), int *info);
  return Qnil;
}

/* bidiagonal factorization workspace query */
// cusolverStatus_t CUDENSEAPI cusolverDnSgebrd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnSgebrd_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgebrd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnDgebrd_bufferSize(VALUE self, VALUE handler_val, VALUE m, VALUE n, VALUE Lwork){
  cusolverStatus_t status = cusolverDnDgebrd_bufferSize(handler->handle, NUM2INT(m), NUM2INT(n), int *Lwork );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgebrd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnCgebrd_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgebrd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnZgebrd_bufferSize(VALUE self){
  return Qnil;
}

/* singular value decomposition, A = U * Sigma * V^H */
// cusolverStatus_t CUDENSEAPI cusolverDnSgesvd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnSgesvd_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgesvd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnDgesvd_bufferSize(VALUE self, VALUE handler_val, VALUE m, VALUE n, VALUE Lwork){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  cusolverStatus_t status = cusolverDnDgesvd_bufferSize(handler->handle, NUM2INT(m), NUM2INT(n), int *Lwork );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgesvd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnCgesvd_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgesvd_bufferSize( cusolverDnHandle_t handle, int m, int n, int *Lwork );
static VALUE rb_cusolverDnZgesvd_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnSgesvd ( cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *Work, int Lwork, float *rwork, int  *devInfo );
static VALUE rb_cusolverDnSgesvd(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDgesvd ( cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *Work, int Lwork, double *rwork, int *devInfo );
static VALUE rb_cusolverDnDgesvd(VALUE self, VALUE handler_val, VALUE jobu, VALUE jobvt, VALUE m, VALUE n, VALUE A, VALUE lda, VALUE S, VALUE U, VALUE ldu, VALUE VT, VALUE ldvt, VALUE Work, VALUE Lwork, VALUE rwork, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  cusolverStatus_t status = cusolverDnDgesvd(handler->handle, signed char jobu, signed char jobvt, NUM2INT(m),
                                              NUM2INT(n), ptr_A->carray, NUM2INT(lda), double *S, double *U,
                                              NUM2INT(ldu), double *VT, NUM2INT(ldvt), double *Work, NUM2INT(Lwork),
                                              double *rwork, int *devInfo );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCgesvd ( cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuComplex *A, int lda, float *S, cuComplex *U, int ldu, cuComplex *VT, int ldvt, cuComplex *Work, int Lwork, float *rwork, int *devInfo );
static VALUE rb_cusolverDnCgesvd(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZgesvd ( cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, cuDoubleComplex *A, int lda, double *S, cuDoubleComplex *U, int ldu, cuDoubleComplex *VT, int ldvt, cuDoubleComplex *Work, int Lwork, double *rwork, int *devInfo );
static VALUE rb_cusolverDnZgesvd(VALUE self){
  return Qnil;
}

/* LDLT,UDUT factorization */
// cusolverStatus_t CUDENSEAPI cusolverDnSsytrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, int *ipiv, float *work, int lwork, int *devInfo );
static VALUE rb_cusolverDnSsytrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDsytrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *ipiv, double *work, int lwork, int *devInfo );
static VALUE rb_cusolverDnDsytrf(VALUE self, VALUE handler_val, VALUE uplo, VALUE n, VALUE A, VALUE lda, VALUE ipiv, VALUE work, VALUE lwork, VALUE devInfo){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  cusolverStatus_t status = cusolverDnDsytrf(handler->handle, rbcu_cublasFillMode_t(uplo), NUM2INT(n), ptr_A->carray,
                                              NUM2INT(lda), int *ipiv, double *work, NUM2INT(lwork), int *devInfo );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCsytrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuComplex *A, int lda, int *ipiv, cuComplex *work, int lwork, int *devInfo );
static VALUE rb_cusolverDnCsytrf(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZsytrf( cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, cuDoubleComplex *A, int lda, int *ipiv, cuDoubleComplex *work, int lwork, int *devInfo );
static VALUE rb_cusolverDnZsytrf(VALUE self){
  return Qnil;
}

/* SYTRF factorization workspace query */
// cusolverStatus_t CUDENSEAPI cusolverDnSsytrf_bufferSize( cusolverDnHandle_t handle, int n, float *A, int lda, int *Lwork );
static VALUE rb_cusolverDnSsytrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnDsytrf_bufferSize( cusolverDnHandle_t handle, int n, double *A, int lda, int *Lwork );
static VALUE rb_cusolverDnDsytrf_bufferSize(VALUE self, VALUE handler_val, VALUE n, VALUE A, VALUE lda, VALUE Lwork){
  rb_cusolver_handle* handler;
  Data_Get_Struct(handler_val, rb_cusolver_handle, handler);

  dev_ptr* ptr_A;
  Data_Get_Struct(A, dev_ptr, ptr_A);

  lwork_size = NUM2INT(Lwork);

  cusolverStatus_t status = cusolverDnDsytrf_bufferSize(handler->handle, NUM2INT(n), ptr_A->carray, NUM2INT(lda), &lwork_size );
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnCsytrf_bufferSize( cusolverDnHandle_t handle, int n, cuComplex *A, int lda, int *Lwork );
static VALUE rb_cusolverDnCsytrf_bufferSize(VALUE self){
  return Qnil;
}

// cusolverStatus_t CUDENSEAPI cusolverDnZsytrf_bufferSize( cusolverDnHandle_t handle, int n, cuDoubleComplex *A, int lda, int *Lwork );
static VALUE rb_cusolverDnZsytrf_bufferSize(VALUE self){
  return Qnil;
}
