// cublasStatus_t cublasXtCreate(cublasXtHandle_t *handle)
// This function initializes the cublasXt API and creates a handle to an opaque structure
// holding the cublasXt API context. It allocates hardware resources on the host and device
// and must be called prior to making any other cublasXt API calls.
// Return Value Meaning
// CUBLAS_STATUS_SUCCESS the initialization succeeded
// CUBLAS_STATUS_ALLOC_FAILED the resources could not be allocated
// CUBLAS_STATUS_NOT_SUPPORTED cublasXt API is only supported on 64-bit platform

static VALUE rb_cublasXtCreate(VALUE self){
  // cublasStatus_t status = cublasXtCreate(cublasXtHandle_t *handle)

  return Qnil;
}

// cublasStatus_t cublasXtDestroy(cublasXtHandle_t handle)
// This function releases hardware resources used by the cublasXt API context. The release
// of GPU resources may be deferred until the application exits. This function is usually the
// last call with a particular handle to the cublasXt API.
// Return Value Meaning
// CUBLAS_STATUS_SUCCESS the shut down succeeded
// CUBLAS_STATUS_NOT_INITIALIZED the library was not initialized

static VALUE rb_cublasXtDestroy(VALUE self){
  // cublasStatus_t status = cublasXtDestroy(cublasXtHandle_t handle)

  return Qnil;
}

// cublasStatus_t cublasXtGetNumBoards (int nbDevices, int* deviceId, int* nbBoards);
static VALUE rb_cublasXtGetNumBoards(VALUE self){
  // cublasStatus_t status = cublasXtGetNumBoards (int nbDevices, int* deviceId, int* nbBoards);

  return Qnil;
}

// cublasStatus_t cublasXtMaxBoards (int* nbGpuBoards);

static VALUE rb_cublasXtMaxBoards(VALUE self){
  // cublasStatus_t cublasXtMaxBoards (int* nbGpuBoards);
  return Qnil;
}

/* This routine selects the Gpus that the user want to use for CUBLAS-XT */
// cublasStatus_t cublasXtDeviceSelect (cublasXtHandle_t handle, int nbDevices, int* deviceId);
static VALUE rb_cublasXtDeviceSelect(VALUE self){
  // cublasStatus_t status = cublasXtDeviceSelect (cublasXtHandle_t handle, int nbDevices, int* deviceId);

  return Qnil;
}

/* This routine allows to change the dimension of the tiles ( blockDim x blockDim ) */
// cublasStatus_t cublasXtSetBlockDim (cublasXtHandle_t handle, int blockDim);
static VALUE rb_cublasXtSetBlockDim(VALUE self, VALUE handle, VALUE blockDim){
  return Qnil;
}

// cublasStatus_t cublasXtGetBlockDim (cublasXtHandle_t handle, int* blockDim);
static VALUE rb_cublasXtGetBlockDim(VALUE self, VALUE handle){
  // cublasStatus_t status = cublasXtGetBlockDim (cublasXtHandle_t handle, int* blockDim);

  return Qnil;
}

const char* const CuBLASXtPinnedMemMode_t[2] = {
  "CUBLASXT_PINNING_DISABLED",
  "CUBLASXT_PINNING_ENABLED"
};

// cublasStatus_t cublasXtGetPinningMemMode (cublasXtHandle_t handle, cublasXtPinnedMemMode_t* mode);
static VALUE rb_cublasXtGetPinningMemMode(VALUE self){
  // cublasStatus_t status = cublasXtGetPinningMemMode (cublasXtHandle_t handle, cublasXtPinnedMemMode_t* mode);
  return Qnil;
}

// cublasStatus_t cublasXtSetPinningMemMode (cublasXtHandle_t handle, cublasXtPinnedMemMode_t mode);
static VALUE rb_cublasXtSetPinningMemMode(VALUE self){
  // cublasStatus_t status = cublasXtSetPinningMemMode (cublasXtHandle_t handle, cublasXtPinnedMemMode_t mode);
  return Qnil;
}

const char* const CuBLASXtOpType_t[4] = {
  "CUBLASXT_FLOAT",
  "CUBLASXT_DOUBLE",
  "CUBLASXT_COMPLEX",
  "CUBLASXT_DOUBLECOMPLEX"
};

const char* const CuBLASXtBlasOp_t[13] = {
  "CUBLASXT_GEMM",
  "CUBLASXT_SYRK",
  "CUBLASXT_HERK",
  "CUBLASXT_SYMM",
  "CUBLASXT_HEMM",
  "CUBLASXT_TRSM",
  "CUBLASXT_SYR2K",
  "CUBLASXT_HER2K",

  "CUBLASXT_SPMM",
  "CUBLASXT_SYRKX",
  "CUBLASXT_HERKX",
  "CUBLASXT_TRMM",
  "CUBLASXT_ROUTINE_MAX"
};

/* Currently only 32-bit integer BLAS routines are supported */
// cublasStatus_t cublasXtSetCpuRoutine (cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, void* blasFunctor);
static VALUE rb_cublasXtSetCpuRoutine(VALUE self){
  // cublasStatus_t status = cublasXtSetCpuRoutine (cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, void* blasFunctor);
  return Qnil;
}

/* Specified the percentage of work that should done by the CPU, default is 0 (no work) */
// cublasStatus_t cublasXtSetCpuRatio (cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, float ratio);
static VALUE rb_cublasXtSetCpuRatio(VALUE self, VALUE handle, VALUE blasOp, VALUE type, VALUE ratio){
  // cublasStatus_t status = cublasXtSetCpuRatio (cublasXtHandle_t handle, cublasXtBlasOp_t blasOp, cublasXtOpType_t type, float ratio);
  return Qnil;
}

/* GEMM */
static VALUE rb_cublasXtSgemm(VALUE self){
  return Qnil;
}

// cublasStatus_t cublasXtDgemm ( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, size_t m, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);

static VALUE rb_cublasXtDgemm(VALUE self, VALUE handle, VALUE transa, VALUE transb, VALUE m, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  // cublasStatus_t status = cublasXtDgemm ( cublasXtHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, size_t m, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);

  return Qnil;
}

static VALUE rb_cublasXtCgemm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZgemm(VALUE self){
  return Qnil;
}

/* ------------------------------------------------------- */
/* SYRK */
static VALUE rb_cublasXtSsyrk(VALUE self){
  return Qnil;
}

// cublasStatus_t cublasXtDsyrk ( cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* beta, double* C, size_t ldc);
static VALUE rb_cublasXtDsyrk(VALUE self, VALUE handle, VALUE uplo, VALUE trans, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE beta, VALUE C, VALUE ldc){
  // cublasStatus_t status = cublasXtDsyrk ( cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* beta, double* C, size_t ldc);
  return Qnil;
}

static VALUE rb_cublasXtCsyrk(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZsyrk(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* HERK */
static VALUE rb_cublasXtCherk(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZherk(VALUE self){
  return Qnil;
}


/* -------------------------------------------------------------------- */
/* SYR2K */
static VALUE rb_cublasXtSsyr2k(VALUE self){
  return Qnil;
}

// cublasStatus_t cublasXtDsyr2k ( cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);

static VALUE rb_cublasXtDsyr2k(VALUE self, VALUE handle, VALUE uplo, VALUE trans, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  // cublasStatus_t status = cublasXtDsyr2k(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);
  return Qnil;
}

static VALUE rb_cublasXtCsyr2k(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZsyr2k(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* HERKX : variant extension of HERK */
static VALUE rb_cublasXtCherkx(VALUE self){
  return Qnil;
}


static VALUE rb_cublasXtZherkx(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* TRSM */
static VALUE rb_cublasXtStrsm(VALUE self){
  return Qnil;
}

// cublasStatus_t cublasXtDtrsm ( cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const(double)* alpha, const(double)* A, size_t lda, double* B, size_t ldb);

static VALUE rb_cublasXtDtrsm(VALUE self, VALUE handle, VALUE side, VALUE uplo, VALUE trans, VALUE diag, VALUE m, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb){
  // cublasStatus_t status = cublasXtDtrsm ( cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const(double)* alpha, const(double)* A, size_t lda, double* B, size_t ldb);
  return Qnil;
}

static VALUE rb_cublasXtCtrsm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZtrsm(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* SYMM : Symmetric Multiply Matrix*/
static VALUE rb_cublasXtSsymm(VALUE self){
  return Qnil;
}


// cublasStatus_t cublasXtDsymm ( cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);

static VALUE rb_cublasXtDsymm(VALUE self, VALUE handle, VALUE side, VALUE uplo, VALUE m, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  // cublasStatus_t status = cublasXtDsymm ( cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);
  return Qnil;
}

static VALUE rb_cublasXtCsymm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZsymm(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* HEMM : Hermitian Matrix Multiply */
static VALUE rb_cublasXtChemm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZhemm(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* SYRKX : variant extension of SYRK  */
static VALUE rb_cublasXtSsyrkx(VALUE self){
  return Qnil;
}

// cublasStatus_t cublasXtDsyrkx (cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);

static VALUE rb_cublasXtDsyrkx(VALUE self, VALUE handle, VALUE uplo, VALUE trans, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  // cublasStatus_t status = cublasXtDsyrkx(cublasXtHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, size_t n, size_t k, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, const(double)* beta, double* C, size_t ldc);
  return Qnil;
}

static VALUE rb_cublasXtCsyrkx(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZsyrkx(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* HER2K : variant extension of HERK  */
static VALUE rb_cublasXtCher2k(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZher2k(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* SPMM : Symmetric Packed Multiply Matrix*/

static VALUE rb_cublasXtSspmm(VALUE self){
  return Qnil;
}

// cublasStatus_t cublasXtDspmm ( cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const(double)* alpha, const(double)* AP, const(double)* B, size_t ldb, const(double)* beta, double* C,size_t ldc);

static VALUE rb_cublasXtDspmm(VALUE self, VALUE handle, VALUE side, VALUE uplo, VALUE m, VALUE n, VALUE alpha, VALUE AP, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  // cublasStatus_t status = cublasXtDspmm ( cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, size_t m, size_t n, const(double)* alpha, const(double)* AP, const(double)* B, size_t ldb, const(double)* beta, double* C,size_t ldc);

  return Qnil;
}

static VALUE rb_cublasXtCspmm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZspmm(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------------------- */
/* TRMM */
static VALUE rb_cublasXtStrmm(VALUE self){
  return Qnil;
}

// cublasStatus_t cublasXtDtrmm ( cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, double* C, size_t ldc);

static VALUE rb_cublasXtDtrmm(VALUE self, VALUE handle, VALUE side, VALUE uplo, VALUE trans, VALUE diag, VALUE m, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE C, VALUE ldc){
  // cublasStatus_t status = cublasXtDtrmm(cublasXtHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, size_t m, size_t n, const(double)* alpha, const(double)* A, size_t lda, const(double)* B, size_t ldb, double* C, size_t ldc);

  return Qnil;
}

static VALUE rb_cublasXtCtrmm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZtrmm(VALUE self){
  return Qnil;
}
