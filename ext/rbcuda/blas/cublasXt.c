static VALUE rb_cublasXtCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtGetNumBoards(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtMaxBoards(VALUE self){
  return Qnil;
}

/* This routine selects the Gpus that the user want to use for CUBLAS-XT */
static VALUE rb_cublasXtDeviceSelect(VALUE self){
  return Qnil;
}

/* This routine allows to change the dimension of the tiles ( blockDim x blockDim ) */
static VALUE rb_cublasXtSetBlockDim(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtGetBlockDim(VALUE self){
  return Qnil;
}

enum cublasXtPinnedMemMode_t
{
  CUBLASXT_PINNING_DISABLED = 0,
  CUBLASXT_PINNING_ENABLED = 1
}

static VALUE rb_cublasXtGetPinningMemMode(VALUE self){
  return Qnil;
}
static VALUE rb_cublasXtSetPinningMemMode(VALUE self){
  return Qnil;
}

enum cublasXtOpType_t
{
  CUBLASXT_FLOAT = 0,
  CUBLASXT_DOUBLE = 1,
  CUBLASXT_COMPLEX = 2,
  CUBLASXT_DOUBLECOMPLEX = 3
}

enum cublasXtBlasOp_t
{
  CUBLASXT_GEMM = 0,
  CUBLASXT_SYRK = 1,
  CUBLASXT_HERK = 2,
  CUBLASXT_SYMM = 3,
  CUBLASXT_HEMM = 4,
  CUBLASXT_TRSM = 5,
  CUBLASXT_SYR2K = 6,
  CUBLASXT_HER2K = 7,

  CUBLASXT_SPMM = 8,
  CUBLASXT_SYRKX = 9,
  CUBLASXT_HERKX = 10,
  CUBLASXT_TRMM = 11,
  CUBLASXT_ROUTINE_MAX = 12
}

/* Currently only 32-bit integer BLAS routines are supported */
static VALUE rb_cublasXtSetCpuRoutine(VALUE self){
  return Qnil;
}

/* Specified the percentage of work that should done by the CPU, default is 0 (no work) */
static VALUE rb_cublasXtSetCpuRatio(VALUE self){
  return Qnil;
}

/* GEMM */
static VALUE rb_cublasXtSgemm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtDgemm(VALUE self){
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

static VALUE rb_cublasXtDsyrk(VALUE self){
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

static VALUE rb_cublasXtDsyr2k(VALUE self){
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

static VALUE rb_cublasXtDtrsm(VALUE self){
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

static VALUE rb_cublasXtDsymm(VALUE self){
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

static VALUE rb_cublasXtDsyrkx(VALUE self){
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

static VALUE rb_cublasXtDspmm(VALUE self){
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

static VALUE rb_cublasXtDtrmm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtCtrmm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasXtZtrmm(VALUE self){
  return Qnil;
}
