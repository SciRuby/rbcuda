/* CUBLAS data types */

static VALUE rb_cublasInit(VALUE self){
  return Qnil;
}

static VALUE rb_cublasShutdown(VALUE self){
  return Qnil;
}

static VALUE rb_cublasGetError(VALUE self){
  return Qnil;
}

static VALUE rb_cublasGetVersion(VALUE self){
  return Qnil;
}

static VALUE rb_cublasAlloc(VALUE self){
  return Qnil;
}

static VALUE rb_cublasFree(VALUE self){
  return Qnil;
}

static VALUE rb_cublasSetKernelStream(VALUE self){
  return Qnil;
}


/* ---------------- CUBLAS BLAS1 functions ---------------- */
/* NRM2 */
static VALUE rb_cublasSnrm2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDnrm2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasScnrm2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDznrm2(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* DOT */
static VALUE rb_cublasSdot(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDdot(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCdotu(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCdotc(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZdotu(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZdotc(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SCAL */
static void rb_cublasSscal(VALUE self){
  return Qnil;
}

static void rb_cublasDscal(VALUE self){
  return Qnil;
}

static void rb_cublasCscal(VALUE self){
  return Qnil;
}

static void rb_cublasZscal(VALUE self){
  return Qnil;
}

static void rb_cublasCsscal(VALUE self){
  return Qnil;
}

static void rb_cublasZdscal(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* AXPY */
static void rb_cublasSaxpy(VALUE self){
  return Qnil;
}

static void rb_cublasDaxpy(VALUE self){
  return Qnil;
}

static void rb_cublasCaxpy(VALUE self){
  return Qnil;
}

static void rb_cublasZaxpy(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* COPY */
static void rb_cublasScopy(VALUE self){
  return Qnil;
}

static void rb_cublasDcopy(VALUE self){
  return Qnil;
}

static void rb_cublasCcopy(VALUE self){
  return Qnil;
}

static void rb_cublasZcopy(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* SWAP */
static void rb_cublasSswap(VALUE self){
  return Qnil;
}

static void rb_cublasDswap(VALUE self){
  return Qnil;
}

static void rb_cublasCswap(VALUE self){
  return Qnil;
}

static void rb_cublasZswap(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* AMAX */
static VALUE rb_cublasIsamax(VALUE self){
  return Qnil;
}

static VALUE rb_cublasIdamax(VALUE self){
  return Qnil;
}

static VALUE rb_cublasIcamax(VALUE self){
  return Qnil;
}

static VALUE rb_cublasIzamax(VALUE self){
  return Qnil;
}
/*------------------------------------------------------------------------*/
/* AMIN */
static VALUE rb_cublasIsamin(VALUE self){
  return Qnil;
}

static VALUE rb_cublasIdamin(VALUE self){
  return Qnil;
}

static VALUE rb_cublasIcamin(VALUE self){
  return Qnil;
}

static VALUE rb_cublasIzamin(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* ASUM */
static VALUE rb_cublasSasum(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDasum(VALUE self){
  return Qnil;
}

static VALUE rb_cublasScasum(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDzasum(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* ROT */
static void rb_cublasSrot(VALUE self){
  return Qnil;
}

static void rb_cublasDrot(VALUE self){
  return Qnil;
}

static void rb_cublasCrot(VALUE self){
  return Qnil;
}

static void rb_cublasZrot(VALUE self){
  return Qnil;
}

static void rb_cublasCsrot(VALUE self){
  return Qnil;
}

static void rb_cublasZdrot(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* ROTG */
static void rb_cublasSrotg(VALUE self){
  return Qnil;
}

static void rb_cublasDrotg(VALUE self){
  return Qnil;
}

static void rb_cublasCrotg(VALUE self){
  return Qnil;
}

static void rb_cublasZrotg(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* ROTM */
static void rb_cublasSrotm(VALUE self){
  return Qnil;
}

static void rb_cublasDrotm(VALUE self){
  return Qnil;
}



/*------------------------------------------------------------------------*/
/* ROTMG */
static void rb_cublasSrotmg(VALUE self){
  return Qnil;
}

static void rb_cublasDrotmg(VALUE self){
  return Qnil;
}



/* --------------- CUBLAS BLAS2 functions  ---------------- */
/* GEMV */
static void rb_cublasSgemv(VALUE self){
  return Qnil;
}

static void rb_cublasDgemv(VALUE self){
  return Qnil;
}

static void rb_cublasCgemv(VALUE self){
  return Qnil;
}

static void rb_cublasZgemv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* GBMV */
static void rb_cublasSgbmv(VALUE self){
  return Qnil;
}

static void rb_cublasDgbmv(VALUE self){
  return Qnil;
}

static void rb_cublasCgbmv(VALUE self){
  return Qnil;
}

static void rb_cublasZgbmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TRMV */
static void rb_cublasStrmv(VALUE self){
  return Qnil;
}

static void rb_cublasDtrmv(VALUE self){
  return Qnil;
}

static void rb_cublasCtrmv(VALUE self){
  return Qnil;
}

static void rb_cublasZtrmv(VALUE self){
  return Qnil;
}



/*------------------------------------------------------------------------*/
/* TBMV */
static void rb_cublasStbmv(VALUE self){
  return Qnil;
}

static void rb_cublasDtbmv(VALUE self){
  return Qnil;
}

static void rb_cublasCtbmv(VALUE self){
  return Qnil;
}

static void rb_cublasZtbmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TPMV */
static void rb_cublasStpmV(VALUE self){
  return Qnil;
}

static void rb_cublasDtpmv(VALUE self){
  return Qnil;
}

static void rb_cublasCtpmv(VALUE self){
  return Qnil;
}

static void rb_cublasZtpmv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* TRSV */
static void rb_cublasStrsv(VALUE self){
  return Qnil;
}

static void rb_cublasDtrsv(VALUE self){
  return Qnil;
}

static void rb_cublasCtrsv(VALUE self){
  return Qnil;
}

static void rb_cublasZtrsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TPSV */
static void rb_cublasStpsv(VALUE self){
  return Qnil;
}

static void rb_cublasDtpsv(VALUE self){
  return Qnil;
}

static void rb_cublasCtpsv(VALUE self){
  return Qnil;
}

static void rb_cublasZtpsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TBSV */
static void rb_cublasStbsv(VALUE self){
  return Qnil;
}

static void rb_cublasDtbsv(VALUE self){
  return Qnil;
}

static void rb_cublasCtbsv(VALUE self){
  return Qnil;
}

static void rb_cublasZtbsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYMV/HEMV */
static void rb_cublasSsymv(VALUE self){
  return Qnil;
}

static void rb_cublasDsymv(VALUE self){
  return Qnil;
}

static void rb_cublasChemv(VALUE self){
  return Qnil;
}

static void rb_cublasZhemv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SBMV/HBMV */
static void rb_cublasSsbmv(VALUE self){
  return Qnil;
}

static void rb_cublasDsbmv(VALUE self){
  return Qnil;
}

static void rb_cublasChbmv(VALUE self){
  return Qnil;
}

static void rb_cublasZhbmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SPMV/HPMV */
static void rb_cublasSspmv(VALUE self){
  return Qnil;
}

static void rb_cublasDspmv(VALUE self){
  return Qnil;
}

static void rb_cublasChpmv(VALUE self){
  return Qnil;
}

static void rb_cublasZhpmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* GER */
static void rb_cublasSger(VALUE self){
  return Qnil;
}

static void rb_cublasDger(VALUE self){
  return Qnil;
}

static void rb_cublasCgeru(VALUE self){
  return Qnil;
}

static void rb_cublasCgerc(VALUE self){
  return Qnil;
}

static void rb_cublasZgeru(VALUE self){
  return Qnil;
}

static void rb_cublasZgerc(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYR/HER */
static void rb_cublasSsyr(VALUE self){
  return Qnil;
}

static void rb_cublasDsyr(VALUE self){
  return Qnil;
}


static void rb_cublasCher(VALUE self){
  return Qnil;
}

static void rb_cublasZher(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SPR/HPR */
static void rb_cublasSspr(VALUE self){
  return Qnil;
}

static void rb_cublasDspr(VALUE self){
  return Qnil;
}

static void rb_cublasChpr(VALUE self){
  return Qnil;
}

static void rb_cublasZhpr(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYR2/HER2 */
static void rb_cublasSsyr2(VALUE self){
  return Qnil;
}

static void rb_cublasDsyr2(VALUE self){
  return Qnil;
}

static void rb_cublasCher2(VALUE self){
  return Qnil;
}

static void rb_cublasZher2(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* SPR2/HPR2 */
static void rb_cublasSspr2(VALUE self){
  return Qnil;
}

static void rb_cublasDspr2(VALUE self){
  return Qnil;
}

static void rb_cublasChpr2(VALUE self){
  return Qnil;
}

static void rb_cublasZhpr2(VALUE self){
  return Qnil;
}


/* ------------------------BLAS3 Functions ------------------------------- */
/* GEMM */
static void rb_cublasSgemm(VALUE self){
  return Qnil;
}

static void rb_cublasDgemm(VALUE self){
  return Qnil;
}

static void rb_cublasCgemm(VALUE self){
  return Qnil;
}

static void rb_cublasZgemm(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------*/
/* SYRK */

static void rb_cublasSsyrk(VALUE self){
  return Qnil;
}

static void rb_cublasDsyrk(VALUE self){
  return Qnil;
}

static void rb_cublasCsyrk(VALUE self){
  return Qnil;
}

static void rb_cublasZsyrk(VALUE self){
  return Qnil;
}



/* ------------------------------------------------------- */
/* HERK */
static void rb_cublasCherk(VALUE self){
  return Qnil;
}

static void rb_cublasZherk(VALUE self){
  return Qnil;
}



/* ------------------------------------------------------- */
/* SYR2K */
static void rb_cublasSsyr2k(VALUE self){
  return Qnil;
}

static void rb_cublasDsyr2k(VALUE self){
  return Qnil;
}

static void rb_cublasCsyr2k(VALUE self){
  return Qnil;
}

static void rb_cublasZsyr2k(VALUE self){
  return Qnil;
}

/* ------------------------------------------------------- */
/* HER2K */
static void rb_cublasCher2k(VALUE self){
  return Qnil;
}

static void rb_cublasZher2k(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYMM*/
static void rb_cublasSsymm(VALUE self){
  return Qnil;
}

static void rb_cublasDsymm(VALUE self){
  return Qnil;
}

static void rb_cublasCsymm(VALUE self){
  return Qnil;
}

static void rb_cublasZsymm(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* HEMM*/
static void rb_cublasChemm(VALUE self){
  return Qnil;
}

static void rb_cublasZhemm(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TRSM*/
static void rb_cublasStrsm(VALUE self){
  return Qnil;
}

static void rb_cublasDtrsm(VALUE self){
  return Qnil;
}

static void rb_cublasCtrsm(VALUE self){
  return Qnil;
}

static void rb_cublasZtrsm(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TRMM*/
static void rb_cublasStrmm(VALUE self){
  return Qnil;
}

static void rb_cublasDtrmm(VALUE self){
  return Qnil;
}

static void rb_cublasCtrmm(VALUE self){
  return Qnil;
}

static void rb_cublasZtrmm(VALUE self){
  return Qnil;
}
