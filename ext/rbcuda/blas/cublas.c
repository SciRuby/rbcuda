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
static VALUE rb_cublasSscal(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDscal(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCscal(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZscal(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCsscal(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZdscal(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* AXPY */
static VALUE rb_cublasSaxpy(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDaxpy(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCaxpy(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZaxpy(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* COPY */
static VALUE rb_cublasScopy(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDcopy(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCcopy(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZcopy(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* SWAP */
static VALUE rb_cublasSswap(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDswap(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCswap(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZswap(VALUE self){
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
static VALUE rb_cublasSrot(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDrot(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCrot(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZrot(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCsrot(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZdrot(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* ROTG */
static VALUE rb_cublasSrotg(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDrotg(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCrotg(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZrotg(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* ROTM */
static VALUE rb_cublasSrotm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDrotm(VALUE self){
  return Qnil;
}



/*------------------------------------------------------------------------*/
/* ROTMG */
static VALUE rb_cublasSrotmg(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDrotmg(VALUE self){
  return Qnil;
}



/* --------------- CUBLAS BLAS2 functions  ---------------- */
/* GEMV */
static VALUE rb_cublasSgemv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDgemv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCgemv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZgemv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* GBMV */
static VALUE rb_cublasSgbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDgbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCgbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZgbmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TRMV */
static VALUE rb_cublasStrmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtrmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtrmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtrmv(VALUE self){
  return Qnil;
}



/*------------------------------------------------------------------------*/
/* TBMV */
static VALUE rb_cublasStbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtbmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TPMV */
static VALUE rb_cublasStpmV(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtpmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtpmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtpmv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* TRSV */
static VALUE rb_cublasStrsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtrsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtrsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtrsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TPSV */
static VALUE rb_cublasStpsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtpsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtpsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtpsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TBSV */
static VALUE rb_cublasStbsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtbsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtbsv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtbsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYMV/HEMV */
static VALUE rb_cublasSsymv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDsymv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasChemv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZhemv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SBMV/HBMV */
static VALUE rb_cublasSsbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDsbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasChbmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZhbmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SPMV/HPMV */
static VALUE rb_cublasSspmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDspmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasChpmv(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZhpmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* GER */
static VALUE rb_cublasSger(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDger(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCgeru(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCgerc(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZgeru(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZgerc(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYR/HER */
static VALUE rb_cublasSsyr(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDsyr(VALUE self){
  return Qnil;
}


static VALUE rb_cublasCher(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZher(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SPR/HPR */
static VALUE rb_cublasSspr(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDspr(VALUE self){
  return Qnil;
}

static VALUE rb_cublasChpr(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZhpr(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYR2/HER2 */
static VALUE rb_cublasSsyr2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDsyr2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCher2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZher2(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* SPR2/HPR2 */
static VALUE rb_cublasSspr2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDspr2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasChpr2(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZhpr2(VALUE self){
  return Qnil;
}


/* ------------------------BLAS3 Functions ------------------------------- */
/* GEMM */
static VALUE rb_cublasSgemm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDgemm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCgemm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZgemm(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------*/
/* SYRK */

static VALUE rb_cublasSsyrk(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDsyrk(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCsyrk(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZsyrk(VALUE self){
  return Qnil;
}



/* ------------------------------------------------------- */
/* HERK */
static VALUE rb_cublasCherk(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZherk(VALUE self){
  return Qnil;
}



/* ------------------------------------------------------- */
/* SYR2K */
static VALUE rb_cublasSsyr2k(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDsyr2k(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCsyr2k(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZsyr2k(VALUE self){
  return Qnil;
}

/* ------------------------------------------------------- */
/* HER2K */
static VALUE rb_cublasCher2k(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZher2k(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYMM*/
static VALUE rb_cublasSsymm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDsymm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCsymm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZsymm(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* HEMM*/
static VALUE rb_cublasChemm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZhemm(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TRSM*/
static VALUE rb_cublasStrsm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtrsm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtrsm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtrsm(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TRMM*/
static VALUE rb_cublasStrmm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasDtrmm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasCtrmm(VALUE self){
  return Qnil;
}

static VALUE rb_cublasZtrmm(VALUE self){
  return Qnil;
}
