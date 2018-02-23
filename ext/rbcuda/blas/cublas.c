// cublasStatus_t cublasInit ();
static VALUE rb_cublasInit(VALUE self){
  cublasStatus_t status = cublasInit();
  return Qnil;
}

// cublasStatus_t cublasShutdown ();
static VALUE rb_cublasShutdown(VALUE self){
  cublasStatus_t status = cublasShutdown();
  return Qnil;
}

// cublasStatus_t cublasGetError ();
static VALUE rb_cublasGetError(VALUE self){
  cublasStatus_t status = cublasGetError();
  return Qnil;
}

// cublasStatus_t cublasGetVersion (int* version_);
static VALUE rb_cublasGetVersion(VALUE self){
  int version;
  // cublasStatus_t status = cublasGetVersion(&version);
  return Qnil;
}

// cublasStatus_t cublasAlloc (int n, int elemSize, void** devicePtr);
static VALUE rb_cublasAlloc(VALUE self, VALUE n, VALUE elem_size, VALUE device_ptr){
  void* devicePtr;
  cublasAlloc(NUM2INT(n), NUM2INT(elem_size), &devicePtr);
  return Qnil;
}

// cublasStatus_t cublasFree (void* devicePtr);
static VALUE rb_cublasFree(VALUE self, VALUE device_ptr){
  cublasStatus_t status = cublasFree((void*)device_ptr);
  return Qnil;
}

// cublasStatus_t cublasSetKernelStream (cudaStream_t stream);
static VALUE rb_cublasSetKernelStream(VALUE self, VALUE stream_val){
  custream_ptr* stream;
  Data_Get_Struct(stream_val, custream_ptr, stream);
  cublasStatus_t status = cublasSetKernelStream(stream->stream);
  return Qnil;
}


/* ---------------- CUBLAS BLAS1 functions ---------------- */
/* NRM2 */

// float cublasSnrm2 (int n, const(float)* x, int incx);
static VALUE rb_cublasSnrm2(VALUE self){
  return Qnil;
}

// double cublasDnrm2 (int n, const(double)* x, int incx);
static VALUE rb_cublasDnrm2(VALUE self, VALUE n, VALUE x_val, VALUE incx){
  dev_ptr* ptr_x;
  Data_Get_Struct(x_val, dev_ptr, ptr_x);

  // double result = cublasDnrm2(NUM2INT(n),  ptr_x->carray, NUM2INT(incx));
  return Qnil;
}

// float cublasScnrm2 (int n, const(cuComplex)* x, int incx);
static VALUE rb_cublasScnrm2(VALUE self){
  return Qnil;
}

// double cublasDznrm2 (int n, const(cuDoubleComplex)* x, int incx);
static VALUE rb_cublasDznrm2(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* DOT */

// float cublasSdot (int n, const(float)* x, int incx, const(float)* y, int incy);
static VALUE rb_cublasSdot(VALUE self){
  return Qnil;
}

// double cublasDdot ( int n, const(double)* x, int incx, const(double)* y, int incy);
static VALUE rb_cublasDdot(VALUE self, VALUE n, VALUE x, VALUE incx, VALUE y, VALUE incy){
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  double result; // = cublasDdot(NUM2INT(n), ptr_x->carray, NUM2INT(incx), ptr_y->carray, NUM2INT(incy));
  return DBL2NUM(result);
}

// cuComplex cublasCdotu ( int n, const(cuComplex)* x, int incx, const(cuComplex)* y, int incy);
static VALUE rb_cublasCdotu(VALUE self){
  return Qnil;
}

// cuComplex cublasCdotc ( int n, const(cuComplex)* x, int incx, const(cuComplex)* y, int incy);
static VALUE rb_cublasCdotc(VALUE self){
  return Qnil;
}

// cuDoubleComplex cublasZdotu ( int n, const(cuDoubleComplex)* x, int incx, const(cuDoubleComplex)* y, int incy);
static VALUE rb_cublasZdotu(VALUE self){
  return Qnil;
}

// cuDoubleComplex cublasZdotc ( int n, const(cuDoubleComplex)* x, int incx, const(cuDoubleComplex)* y, int incy);
static VALUE rb_cublasZdotc(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SCAL */

// void cublasSscal (int n, float alpha, float* x, int incx);
static VALUE rb_cublasSscal(VALUE self){
  return Qnil;
}

// void cublasDscal (int n, double alpha, double* x, int incx);
static VALUE rb_cublasDscal(VALUE self, VALUE n, VALUE alpha, VALUE x, VALUE incx){
  dev_ptr* ptr_x;
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDscal(NUM2INT(n), NUM2DBL(alpha), ptr_x->carray, NUM2INT(incx));
  return Qnil;
}

// void cublasCscal (int n, cuComplex alpha, cuComplex* x, int incx);
static VALUE rb_cublasCscal(VALUE self){
  return Qnil;
}

// void cublasZscal (int n, cuDoubleComplex alpha, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZscal(VALUE self){
  return Qnil;
}

// void cublasCsscal (int n, float alpha, cuComplex* x, int incx);
static VALUE rb_cublasCsscal(VALUE self){
  return Qnil;
}

// void cublasZdscal (int n, double alpha, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZdscal(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* AXPY */

// void cublasSaxpy ( int n, float alpha, const(float)* x, int incx, float* y, int incy);
static VALUE rb_cublasSaxpy(VALUE self){
  return Qnil;
}

// void cublasDaxpy ( int n, double alpha, const(double)* x, int incx, double* y, int incy);
static VALUE rb_cublasDaxpy(VALUE self, VALUE alpha, VALUE x, VALUE incx, VALUE y, VALUE incy){
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDaxpy(NUM2INT(n), NUM2DBL(alpha), ptr_x->carray, NUM2INT(incx),  ptr_y->carray, NUM2INT(incy));
  return Qnil;
}

// void cublasCaxpy ( int n, cuComplex alpha, const(cuComplex)* x, int incx, cuComplex* y, int incy);
static VALUE rb_cublasCaxpy(VALUE self){
  return Qnil;
}

// void cublasZaxpy ( int n, cuDoubleComplex alpha, const(cuDoubleComplex)* x, int incx, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZaxpy(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* COPY */

// void cublasScopy (int n, const(float)* x, int incx, float* y, int incy);
static VALUE rb_cublasScopy(VALUE self){
  return Qnil;
}

// void cublasDcopy (int n, const(double)* x, int incx, double* y, int incy);
static VALUE rb_cublasDcopy(VALUE self, VALUE x, VALUE incx, VALUE y, VALUE incy){
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDcopy(NUM2INT(n),  ptr_x->carray, NUM2INT(incx),  ptr_y->carray, NUM2INT(incy));
  return Qnil;
}

// void cublasCcopy (int n, const(cuComplex)* x, int incx, cuComplex* y, int incy);
static VALUE rb_cublasCcopy(VALUE self){
  return Qnil;
}

// void cublasZcopy (int n, const(cuDoubleComplex)* x, int incx, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZcopy(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* SWAP */

// void cublasSswap (int n, float* x, int incx, float* y, int incy);
static VALUE rb_cublasSswap(VALUE self){
  return Qnil;
}

// void cublasDswap (int n, double* x, int incx, double* y, int incy);
static VALUE rb_cublasDswap(VALUE self, VALUE n, VALUE  x, VALUE incx, VALUE y, VALUE incy){
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDswap(NUM2INT(n), x->carray, NUM2INT(incx), y->carray, NUM2INT(incy));
  return Qnil;
}

// void cublasCswap (int n, cuComplex* x, int incx, cuComplex* y, int incy);
static VALUE rb_cublasCswap(VALUE self){
  return Qnil;
}

// void cublasZswap (int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZswap(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* AMAX */


// int cublasIsamax (int n, const(float)* x, int incx);
static VALUE rb_cublasIsamax(VALUE self){
  return Qnil;
}

// int cublasIdamax (int n, const(double)* x, int incx);
static VALUE rb_cublasIdamax(VALUE self, VALUE n, VALUE x, VALUE incx){
  dev_ptr* ptr_x;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  int result; // = cublasIdamax(NUM2INT(n), ptr_x->carray, NUM2INT(incx));
  return INT2NUM(result);
}

// int cublasIcamax (int n, const(cuComplex)* x, int incx);
static VALUE rb_cublasIcamax(VALUE self){
  return Qnil;
}

// int cublasIzamax (int n, const(cuDoubleComplex)* x, int incx);
static VALUE rb_cublasIzamax(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* AMIN */

// int cublasIsamin (int n, const(float)* x, int incx);
static VALUE rb_cublasIsamin(VALUE self){
  return Qnil;
}

// int cublasIdamin (int n, const(double)* x, int incx);
static VALUE rb_cublasIdamin(VALUE self, VALUE n, VALUE x, VALUE incx){
  dev_ptr* ptr_x;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  // int result = cublasIdamin(NUM2INT(n), ptr_x->carray, NUM2INT(incx));
  return Qnil;
}

// int cublasIcamin (int n, const(cuComplex)* x, int incx);
static VALUE rb_cublasIcamin(VALUE self){
  return Qnil;
}

// int cublasIzamin (int n, const(cuDoubleComplex)* x, int incx);
static VALUE rb_cublasIzamin(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* ASUM */

// double cublasDasum (int n, const(double)* x, int incx);
static VALUE rb_cublasSasum(VALUE self, VALUE n, VALUE x, VALUE incx){
  dev_ptr* ptr_x;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  // double asum = cublasDasum(NUM2INT(n), ptr_x->carray, NUM2INT(incx));
  return Qnil;
}

// float cublasScasum (int n, const(cuComplex)* x, int incx);
static VALUE rb_cublasDasum(VALUE self){
  return Qnil;
}

// double cublasDzasum (int n, const(cuDoubleComplex)* x, int incx);
static VALUE rb_cublasScasum(VALUE self){
  return Qnil;
}

// float cublasSasum (int n, const(float)* x, int incx);
static VALUE rb_cublasDzasum(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* ROT */

// void cublasSrot ( int n, float* x, int incx, float* y, int incy, float sc, float ss);
static VALUE rb_cublasSrot(VALUE self){
  return Qnil;
}

// void cublasDrot ( int n, double* x, int incx, double* y, int incy, double sc, double ss);
static VALUE rb_cublasDrot(VALUE self, VALUE n, VALUE incx, VALUE y, VALUE incy, VALUE sc, VALUE ss){
  dev_ptr* ptr_y;
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDrot( NUM2INT(n), x->carray, NUM2INT(incx), y->carray, NUM2INT(incy), NUM2DBL(sc), NUM2DBL(ss));
  return Qnil;
}

// void cublasCrot ( int n, cuComplex* x, int incx, cuComplex* y, int incy, float c, cuComplex s);
static VALUE rb_cublasCrot(VALUE self){
  return Qnil;
}

// void cublasZrot ( int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, double sc, cuDoubleComplex cs);
static VALUE rb_cublasZrot(VALUE self){
  return Qnil;
}

// void cublasCsrot ( int n, cuComplex* x, int incx, cuComplex* y, int incy, float c, float s);
static VALUE rb_cublasCsrot(VALUE self){
  return Qnil;
}

// void cublasZdrot ( int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, double c, double s);
static VALUE rb_cublasZdrot(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* ROTG */

// void cublasSrotg (float* sa, float* sb, float* sc, float* ss);
static VALUE rb_cublasSrotg(VALUE self){
  return Qnil;
}

// void cublasDrotg (double* sa, double* sb, double* sc, double* ss);
static VALUE rb_cublasDrotg(VALUE self, VALUE sa, VALUE sb, VALUE sc, VALUE ss){
  // cublasDrotg((double*)sa, (double*)sb, (double*)sc, (double*)ss);
  return Qnil;
}

// void cublasCrotg (cuComplex* ca, cuComplex cb, float* sc, cuComplex* cs);
static VALUE rb_cublasCrotg(VALUE self){
  return Qnil;
}

// void cublasZrotg ( cuDoubleComplex* ca, cuDoubleComplex cb, double* sc, cuDoubleComplex* cs);
static VALUE rb_cublasZrotg(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* ROTM */

// void cublasSrotm ( int n, float* x, int incx, float* y, int incy, const(float)* sparam);
static VALUE rb_cublasSrotm(VALUE self){
  return Qnil;
}

// void cublasDrotm ( int n, double* x, int incx, double* y, int incy, const(double)* sparam);
static VALUE rb_cublasDrotm(VALUE self, VALUE n, VALUE x, VALUE incx, VALUE y, VALUE incy, VALUE sparam){
  dev_ptr* ptr_sparam;
  Data_Get_Struct(sparam, dev_ptr, ptr_sparam);

  dev_ptr* ptr_x;
  dev_ptr* ptr_y;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDrotm(NUM2INT(n), x->carray, NUM2INT(incx), y->carray, NUM2INT(incy), ptr_sparam->carray);
  return Qnil;
}

// /*------------------------------------------------------------------------*/
// /* ROTMG */

// void cublasSrotmg ( float* sd1, float* sd2, float* sx1, const(float)* sy1, float* sparam);
static VALUE rb_cublasSrotmg(VALUE self){
  return Qnil;
}

// void cublasDrotmg ( double* sd1, double* sd2, double* sx1, const(double)* sy1, double* sparam);
static VALUE rb_cublasDrotmg(VALUE self){
  // cublasDrotmg(double* sd1, double* sd2, double* sx1, const(double)* sy1, double* sparam);
  return Qnil;
}



/* --------------- CUBLAS BLAS2 functions  ---------------- */
/* GEMV */

// void cublasSgemv ( char trans, int m,int n, float alpha, const(float)* A, int lda, const(float)* x, int incx, float beta, float* y, int incy);
static VALUE rb_cublasSgemv(VALUE self){
  return Qnil;
}
// void cublasDgemv ( char trans, int m, int n, double alpha, const(double)* A, int lda, const(double)* x, int incx, double beta, double* y, int incy);
static VALUE rb_cublasDgemv(VALUE self, VALUE trans, VALUE m, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE x, VALUE incx, VALUE beta, VALUE y, VALUE incy){
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;
  dev_ptr* ptr_A;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);
  Data_Get_Struct(A, dev_ptr, ptr_A);

  // cublasDgemv( char trans, NUM2INT(m), NUM2INT(n), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx), NUM2DBL(beta), ptr_y->carray, NUM2INT(incy));

  return Qnil;
}

// void cublasCgemv ( char trans, int m, int n, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* x, int incx, cuComplex beta, cuComplex* y, int incy);
static VALUE rb_cublasCgemv(VALUE self){
  return Qnil;
}

// void cublasZgemv ( char trans, int m, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZgemv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* GBMV */

// void cublasSgbmv ( char trans, int m, int n, int kl, int ku, float alpha, const(float)* A, int lda, const(float)* x, int incx, float beta, float* y, int incy);
static VALUE rb_cublasSgbmv(VALUE self){
  return Qnil;
}

// void cublasDgbmv ( char trans, int m, int n, int kl, int ku, double alpha, const(double)* A, int lda, const(double)* x, int incx, double beta, double* y, int incy);
static VALUE rb_cublasDgbmv(VALUE self, VALUE trans, VALUE m, VALUE n, VALUE kl, VALUE ku, VALUE alpha, VALUE A, VALUE lda, VALUE x, VALUE incx, VALUE beta, VALUE y, VALUE incy){
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;
  dev_ptr* ptr_A;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);
  Data_Get_Struct(A, dev_ptr, ptr_A);

  // cublasDgbmv(char trans, NUM2INT(m), NUM2INT(n), NUM2INT(kl), NUM2INT(ku), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx), NUM2DBL(beta), ptr_y->carray, NUM2INT(incy));

  return Qnil;
}

// void cublasCgbmv ( char trans, int m, int n, int kl, int ku, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* x, int incx, cuComplex beta, cuComplex* y, int incy);
static VALUE rb_cublasCgbmv(VALUE self){
  return Qnil;
}

// void cublasZgbmv ( char trans, int m, int n, int kl, int ku, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZgbmv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* TRMV */

// void cublasStrmv ( char uplo, char trans, char diag, int n, const(float)* A, int lda, float* x, int incx);
static VALUE rb_cublasStrmv(VALUE self){
  return Qnil;
}

// void cublasDtrmv ( char uplo, char trans, char diag, int n, const(double)* A, int lda, double* x, int incx);
static VALUE rb_cublasDtrmv(VALUE self, VALUE uplo, VALUE trans, VALUE diag, VALUE n, VALUE A, VALUE lda, VALUE x, VALUE incx){
  dev_ptr* ptr_x;
  dev_ptr* ptr_A;
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(A, dev_ptr, ptr_A);

  // cublasDtrmv(char uplo, char trans, char diag, NUM2INT(n), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx));

  return Qnil;
}

// void cublasCtrmv ( char uplo, char trans, char diag, int n, const(cuComplex)* A, int lda, cuComplex* x, int incx);
static VALUE rb_cublasCtrmv(VALUE self){
  return Qnil;
}

// void cublasZtrmv ( char uplo, char trans, char diag, int n, const(cuDoubleComplex)* A, int lda, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZtrmv(VALUE self){
  return Qnil;
}



/*------------------------------------------------------------------------*/
/* TBMV */

// void cublasStbmv ( char uplo, char trans, char diag, int n, int k, const(float)* A, int lda, float* x, int incx);
static VALUE rb_cublasStbmv(VALUE self){
  return Qnil;
}

// void cublasDtbmv ( char uplo, char trans, char diag, int n, int k, const(double)* A, int lda, double* x, int incx);
static VALUE rb_cublasDtbmv(VALUE self, VALUE uplo, VALUE trans, VALUE diag, VALUE n, VALUE k, VALUE A, VALUE lda, VALUE x, VALUE incx){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;
  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDtbmv ( char uplo, char trans, char diag, NUM2INT(n), NUM2INT(k), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx));

  return Qnil;
}

// void cublasCtbmv ( char uplo, char trans, char diag, int n, int k, const(cuComplex)* A, int lda, cuComplex* x, int incx);
static VALUE rb_cublasCtbmv(VALUE self){
  return Qnil;
}

// void cublasZtbmv ( char uplo, char trans, char diag, int n, int k, const(cuDoubleComplex)* A, int lda, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZtbmv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* TPMV */

// void cublasStpmv ( char uplo, char trans, char diag, int n, const(float)* AP, float* x, int incx);
static VALUE rb_cublasStpmV(VALUE self){
  return Qnil;
}

// void cublasDtpmv ( char uplo, char trans, char diag, int n, const(double)* AP, double* x, int incx);
static VALUE rb_cublasDtpmv(VALUE self, VALUE uplo, VALUE trans, VALUE diag, VALUE n, VALUE AP, VALUE x, VALUE incx){
  dev_ptr* ptr_AP;
  dev_ptr* ptr_x;
  Data_Get_Struct(AP, dev_ptr, ptr_AP);
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDtpmv(char uplo, char trans, char diag, NUM2INT(n), ptr_AP->carray, ptr_x->carray, NUM2INT(incx));

  return Qnil;
}

// void cublasCtpmv ( char uplo, char trans, char diag, int n, const(cuComplex)* AP, cuComplex* x, int incx);
static VALUE rb_cublasCtpmv(VALUE self){
  return Qnil;
}

// void cublasZtpmv ( char uplo, char trans, char diag, int n, const(cuDoubleComplex)* AP, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZtpmv(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* TRSV */

// void cublasStrsv ( char uplo, char trans, char diag, int n, const(float)* A, int lda, float* x, int incx);
static VALUE rb_cublasStrsv(VALUE self){
  return Qnil;
}

// void cublasDtrsv ( char uplo, char trans, char diag, int n, const(double)* A, int lda, double* x, int incx);
static VALUE rb_cublasDtrsv(VALUE self, VALUE uplo, VALUE trans, VALUE diag, VALUE n, VALUE A, VALUE lda, VALUE x, VALUE incx){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;
  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDtrsv(char uplo, char trans, char diag, NUM2INT(n), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx));

  return Qnil;
}

// void cublasCtrsv ( char uplo, char trans, char diag, int n, const(cuComplex)* A, int lda, cuComplex* x, int incx);
static VALUE rb_cublasCtrsv(VALUE self){
  return Qnil;
}

// void cublasZtrsv ( char uplo, char trans, char diag, int n, const(cuDoubleComplex)* A, int lda, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZtrsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TPSV */

// void cublasStpsv ( char uplo, char trans, char diag, int n, const(float)* AP, float* x, int incx);
static VALUE rb_cublasStpsv(VALUE self){
  return Qnil;
}

// void cublasDtpsv ( char uplo, char trans, char diag, int n, const(double)* AP, double* x, int incx);
static VALUE rb_cublasDtpsv(VALUE self, VALUE uplo, VALUE trans, VALUE diag, VALUE n, VALUE AP, VALUE x, VALUE incx){
  dev_ptr* ptr_AP;
  dev_ptr* ptr_x;
  Data_Get_Struct(AP, dev_ptr, ptr_AP);
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDtpsv(char uplo, char trans, char diag, NUM2INT(n), ptr_AP->carray, ptr_x->carray, NUM2INT(incx));

  return Qnil;
}

// void cublasCtpsv ( char uplo, char trans, char diag, int n, const(cuComplex)* AP, cuComplex* x, int incx);
static VALUE rb_cublasCtpsv(VALUE self){
  return Qnil;
}

// void cublasZtpsv ( char uplo, char trans, char diag, int n, const(cuDoubleComplex)* AP, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZtpsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TBSV */

// void cublasStbsv ( char uplo, char trans, char diag, int n, int k, const(float)* A, int lda, float* x, int incx);
static VALUE rb_cublasStbsv(VALUE self){
  return Qnil;
}

// void cublasDtbsv ( char uplo, char trans, char diag, int n, int k, const(double)* A, int lda, double* x, int incx);
static VALUE rb_cublasDtbsv(VALUE self, VALUE uplo, VALUE trans, VALUE diag, VALUE n, VALUE k, VALUE A, VALUE lda, VALUE x, VALUE incx){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;
  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDtbsv(char uplo, char trans, char diag, NUM2INT(n), NUM2INT(k), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx));

  return Qnil;
}

// void cublasCtbsv ( char uplo, char trans, char diag, int n, int k, const(cuComplex)* A, int lda, cuComplex* x, int incx);
static VALUE rb_cublasCtbsv(VALUE self){
  return Qnil;
}

// void cublasZtbsv ( char uplo, char trans, char diag, int n, int k, const(cuDoubleComplex)* A, int lda, cuDoubleComplex* x, int incx);
static VALUE rb_cublasZtbsv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYMV/HEMV */

// void cublasSsymv ( char uplo, int n, float alpha, const(float)* A, int lda, const(float)* x, int incx, float beta, float* y, int incy);
static VALUE rb_cublasSsymv(VALUE self){
  return Qnil;
}

// void cublasDsymv ( char uplo, int n, double alpha, const(double)* A, int lda, const(double)* x, int incx, double beta, double* y, int incy);
static VALUE rb_cublasDsymv(VALUE self, VALUE uplo, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE x, VALUE incx, VALUE beta, VALUE y, VALUE incy){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDsymv(char uplo, NUM2INT(n), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx), NUM2DBL(beta), ptr_y->carray, NUM2INT(incy));

  return Qnil;
}

// void cublasChemv ( char uplo, int n, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* x, int incx, cuComplex beta, cuComplex* y, int incy);
static VALUE rb_cublasChemv(VALUE self){
  return Qnil;
}

// void cublasZhemv ( char uplo, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZhemv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SBMV/HBMV */

// void cublasSsbmv ( char uplo, int n, int k, float alpha, const(float)* A, int lda, const(float)* x, int incx, float beta, float* y, int incy);
static VALUE rb_cublasSsbmv(VALUE self){
  return Qnil;
}

// void cublasDsbmv ( char uplo, int n, int k, double alpha, const(double)* A, int lda, const(double)* x, int incx, double beta, double* y, int incy);
static VALUE rb_cublasDsbmv(VALUE self, VALUE uplo, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE x, VALUE incx, VALUE beta, VALUE y, VALUE incy){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDsbmv(char uplo, NUM2INT(n), NUM2INT(k), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_x->carray, NUM2INT(incx), NUM2DBL(beta), ptr_y->carray, NUM2INT(incy));

  return Qnil;
}

// void cublasChbmv ( char uplo, int n, int k, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* x, int incx, cuComplex beta, cuComplex* y, int incy);
static VALUE rb_cublasChbmv(VALUE self){
  return Qnil;
}

// void cublasZhbmv ( char uplo, int n, int k, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZhbmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SPMV/HPMV */

// void cublasSspmv ( char uplo, int n, float alpha, const(float)* AP, const(float)* x, int incx, float beta, float* y, int incy);
static VALUE rb_cublasSspmv(VALUE self){
  return Qnil;
}

// void cublasDspmv ( char uplo, int n, double alpha, const(double)* AP, const(double)* x, int incx, double beta, double* y, int incy);
static VALUE rb_cublasDspmv(VALUE self, VALUE uplo, VALUE n, VALUE alpha, VALUE AP, VALUE x, VALUE incx, VALUE beta, VALUE y, VALUE incy){
  dev_ptr* ptr_AP;
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;

  Data_Get_Struct(AP, dev_ptr, ptr_AP);
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDspmv(char uplo, NUM2INT(n), NUM2DBL(alpha), ptr_AP->carray, ptr_x->carray, NUM2INT(incx), NUM2DBL(beta), ptr_y->carray, NUM2INT(incy));

  return Qnil;
}

// void cublasChpmv ( char uplo, int n, cuComplex alpha, const(cuComplex)* AP, const(cuComplex)* x, int incx, cuComplex beta, cuComplex* y, int incy);
static VALUE rb_cublasChpmv(VALUE self){
  return Qnil;
}

// void cublasZhpmv ( char uplo, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* AP, const(cuDoubleComplex)* x, int incx, cuDoubleComplex beta, cuDoubleComplex* y, int incy);
static VALUE rb_cublasZhpmv(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* GER */

// void cublasSger ( int m, int n, float alpha, const(float)* x, int incx, const(float)* y, int incy, float* A, int lda);
static VALUE rb_cublasSger(VALUE self){
  return Qnil;
}

// void cublasDger ( int m, int n, double alpha, const(double)* x, int incx, const(double)* y, int incy, double* A, int lda);
static VALUE rb_cublasDger(VALUE self, VALUE m, VALUE n, VALUE alpha, VALUE x, VALUE incx, VALUE y, VALUE incy, VALUE A, VALUE lda){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDger(NUM2INT(m), NUM2INT(n), NUM2DBL(alpha), ptr_x->carray, NUM2INT(incx), ptr_y->carray, NUM2INT(incy), ptr_A->carray, NUM2INT(lda));

  return Qnil;
}

// void cublasCgeru ( int m, int n, cuComplex alpha, const(cuComplex)* x, int incx, const(cuComplex)* y, int incy, cuComplex* A, int lda);
static VALUE rb_cublasCgeru(VALUE self){
  return Qnil;
}

// void cublasCgerc ( int m, int n, cuComplex alpha, const(cuComplex)* x, int incx, const(cuComplex)* y, int incy, cuComplex* A, int lda);
static VALUE rb_cublasCgerc(VALUE self){
  return Qnil;
}

// void cublasZgeru ( int m, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* x, int incx, const(cuDoubleComplex)* y, int incy, cuDoubleComplex* A, int lda);
static VALUE rb_cublasZgeru(VALUE self){
  return Qnil;
}

// void cublasZgerc ( int m, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* x, int incx, const(cuDoubleComplex)* y, int incy, cuDoubleComplex* A, int lda);
static VALUE rb_cublasZgerc(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* SYR/HER */

// void cublasSsyr ( char uplo, int n, float alpha, const(float)* x, int incx, float* A, int lda);
static VALUE rb_cublasSsyr(VALUE self){
  return Qnil;
}

// void cublasDsyr ( char uplo, int n, double alpha, const(double)* x, int incx, double* A, int lda);
static VALUE rb_cublasDsyr(VALUE self, VALUE uplo, VALUE n, VALUE alpha, VALUE x, VALUE incx, VALUE A, VALUE lda){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDsyr(char uplo, NUM2INT(n), NUM2DBL(alpha), ptr_x->carray, NUM2INT(incx), ptr_A->carray, NUM2INT(lda));

  return Qnil;
}

// void cublasCher ( char uplo, int n, float alpha, const(cuComplex)* x, int incx, cuComplex* A, int lda);
static VALUE rb_cublasCher(VALUE self){
  return Qnil;
}

// void cublasZher ( char uplo, int n, double alpha, const(cuDoubleComplex)* x, int incx, cuDoubleComplex* A, int lda);
static VALUE rb_cublasZher(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SPR/HPR */

// void cublasSspr ( char uplo, int n, float alpha, const(float)* x, int incx, float* AP);
static VALUE rb_cublasSspr(VALUE self){
  return Qnil;
}
// void cublasDspr ( char uplo, int n, double alpha, const(double)* x, int incx, double* AP);
static VALUE rb_cublasDspr(VALUE self, VALUE uplo, VALUE n, VALUE alpha, VALUE x, VALUE incx, VALUE AP){
  dev_ptr* ptr_AP;
  dev_ptr* ptr_x;

  Data_Get_Struct(AP, dev_ptr, ptr_AP);
  Data_Get_Struct(x, dev_ptr, ptr_x);

  // cublasDspr(char uplo, NUM2INT(n), NUM2DBL(alpha), ptr_x->carray, NUM2INT(incx), ptr_AP->carray);

  return Qnil;
}

// void cublasChpr ( char uplo, int n, float alpha, const(cuComplex)* x, int incx, cuComplex* AP);
static VALUE rb_cublasChpr(VALUE self){
  return Qnil;
}

// void cublasZhpr ( char uplo, int n, double alpha, const(cuDoubleComplex)* x, int incx, cuDoubleComplex* AP);
static VALUE rb_cublasZhpr(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYR2/HER2 */

// void cublasSsyr2 ( char uplo, int n, float alpha, const(float)* x, int incx, const(float)* y, int incy, float* A, int lda);
static VALUE rb_cublasSsyr2(VALUE self){
  return Qnil;
}

// void cublasDsyr2 ( char uplo, int n, double alpha, const(double)* x, int incx, const(double)* y, int incy, double* A, int lda);
static VALUE rb_cublasDsyr2(VALUE self, VALUE uplo, VALUE n, VALUE alpha, VALUE x, VALUE incx, VALUE y, VALUE incy, VALUE A, VALUE lda){
  dev_ptr* ptr_A;
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDsyr2(char uplo, NUM2INT(n), NUM2DBL(alpha), ptr_x->carray, NUM2INT(incx), ptr_y->carray, NUM2INT(incy), ptr_A->carray, NUM2INT(lda));

  return Qnil;
}

// void cublasCher2 ( char uplo, int n, cuComplex alpha, const(cuComplex)* x, int incx, const(cuComplex)* y, int incy, cuComplex* A, int lda);
static VALUE rb_cublasCher2(VALUE self){
  return Qnil;
}

// void cublasZher2 ( char uplo, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* x, int incx, const(cuDoubleComplex)* y, int incy, cuDoubleComplex* A, int lda);
static VALUE rb_cublasZher2(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* SPR2/HPR2 */

// void cublasSspr2 ( char uplo, int n, float alpha, const(float)* x, int incx, const(float)* y, int incy, float* AP);
static VALUE rb_cublasSspr2(VALUE self){
  return Qnil;
}

// void cublasDspr2 ( char uplo, int n, double alpha, const(double)* x, int incx, const(double)* y, int incy, double* AP);
static VALUE rb_cublasDspr2(VALUE self, VALUE uplo, VALUE n, VALUE alpha, VALUE x, VALUE incx, VALUE y, VALUE incy, VALUE AP){
  dev_ptr* ptr_AP;
  dev_ptr* ptr_x;
  dev_ptr* ptr_y;

  Data_Get_Struct(AP, dev_ptr, ptr_AP);
  Data_Get_Struct(x, dev_ptr, ptr_x);
  Data_Get_Struct(y, dev_ptr, ptr_y);

  // cublasDspr2(char uplo, NUM2INT(n), NUM2DBL(alpha), ptr_x->carray, NUM2INT(incx), ptr_y->carray, NUM2INT(incy), ptr_AP->carray);

  return Qnil;
}

// void cublasChpr2 ( char uplo, int n, cuComplex alpha, const(cuComplex)* x, int incx, const(cuComplex)* y, int incy, cuComplex* AP);
static VALUE rb_cublasChpr2(VALUE self){
  return Qnil;
}

// void cublasZhpr2 ( char uplo, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* x, int incx, const(cuDoubleComplex)* y, int incy, cuDoubleComplex* AP);
static VALUE rb_cublasZhpr2(VALUE self){
  return Qnil;
}


/* ------------------------BLAS3 Functions ------------------------------- */
/* GEMM */

// void cublasSgemm ( char transa, char transb, int m, int n, int k, float alpha, const(float)* A, int lda, const(float)* B, int ldb, float beta, float* C, int ldc);
static VALUE rb_cublasSgemm(VALUE self){
  return Qnil;
}

// void cublasDgemm ( char transa, char transb, int m, int n, int k, double alpha, const(double)* A, int lda, const(double)* B, int ldb, double beta, double* C, int ldc);
static VALUE rb_cublasDgemm(VALUE self, VALUE transa, VALUE transb, VALUE m, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  dev_ptr* ptr_A;
  dev_ptr* ptr_B;
  dev_ptr* ptr_C;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(B, dev_ptr, ptr_B);
  Data_Get_Struct(C, dev_ptr, ptr_C);

  // cublasDgemm(char transa, char transb, NUM2INT(m), NUM2INT(n), NUM2INT(k), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_B->carray, NUM2INT(ldb), NUM2DBL(beta), ptr_C->carray, NUM2INT(ldc));

  return Qnil;
}

// void cublasCgemm ( char transa, char transb, int m, int n, int k, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* B, int ldb, cuComplex beta, cuComplex* C, int ldc);
static VALUE rb_cublasCgemm(VALUE self){
  return Qnil;
}

// void cublasZgemm ( char transa, char transb, int m, int n, int k, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc);
static VALUE rb_cublasZgemm(VALUE self){
  return Qnil;
}

/* -------------------------------------------------------*/
/* SYRK */

// void cublasSsyrk ( char uplo, char trans, int n, int k, float alpha, const(float)* A, int lda, float beta, float* C, int ldc);
static VALUE rb_cublasSsyrk(VALUE self){
  return Qnil;
}

// void cublasDsyrk ( char uplo, char trans, int n, int k, double alpha, const(double)* A, int lda, double beta, double* C, int ldc);
static VALUE rb_cublasDsyrk(VALUE self, VALUE uplo, VALUE trans, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE beta, VALUE C, VALUE ldc){
  dev_ptr* ptr_A;
  dev_ptr* ptr_C;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(C, dev_ptr, ptr_C);

  // cublasDsyrk(char uplo, char trans, NUM2INT(n), NUM2INT(k), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), NUM2DBL(beta), ptr_C->carray, NUM2INT(ldc));

  return Qnil;
}

// void cublasCsyrk ( char uplo, char trans, int n, int k, cuComplex alpha, const(cuComplex)* A, int lda, cuComplex beta, cuComplex* C, int ldc);
static VALUE rb_cublasCsyrk(VALUE self){
  return Qnil;
}

// void cublasZsyrk ( char uplo, char trans, int n, int k, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, cuDoubleComplex beta, cuDoubleComplex* C, int ldc);
static VALUE rb_cublasZsyrk(VALUE self){
  return Qnil;
}

/* ------------------------------------------------------- */
/* HERK */

// void cublasCherk ( char uplo, char trans, int n, int k, float alpha, const(cuComplex)* A, int lda, float beta, cuComplex* C, int ldc);
static VALUE rb_cublasCherk(VALUE self){
  return Qnil;
}

// void cublasZherk ( char uplo, char trans, int n, int k, double alpha, const(cuDoubleComplex)* A, int lda, double beta, cuDoubleComplex* C, int ldc);
static VALUE rb_cublasZherk(VALUE self){
  return Qnil;
}

/* ------------------------------------------------------- */
/* SYR2K */

// void cublasSsyr2k ( char uplo, char trans, int n, int k, float alpha, const(float)* A, int lda, const(float)* B, int ldb, float beta, float* C, int ldc);
static VALUE rb_cublasSsyr2k(VALUE self){
  return Qnil;
}

// void cublasDsyr2k ( char uplo, char trans, int n, int k, double alpha, const(double)* A, int lda, const(double)* B, int ldb, double beta, double* C, int ldc);
static VALUE rb_cublasDsyr2k(VALUE self, VALUE uplo, VALUE trans, VALUE n, VALUE k, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  dev_ptr* ptr_A;
  dev_ptr* ptr_B;
  dev_ptr* ptr_C;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(B, dev_ptr, ptr_B);
  Data_Get_Struct(C, dev_ptr, ptr_C);

  // cublasDsyr2k(char uplo, char trans, NUM2INT(n), NUM2INT(k), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_B->carray, NUM2INT(ldb), NUM2DBL(beta), ptr_C->carray, NUM2INT(ldc));

  return Qnil;
}

// void cublasCsyr2k ( char uplo, char trans, int n, int k, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* B, int ldb, cuComplex beta, cuComplex* C, int ldc);
static VALUE rb_cublasCsyr2k(VALUE self){
  return Qnil;
}

// void cublasZsyr2k ( char uplo, char trans, int n, int k, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc);
static VALUE rb_cublasZsyr2k(VALUE self){
  return Qnil;
}

/* ------------------------------------------------------- */
/* HER2K */

// void cublasCher2k ( char uplo, char trans, int n, int k, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* B, int ldb, float beta, cuComplex* C, int ldc);
static VALUE rb_cublasCher2k(VALUE self){
  return Qnil;
}

// void cublasZher2k ( char uplo, char trans, int n, int k, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* B, int ldb, double beta, cuDoubleComplex* C, int ldc);
static VALUE rb_cublasZher2k(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* SYMM*/

// void cublasSsymm ( char side, char uplo, int m, int n, float alpha, const(float)* A, int lda, const(float)* B, int ldb, float beta, float* C, int ldc);
static VALUE rb_cublasSsymm(VALUE self){
  return Qnil;
}

// void cublasDsymm ( char side, char uplo, int m, int n, double alpha, const(double)* A, int lda, const(double)* B, int ldb, double beta, double* C, int ldc);
static VALUE rb_cublasDsymm(VALUE self, VALUE side, VALUE uplo, VALUE m, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb, VALUE beta, VALUE C, VALUE ldc){
  dev_ptr* ptr_A;
  dev_ptr* ptr_B;
  dev_ptr* ptr_C;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(B, dev_ptr, ptr_B);
  Data_Get_Struct(C, dev_ptr, ptr_C);

  // cublasDsymm(char side, char uplo, NUM2INT(m), NUM2INT(n), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_B->carray, NUM2INT(ldb), NUM2DBL(beta), ptr_C->carray, NUM2INT(ldc));

  return Qnil;
}

// void cublasCsymm ( char side, char uplo, int m, int n, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* B, int ldb, cuComplex beta, cuComplex* C, int ldc);
static VALUE rb_cublasCsymm(VALUE self){
  return Qnil;
}

// void cublasZsymm ( char side, char uplo, int m, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc);
static VALUE rb_cublasZsymm(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* HEMM*/

// void cublasChemm ( char side, char uplo, int m, int n, cuComplex alpha, const(cuComplex)* A, int lda, const(cuComplex)* B, int ldb, cuComplex beta, cuComplex* C, int ldc);
static VALUE rb_cublasChemm(VALUE self){
  return Qnil;
}

// void cublasZhemm ( char side, char uplo, int m, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, const(cuDoubleComplex)* B, int ldb, cuDoubleComplex beta, cuDoubleComplex* C, int ldc);
static VALUE rb_cublasZhemm(VALUE self){
  return Qnil;
}

/*------------------------------------------------------------------------*/
/* TRSM*/

// void cublasStrsm ( char side, char uplo, char transa, char diag, int m, int n, float alpha, const(float)* A, int lda, float* B, int ldb);
static VALUE rb_cublasStrsm(VALUE self){
  return Qnil;
}

// void cublasDtrsm ( char side, char uplo, char transa, char diag, int m, int n, double alpha, const(double)* A, int lda, double* B, int ldb);
static VALUE rb_cublasDtrsm(VALUE self, VALUE side, VALUE uplo, VALUE transa, VALUE diag, VALUE m, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb){
  dev_ptr* ptr_A;
  dev_ptr* ptr_B;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(B, dev_ptr, ptr_B);

  // cublasDtrsm(char side, char uplo, char transa, char diag, NUM2INT(m), NUM2INT(n), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_B->carray, NUM2INT(ldb));

  return Qnil;
}

// void cublasCtrsm ( char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const(cuComplex)* A, int lda, cuComplex* B, int ldb);
static VALUE rb_cublasCtrsm(VALUE self){
  return Qnil;
}

// void cublasZtrsm ( char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, cuDoubleComplex* B, int ldb);
static VALUE rb_cublasZtrsm(VALUE self){
  return Qnil;
}


/*------------------------------------------------------------------------*/
/* TRMM*/

// void cublasStrmm ( char side, char uplo, char transa, char diag, int m, int n, float alpha, const(float)* A, int lda, float* B, int ldb);
static VALUE rb_cublasStrmm(VALUE self){
  return Qnil;
}

// void cublasDtrmm ( char side, char uplo, char transa, char diag, int m, int n, double alpha, const(double)* A, int lda, double* B, int ldb);
static VALUE rb_cublasDtrmm(VALUE self, VALUE side, VALUE uplo, VALUE transa, VALUE diag, VALUE m, VALUE n, VALUE alpha, VALUE A, VALUE lda, VALUE B, VALUE ldb){
  dev_ptr* ptr_A;
  dev_ptr* ptr_B;

  Data_Get_Struct(A, dev_ptr, ptr_A);
  Data_Get_Struct(B, dev_ptr, ptr_B);

  // cublasDtrmm((char)FIX2INT(side), (char)FIX2INT(uplo), (char)FIX2INT(transa), (char)FIX2INT(diag), NUM2INT(m), NUM2INT(n), NUM2DBL(alpha), ptr_A->carray, NUM2INT(lda), ptr_B->carray, NUM2INT(ldb));

  return Qnil;
}

// void cublasCtrmm ( char side, char uplo, char transa, char diag, int m, int n, cuComplex alpha, const(cuComplex)* A, int lda, cuComplex* B, int ldb);
static VALUE rb_cublasCtrmm(VALUE self){
  return Qnil;
}

// void cublasZtrmm ( char side, char uplo, char transa, char diag, int m, int n, cuDoubleComplex alpha, const(cuDoubleComplex)* A, int lda, cuDoubleComplex* B, int ldb);
static VALUE rb_cublasZtrmm(VALUE self){
  return Qnil;
}
