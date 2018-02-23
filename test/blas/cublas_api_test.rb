require 'test_helper'

class RbCUDA::CuBLAS_apiTest < Minitest::Test

  def setup
    shape = [2,2]
  end

  def test_cublasDnrm2_v2
    assert true
  end

  def test_cublasDdot_v2
    assert true
  end

  def test_cublasDscal_v2
    assert true
  end

  def test_cublasDaxpy_v2
    assert true
  end

  def test_cublasDcopy_v2
    assert true
  end

  def test_cublasDswap_v2
    assert true
  end

  def test_cublasIdamax_v2
    assert true
  end

  def test_cublasIdamin_v2
    assert true
  end

  def test_cublasDasum_v2
    assert true
  end

  def test_cublasDrot_v2
    assert true
  end

  def test_cublasDrotg_v2
    assert true
  end

  def test_cublasDrotm_v2
    assert true
  end

  def test_cublasDgemv_v2
    assert true
  end

  def test_cublasDgbmv_v2
    assert true
  end

  def test_cublasDtrmv_v2
    assert true
  end

  def test_cublasDtbmv_v2
    assert true
  end

  def test_cublasDtpmv_v2
    assert true
  end

  def test_cublasDtrsv_v2
    assert true
  end

  def test_cublasDtpsv_v2
    assert true
  end

  def test_cublasDtbsv_v2
    assert true
  end

  def test_cublasDsymv_v2
    assert true
  end

  def test_cublasDsbmv_v2
    assert true
  end

  def test_cublasDspmv_v2
    assert true
  end

  def test_cublasDger_v2
    assert true
  end

  def test_cublasDsyr_v2
    assert true
  end

  def test_cublasDspr_v2
    assert true
  end

  def test_cublasDsyr2_v2
    assert true
  end

  def test_cublasDspr2_v2
    assert true
  end

  def test_cublasDgemm_v2
    shape = [2,2]
    cpu_ary1 = [2, 2, 2, 2]
    cpu_ary2 = [2, 2, 2, 2]
    cpu_ary_res = []


    gpu_ary1    = RbCUDA::Runtime.cudaMalloc(shape);
    gpu_ary2    = RbCUDA::Runtime.cudaMalloc(shape);
    gpu_ary_res = RbCUDA::Runtime.cudaMalloc(shape);

    RbCUDA::Runtime.cudaMemcpy(gpu_ary1, cpu_ary1, shape[0]*shape[1], :cudaMemcpyHostToDevice);
    RbCUDA::Runtime.cudaMemcpy(gpu_ary2, cpu_ary2, shape[0]*shape[1], :cudaMemcpyHostToDevice);

    lda= m = shape[0]
    ldb= n = shape[0]
    ldc= k = shape[0]
    alf = 1
    bet = 0
    alpha = alf
    beta = bet
    handle = RbCUDA::CuBLAS_v2.cublasCreate_v2()

    RbCUDA::CuBLAS_v2.cublasDgemm_v2(handle, :CUBLAS_OP_N, :CUBLAS_OP_N, m, n, k, alpha, gpu_ary1, lda, gpu_ary2, ldb, beta, gpu_ary_res, ldc)

    puts RbCUDA::Runtime.cudaMemcpy([], gpu_ary_res, shape[0]*shape[1], :cudaMemcpyDeviceToHost);
    RbCUDA::CuBLAS_v2.cublasDestroy_v2(handle)
    RbCUDA::Runtime.cudaFree(gpu_ary1)
    RbCUDA::Runtime.cudaFree(gpu_ary2)
    RbCUDA::Runtime.cudaFree(gpu_ary_res)
    assert true
  end

  def test_cublasDsyrk_v2
    assert true
  end

  def test_cublasDsyr2k_v2
    assert true
  end

  def test_cublasDsyrkx
    assert true
  end

  def test_cublasDsymm_v2
    assert true
  end

  def test_cublasDtrsm_v2
    assert true
  end

  def test_cublasDtrmm_v2
    assert true
  end

  def test_cublasDgemmBatched
    assert true
  end

  def test_cublasDgeam
    assert true
  end

  def test_cublasDgetrfBatched
    assert true
  end

  def test_cublasDgetriBatched
    assert true
  end

  def test_cublasDgetrsBatched
    assert true
  end

  def test_cublasDtrsmBatched
    assert true
  end

  def test_cublasDmatinvBatched
    assert true
  end

  def test_cublasDgeqrfBatched
    assert true
  end

  def test_cublasDgelsBatched
    assert true
  end

  def test_cublasDdgmm
    assert true
  end

  def test_cublasDtpttr
    assert true
  end

  def test_cublasDtrttp
    assert true
  end

  def test_it_does_something_useful
    assert true
  end

end
