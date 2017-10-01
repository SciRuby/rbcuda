static VALUE rb_cudaDeviceReset(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceSynchronize(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceSetLimit(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceGetLimit(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceGetCacheConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceGetStreamPriorityRange(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceSetCacheConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceGetSharedMemConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceSetSharedMemConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceGetByPCIBusId(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceGetPCIBusId(VALUE self){
  return Qnil;
}

static VALUE rb_cudaIpcGetEventHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cudaIpcOpenEventHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cudaIpcGetMemHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cudaIpcOpenMemHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cudaIpcCloseMemHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cudaThreadExit(VALUE self){
  return Qnil;
}

static VALUE rb_cudaThreadSynchronize(VALUE self){
  return Qnil;
}

static VALUE rb_cudaThreadSetLimit(VALUE self){
  return Qnil;
}

static VALUE rb_cudaThreadGetLimit(VALUE self){
  return Qnil;
}

static VALUE rb_cudaThreadGetCacheConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaThreadSetCacheConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetLastError(VALUE self){
  return Qnil;
}

static VALUE rb_cudaPeekAtLastError(VALUE self){
  return Qnil;
}


static VALUE rb_cudaGetDeviceCount(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetDeviceProperties(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceGetAttribute(VALUE self){
  return Qnil;
}

static VALUE rb_cudaChooseDevice(VALUE self){
  return Qnil;
}

static VALUE rb_cudaSetDevice(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetDevice(VALUE self){
  return Qnil;
}

static VALUE rb_cudaSetValidDevices(VALUE self){
  return Qnil;
}

static VALUE rb_cudaSetDeviceFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetDeviceFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamCreateWithFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamCreateWithPriority(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamGetPriority(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamGetFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamWaitEvent(VALUE self){
  return Qnil;
}


static VALUE rb_cudaStreamAddCallback(VALUE self){
  return Qnil;
}


static VALUE rb_cudaStreamSynchronize(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamQuery(VALUE self){
  return Qnil;
}

static VALUE rb_cudaStreamAttachMemAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaEventCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cudaEventCreateWithFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cudaEventRecord(VALUE self){
  return Qnil;
}


static VALUE rb_cudaEventQuery(VALUE self){
  return Qnil;
}

static VALUE rb_cudaEventSynchronize(VALUE self){
  return Qnil;
}

static VALUE rb_cudaEventDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cudaEventElapsedTime(VALUE self){
  return Qnil;
}

static VALUE rb_cudaLaunchKernel(VALUE self){
  return Qnil;
}

static VALUE rb_cudaFuncSetCacheConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaFuncSetSharedMemConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cudaFuncGetAttributes(VALUE self){
  return Qnil;
}

static VALUE rb_cudaSetDoubleForDevice(VALUE self){
  return Qnil;
}

static VALUE rb_cudaSetDoubleForHost(VALUE self){
  return Qnil;
}

static VALUE rb_cudaOccupancyMaxActiveBlocksPerMultiprocessor(VALUE self){
  return Qnil;
}

static VALUE rb_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cudaConfigureCall(VALUE self){
  return Qnil;
}

static VALUE rb_cudaSetupArgument(VALUE self){
  return Qnil;
}

static VALUE rb_cudaLaunch(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMallocManaged(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMalloc(VALUE self, VALUE shape){
  dev_ptr* ptr = ALLOC(dev_ptr);
  size_t count = 1;
  for (size_t index = 0; index < 2; index++) {
    count *= (size_t)NUM2LONG(RARRAY_AREF(shape, index));
  }
  size_t size = sizeof(double)*count;
  cudaMalloc((void **)&ptr->carray, size);
  return Data_Wrap_Struct(Dev_Array, NULL, rbcu_free, ptr);
}

static VALUE rb_cudaMallocHost(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMallocPitch(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMallocArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaFree(VALUE self, VALUE ptr_val){
  dev_ptr* ptr;
  Data_Get_Struct(ptr_val, dev_ptr, ptr);
  cudaFree(ptr->carray);
  return Qnil;
}

static VALUE rb_cudaFreeHost(VALUE self){
  return Qnil;
}

static VALUE rb_cudaFreeArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaFreeMipmappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaHostAlloc(VALUE self){
  return Qnil;
}

static VALUE rb_cudaHostRegister(VALUE self){
  return Qnil;
}

static VALUE rb_cudaHostUnregister(VALUE self){
  return Qnil;
}

static VALUE rb_cudaHostGetDevicePointer(VALUE self){
  return Qnil;
}

static VALUE rb_cudaHostGetFlags(VALUE self){
  return Qnil;
}
//
static VALUE rb_cudaMalloc3D(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMalloc3DArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMallocMipmappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetMipmappedArrayLevel(VALUE self){
  return Qnil;
}
// http://horacio9573.no-ip.org/cuda/structcudaMemcpy3DParms.html

static VALUE rb_cudaMemcpy3D(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy3DPeer(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy3DAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy3DPeerAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemGetInfo(VALUE self){
  return Qnil;
}

static VALUE rb_cudaArrayGetInfo(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy(VALUE self, VALUE dest_array, VALUE source_ary, VALUE count_val, VALUE kind){
  cudaMemcpyKind flag = rbcu_memcopy_kind(kind);
  size_t count = NUM2UINT(count_val);

  if(flag == cudaMemcpyHostToDevice){
    dev_ptr* ptr;
    Data_Get_Struct(dest_array, dev_ptr, ptr);
    double* host_array = ALLOC_N(double, count);
    for (size_t index = 0; index < count; index++) {
      host_array[index] = (double)NUM2DBL(RARRAY_AREF(source_ary, index));
    }

    cudaMemcpy((void*)ptr->carray, (void*)host_array, sizeof(double)*count, rbcu_memcopy_kind(kind));
  }else{
    dev_ptr* ptr;
    Data_Get_Struct(source_ary, dev_ptr, ptr);
    double* host_array = ALLOC_N(double, count);
    cudaMemcpy((void*)host_array, (void*)ptr->carray, sizeof(double)*count, rbcu_memcopy_kind(kind));
    VALUE* tem = ALLOC_N(VALUE, count);
    for (size_t index = 0; index < count; index++){
      tem[index] = DBL2NUM(host_array[index]);
    }
    dest_array = rb_ary_new4(count, tem);
    return dest_array;
  }
  return Qnil;
}

static VALUE rb_cudaMemcpyPeer(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyToArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyFromArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyArrayToArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy2D(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy2DToArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy2DFromArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy2DArrayToArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyToSymbol(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyFromSymbol(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyPeerAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyToArrayAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyFromArrayAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy2DAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy2DToArrayAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpy2DFromArrayAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyToSymbolAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemcpyFromSymbolAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemset(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemset2D(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemset3D(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemsetAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemset2DAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaMemset3DAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetSymbolAddress(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetSymbolSize(VALUE self){
  return Qnil;
}

static VALUE rb_cudaPointerGetAttributes(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceCanAccessPeer(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceEnablePeerAccess(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDeviceDisablePeerAccess(VALUE self){
  return Qnil;
}

static VALUE rb_cudaCreateChannelDesc (VALUE self){
  return Qnil;
}

static VALUE rb_cudaGraphicsUnregisterResource(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGraphicsResourceSetMapFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGraphicsMapResources(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGraphicsUnmapResources(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGraphicsResourceGetMappedPointer(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGraphicsSubResourceGetMappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGraphicsResourceGetMappedMipmappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetChannelDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cudaBindTexture(VALUE self){
  return Qnil;
}

static VALUE rb_cudaBindTexture2D(VALUE self){
  return Qnil;
}

static VALUE rb_cudaBindTextureToArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaBindTextureToMipmappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaUnbindTexture(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetTextureAlignmentOffset(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetTextureReference(VALUE self){
  return Qnil;
}

static VALUE rb_cudaBindSurfaceToArray(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetSurfaceReference(VALUE self){
  return Qnil;
}

static VALUE rb_cudaCreateTextureObject(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDestroyTextureObject(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetTextureObjectResourceDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetTextureObjectTextureDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetTextureObjectResourceViewDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cudaCreateSurfaceObject(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDestroySurfaceObject(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetSurfaceObjectResourceDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cudaDriverGetVersion(VALUE self){
  return Qnil;
}

static VALUE rb_cudaRuntimeGetVersion(VALUE self){
  return Qnil;
}

static VALUE rb_cudaGetExportTable(VALUE self){
  return Qnil;
}