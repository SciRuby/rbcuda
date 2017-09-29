static VALUE rb_cuGetErrorString(VALUE self){
  return Qnil;
}

static VALUE rb_cuGetErrorName(VALUE self){
  return Qnil;
}

static VALUE rb_cuInit(VALUE self){
  return Qnil;
}

static VALUE rb_cuDriverGetVersion(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceGet(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceGetCount(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceGetName(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceTotalMem_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceGetAttribute(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceGetProperties(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceComputeCapability(VALUE self){
  return Qnil;
}

static VALUE rb_cuDevicePrimaryCtxRetain(VALUE self){
  return Qnil;
}

static VALUE rb_cuDevicePrimaryCtxRelease(VALUE self){
  return Qnil;
}

static VALUE rb_cuDevicePrimaryCtxSetFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cuDevicePrimaryCtxGetState(VALUE self){
  return Qnil;
}

static VALUE rb_cuDevicePrimaryCtxReset(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxCreate_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxDestroy_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxPushCurrent_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxPopCurrent_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxSetCurrent(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxGetCurrent(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxGetDevice(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxGetFlags(VALUE self){
  return Qnil;
}


static VALUE rb_cuCtxSynchronize(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxSetLimit(VALUE self){
  return Qnil;
}


static VALUE rb_cuCtxGetLimit(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxGetCacheConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxSetCacheConfig(VALUE self){
  return Qnil;
}


static VALUE rb_cuCtxGetSharedMemConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxSetSharedMemConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxGetApiVersion(VALUE self){
  return Qnil;
}


static VALUE rb_cuCtxGetStreamPriorityRange(VALUE self){
  return Qnil;
}


static VALUE rb_cuCtxAttach(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxDetach(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleLoad(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleLoadData(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleLoadDataEx(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleLoadFatBinary(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleUnload(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleGetFunction(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleGetGlobal_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleGetTexRef(VALUE self){
  return Qnil;
}

static VALUE rb_cuModuleGetSurfRef(VALUE self){
  return Qnil;
}

static VALUE rb_cuLinkCreate_v2(VALUE self){
  return Qnil;
}


static VALUE rb_cuLinkAddData_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuLinkAddFile_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuLinkComplete(VALUE self){
  return Qnil;
}

static VALUE rb_cuLinkDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemGetInfo_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemAlloc_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemAllocPitch_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemFree_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemGetAddressRange_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemAllocHost_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemFreeHost(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemHostAlloc(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemHostGetDevicePointer_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemHostGetFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemAllocManaged(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceGetByPCIBusId(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceGetPCIBusId(VALUE self){
  return Qnil;
}


static VALUE rb_cuIpcGetEventHandle(VALUE self){
  return Qnil;
}


static VALUE rb_cuIpcOpenEventHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cuIpcGetMemHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cuIpcOpenMemHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cuIpcCloseMemHandle(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemHostRegister_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemHostUnregister(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpy(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyPeer(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyHtoD_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyDtoH_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyDtoD_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyDtoA_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyAtoD_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyHtoA_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyAtoH_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyAtoA_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpy2D_v2(VALUE self){
  return Qnil;
}


static VALUE rb_cuMemcpy2DUnaligned_v2(VALUE self){
  return Qnil;
}


static VALUE rb_cuMemcpy3D_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpy3DPeer(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyPeerAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyHtoDAsync_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyDtoHAsync_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyDtoDAsync_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyHtoAAsync_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpyAtoHAsync_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpy2DAsync_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpy3DAsync_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemcpy3DPeerAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD8_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD16_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD32_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD2D8_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD2D16_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD2D32_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD8Async(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD16Async(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD32Async(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD2D8Async(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD2D16Async(VALUE self){
  return Qnil;
}

static VALUE rb_cuMemsetD2D32Async(VALUE self){
  return Qnil;
}

static VALUE rb_cuArrayCreate_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuArrayGetDescriptor_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuArrayDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cuArray3DCreate_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuArray3DGetDescriptor_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuMipmappedArrayCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cuMipmappedArrayGetLevel(VALUE self){
  return Qnil;
}

static VALUE rb_cuMipmappedArrayDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cuPointerGetAttribute(VALUE self){
  return Qnil;
}

static VALUE rb_cuPointerSetAttribute(VALUE self){
  return Qnil;
}

static VALUE rb_cuPointerGetAttributes(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamCreateWithPriority(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamGetPriority(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamGetFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamWaitEvent(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamAddCallback(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamAttachMemAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamQuery(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamSynchronize(VALUE self){
  return Qnil;
}

static VALUE rb_cuStreamDestroy_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuEventCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cuEventRecord(VALUE self){
  return Qnil;
}

static VALUE rb_cuEventQuery(VALUE self){
  return Qnil;
}

static VALUE rb_cuEventSynchronize(VALUE self){
  return Qnil;
}

static VALUE rb_cuEventDestroy_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuEventElapsedTime(VALUE self){
  return Qnil;
}

static VALUE rb_cuFuncGetAttribute(VALUE self){
  return Qnil;
}

static VALUE rb_cuFuncSetCacheConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cuFuncSetSharedMemConfig(VALUE self){
  return Qnil;
}

static VALUE rb_cuLaunchKernel(VALUE self){
  return Qnil;
}

static VALUE rb_cuFuncSetBlockShape(VALUE self){
  return Qnil;
}

static VALUE rb_cuFuncSetSharedSize(VALUE self){
  return Qnil;
}

static VALUE rb_cuParamSetSize(VALUE self){
  return Qnil;
}

static VALUE rb_cuParamSeti(VALUE self){
  return Qnil;
}

static VALUE rb_cuParamSetf(VALUE self){
  return Qnil;
}

static VALUE rb_cuParamSetv(VALUE self){
  return Qnil;
}

static VALUE rb_cuLaunch(VALUE self){
  return Qnil;
}

static VALUE rb_cuLaunchGrid(VALUE self){
  return Qnil;
}

static VALUE rb_cuLaunchGridAsync(VALUE self){
  return Qnil;
}

static VALUE rb_cuParamSetTexRef(VALUE self){
  return Qnil;
}

static VALUE rb_cuOccupancyMaxActiveBlocksPerMultiprocessor(VALUE self){
  return Qnil;
}

static VALUE rb_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cuOccupancyMaxPotentialBlockSize(VALUE self){
  return Qnil;
}

static VALUE rb_cuOccupancyMaxPotentialBlockSizeWithFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetAddress_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetAddress2D_v3(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetFormat(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetAddressMode(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetFilterMode(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapFilterMode(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapLevelBias(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapLevelClamp(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetMaxAnisotropy(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefSetFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetAddress_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetAddressMode(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetFilterMode(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetFormat(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapFilterMode(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapLevelBias(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapLevelClamp(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetMaxAnisotropy(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefGetFlags(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexRefDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cuSurfRefSetArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuSurfRefGetArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexObjectCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexObjectDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexObjectGetResourceDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexObjectGetTextureDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cuTexObjectGetResourceViewDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cuSurfObjectCreate(VALUE self){
  return Qnil;
}

static VALUE rb_cuSurfObjectDestroy(VALUE self){
  return Qnil;
}

static VALUE rb_cuSurfObjectGetResourceDesc(VALUE self){
  return Qnil;
}

static VALUE rb_cuDeviceCanAccessPeer(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxEnablePeerAccess(VALUE self){
  return Qnil;
}

static VALUE rb_cuCtxDisablePeerAccess(VALUE self){
  return Qnil;
}

static VALUE rb_cuGraphicsUnregisterResource(VALUE self){
  return Qnil;
}

static VALUE rb_cuGraphicsSubResourceGetMappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuGraphicsResourceGetMappedMipmappedArray(VALUE self){
  return Qnil;
}

static VALUE rb_cuGraphicsResourceGetMappedPointer_v2(VALUE self){
  return Qnil;
}

static VALUE rb_cuGraphicsResourceSetMapFlags_v2(VALUE self){
  return Qnil;
}


static VALUE rb_cuGraphicsMapResources(VALUE self){
  return Qnil;
}

static VALUE rb_cuGraphicsUnmapResources(VALUE self){
  return Qnil;
}

static VALUE rb_cuGetExportTable(VALUE self){
  return Qnil;
}
