/*
 * Returns the message string from an error code.
 *
 * Parameters:
 * error   - Error code to convert to string
 * Returns:
 * char* pointer to a NULL-terminated string
*/

static VALUE rb_cuGetErrorString(VALUE self, VALUE error_val){
  const char* pStr;
  CUresult error = rb_cuda_cu_result_from_rbsymbol(error_val);
  CUresult result = cuGetErrorString(error, &pStr);
  return rb_str_new_cstr(pStr);
}

// CUresult cuGetErrorName ( CUresult error, const char** pStr )
// Gets the string representation of an error code enum name.
// Parameters
// error
// - Error code to convert to string
// pStr
// - Address of the string pointer.
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuGetErrorName(VALUE self, VALUE error_val){
  const char* pStr;
  CUresult error = rb_cuda_cu_result_from_rbsymbol(error_val);
  CUresult result = cuGetErrorName(error, &pStr);
  return rb_str_new_cstr(pStr);
}

// CUresult cuInit ( unsigned int  Flags )
// Initialize the CUDA driver API.
// Parameters
// Flags
// - Initialization flag for CUDA.
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuInit(VALUE self, VALUE flags){
  CUresult result = cuInit(NUM2UINT(flags));
  return Qtrue;
}

// CUresult cuDriverGetVersion ( int* driverVersion )
// Returns the CUDA driver version.
// Parameters
// driverVersion
// - Returns the CUDA driver version
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuDriverGetVersion(VALUE self, VALUE driver_version_val){
  int driverVersion;
  CUresult result = cuDriverGetVersion(&driverVersion);
  return INT2NUM(driverVersion);
}

// CUresult cuDeviceGet ( CUdevice* device, int  ordinal )
// Returns a handle to a compute device.
// Parameters
// device
// - Returned device handle
// ordinal
// - Device number to get handle for
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDeviceGet(VALUE self, VALUE ordinal){
  CUdevice device;
  CUresult result = cuDeviceGet(&device, NUM2INT(ordinal));
  return UINT2NUM(device);
}

// CUresult cuDeviceGetCount ( int* count )
// Returns the number of compute-capable devices.
// Parameters
// count
// - Returned number of compute-capable devices
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuDeviceGetCount(VALUE self){
  int count;
  CUresult result = cuDeviceGetCount(&count);
  return INT2NUM(count);
}

// CUresult cuDeviceGetName ( char* name, int  len, CUdevice dev )
// Returns an identifer string for the device.
// Parameters
// name
// - Returned identifier string for the device
// len
// - Maximum length of string to store in name
// dev
// - Device to get identifier string for
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDeviceGetName(VALUE self, VALUE len_val, VALUE device_val){
  char* name;
  CUdevice dev = NUM2ULONG(device_val);
  CUresult result = cuDeviceGetName(name, NUM2INT(len_val), dev);
  return rb_str_new_cstr(name);
}

// CUresult cuDeviceTotalMem ( size_t* bytes, CUdevice dev )
// Returns the total amount of memory on the device.
// Parameters
// bytes
// - Returned memory available on device in bytes
// dev
// - Device handle
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDeviceTotalMem_v2(VALUE self, VALUE device_val){
  size_t bytes;
  CUdevice dev = NUM2ULONG(device_val);
  CUresult result = cuDeviceTotalMem_v2(&bytes, dev);
  return ULONG2NUM(bytes);
}

// CUresult cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev )
// Returns information about the device.
// Parameters
// pi
// - Returned device attribute value
// attrib
// - Device attribute to query
// dev
// - Device handle
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDeviceGetAttribute(VALUE self, VALUE pi_val, VALUE attrib_val, VALUE device_val){
  int pi;
  CUdevice dev = NUM2ULONG(device_val);
  CUdevice_attribute attrib = rb_cu_get_attrib_value(attrib_val);
  CUresult result = cuDeviceGetAttribute(&pi, attrib, dev);
  return INT2NUM(pi);
}

// CUresult cuDeviceGetProperties ( CUdevprop* prop, CUdevice dev )
// Returns properties for a selected device.
// Parameters
// prop
// - Returned properties of device
// dev
// - Device to get properties for

static VALUE rb_cuDeviceGetProperties(VALUE self, VALUE device_val){
  CUdevice dev = NUM2ULONG(device_val);
  CUdevprop* prop;
  CUresult cuDeviceGetProperties(prop, dev);
  return Qnil;
}

// CUresult cuDeviceComputeCapability ( int* major, int* minor, CUdevice dev )
// Returns the compute capability of the device.
// Parameters
// major
// - Major revision number
// minor
// - Minor revision number
// dev
// - Device handle
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDeviceComputeCapability(VALUE self, VALUE device_val){
  int major, minor;
  CUdevice dev = NUM2ULONG(device_val);
  CUresult result = cuDeviceComputeCapability(&major, &minor, dev);
  return Qnil;
}

// CUresult cuDevicePrimaryCtxRetain ( CUcontext* pctx, CUdevice dev )
// Retain the primary context on the GPU.
// Parameters
// pctx
// - Returned context handle of the new context
// dev
// - Device for which primary context is requested
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN

static VALUE rb_cuDevicePrimaryCtxRetain(VALUE self, VALUE device_val){
  CUdevice dev = NUM2ULONG(device_val);
  CUcontext* pctx;
  CUresult result = cuDevicePrimaryCtxRetain(&pctx, CUdevice dev);
  return Qnil;
}

// CUresult cuDevicePrimaryCtxRelease ( CUdevice dev )
// Release the primary context on the GPU.
// Parameters
// dev
// - Device which primary context is released
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDevicePrimaryCtxRelease(VALUE self, VALUE device_val){
  CUdevice dev = NUM2ULONG(device_val);
  CUresult result = cuDevicePrimaryCtxRelease(CUdevice dev);
  return Qnil;
}

// CUresult cuDevicePrimaryCtxSetFlags ( CUdevice dev, unsigned int  flags )
// Set flags for the primary context.
// Parameters
// dev
// - Device for which the primary context flags are set
// flags
// - New flags for the device
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_DEVICE,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE


static VALUE rb_cuDevicePrimaryCtxSetFlags(VALUE self, VALUE  device_val, VALUE flags){
  CUdevice dev = NUM2ULONG(device_val);
  CUresult result = cuDevicePrimaryCtxSetFlags (CUdevice dev, NUM2UINT(flags));
  return Qnil;
}

// CUresult cuDevicePrimaryCtxGetState ( CUdevice dev, unsigned int* flags, int* active )
// Get the state of the primary context.
// Parameters
// dev
// - Device to get primary context flags for
// flags
// - Pointer to store flags
// active
// - Pointer to store context state; 0 = inactive, 1 = active
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE,

static VALUE rb_cuDevicePrimaryCtxGetState(VALUE self, VALUE device_val){
  CUdevice dev = NUM2ULONG(device_val);
  uint flags;
  int active;
  CUresult cuDevicePrimaryCtxGetState(CUdevice dev, &flags, &active);
  return Qnil;
}

// CUresult cuDevicePrimaryCtxReset ( CUdevice dev )
// Destroy all allocations and reset all state on the primary context.
// Parameters
// dev
// - Device for which primary context is destroyed
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_DEVICE,
// CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE

static VALUE rb_cuDevicePrimaryCtxReset(VALUE self, VALUE device_val){
  CUdevice dev = NUM2ULONG(device_val);
  CUresult cuDevicePrimaryCtxReset(dev);
  return Qnil;
}

// CUresult cuCtxCreate ( CUcontext* pctx, unsigned int  flags, CUdevice dev )
// Create a CUDA context.
// Parameters
// pctx
// - Returned context handle of the new context
// flags
// - Context creation flags
// dev
// - Device to create context on
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN


/////////////////////////////////////////////////////////
//////             Context Management              //////
/////////////////////////////////////////////////////////

static VALUE rb_cuCtxCreate_v2(VALUE self){
  CUresult cuCtxCreate_v2 (CUcontext* pctx, uint flags, CUdevice dev);
  return Qnil;
}

// CUresult cuCtxDestroy ( CUcontext ctx )
// Destroy a CUDA context.
// Parameters
// ctx
// - Context to destroy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxDestroy_v2(VALUE self){
  CUresult cuCtxDestroy_v2 (CUcontext ctx);
  return Qnil;
}

// CUresult cuCtxPushCurrent ( CUcontext ctx )
// Pushes a context on the current CPU thread.
// Parameters
// ctx
// - Context to push

static VALUE rb_cuCtxPushCurrent_v2(VALUE self){
  CUresult cuCtxPushCurrent_v2 (CUcontext ctx);
  return Qnil;
}

// CUresult cuCtxPopCurrent ( CUcontext* pctx )
// Pops the current CUDA context from the current CPU thread.
// Parameters
// pctx
// - Returned new context handle
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

static VALUE rb_cuCtxPopCurrent_v2(VALUE self){
  CUresult cuCtxPopCurrent_v2 (CUcontext* pctx);
  return Qnil;
}

// CUresult cuCtxSetCurrent ( CUcontext ctx )
// Binds the specified CUDA context to the calling CPU thread.
// Parameters
// ctx
// - Context to bind to the calling CPU thread
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

static VALUE rb_cuCtxSetCurrent(VALUE self){
  CUresult cuCtxSetCurrent (CUcontext ctx);
  return Qnil;
}

static VALUE rb_cuCtxGetCurrent(VALUE self){
  CUresult cuCtxGetCurrent (CUcontext* pctx);
  return Qnil;
}

static VALUE rb_cuCtxGetDevice(VALUE self){
  CUresult cuCtxGetDevice (CUdevice* device);
  return Qnil;
}

static VALUE rb_cuCtxGetFlags(VALUE self){
  CUresult cuCtxGetFlags (uint* flags);
  return Qnil;
}

// CUresult cuCtxSynchronize ( void )
// Block for a context's tasks to complete.
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

static VALUE rb_cuCtxSynchronize(VALUE self){
  CUresult result = cuCtxSynchronize();
  return Qtrue;
}

// CUresult cuCtxSetLimit ( CUlimit limit, size_t value )
// Set resource limits.
// Parameters
// limit
// - Limit to set
// value
// - Size of limit
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNSUPPORTED_LIMIT, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuCtxSetLimit(VALUE self){
  CUresult cuCtxSetLimit (CUlimit limit, size_t value);
  return Qnil;
}

// CUresult cuCtxGetLimit ( size_t* pvalue, CUlimit limit )
// Returns resource limits.
// Parameters
// pvalue
// - Returned size of limit
// limit
// - Limit to query
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNSUPPORTED_LIMIT

static VALUE rb_cuCtxGetLimit(VALUE self){
  CUresult cuCtxGetLimit (size_t* pvalue, CUlimit limit);
  return Qnil;
}

// CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig )
// Returns the preferred cache configuration for the current context.
// Parameters
// pconfig
// - Returned cache configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxGetCacheConfig(VALUE self){
  CUresult cuCtxGetCacheConfig (CUfunc_cache* pconfig);
  return Qnil;
}

// CUresult cuCtxSetCacheConfig ( CUfunc_cache config )
// Sets the preferred cache configuration for the current context.
// Parameters
// config
// - Requested cache configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxSetCacheConfig(VALUE self){
  CUresult cuCtxSetCacheConfig (CUfunc_cache config);
  return Qnil;
}

// CUresult cuCtxGetSharedMemConfig ( CUsharedconfig* pConfig )
// Returns the current shared memory configuration for the current context.
// Parameters
// pConfig
// - returned shared memory configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxGetSharedMemConfig(VALUE self){
  CUresult cuCtxGetSharedMemConfig (CUsharedconfig* pConfig);
  return Qnil;
}

// CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config )
// Sets the shared memory configuration for the current context.
// Parameters
// config
// - requested shared memory configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxSetSharedMemConfig(VALUE self){
  CUresult cuCtxSetSharedMemConfig (CUsharedconfig config);
  return Qnil;
}

// CUresult cuCtxGetApiVersion ( CUcontext ctx, unsigned int* version )
// Gets the context's API version.
// Parameters
// ctx
// - Context to check
// version
// - Pointer to version
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_UNKNOWN

static VALUE rb_cuCtxGetApiVersion(VALUE self){
  CUresult cuCtxGetApiVersion (CUcontext ctx, uint* version_);
  return Qnil;
}

// CUresult cuCtxGetStreamPriorityRange ( int* leastPriority, int* greatestPriority )
// Returns numerical values that correspond to the least and greatest stream priorities.
// Parameters
// leastPriority
// - Pointer to an int in which the numerical value for least stream priority is returned
// greatestPriority
// - Pointer to an int in which the numerical value for greatest stream priority is returned
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,


static VALUE rb_cuCtxGetStreamPriorityRange(VALUE self){
  CUresult cuCtxGetStreamPriorityRange (int* leastPriority, int* greatestPriority);
  return Qnil;
}



static VALUE rb_cuCtxAttach(VALUE self){
  CUresult cuCtxAttach (CUcontext* pctx, uint flags);
  return Qnil;
}

static VALUE rb_cuCtxDetach(VALUE self){
  CUresult cuCtxDetach(CUcontext ctx);
  return Qnil;
}

static VALUE rb_cuModuleLoad(VALUE self){
  CUresult cuModuleLoad (CUmodule* module_, const(char)* fname);
  return Qnil;
}

static VALUE rb_cuModuleLoadData(VALUE self){
  CUresult cuModuleLoadData (CUmodule* module_, const(void)* image);
  return Qnil;
}

static VALUE rb_cuModuleLoadDataEx(VALUE self){
  CUresult cuModuleLoadDataEx (CUmodule* module_, const(void)* image, uint numOptions, CUjit_option* options, void** optionValues);
  return Qnil;
}

static VALUE rb_cuModuleLoadFatBinary(VALUE self){
  CUresult cuModuleLoadFatBinary (CUmodule* module_, const(void)* fatCubin);
  return Qnil;
}

static VALUE rb_cuModuleUnload(VALUE self){
  CUresult cuModuleUnload (CUmodule hmod);
  return Qnil;
}

static VALUE rb_cuModuleGetFunction(VALUE self){
  CUresult cuModuleGetFunction (CUfunction* hfunc, CUmodule hmod, const(char)* name);
  return Qnil;
}

static VALUE rb_cuModuleGetGlobal_v2(VALUE self){
  CUresult cuModuleGetGlobal_v2 (CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const(char)* name);
  return Qnil;
}

static VALUE rb_cuModuleGetTexRef(VALUE self){
  CUresult cuModuleGetTexRef (CUtexref* pTexRef, CUmodule hmod, const(char)* name);
  return Qnil;
}

static VALUE rb_cuModuleGetSurfRef(VALUE self){
  CUresult cuModuleGetSurfRef (CUsurfref* pSurfRef, CUmodule hmod, const(char)* name);
  return Qnil;
}

static VALUE rb_cuLinkCreate_v2(VALUE self){
  CUresult cuLinkCreate_v2 ( uint numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut);
  return Qnil;
}


static VALUE rb_cuLinkAddData_v2(VALUE self){
  CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void* data, size_t size, const(char)* name, uint numOptions, CUjit_option* options, void** optionValues);
  return Qnil;
}

static VALUE rb_cuLinkAddFile_v2(VALUE self){
  CUresult cuLinkAddFile_v2 ( CUlinkState state, CUjitInputType type, const(char)* path, uint numOptions, CUjit_option* options, void** optionValues);
  return Qnil;
}

static VALUE rb_cuLinkComplete(VALUE self){
  CUresult cuLinkComplete (CUlinkState state, void** cubinOut, size_t* sizeOut);
  return Qnil;
}

static VALUE rb_cuLinkDestroy(VALUE self){
  CUresult cuLinkDestroy (CUlinkState state);
  return Qnil;
}

static VALUE rb_cuMemGetInfo_v2(VALUE self){
  CUresult cuMemGetInfo_v2 (size_t* free, size_t* total);
  return Qnil;
}

static VALUE rb_cuMemAlloc_v2(VALUE self){
  CUresult cuMemAlloc_v2 (CUdeviceptr* dptr, size_t bytesize);
  return Qnil;
}

static VALUE rb_cuMemAllocPitch_v2(VALUE self){
  CUresult cuMemAllocPitch_v2 (CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, uint ElementSizeBytes);
  return Qnil;
}

static VALUE rb_cuMemFree_v2(VALUE self){
  CUresult cuMemFree_v2 (CUdeviceptr dptr);
  return Qnil;
}

static VALUE rb_cuMemGetAddressRange_v2(VALUE self){
  CUresult cuMemGetAddressRange_v2 (CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
  return Qnil;
}

static VALUE rb_cuMemAllocHost_v2(VALUE self){
  CUresult cuMemAllocHost_v2 (void** pp, size_t bytesize);
  return Qnil;
}

static VALUE rb_cuMemFreeHost(VALUE self){
  CUresult cuMemFreeHost (void* p);
  return Qnil;
}

static VALUE rb_cuMemHostAlloc(VALUE self){
  CUresult cuMemHostAlloc (void** pp, size_t bytesize, uint Flags);
  return Qnil;
}

static VALUE rb_cuMemHostGetDevicePointer_v2(VALUE self){
  CUresult cuMemHostGetDevicePointer_v2 (CUdeviceptr* pdptr, void* p, uint Flags);
  return Qnil;
}

static VALUE rb_cuMemHostGetFlags(VALUE self){
  CUresult cuMemHostGetFlags (uint* pFlags, void* p);
  return Qnil;
}

static VALUE rb_cuMemAllocManaged(VALUE self){
  CUresult cuMemAllocManaged (CUdeviceptr* dptr, size_t bytesize, uint flags);
  return Qnil;
}

static VALUE rb_cuDeviceGetByPCIBusId(VALUE self){
  CUresult cuDeviceGetByPCIBusId (CUdevice* dev, const(char)* pciBusId);
  return Qnil;
}

static VALUE rb_cuDeviceGetPCIBusId(VALUE self){
  CUresult cuDeviceGetPCIBusId (char* pciBusId, int len, CUdevice dev);
  return Qnil;
}


static VALUE rb_cuIpcGetEventHandle(VALUE self){
  CUresult cuIpcGetEventHandle (CUipcEventHandle* pHandle, CUevent event);
  return Qnil;
}


static VALUE rb_cuIpcOpenEventHandle(VALUE self){
  CUresult cuIpcOpenEventHandle (CUevent* phEvent, CUipcEventHandle handle);
  return Qnil;
}

static VALUE rb_cuIpcGetMemHandle(VALUE self){
  CUresult cuIpcGetMemHandle (CUipcMemHandle* pHandle, CUdeviceptr dptr);
  return Qnil;
}

static VALUE rb_cuIpcOpenMemHandle(VALUE self){
  CUresult cuIpcOpenMemHandle (CUdeviceptr* pdptr, CUipcMemHandle handle, uint Flags);
  return Qnil;
}

static VALUE rb_cuIpcCloseMemHandle(VALUE self){
  CUresult cuIpcCloseMemHandle (CUdeviceptr dptr);
  return Qnil;
}

static VALUE rb_cuMemHostRegister_v2(VALUE self){
  CUresult cuMemHostRegister_v2 (void* p, size_t bytesize, uint Flags);
  return Qnil;
}

static VALUE rb_cuMemHostUnregister(VALUE self){
  CUresult cuMemHostUnregister (void* p);
  return Qnil;
}

static VALUE rb_cuMemcpy(VALUE self){
  CUresult cuMemcpy (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyPeer(VALUE self){
  CUresult cuMemcpyPeer (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyHtoD_v2(VALUE self){
  CUresult cuMemcpyHtoD_v2 (CUdeviceptr dstDevice, const(void)* srcHost, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyDtoH_v2(VALUE self){
  CUresult cuMemcpyDtoH_v2 (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyDtoD_v2(VALUE self){
  CUresult cuMemcpyDtoD_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyDtoA_v2(VALUE self){
  CUresult cuMemcpyDtoA_v2 (CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyAtoD_v2(VALUE self){
  CUresult cuMemcpyAtoD_v2 (CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyHtoA_v2(VALUE self){
  CUresult cuMemcpyHtoA_v2 (CUarray dstArray, size_t dstOffset, const(void)* srcHost, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyAtoH_v2(VALUE self){
  CUresult cuMemcpyAtoH_v2 (void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpyAtoA_v2(VALUE self){
  CUresult cuMemcpyAtoA_v2 (CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
  return Qnil;
}

static VALUE rb_cuMemcpy2D_v2(VALUE self){
  CUresult cuMemcpy2D_v2 (const(CUDA_MEMCPY2D)* pCopy);
  return Qnil;
}


static VALUE rb_cuMemcpy2DUnaligned_v2(VALUE self){
  CUresult cuMemcpy2DUnaligned_v2 (const(CUDA_MEMCPY2D)* pCopy);
  return Qnil;
}


static VALUE rb_cuMemcpy3D_v2(VALUE self){
  CUresult cuMemcpy3D_v2 (const(CUDA_MEMCPY3D)* pCopy);
  return Qnil;
}

static VALUE rb_cuMemcpy3DPeer(VALUE self){
  CUresult cuMemcpy3DPeer (const(CUDA_MEMCPY3D_PEER)* pCopy);
  return Qnil;
}

static VALUE rb_cuMemcpyAsync(VALUE self){
  CUresult cuMemcpyAsync (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpyPeerAsync(VALUE self){
  CUresult cuMemcpyPeerAsync (CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpyHtoDAsync_v2(VALUE self){
  CUresult cuMemcpyHtoDAsync_v2 (CUdeviceptr dstDevice, const(void)* srcHost, size_t ByteCount, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpyDtoHAsync_v2(VALUE self){
  CUresult cuMemcpyDtoHAsync_v2 (void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpyDtoDAsync_v2(VALUE self){
  CUresult cuMemcpyDtoDAsync_v2 (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpyHtoAAsync_v2(VALUE self){
  CUresult cuMemcpyHtoAAsync_v2 (CUarray dstArray, size_t dstOffset, const(void)* srcHost, size_t ByteCount, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpyAtoHAsync_v2(VALUE self){
  CUresult cuMemcpyAtoHAsync_v2 (void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpy2DAsync_v2(VALUE self){
  CUresult cuMemcpy2DAsync_v2 (const(CUDA_MEMCPY2D)* pCopy, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpy3DAsync_v2(VALUE self){
  CUresult cuMemcpy3DAsync_v2 (const(CUDA_MEMCPY3D)* pCopy, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemcpy3DPeerAsync(VALUE self){
  CUresult cuMemcpy3DPeerAsync (const(CUDA_MEMCPY3D_PEER)* pCopy, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemsetD8_v2(VALUE self){
  CUresult cuMemsetD8_v2 (CUdeviceptr dstDevice, ubyte uc, size_t N);
  return Qnil;
}

static VALUE rb_cuMemsetD16_v2(VALUE self){
  CUresult cuMemsetD16_v2 (CUdeviceptr dstDevice, ushort us, size_t N);
  return Qnil;
}

static VALUE rb_cuMemsetD32_v2(VALUE self){
  CUresult cuMemsetD32_v2 (CUdeviceptr dstDevice, uint ui, size_t N);
  return Qnil;
}

static VALUE rb_cuMemsetD2D8_v2(VALUE self){
  CUresult cuMemsetD2D8_v2 (CUdeviceptr dstDevice, size_t dstPitch, ubyte uc, size_t Width, size_t Height);
  return Qnil;
}

static VALUE rb_cuMemsetD2D16_v2(VALUE self){
  CUresult cuMemsetD2D16_v2 (CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height);
  return Qnil;
}

static VALUE rb_cuMemsetD2D32_v2(VALUE self){
  CUresult cuMemsetD2D32_v2 (CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height);
  return Qnil;
}

static VALUE rb_cuMemsetD8Async(VALUE self){
  CUresult cuMemsetD8Async (CUdeviceptr dstDevice, ubyte uc, size_t N, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemsetD16Async(VALUE self){
  CUresult cuMemsetD16Async (CUdeviceptr dstDevice, ushort us, size_t N, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemsetD32Async(VALUE self){
  CUresult cuMemsetD32Async (CUdeviceptr dstDevice, uint ui, size_t N, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemsetD2D8Async(VALUE self){
  CUresult cuMemsetD2D8Async (CUdeviceptr dstDevice, size_t dstPitch, ubyte uc, size_t Width, size_t Height, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemsetD2D16Async(VALUE self){
  CUresult cuMemsetD2D16Async (CUdeviceptr dstDevice, size_t dstPitch, ushort us, size_t Width, size_t Height, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuMemsetD2D32Async(VALUE self){
  CUresult cuMemsetD2D32Async (CUdeviceptr dstDevice, size_t dstPitch, uint ui, size_t Width, size_t Height, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuArrayCreate_v2(VALUE self){
  CUresult cuArrayCreate_v2 (CUarray* pHandle, const(CUDA_ARRAY_DESCRIPTOR)* pAllocateArray);
  return Qnil;
}

static VALUE rb_cuArrayGetDescriptor_v2(VALUE self){
  CUresult cuArrayGetDescriptor_v2 (CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
  return Qnil;
}

static VALUE rb_cuArrayDestroy(VALUE self){
  CUresult cuArrayDestroy (CUarray hArray);
  return Qnil;
}

static VALUE rb_cuArray3DCreate_v2(VALUE self){
  CUresult cuArray3DCreate_v2 (CUarray* pHandle, const(CUDA_ARRAY3D_DESCRIPTOR)* pAllocateArray);
  return Qnil;
}

static VALUE rb_cuArray3DGetDescriptor_v2(VALUE self){
  CUresult cuArray3DGetDescriptor_v2 (CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
  return Qnil;
}

static VALUE rb_cuMipmappedArrayCreate(VALUE self){
  CUresult cuMipmappedArrayCreate (CUmipmappedArray* pHandle, const(CUDA_ARRAY3D_DESCRIPTOR)* pMipmappedArrayDesc, uint numMipmapLevels);
  return Qnil;
}

static VALUE rb_cuMipmappedArrayGetLevel(VALUE self){
  CUresult cuMipmappedArrayGetLevel (CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, uint level);
  return Qnil;
}

static VALUE rb_cuMipmappedArrayDestroy(VALUE self){
  CUresult cuMipmappedArrayDestroy (CUmipmappedArray hMipmappedArray);
  return Qnil;
}

static VALUE rb_cuPointerGetAttribute(VALUE self){
  CUresult cuPointerGetAttribute (void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
  return Qnil;
}

static VALUE rb_cuPointerSetAttribute(VALUE self){
  CUresult cuPointerSetAttribute (const(void)* value, CUpointer_attribute attribute, CUdeviceptr ptr);
  return Qnil;
}

static VALUE rb_cuPointerGetAttributes(VALUE self){
  CUresult cuPointerGetAttributes (uint numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr);
  return Qnil;
}

static VALUE rb_cuStreamCreate(VALUE self){
  CUresult cuStreamCreate (CUstream* phStream, uint Flags);
  return Qnil;
}

static VALUE rb_cuStreamCreateWithPriority(VALUE self){
  CUresult cuStreamCreateWithPriority (CUstream* phStream, uint flags, int priority);
  return Qnil;
}

static VALUE rb_cuStreamGetPriority(VALUE self){
  CUresult cuStreamGetPriority (CUstream hStream, int* priority);
  return Qnil;
}

static VALUE rb_cuStreamGetFlags(VALUE self){
  CUresult cuStreamGetFlags (CUstream hStream, uint* flags);
  return Qnil;
}

static VALUE rb_cuStreamWaitEvent(VALUE self){
  CUresult cuStreamWaitEvent (CUstream hStream, CUevent hEvent, uint Flags);
  return Qnil;
}

static VALUE rb_cuStreamAddCallback(VALUE self){
  CUresult cuStreamAddCallback (CUstream hStream, CUstreamCallback callback, void* userData, uint flags);
  return Qnil;
}

static VALUE rb_cuStreamAttachMemAsync(VALUE self){
  CUresult cuStreamAttachMemAsync (CUstream hStream, CUdeviceptr dptr, size_t length, uint flags);
  return Qnil;
}

static VALUE rb_cuStreamQuery(VALUE self){
  CUresult cuStreamQuery (CUstream hStream);
  return Qnil;
}

static VALUE rb_cuStreamSynchronize(VALUE self){
  CUresult cuStreamSynchronize (CUstream hStream);
  return Qnil;
}

static VALUE rb_cuStreamDestroy_v2(VALUE self){
  CUresult cuStreamDestroy_v2 (CUstream hStream);
  return Qnil;
}

static VALUE rb_cuEventCreate(VALUE self){
  CUresult cuEventCreate (CUevent* phEvent, uint Flags);
  return Qnil;
}

static VALUE rb_cuEventRecord(VALUE self){
  CUresult cuEventRecord (CUevent hEvent, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuEventQuery(VALUE self){
  CUresult cuEventQuery (CUevent hEvent);
  return Qnil;
}

static VALUE rb_cuEventSynchronize(VALUE self){
  CUresult cuEventSynchronize (CUevent hEvent);
  return Qnil;
}

static VALUE rb_cuEventDestroy_v2(VALUE self){
  CUresult cuEventDestroy_v2 (CUevent hEvent);
  return Qnil;
}

static VALUE rb_cuEventElapsedTime(VALUE self){
  CUresult cuEventElapsedTime (float* pMilliseconds, CUevent hStart, CUevent hEnd);
  return Qnil;
}

static VALUE rb_cuFuncGetAttribute(VALUE self){
  CUresult cuFuncGetAttribute (int* pi, CUfunction_attribute attrib, CUfunction hfunc);
  return Qnil;
}

static VALUE rb_cuFuncSetCacheConfig(VALUE self){
  CUresult cuFuncSetCacheConfig (CUfunction hfunc, CUfunc_cache config);
  return Qnil;
}

static VALUE rb_cuFuncSetSharedMemConfig(VALUE self){
  CUresult cuFuncSetSharedMemConfig (CUfunction hfunc, CUsharedconfig config);
  return Qnil;
}

// IMPORTANT
static VALUE rb_cuLaunchKernel(VALUE self){

  CUresult cuLaunchKernel (
    CUfunction f,
    uint gridDimX,
    uint gridDimY,
    uint gridDimZ,
    uint blockDimX,
    uint blockDimY,
    uint blockDimZ,
    uint sharedMemBytes,
    CUstream hStream,
    void** kernelParams,
    void** extra);

  return Qnil;
}

static VALUE rb_cuFuncSetBlockShape(VALUE self){
  CUresult cuFuncSetBlockShape (CUfunction hfunc, int x, int y, int z);
  return Qnil;
}

static VALUE rb_cuFuncSetSharedSize(VALUE self){
  CUresult cuFuncSetSharedSize (CUfunction hfunc, uint bytes);
  return Qnil;
}

static VALUE rb_cuParamSetSize(VALUE self){
  CUresult cuParamSetSize (CUfunction hfunc, uint numbytes);
  return Qnil;
}

static VALUE rb_cuParamSeti(VALUE self){
  CUresult cuParamSeti (CUfunction hfunc, int offset, uint value);
  return Qnil;
}

static VALUE rb_cuParamSetf(VALUE self){
  CUresult cuParamSetf (CUfunction hfunc, int offset, float value);
  return Qnil;
}





static VALUE rb_cuParamSetv(VALUE self){
  CUresult cuParamSetv (CUfunction hfunc, int offset, void* ptr, uint numbytes);
  return Qnil;
}

static VALUE rb_cuLaunch(VALUE self){
  CUresult cuLaunch (CUfunction f);
  return Qnil;
}

static VALUE rb_cuLaunchGrid(VALUE self){
  CUresult cuLaunchGrid (CUfunction f, int grid_width, int grid_height);
  return Qnil;
}

static VALUE rb_cuLaunchGridAsync(VALUE self){
  CUresult cuLaunchGridAsync (CUfunction f, int grid_width, int grid_height, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuParamSetTexRef(VALUE self){
  CUresult cuParamSetTexRef (CUfunction hfunc, int texunit, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuOccupancyMaxActiveBlocksPerMultiprocessor(VALUE self){
  CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor (int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
  return Qnil;
}

static VALUE rb_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(VALUE self){
  CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (int* numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize, uint flags);
  return Qnil;
}

static VALUE rb_cuOccupancyMaxPotentialBlockSize(VALUE self){
  CUresult cuOccupancyMaxPotentialBlockSize (int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit);
  return Qnil;
}

static VALUE rb_cuOccupancyMaxPotentialBlockSizeWithFlags(VALUE self){
  CUresult cuOccupancyMaxPotentialBlockSizeWithFlags (int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit, uint flags);
  return Qnil;
}

static VALUE rb_cuTexRefSetArray(VALUE self){
  CUresult cuTexRefSetArray (CUtexref hTexRef, CUarray hArray, uint Flags);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmappedArray(VALUE self){
  CUresult cuTexRefSetMipmappedArray (CUtexref hTexRef, CUmipmappedArray hMipmappedArray, uint Flags);
  return Qnil;
}

static VALUE rb_cuTexRefSetAddress_v2(VALUE self){
  CUresult cuTexRefSetAddress_v2 (size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
  return Qnil;
}

static VALUE rb_cuTexRefSetAddress2D_v3(VALUE self){
  CUresult cuTexRefSetAddress2D_v3 (CUtexref hTexRef, const(CUDA_ARRAY_DESCRIPTOR)* desc, CUdeviceptr dptr, size_t Pitch);
  return Qnil;
}

static VALUE rb_cuTexRefSetFormat(VALUE self){
  CUresult cuTexRefSetFormat (CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
  return Qnil;
}

static VALUE rb_cuTexRefSetAddressMode(VALUE self){
  CUresult cuTexRefSetAddressMode (CUtexref hTexRef, int dim, CUaddress_mode am);
  return Qnil;
}

static VALUE rb_cuTexRefSetFilterMode(VALUE self){
  CUresult cuTexRefSetFilterMode (CUtexref hTexRef, CUfilter_mode fm);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapFilterMode(VALUE self){
  CUresult cuTexRefSetMipmapFilterMode (CUtexref hTexRef, CUfilter_mode fm);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapLevelBias(VALUE self){
  CUresult cuTexRefSetMipmapLevelBias (CUtexref hTexRef, float bias);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapLevelClamp(VALUE self){
  CUresult cuTexRefSetMipmapLevelClamp (CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
  return Qnil;
}

static VALUE rb_cuTexRefSetMaxAnisotropy(VALUE self){
  CUresult cuTexRefSetMaxAnisotropy (CUtexref hTexRef, uint maxAniso);
  return Qnil;
}

static VALUE rb_cuTexRefSetFlags(VALUE self){
  CUresult cuTexRefSetFlags (CUtexref hTexRef, uint Flags);
  return Qnil;
}

static VALUE rb_cuTexRefGetAddress_v2(VALUE self){
  CUresult cuTexRefGetAddress_v2 (CUdeviceptr* pdptr, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetArray(VALUE self){
  CUresult cuTexRefGetArray (CUarray* phArray, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmappedArray(VALUE self){
  CUresult cuTexRefGetMipmappedArray (CUmipmappedArray* phMipmappedArray, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetAddressMode(VALUE self){
  CUresult cuTexRefGetAddressMode (CUaddress_mode* pam, CUtexref hTexRef, int dim);
  return Qnil;
}

static VALUE rb_cuTexRefGetFilterMode(VALUE self){
  CUresult cuTexRefGetFilterMode (CUfilter_mode* pfm, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetFormat(VALUE self){
  CUresult cuTexRefGetFormat (CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapFilterMode(VALUE self){
  CUresult cuTexRefGetMipmapFilterMode (CUfilter_mode* pfm, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapLevelBias(VALUE self){
  CUresult cuTexRefGetMipmapLevelBias (float* pbias, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapLevelClamp(VALUE self){
  CUresult cuTexRefGetMipmapLevelClamp (float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMaxAnisotropy(VALUE self){
  CUresult cuTexRefGetMaxAnisotropy (int* pmaxAniso, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetFlags(VALUE self){
  CUresult cuTexRefGetFlags (uint* pFlags, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefCreate(VALUE self){
  CUresult cuTexRefCreate (CUtexref* pTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefDestroy(VALUE self){
  CUresult cuTexRefDestroy (CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuSurfRefSetArray(VALUE self){
  CUresult cuSurfRefSetArray (CUsurfref hSurfRef, CUarray hArray, uint Flags);
  return Qnil;
}

static VALUE rb_cuSurfRefGetArray(VALUE self){
  CUresult cuSurfRefGetArray (CUarray* phArray, CUsurfref hSurfRef);
  return Qnil;
}

static VALUE rb_cuTexObjectCreate(VALUE self){
  CUresult cuTexObjectCreate (CUtexObject* pTexObject, const(CUDA_RESOURCE_DESC)* pResDesc, const(CUDA_TEXTURE_DESC)* pTexDesc, const(CUDA_RESOURCE_VIEW_DESC)* pResViewDesc);
  return Qnil;
}

static VALUE rb_cuTexObjectDestroy(VALUE self){
  CUresult cuTexObjectDestroy (CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuTexObjectGetResourceDesc(VALUE self){
  CUresult cuTexObjectGetResourceDesc (CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuTexObjectGetTextureDesc(VALUE self){
  CUresult cuTexObjectGetTextureDesc (CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuTexObjectGetResourceViewDesc(VALUE self){
  CUresult cuTexObjectGetResourceViewDesc (CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuSurfObjectCreate(VALUE self){
  CUresult cuSurfObjectCreate (CUsurfObject* pSurfObject, const(CUDA_RESOURCE_DESC)* pResDesc);
  return Qnil;
}

static VALUE rb_cuSurfObjectDestroy(VALUE self){
  CUresult cuSurfObjectDestroy (CUsurfObject surfObject);
  return Qnil;
}

static VALUE rb_cuSurfObjectGetResourceDesc(VALUE self){
  CUresult cuSurfObjectGetResourceDesc (CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject);
  return Qnil;
}

static VALUE rb_cuDeviceCanAccessPeer(VALUE self){
  CUresult cuDeviceCanAccessPeer (int* canAccessPeer, CUdevice dev, CUdevice peerDev);
  return Qnil;
}

static VALUE rb_cuCtxEnablePeerAccess(VALUE self){
  CUresult cuCtxEnablePeerAccess (CUcontext peerContext, uint Flags);
  return Qnil;
}

static VALUE rb_cuCtxDisablePeerAccess(VALUE self){
  CUresult cuCtxDisablePeerAccess (CUcontext peerContext);
  return Qnil;
}

static VALUE rb_cuGraphicsUnregisterResource(VALUE self){
  CUresult cuGraphicsUnregisterResource (CUgraphicsResource resource);
  return Qnil;
}

static VALUE rb_cuGraphicsSubResourceGetMappedArray(VALUE self){
  CUresult cuGraphicsSubResourceGetMappedArray (CUarray* pArray, CUgraphicsResource resource, uint arrayIndex, uint mipLevel);
  return Qnil;
}

static VALUE rb_cuGraphicsResourceGetMappedMipmappedArray(VALUE self){
  CUresult cuGraphicsResourceGetMappedMipmappedArray (CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource);
  return Qnil;
}

static VALUE rb_cuGraphicsResourceGetMappedPointer_v2(VALUE self){
  CUresult cuGraphicsResourceGetMappedPointer_v2 (CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
  return Qnil;
}

static VALUE rb_cuGraphicsResourceSetMapFlags_v2(VALUE self){
  CUresult cuGraphicsResourceSetMapFlags_v2 (CUgraphicsResource resource, uint flags);
  return Qnil;
}


static VALUE rb_cuGraphicsMapResources(VALUE self){
  CUresult cuGraphicsMapResources (uint count, CUgraphicsResource* resources, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuGraphicsUnmapResources(VALUE self){
  CUresult cuGraphicsUnmapResources (uint count, CUgraphicsResource* resources, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuGetExportTable(VALUE self){
  CUresult cuGetExportTable (const(void*)* ppExportTable, const(CUuuid)* pExportTableId);
  return Qnil;
}
