// __host__ ​cudaError_t cudaDeviceReset ( void )
// Destroy all allocations and reset all state on the current device in the current process.
// Returns
// cudaSuccess

static VALUE rb_cudaDeviceReset(VALUE self){
  cudaError error = cudaDeviceReset();
  return Qtrue;
}

// __host__ ​cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig )
// Sets the preferred cache configuration for the current device.
// Parameters
// cacheConfig
// - Requested cache configuration
// Returns
// cudaSuccess, cudaErrorInitializationError

static VALUE rb_cudaDeviceSynchronize(VALUE self){
  cudaError error = cudaDeviceSynchronize();
  return Qtrue;
}

// __host__ ​cudaError_t cudaDeviceSetLimit ( cudaLimit limit, size_t value )
// Set resource limits.
// Parameters
// limit
// - Limit to set
// value
// - Size of limit
// Returns
// cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue, cudaErrorMemoryAllocation

static VALUE rb_cudaDeviceSetLimit(VALUE self, VALUE limit, VALUE value){
  cudaError error = cudaDeviceSetLimit(rb_cudaLimit_from_rbsymbol(limit), NUM2UINT(value));
  return Qtrue;
}

// __host__ ​ __device__ ​cudaError_t cudaDeviceGetLimit ( size_t* pValue, cudaLimit limit )
// Returns resource limits.
// Parameters
// pValue
// - Returned size of the limit
// limit
// - Limit to query
// Returns
// cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue

static VALUE rb_cudaDeviceGetLimit(VALUE self, VALUE pValue, VALUE limit){
  size_t p_value;
  cudaError error = cudaDeviceGetLimit(&p_value, rb_cudaLimit_from_rbsymbol(limit));
  return Qtrue;
}

// __host__ ​ __device__ ​cudaError_t cudaDeviceGetCacheConfig ( cudaFuncCache ** pCacheConfig )
// Returns the preferred cache configuration for the current device.
// Parameters
// pCacheConfig
// - Returned cache configuration
// Returns
// cudaSuccess, cudaErrorInitializationError

static VALUE rb_cudaDeviceGetCacheConfig(VALUE self){
  cudaFuncCache p_cache_config;
  cudaError error = cudaDeviceGetCacheConfig(&p_cache_config);
  return rb_str_new_cstr(get_function_cache_name(p_cache_config));
}

// __host__ ​cudaError_t cudaDeviceGetStreamPriorityRange ( int* leastPriority, int* greatestPriority )
// Returns numerical values that correspond to the least and greatest stream priorities.
// Parameters
// leastPriority
// - Pointer to an int in which the numerical value for least stream priority is returned
// greatestPriority
// - Pointer to an int in which the numerical value for greatest stream priority is returned
// Returns
// cudaSuccess, cudaErrorInitializationError

static VALUE rb_cudaDeviceGetStreamPriorityRange(VALUE self){
  int least_priority, greatest_priority;
  cudaError error = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  return Qtrue;
}

// __host__ ​cudaError_t cudaDeviceSetCacheConfig ( cudaFuncCache cacheConfig )
// Sets the preferred cache configuration for the current device.
// Parameters
// cacheConfig
// - Requested cache configuration
// Returns
// cudaSuccess, cudaErrorInitializationError

static VALUE rb_cudaDeviceSetCacheConfig(VALUE self, VALUE cache_config){
  cudaError error = cudaDeviceSetCacheConfig(rb_cu_function_cache_from_rbsymbol(cache_config));
  return Qtrue;
}

// __host__ ​cudaError_t cudaDeviceSetSharedMemConfig ( cudaSharedMemConfig config )
// Sets the shared memory configuration for the current device.
// Parameters
// config
// - Requested cache configuration
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError

static VALUE rb_cudaDeviceGetSharedMemConfig(VALUE self){
  cudaSharedMemConfig p_config;
  cudaError error = cudaDeviceGetSharedMemConfig(&p_config);
  const char* config = get_shared_mem_name(p_config);
  return rb_str_new_cstr(config);
}

// __host__ ​cudaError_t cudaDeviceSetSharedMemConfig ( cudaSharedMemConfig config )
// Sets the shared memory configuration for the current device.
// Parameters
// config
// - Requested cache configuration
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError

static VALUE rb_cudaDeviceSetSharedMemConfig(VALUE self, VALUE config){
  cudaError error = cudaDeviceSetSharedMemConfig(rb_cu_shared_mem_from_rbsymbol(config));
  return Qtrue;
}

// __host__ ​cudaError_t cudaDeviceGetByPCIBusId ( int* device, const char* pciBusId )
// Returns a handle to a compute device.
// Parameters
// device
// - Returned device ordinal
// pciBusId
// - String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

static VALUE rb_cudaDeviceGetByPCIBusId(VALUE self, VALUE pci_bus_id){
  int device;
  cudaError error = cudaDeviceGetByPCIBusId(&device, StringValueCStr(pci_bus_id));
  return INT2NUM(device);
}

// __host__ ​cudaError_t cudaDeviceGetPCIBusId ( char* pciBusId, int  len, int  device )
// Returns a PCI Bus Id string for the device.
// Parameters
// pciBusId
// - Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values. pciBusId should be large enough to store 13 characters including the NULL-terminator.
// len
// - Maximum length of string to store in name
// device
// - Device to get identifier string for
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

static VALUE rb_cudaDeviceGetPCIBusId(VALUE self, VALUE len, VALUE device){
  char* pci_bus_id;
  cudaError error = cudaDeviceGetPCIBusId(pci_bus_id, len, device);
  return rb_str_new_cstr(pci_bus_id);
}

// __host__ ​cudaError_t cudaIpcGetEventHandle ( cudaIpcEventHandle_t* handle, cudaEvent_t event )
// Gets an interprocess handle for a previously allocated event.
// Parameters
// handle
// - Pointer to a user allocated cudaIpcEventHandle in which to return the opaque event handle
// event
// - Event allocated with cudaEventInterprocess and cudaEventDisableTiming flags.
// Returns
// cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation, cudaErrorMapBufferObjectFailed, cudaErrorNotSupported

static VALUE rb_cudaIpcGetEventHandle(VALUE self, VALUE event_val){
  cuda_ipc_event_handler* handler = ALLOC(cuda_ipc_event_handler);
  cu_event* event_ptr;
  Data_Get_Struct(event_val, cu_event, event_ptr);
  cudaIpcGetEventHandle(&handler->handle, event_ptr->event);
  return Data_Wrap_Struct(RbCuCUDAIPCEventHandler, NULL, rbcu_free, handler);
}

// __host__ ​cudaError_t cudaIpcOpenEventHandle ( cudaEvent_t* event, cudaIpcEventHandle_t handle )
// Opens an interprocess event handle for use in the current process.
// Parameters
// event
// - Returns the imported event
// handle
// - Interprocess handle to open
// Returns
// cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle, cudaErrorNotSupported

static VALUE rb_cudaIpcOpenEventHandle(VALUE self, VALUE handler_val){
  cuda_ipc_event_handler* handler;
  Data_Get_Struct(handler_val, cuda_ipc_event_handler, handler);
  cu_event* event_ptr = ALLOC(cu_event);
  cudaError error = cudaIpcOpenEventHandle(&event_ptr->event, handler->handle);
  return Data_Wrap_Struct(RbCuEvent, NULL, rbcu_free, event_ptr);
}

static VALUE rb_cudaIpcGetMemHandle(VALUE self, VALUE handler_val, VALUE dev_ptr){
  cuda_ipc_mem_handler* handler = ALLOC(cuda_ipc_mem_handler);
  cudaError error = cudaIpcGetMemHandle(&handler->handle, (void*)dev_ptr);
  return Data_Wrap_Struct(RbCuCUDAIPCMemHandler, NULL, rbcu_free, handler);
}

// __host__ ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags )
// Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
// Parameters
// devPtr
// - Returned device pointer
// handle
// - cudaIpcMemHandle to open
// flags
// - Flags for this operation. Must be specified as cudaIpcMemLazyEnablePeerAccess
// Returns
// cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle, cudaErrorTooManyPeers, cudaErrorNotSupported

static VALUE rb_cudaIpcOpenMemHandle(VALUE self, VALUE handler_val, VALUE flags){
  cuda_ipc_mem_handler* handler;
  Data_Get_Struct(handler_val, cuda_ipc_mem_handler, handler);
  void* dev_ptr;
  cudaError error = cudaIpcOpenMemHandle(&dev_ptr, handler->handle, NUM2UINT(flags));
  return VALUE(dev_ptr);
}

// __host__ ​cudaError_t cudaIpcCloseMemHandle ( void* devPtr )
// Close memory mapped with cudaIpcOpenMemHandle.
// Parameters
// devPtr
// - Device pointer returned by cudaIpcOpenMemHandle
// Returns
// cudaSuccess, cudaErrorMapBufferObjectFailed, cudaErrorInvalidResourceHandle, cudaErrorNotSupported

static VALUE rb_cudaIpcCloseMemHandle(VALUE self, VALUE dev_ptr){
  cudaError error = cudaIpcCloseMemHandle( (void*)dev_ptr);
  return Qtrue;
}

// __host__ ​cudaError_t cudaThreadExit ( void )
// Exit and clean up from CUDA launches.
// Returns
// cudaSuccess

static VALUE rb_cudaThreadExit(VALUE self){
  cudaError error = cudaThreadExit();
  return Qtrue;
}

// __host__ ​cudaError_t cudaThreadSynchronize ( void )
// Wait for compute device to finish.
// Returns
// cudaSuccess

static VALUE rb_cudaThreadSynchronize(VALUE self){
  cudaError error = cudaThreadSynchronize();
  return Qtrue;
}

// __host__ ​cudaError_t cudaThreadSetLimit ( cudaLimit limit, size_t value )
// Set resource limits.
// Parameters
// limit
// - Limit to set
// value
// - Size in bytes of limit
// Returns
// cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue

static VALUE rb_cudaThreadSetLimit(VALUE self, VALUE limit, VALUE value){
  cudaError error = cudaThreadSetLimit(rb_cudaLimit_from_rbsymbol(limit), NUM2LONG(value));
  return Qtrue;
}

// __host__ ​cudaError_t cudaThreadGetLimit ( size_t* pValue, cudaLimit limit )
// Returns resource limits.
// Parameters
// pValue
// - Returned size in bytes of limit
// limit
// - Limit to query
// Returns
// cudaSuccess, cudaErrorUnsupportedLimit, cudaErrorInvalidValue

static VALUE rb_cudaThreadGetLimit(VALUE self, VALUE limit){
  size_t p_value;
  cudaError error = cudaThreadGetLimit(&p_value, rb_cudaLimit_from_rbsymbol(limit));
  return ULONG2NUM(p_value);
}

// __host__ ​cudaError_t cudaThreadGetCacheConfig ( cudaFuncCache ** pCacheConfig )
// Returns the preferred cache configuration for the current device.
// Parameters
// pCacheConfig
// - Returned cache configuration
// Returns
// cudaSuccess, cudaErrorInitializationError

static VALUE rb_cudaThreadGetCacheConfig(VALUE self, VALUE pCacheConfig){
  cudaError error = cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig);
  return Qtrue;
}

// __host__ ​cudaError_t cudaThreadSetCacheConfig ( cudaFuncCache cacheConfig )
// Sets the preferred cache configuration for the current device.
// Parameters
// cacheConfig
// - Requested cache configuration
// Returns
// cudaSuccess, cudaErrorInitializationError

static VALUE rb_cudaThreadSetCacheConfig(VALUE self, VALUE cacheConfig){
  cudaError error = cudaThreadSetCacheConfig(cudaFuncCache cacheConfig);
  return Qtrue;
}

// __host__ ​ __device__ ​cudaError_t cudaGetLastError ( void )
// Returns the last error from a runtime call.
// Returns
// cudaSuccess, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation, cudaErrorInitializationError,
// cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction,
// cudaErrorInvalidConfiguration, cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidPitchValue,
// cudaErrorInvalidSymbol, cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture,
// cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, cudaErrorInvalidMemcpyDirection,
// cudaErrorInvalidFilterSetting, cudaErrorInvalidNormSetting, cudaErrorUnknown, cudaErrorInvalidResourceHandle,
// cudaErrorInsufficientDriver, cudaErrorSetOnActiveProcess, cudaErrorStartupFailure, cudaErrorInvalidPtx,
// cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound

static VALUE rb_cudaGetLastError(VALUE self){
  cudaError error = cudaGetLastError();
  return Qtrue;
}

// _host__ ​ __device__ ​cudaError_t cudaPeekAtLastError ( void )
// Returns the last error from a runtime call.
// Returns
// cudaSuccess, cudaErrorMissingConfiguration, cudaErrorMemoryAllocation, cudaErrorInitializationError,
// cudaErrorLaunchFailure, cudaErrorLaunchTimeout, cudaErrorLaunchOutOfResources, cudaErrorInvalidDeviceFunction,
// cudaErrorInvalidConfiguration, cudaErrorInvalidDevice, cudaErrorInvalidValue, cudaErrorInvalidPitchValue,
// cudaErrorInvalidSymbol, cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture,
// cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, cudaErrorInvalidMemcpyDirection, cudaErrorInvalidFilterSetting,
// cudaErrorInvalidNormSetting, cudaErrorUnknown, cudaErrorInvalidResourceHandle, cudaErrorInsufficientDriver,
// cudaErrorSetOnActiveProcess, cudaErrorStartupFailure, cudaErrorInvalidPtx, cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound

static VALUE rb_cudaPeekAtLastError(VALUE self){
  cudaError error = cudaPeekAtLastError();
  return Qtrue;
}

// __host__ ​ __device__ ​const char* cudaGetErrorName ( cudaError_t error )
// Returns the string representation of an error code enum name.
// Parameters
// error
// - Error code to convert to string
// Returns
// char* pointer to a NULL-terminated string

static VALUE rb_cudaGetErrorName(VALUE self, VALUE error){
  const char* error = cudaGetErrorName(cudaError_t error);
  return Qnil;
}

// __host__ ​ __device__ ​const char* cudaGetErrorString ( cudaError_t error )
// Returns the description string for an error code.
// Parameters
// error
// - Error code to convert to string
// Returns
// char* pointer to a NULL-terminated string

static VALUE rb_cudaGetErrorString(VALUE self, VALUE error){
  const char* error = cudaGetErrorString(cudaError_t error);
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaGetDeviceCount ( int* count )
// Returns the number of compute-capable devices.
// Parameters
// count
// - Returns the number of devices with compute capability greater or equal to 2.0
// Returns
// cudaSuccess, cudaErrorNoDevice, cudaErrorInsufficientDriver

static VALUE rb_cudaGetDeviceCount(VALUE self){
  cudaGetDeviceCount(int* count);
  return Qnil;
}

// __host__ ​cudaError_t cudaGetDeviceProperties ( cudaDeviceProp* prop, int  device )
// Returns information about the compute-device.
// Parameters
// prop
// - Properties for the specified device
// device
// - Device number to get properties for
// Returns
// cudaSuccess, cudaErrorInvalidDevice

static VALUE rb_cudaGetDeviceProperties(VALUE self, VALUE device){
  cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaDeviceGetAttribute ( int* value, cudaDeviceAttr attr, int  device )
// Returns information about the device.
// Parameters
// value
// - Returned device attribute value
// attr
// - Device attribute to query
// device
// - Device number to query
// Returns
// cudaSuccess, cudaErrorInvalidDevice, cudaErrorInvalidValue

static VALUE rb_cudaDeviceGetAttribute(VALUE self, VALUE device){
  cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
  return Qnil;
}

// __host__ ​cudaError_t cudaChooseDevice ( int* device, const cudaDeviceProp* prop )
// Select compute-device which best matches criteria.
// Parameters
// device
// - Device with best match
// prop
// - Desired device properties
// Returns
// cudaSuccess, cudaErrorInvalidValue

static VALUE rb_cudaChooseDevice(VALUE self){
  cudaChooseDevice(int* device, const(cudaDeviceProp)* prop);
  return Qnil;
}

// __host__ ​cudaError_t cudaSetDevice ( int  device )
// Set device to be used for GPU executions.
// Parameters
// device
// - Device on which the active host thread should execute the device code.
// Returns
// cudaSuccess, cudaErrorInvalidDevice, cudaErrorDeviceAlreadyInUse

static VALUE rb_cudaSetDevice(VALUE self, VALUE device){
  cudaSetDevice(int device);
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaGetDevice ( int* device )
// Returns which device is currently being used.
// Parameters
// device
// - Returns the device on which the active host thread executes the device code.
// Returns
// cudaSuccess

static VALUE rb_cudaGetDevice(VALUE self){
  cudaGetDevice(int* device);
  return Qnil;
}

// __host__ ​cudaError_t cudaSetValidDevices ( int* device_arr, int  len )
// Set a list of devices that can be used for CUDA.
// Parameters
// device_arr
// - List of devices to try
// len
// - Number of devices in specified list
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

static VALUE rb_cudaSetValidDevices(VALUE self, VALUE len){
  cudaSetValidDevices(int* device_arr, int len);
  return Qnil;
}

// __host__ ​cudaError_t cudaGetDeviceFlags ( unsigned int* flags )
// Gets the flags for the current device.
// Parameters
// flags
// - Pointer to store the device flags
// Returns
// cudaSuccess, cudaErrorInvalidDevice

static VALUE rb_cudaSetDeviceFlags(VALUE self, VALUE flags){
  cudaSetDeviceFlags(uint flags);
  return Qnil;
}

// __host__ ​cudaError_t cudaGetDeviceFlags ( unsigned int* flags )
// Gets the flags for the current device.
// Parameters
// flags
// - Pointer to store the device flags
// Returns
// cudaSuccess, cudaErrorInvalidDevice

static VALUE rb_cudaGetDeviceFlags(VALUE self){
  cudaGetDeviceFlags(uint* flags);
  return Qnil;
}

static VALUE rb_cudaStreamCreate(VALUE self){
  cudaStreamCreate(cudaStream_t* pStream);
  return Qnil;
}

static VALUE rb_cudaStreamCreateWithFlags(VALUE self, VALUE flags){
  cudaStreamCreateWithFlags(cudaStream_t* pStream, uint flags);
  return Qnil;
}

static VALUE rb_cudaStreamCreateWithPriority(VALUE self){
  cudaStreamCreateWithPriority(cudaStream_t* pStream, uint flags, int priority);
  return Qnil;
}

static VALUE rb_cudaStreamGetPriority(VALUE self){
  cudaStreamGetPriority(cudaStream_t hStream, int* priority);
  return Qnil;
}

static VALUE rb_cudaStreamGetFlags(VALUE self, VALUE hStream){
  cudaStreamGetFlags(cudaStream_t hStream, uint* flags);
  return Qnil;
}

static VALUE rb_cudaStreamDestroy(VALUE self, VALUE stream){
  cudaStreamDestroy(cudaStream_t stream);
  return Qnil;
}

static VALUE rb_cudaStreamWaitEvent(VALUE self, VALUE stream, VALUE event, VALUE flags){
  cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, uint flags);
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