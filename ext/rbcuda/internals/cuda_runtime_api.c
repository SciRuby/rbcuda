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

static VALUE rb_cudaThreadGetCacheConfig(VALUE self){
  cudaFuncCache p_cache_config;
  cudaError error = cudaThreadGetCacheConfig(&p_cache_config);
  return rb_str_new_cstr(get_function_cache_name(p_cache_config));
}

// __host__ ​cudaError_t cudaThreadSetCacheConfig ( cudaFuncCache cacheConfig )
// Sets the preferred cache configuration for the current device.
// Parameters
// cacheConfig
// - Requested cache configuration
// Returns
// cudaSuccess, cudaErrorInitializationError

static VALUE rb_cudaThreadSetCacheConfig(VALUE self, VALUE p_cache_config){
  cudaError error = cudaThreadSetCacheConfig(rb_cu_function_cache_from_rbsymbol(p_cache_config));
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
  const char* error_string = cudaGetErrorName((cudaError)rb_cuda_cu_result_from_rbsymbol(error));
  return rb_str_new_cstr(error_string);
}

// __host__ ​ __device__ ​const char* cudaGetErrorString ( cudaError_t error )
// Returns the description string for an error code.
// Parameters
// error
// - Error code to convert to string
// Returns
// char* pointer to a NULL-terminated string

static VALUE rb_cudaGetErrorString(VALUE self, VALUE error){
  const char* error_string = cudaGetErrorString((cudaError)rb_cuda_cu_result_from_rbsymbol(error));
  return rb_str_new_cstr(error_string);
}

// __host__ ​ __device__ ​cudaError_t cudaGetDeviceCount ( int* count )
// Returns the number of compute-capable devices.
// Parameters
// count
// - Returns the number of devices with compute capability greater or equal to 2.0
// Returns
// cudaSuccess, cudaErrorNoDevice, cudaErrorInsufficientDriver

static VALUE rb_cudaGetDeviceCount(VALUE self){
  int count;
  cudaError error = cudaGetDeviceCount(&count);
  return INT2NUM(count);
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
  ///////////////////////////////////////////////////////
  //                                                   //
  //                       TODO                        //
  //                                                   //
  ///////////////////////////////////////////////////////
  // cudaError error = cudaGetDeviceProperties(cudaDeviceProp* prop, INT2NUM(device));
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

static VALUE rb_cudaDeviceGetAttribute(VALUE self, VALUE attr, VALUE device){
  int value;
  cudaError error = cudaDeviceGetAttribute(&value, rb_cudaDeviceAttr_from_rbsymbol(attr), NUM2INT(device));
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
  int device;
   ///////////////////////////////////////////////////////
  //                                                   //
  //                       TODO                        //
  //                                                   //
  ///////////////////////////////////////////////////////
  // cudaError error = cudaChooseDevice(&device, const(cudaDeviceProp)* prop);
  return INT2NUM(device);
}

// __host__ ​cudaError_t cudaSetDevice ( int  device )
// Set device to be used for GPU executions.
// Parameters
// device
// - Device on which the active host thread should execute the device code.
// Returns
// cudaSuccess, cudaErrorInvalidDevice, cudaErrorDeviceAlreadyInUse

static VALUE rb_cudaSetDevice(VALUE self, VALUE device){
  cudaError error = cudaSetDevice(NUM2INT(device));
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
  int device;
  cudaError error = cudaGetDevice(&device);
  return INT2NUM(device);
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
  int device_arr;
  cudaError error = cudaSetValidDevices(&device_arr, NUM2INT(len));
  return INT2NUM(device_arr);
}

// __host__ ​cudaError_t cudaGetDeviceFlags ( unsigned int* flags )
// Gets the flags for the current device.
// Parameters
// flags
// - Pointer to store the device flags
// Returns
// cudaSuccess, cudaErrorInvalidDevice

static VALUE rb_cudaSetDeviceFlags(VALUE self, VALUE flags){
  cudaError error = cudaSetDeviceFlags(NUM2UINT(flags));
  return Qtrue;
}

// __host__ ​cudaError_t cudaGetDeviceFlags ( unsigned int* flags )
// Gets the flags for the current device.
// Parameters
// flags
// - Pointer to store the device flags
// Returns
// cudaSuccess, cudaErrorInvalidDevice

static VALUE rb_cudaGetDeviceFlags(VALUE self){
  uint flags;
  cudaError error = cudaGetDeviceFlags(&flags);
  return UINT2NUM(flags);
}

// __host__ ​cudaError_t cudaStreamCreate ( cudaStream_t* pStream )
// Create an asynchronous stream.
// Parameters
// pStream
// - Pointer to new stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidValue

static VALUE rb_cudaStreamCreate(VALUE self){
  custream_ptr* p_stream = ALLOC(custream_ptr);
  cudaError error = cudaStreamCreate(&p_stream->stream);
  return Data_Wrap_Struct(RbCuStream, NULL, rbcu_free, p_stream);
}

// __host__ ​ __device__ ​cudaError_t cudaStreamCreateWithFlags ( cudaStream_t* pStream, unsigned int  flags )
// Create an asynchronous stream.
// Parameters
// pStream
// - Pointer to new stream identifier
// flags
// - Parameters for stream creation
// Returns
// cudaSuccess, cudaErrorInvalidValue

static VALUE rb_cudaStreamCreateWithFlags(VALUE self, VALUE flags){
  custream_ptr* p_stream = ALLOC(custream_ptr);
  cudaError error = cudaStreamCreateWithFlags(&p_stream->stream, NUM2UINT(flags));
  return Data_Wrap_Struct(RbCuStream, NULL, rbcu_free, p_stream);
}

// __host__ ​cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority )
// Create an asynchronous stream with the specified priority.
// Parameters
// pStream
// - Pointer to new stream identifier
// flags
// - Flags for stream creation. See cudaStreamCreateWithFlags for a list of valid flags that can be passed
// priority
// - Priority of the stream. Lower numbers represent higher priorities. See cudaDeviceGetStreamPriorityRange for more information about the meaningful stream priorities that can be passed.
// Returns
// cudaSuccess, cudaErrorInvalidValue

static VALUE rb_cudaStreamCreateWithPriority(VALUE self, VALUE flags, VALUE priority){
  custream_ptr* p_stream = ALLOC(custream_ptr);
  cudaError error = cudaStreamCreateWithPriority(&p_stream->stream, NUM2UINT(flags), NUM2INT(priority));
  return Data_Wrap_Struct(RbCuStream, NULL, rbcu_free, p_stream);
}

// __host__ ​cudaError_t cudaStreamGetPriority ( cudaStream_t hStream, int* priority )
// Query the priority of a stream.
// Parameters
// hStream
// - Handle to the stream to be queried
// priority
// - Pointer to a signed integer in which the stream's priority is returned
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

static VALUE rb_cudaStreamGetPriority(VALUE self, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  int priority;
  cudaError error = cudaStreamGetPriority(h_stream->stream, &priority);
  return INT2NUM(priority);
}

// __host__ ​cudaError_t cudaStreamGetFlags ( cudaStream_t hStream, unsigned int* flags )
// Query the flags of a stream.
// Parameters
// hStream
// - Handle to the stream to be queried
// flags
// - Pointer to an unsigned integer in which the stream's flags are returned
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

static VALUE rb_cudaStreamGetFlags(VALUE self, VALUE h_stream_val){
  uint flags;
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  cudaError error = cudaStreamGetFlags(h_stream->stream, &flags);
  return UINT2NUM(flags);
}

// __host__ ​ __device__ ​cudaError_t cudaStreamDestroy ( cudaStream_t stream )
// Destroys and cleans up an asynchronous stream.
// Parameters
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

static VALUE rb_cudaStreamDestroy(VALUE self, VALUE stream_val){
  custream_ptr* stream;
  Data_Get_Struct(stream_val, custream_ptr, stream);
  cudaError error = cudaStreamDestroy(stream->stream);
  return Qtrue;
}

// __host__ ​ __device__ ​cudaError_t cudaStreamWaitEvent ( cudaStream_t stream, cudaEvent_t event, unsigned int  flags )
// Make a compute stream wait on an event.
// Parameters
// stream
// - Stream to wait
// event
// - Event to wait on
// flags
// - Parameters for the operation (must be 0)
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

static VALUE rb_cudaStreamWaitEvent(VALUE self, VALUE stream_val, VALUE event, VALUE flags){
  custream_ptr* stream;
  Data_Get_Struct(stream_val, custream_ptr, stream);
  cu_event* event_ptr;
  Data_Get_Struct(event, cu_event, event_ptr);
  cudaError error = cudaStreamWaitEvent(stream->stream, event_ptr->event, NUM2UINT(flags));
  return Qnil;
}

// __host__ ​cudaError_t cudaStreamAddCallback ( cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int  flags )
// Add a callback to a compute stream.
// Parameters
// stream
// - Stream to add callback to
// callback
// - The function to call once preceding stream operations are complete
// userData
// - User specified data to be passed to the callback function
// flags
// - Reserved for future use, must be 0
// Returns
// cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorNotSupported

static VALUE rb_cudaStreamAddCallback(VALUE self, VALUE stream_val, VALUE call_back, VALUE user_data, VALUE flags){
  custream_ptr* stream;
  Data_Get_Struct(stream_val, custream_ptr, stream);
  ////////////////////////////////////
  //                                //
  //              todo              //
  //                                //
  ////////////////////////////////////
  // cudaError error = cudaStreamAddCallback( stream->stream, cudaStreamCallback_t callback, (void*)user_data, NUM2UINT(flags));
  return Qnil;
}

// __host__ ​cudaError_t cudaStreamSynchronize ( cudaStream_t stream )
// Waits for stream tasks to complete.
// Parameters
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidResourceHandle

static VALUE rb_cudaStreamSynchronize(VALUE self, VALUE stream_val){
  custream_ptr* stream;
  Data_Get_Struct(stream_val, custream_ptr, stream);
  cudaError error = cudaStreamSynchronize(stream->stream);
  return Qtrue;
}

// __host__ ​cudaError_t cudaStreamQuery ( cudaStream_t stream )
// Queries an asynchronous stream for completion status.
// Parameters
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorNotReady, cudaErrorInvalidResourceHandle

static VALUE rb_cudaStreamQuery(VALUE self, VALUE stream_val){
  custream_ptr* stream;
  Data_Get_Struct(stream_val, custream_ptr, stream);
  cudaError error = cudaStreamQuery(stream->stream);
  return Qtrue;
}

// __host__ ​cudaError_t cudaStreamAttachMemAsync ( cudaStream_t stream, void* devPtr, size_t length = 0, unsigned int  flags = cudaMemAttachSingle )
// Attach memory to a stream asynchronously.
// Parameters
// stream
// - Stream in which to enqueue the attach operation
// devPtr
// - Pointer to memory (must be a pointer to managed memory)
// length
// - Length of memory (must be zero, defaults to zero)
// flags
// - Must be one of cudaMemAttachGlobal, cudaMemAttachHost or cudaMemAttachSingle (defaults to cudaMemAttachSingle)
// Returns
// cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

static VALUE rb_cudaStreamAttachMemAsync(VALUE self, VALUE stream_val){
  custream_ptr* stream;
  Data_Get_Struct(stream_val, custream_ptr, stream);
  return Qtrue;
}

// __host__ ​cudaError_t cudaEventCreate ( cudaEvent_t* event )
// Creates an event object.
// Parameters
// event
// - Newly created event
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorLaunchFailure, cudaErrorMemoryAllocation

static VALUE rb_cudaEventCreate(VALUE self){
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaEventCreateWithFlags ( cudaEvent_t* event, unsigned int  flags )
// Creates an event object with the specified flags.
// Parameters
// event
// - Newly created event
// flags
// - Flags for new event
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorLaunchFailure, cudaErrorMemoryAllocation

static VALUE rb_cudaEventCreateWithFlags(VALUE self){
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaEventRecord ( cudaEvent_t event, cudaStream_t stream = 0 )
// Records an event.
// Parameters
// event
// - Event to record
// stream
// - Stream in which to record event
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

static VALUE rb_cudaEventRecord(VALUE self){
  return Qnil;
}

// __host__ ​cudaError_t cudaEventQuery ( cudaEvent_t event )
// Queries an event's status.
// Parameters
// event
// - Event to query
// Returns
// cudaSuccess, cudaErrorNotReady, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

static VALUE rb_cudaEventQuery(VALUE self){
  return Qnil;
}

// __host__ ​cudaError_t cudaEventSynchronize ( cudaEvent_t event )
// Waits for an event to complete.
// Parameters
// event
// - Event to wait for
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

static VALUE rb_cudaEventSynchronize(VALUE self){
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaEventDestroy ( cudaEvent_t event )
// Destroys an event object.
// Parameters
// event
// - Event to destroy
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorLaunchFailure

static VALUE rb_cudaEventDestroy(VALUE self){
  return Qnil;
}

// __host__ ​cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start, cudaEvent_t end )
// Computes the elapsed time between events.
// Parameters
// ms
// - Time between start and end in ms
// start
// - Starting event
// end
// - Ending event
// Returns
// cudaSuccess, cudaErrorNotReady, cudaErrorInvalidValue, cudaErrorInitializationError, cudaErrorInvalidResourceHandle, cudaErrorLaunchFailure

static VALUE rb_cudaEventElapsedTime(VALUE self){
  return Qnil;
}

// __host__ ​cudaError_t cudaLaunchKernel ( const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream )
// Launches a device function.
// Parameters
// func
// - Device function symbol
// gridDim
// - Grid dimentions
// blockDim
// - Block dimentions
// args
// - Arguments
// sharedMem
// - Shared memory
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
// cudaErrorLaunchOutOfResources, cudaErrorSharedObjectInitFailed, cudaErrorInvalidPtx, cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound

static VALUE rb_cudaLaunchKernel(VALUE self){
  return Qnil;
}

// __host__ ​cudaError_t cudaFuncSetCacheConfig ( const void* func, cudaFuncCache cacheConfig )
// Sets the preferred cache configuration for a device function.
// Parameters
// func
// - Device function symbol
// cacheConfig
// - Requested cache configuration
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidDeviceFunction

static VALUE rb_cudaFuncSetCacheConfig(VALUE self){
  return Qnil;
}

// __host__ ​cudaError_t cudaFuncSetSharedMemConfig ( const void* func, cudaSharedMemConfig config )
// Sets the shared memory configuration for a device function.
// Parameters
// func
// - Device function symbol
// config
// - Requested shared memory configuration
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue,

static VALUE rb_cudaFuncSetSharedMemConfig(VALUE self, VALUE func, VALUE config){
  cudaError error = cudaFuncSetSharedMemConfig((void*)func, rb_cu_shared_mem_from_rbsymbol(config));
  return Qtrue;
}

// __host__ ​ __device__ ​cudaError_t cudaFuncGetAttributes ( cudaFuncAttributes* attr, const void* func )
// Find out attributes for a given function.
// Parameters
// attr
// - Return pointer to function's attributes
// func
// - Device function symbol
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidDeviceFunction

static VALUE rb_cudaFuncGetAttributes(VALUE self){
  //////////////////////////////////
  //             TODO             //
  //////////////////////////////////
  // cudaError error = cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func);
  return Qnil;
}

// __host__ ​cudaError_t cudaSetDoubleForDevice ( double* d )
// Converts a double argument to be executed on a device.
// Parameters
// d
// - Double to convert
// Returns
// cudaSuccess

static VALUE rb_cudaSetDoubleForDevice(VALUE self){
  double d;
  cudaError error = cudaSetDoubleForDevice(&d);
  return DBL2NUM(d);
}

// __host__ ​cudaError_t cudaSetDoubleForHost ( double* d )
// Converts a double argument after execution on a device.
// Parameters
// d
// - Double to convert
// Returns
// cudaSuccess

static VALUE rb_cudaSetDoubleForHost(VALUE self){
  double d;
  cudaError error = cudaSetDoubleForHost(&d);
  return DBL2NUM(d);
}

// __host__ ​ __device__ ​cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize )
// Returns occupancy for a device function.
// Parameters
// numBlocks
// - Returned occupancy
// func
// - Kernel function for which occupancy is calculated
// blockSize
// - Block size the kernel is intended to be launched with
// dynamicSMemSize
// - Per-block dynamic shared memory usage intended, in bytes
// Returns
// cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue, cudaErrorUnknown,

static VALUE rb_cudaOccupancyMaxActiveBlocksPerMultiprocessor(VALUE self, VALUE func, VALUE block_size, VALUE dynamic_smem_size){
  int num_blocks;
  cudaError error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, (void*)func, NUM2UINT(block_size),  NUM2ULONG(dynamic_smem_size));
  return INT2NUM(num_blocks);
}

// __host__ ​cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, const void* func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags )
// Returns occupancy for a device function with the specified flags.
// Parameters
// numBlocks
// - Returned occupancy
// func
// - Kernel function for which occupancy is calculated
// blockSize
// - Block size the kernel is intended to be launched with
// dynamicSMemSize
// - Per-block dynamic shared memory usage intended, in bytes
// flags
// - Requested behavior for the occupancy calculator
// Returns
// cudaSuccess, cudaErrorCudartUnloading, cudaErrorInitializationError, cudaErrorInvalidDevice, cudaErrorInvalidDeviceFunction, cudaErrorInvalidValue, cudaErrorUnknown,

static VALUE rb_cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(VALUE self, VALUE func, VALUE block_size, VALUE dynamic_smem_size, VALUE flags){
  int num_blocks;
  cudaError error = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&num_blocks, (void*)func, NUM2INT(block_size), NUM2ULONG(dynamic_smem_size), NUM2UINT(flags));
  return Qnil;
}

// __host__ ​cudaError_t cudaConfigureCall ( dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, cudaStream_t stream = 0 )
// Configure a device-launch.
// Parameters
// gridDim
// - Grid dimensions
// blockDim
// - Block dimensions
// sharedMem
// - Shared memory
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidConfiguration

static VALUE rb_cudaConfigureCall(VALUE self){
  // cudaError error = cudaConfigureCall(dim3 grid_dim, dim3 bloc_dim, NUM2ULONG(shared_mem), cudaStream_t stream);
  return Qnil;
}

// __host__ ​cudaError_t cudaSetupArgument ( const void* arg, size_t size, size_t offset )
// Configure a device launch.
// Parameters
// arg
// - Argument to push for a kernel launch
// size
// - Size of argument
// offset
// - Offset in argument stack to push new arg
// Returns
// cudaSuccess

static VALUE rb_cudaSetupArgument(VALUE self){
  // ​cudaError error = cudaSetupArgument ( const void* arg, size_t size, size_t offset );
  return Qnil;
}

// __host__ ​cudaError_t cudaLaunch ( const void* func )
// Launches a device function.
// Parameters
// func
// - Device function symbol
// Returns
// cudaSuccess, cudaErrorInvalidDeviceFunction, cudaErrorInvalidConfiguration, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
// cudaErrorLaunchOutOfResources, cudaErrorSharedObjectInitFailed, cudaErrorInvalidPtx, cudaErrorNoKernelImageForDevice, cudaErrorJitCompilerNotFound

static VALUE rb_cudaLaunch(VALUE self, VALUE func){
  cudaError error = cudaLaunch((void*)func);
  return Qtrue;
}

// __host__ ​cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal )
// Allocates memory that will be automatically managed by the Unified Memory system.
// Parameters
// devPtr
// - Pointer to allocated device memory
// size
// - Requested allocation size in bytes
// flags
// - Must be either cudaMemAttachGlobal or cudaMemAttachHost (defaults to cudaMemAttachGlobal)

static VALUE rb_cudaMallocManaged(VALUE self){
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaMalloc ( void** devPtr, size_t size )
// Allocate memory on the device.
// Parameters
// devPtr
// - Pointer to allocated device memory
// size
// - Requested allocation size in bytes
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

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

// __host__ ​cudaError_t cudaMallocHost ( void** ptr, size_t size )
// Allocates page-locked memory on the host.
// Parameters
// ptr
// - Pointer to allocated host memory
// size
// - Requested allocation size in bytes
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

static VALUE rb_cudaMallocHost(VALUE self){
  // ​cudaError error = cudaMallocHost ( void** ptr, size_t size );
  return Qnil;
}

// __host__ ​cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal )
// Allocates memory that will be automatically managed by the Unified Memory system.
// Parameters
// devPtr
// - Pointer to allocated device memory
// size
// - Requested allocation size in bytes
// flags
// - Must be either cudaMemAttachGlobal or cudaMemAttachHost (defaults to cudaMemAttachGlobal)
// Returns
// cudaSuccess, cudaErrorMemoryAllocation, cudaErrorNotSupported, cudaErrorInvalidValue

static VALUE rb_cudaMallocPitch(VALUE self){
  // ​cudaError error = cudaMallocManaged( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal);
  return Qnil;
}

// __host__ ​cudaError_t cudaMallocArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height = 0, unsigned int  flags = 0 )
// Allocate an array on the device.
// Parameters
// array
// - Pointer to allocated array in device memory
// desc
// - Requested channel format
// width
// - Requested array allocation width
// height
// - Requested array allocation height
// flags
// - Requested properties of allocated array
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

static VALUE rb_cudaMallocArray(VALUE self){
  // ​cudaError error = cudaMallocArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height = 0, unsigned int  flags = 0 );
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaFree ( void* devPtr )
// Frees memory on the device.
// Parameters
// devPtr
// - Device pointer to memory to free
// Returns
// cudaSuccess, cudaErrorInvalidDevicePointer, cudaErrorInitializationError

static VALUE rb_cudaFree(VALUE self, VALUE ptr_val){
  dev_ptr* ptr;
  Data_Get_Struct(ptr_val, dev_ptr, ptr);
  // ​cudaError error = cudaFree(ptr->carray);
  return Qnil;
}

// __host__ ​cudaError_t cudaFreeHost ( void* ptr )
// Frees page-locked memory.
// Parameters
// ptr
// - Pointer to memory to free
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError

static VALUE rb_cudaFreeHost(VALUE self){
  // ​cudaError error = cudaFreeHost((void*)ptr);
  return Qnil;
}

// __host__ ​cudaError_t cudaFreeArray ( cudaArray_t array )
// Frees an array on the device.
// Parameters
// array
// - Pointer to array to free
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError

static VALUE rb_cudaFreeArray(VALUE self){
  // ​cudaError error = cudaFreeArray ( cudaArray_t array );
  return Qnil;
}

// __host__ ​cudaError_t cudaFreeMipmappedArray ( cudaMipmappedArray_t mipmappedArray )
// Frees a mipmapped array on the device.
// Parameters
// mipmappedArray
// - Pointer to mipmapped array to free
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInitializationError

static VALUE rb_cudaFreeMipmappedArray(VALUE self){
  // cudaError error = cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
  return Qnil;
}

// __host__ ​cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags )
// Allocates page-locked memory on the host.
// Parameters
// pHost
// - Device pointer to allocated memory
// size
// - Requested allocation size in bytes
// flags
// - Requested properties of allocated memory
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

static VALUE rb_cudaHostAlloc(VALUE self){
  // ​cudaError error = cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
  return Qnil;
}

// __host__ ​cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags )
// Registers an existing host memory range for use by CUDA.
// Parameters
// ptr
// - Host pointer to memory to page-lock
// size
// - Size in bytes of the address range to page-lock in bytes
// flags
// - Flags for allocation request
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation, cudaErrorHostMemoryAlreadyRegistered, cudaErrorNotSupported

static VALUE rb_cudaHostRegister(VALUE self){
  // ​cudaError error = cudaHostRegister( void* ptr, size_t size, unsigned int  flags);
  return Qnil;
}

// __host__ ​cudaError_t cudaHostUnregister ( void* ptr )
// Unregisters a memory range that was registered with cudaHostRegister.
// Parameters
// ptr
// - Host pointer to memory to unregister
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorHostMemoryNotRegistered

static VALUE rb_cudaHostUnregister(VALUE self){
  // ​cudaError error = cudaHostUnregister ( void* ptr );
  return Qnil;
}

// __host__ ​cudaError_t cudaHostGetDevicePointer ( void** pDevice, void* pHost, unsigned int  flags )
// Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
// Parameters
// pDevice
// - Returned device pointer for mapped memory
// pHost
// - Requested host pointer mapping
// flags
// - Flags for extensions (must be 0 for now)
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

static VALUE rb_cudaHostGetDevicePointer(VALUE self){
  // ​cudaError error = cudaHostGetDevicePointer ( void** pDevice, void* pHost, unsigned int  flags );
  return Qnil;
}

// __host__ ​cudaError_t cudaHostGetFlags ( unsigned int* pFlags, void* pHost )
// Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.
// Parameters
// pFlags
// - Returned flags word
// pHost
// - Host pointer
// Returns
// cudaSuccess, cudaErrorInvalidValue

static VALUE rb_cudaHostGetFlags(VALUE self){
  // ​cudaError_t error = cudaHostGetFlags ( unsigned int* pFlags, void* pHost )
  return Qnil;
}

// __host__ ​cudaError_t cudaMalloc3D ( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent )
// Allocates logical 1D, 2D, or 3D memory objects on the device.
// Parameters
// pitchedDevPtr
// - Pointer to allocated pitched device memory
// extent
// - Requested allocation size (width field in bytes)
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

static VALUE rb_cudaMalloc3D(VALUE self){
// / ​cudaError error = cudaMalloc3D ( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent )
  return Qnil;
}

// __host__ ​cudaError_t cudaMalloc3DArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  flags = 0 )
// Allocate an array on the device.
// Parameters
// array
// - Pointer to allocated array in device memory
// desc
// - Requested channel format
// extent
// - Requested allocation size (width field in elements)
// flags
// - Flags for extensions
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation

static VALUE rb_cudaMalloc3DArray(VALUE self){
  // ​cudaError error = cudaMalloc3DArray( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  flags = 0 )
  return Qnil;
}

// __host__ ​cudaError_t cudaMallocMipmappedArray ( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags = 0 )
// Allocate a mipmapped array on the device.
// Parameters
// mipmappedArray
// - Pointer to allocated mipmapped array in device memory
// desc
// - Requested channel format
// extent
// - Requested allocation size (width field in elements)
// numLevels
// - Number of mipmap levels to allocate
// flags
// - Flags for extensions
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation


static VALUE rb_cudaMallocMipmappedArray(VALUE self){
  // ​cudaError error = cudaMallocMipmappedArray ( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags = 0 )
  return Qnil;
}

// __host__ ​cudaError_t cudaGetMipmappedArrayLevel ( cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level )
// Gets a mipmap level of a CUDA mipmapped array.
// Parameters
// levelArray
// - Returned mipmap level CUDA array
// mipmappedArray
// - CUDA mipmapped array
// level
// - Mipmap level
// Returns
// cudaSuccess, cudaErrorInvalidValue

static VALUE rb_cudaGetMipmappedArrayLevel(VALUE self){
  // ​cudaError_t error = cudaGetMipmappedArrayLevel ( cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy3D ( const cudaMemcpy3DParms* p )
// Copies data between 3D objects.
// Parameters
// p
// - 3D memory copy parameters
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpy3D(VALUE self){
  // ​cudaError error = cudaMemcpy3D ( const cudaMemcpy3DParms* p );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p )
// Copies memory between devices.
// Parameters
// p
// - Parameters for the memory copy
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

static VALUE rb_cudaMemcpy3DPeer(VALUE self){
  // ​cudaError error = cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p );
  return Qnil;
}

// __host__ ​ __device__ ​cudaError_t cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p, cudaStream_t stream = 0 )
// Copies data between 3D objects.
// Parameters
// p
// - 3D memory copy parameters
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpy3DAsync(VALUE self){
  // ​cudaError error = cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p, cudaStream_t stream = 0 );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p, cudaStream_t stream = 0 )
// Copies memory between devices asynchronously.
// Parameters
// p
// - Parameters for the memory copy
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

static VALUE rb_cudaMemcpy3DPeerAsync(VALUE self){
  // ​cudaError error = cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p, cudaStream_t stream = 0 );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemGetInfo ( size_t* free, size_t* total )
// Gets free and total device memory.
// Parameters
// free
// - Returned free memory in bytes
// total
// - Returned total memory in bytes
// Returns
// cudaSuccess, cudaErrorInitializationError, cudaErrorInvalidValue, cudaErrorLaunchFailure

static VALUE rb_cudaMemGetInfo(VALUE self){
  // ​cudaError error = cudaMemGetInfo ( size_t* free, size_t* total );
  return Qnil;
}

// __host__ ​cudaError_t cudaArrayGetInfo ( cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array )
// Gets info about the specified cudaArray.
// Parameters
// desc
// - Returned array type
// extent
// - Returned array shape. 2D arrays will have depth of zero
// flags
// - Returned array flags
// array
// - The cudaArray to get info for
// Returns
// cudaSuccess, cudaErrorInvalidValue

static VALUE rb_cudaArrayGetInfo(VALUE self){
  // ​cudaError error = cudaArrayGetInfo ( cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// src
// - Source memory address
// count
// - Size in bytes to copy
// kind
// - Type of transfer
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

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

// __host__ ​cudaError_t cudaMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count )
// Copies memory between two devices.
// Parameters
// dst
// - Destination device pointer
// dstDevice
// - Destination device
// src
// - Source device pointer
// srcDevice
// - Source device
// count
// - Size of memory copy in bytes
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDevice

static VALUE rb_cudaMemcpyPeer(VALUE self){
  // ​cudaError error = cudaMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpyToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// wOffset
// - Destination starting X offset
// hOffset
// - Destination starting Y offset
// src
// - Source memory address
// count
// - Size in bytes to copy
// kind
// - Type of transfer
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpyToArray(VALUE self){
  // ​cudaError error = cudaMemcpyToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t count, cudaMemcpyKind kind );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpyFromArray ( void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// src
// - Source memory address
// wOffset
// - Source starting X offset
// hOffset
// - Source starting Y offset
// count
// - Size in bytes to copy
// kind
// - Type of transfer
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpyFromArray(VALUE self){
  // ​cudaError error = cudaMemcpyFromArray ( void* dst, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpyArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// wOffsetDst
// - Destination starting X offset
// hOffsetDst
// - Destination starting Y offset
// src
// - Source memory address
// wOffsetSrc
// - Source starting X offset
// hOffsetSrc
// - Source starting Y offset
// count
// - Size in bytes to copy
// kind
// - Type of transfer
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpyArrayToArray(VALUE self){
  // ​cudaError error = cudaMemcpyArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// dpitch
// - Pitch of destination memory
// src
// - Source memory address
// spitch
// - Pitch of source memory
// width
// - Width of matrix transfer (columns in bytes)
// height
// - Height of matrix transfer (rows)
// kind
// - Type of transfer
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpy2D(VALUE self){
  // ​cudaError_t error = cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// wOffsetDst
// - Destination starting X offset
// hOffsetDst
// - Destination starting Y offset
// src
// - Source memory address
// wOffsetSrc
// - Source starting X offset
// hOffsetSrc
// - Source starting Y offset
// width
// - Width of matrix transfer (columns in bytes)
// height
// - Height of matrix transfer (rows)
// kind
// - Type of transfer
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpy2DToArray(VALUE self){
  // ​cudaError_t error = cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy2DFromArray ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// dpitch
// - Pitch of destination memory
// src
// - Source memory address
// wOffset
// - Source starting X offset
// hOffset
// - Source starting Y offset
// width
// - Width of matrix transfer (columns in bytes)
// height
// - Height of matrix transfer (rows)
// kind
// - Type of transfer
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

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

// __host__ ​cudaError_t cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// wOffset
// - Destination starting X offset
// hOffset
// - Destination starting Y offset
// src
// - Source memory address
// spitch
// - Pitch of source memory
// width
// - Width of matrix transfer (columns in bytes)
// height
// - Height of matrix transfer (rows)
// kind
// - Type of transfer
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpy2DToArrayAsync(VALUE self){
  // ​cudaError error = cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 );
  return Qnil;
}

// __host__ ​cudaError_t cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
// Copies data between host and device.
// Parameters
// dst
// - Destination memory address
// dpitch
// - Pitch of destination memory
// src
// - Source memory address
// wOffset
// - Source starting X offset
// hOffset
// - Source starting Y offset
// width
// - Width of matrix transfer (columns in bytes)
// height
// - Height of matrix transfer (rows)
// kind
// - Type of transfer
// stream
// - Stream identifier
// Returns
// cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidPitchValue, cudaErrorInvalidMemcpyDirection

static VALUE rb_cudaMemcpy2DFromArrayAsync(VALUE self){
  // ​cudaError error = cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 );
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