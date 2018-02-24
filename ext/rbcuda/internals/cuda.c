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
  device_ptr* pdevice = ALLOC(device_ptr);
  CUresult result = cuDeviceGet(&pdevice->device, NUM2INT(ordinal));
  return Data_Wrap_Struct(RbCuDevice, NULL, rbcu_free, pdevice);
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
  char* name = (char *)malloc(NUM2ULONG(len_val) * sizeof(char));
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDeviceGetName(name, NUM2ULONG(len_val), device->device);
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
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDeviceTotalMem_v2(&bytes, device->device);
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
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUdevice_attribute attrib = rb_cu_get_attrib_from_rbsymbol(attrib_val);
  CUresult result = cuDeviceGetAttribute(&pi, attrib, device->device);
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
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUdevprop* prop;
  CUresult result = cuDeviceGetProperties(prop, device->device);
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
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDeviceComputeCapability(&major, &minor, device->device);
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
  ctx_ptr* pctx = ALLOC(ctx_ptr);
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDevicePrimaryCtxRetain(&pctx->ctx, device->device);
  return Data_Wrap_Struct(RbCuContext, NULL, rbcu_free, pctx);
}

// CUresult cuDevicePrimaryCtxRelease ( CUdevice dev )
// Release the primary context on the GPU.
// Parameters
// dev
// - Device which primary context is released
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDevicePrimaryCtxRelease(VALUE self, VALUE device_val){
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDevicePrimaryCtxRelease(device->device);
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
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDevicePrimaryCtxSetFlags(device->device, NUM2UINT(flags));
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
  uint flags;
  int active;
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDevicePrimaryCtxGetState(device->device, &flags, &active);
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
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  CUresult result = cuDevicePrimaryCtxReset(device->device);
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

static VALUE rb_cuCtxCreate_v2(VALUE self, VALUE flags, VALUE device_val){
  device_ptr* device;
  Data_Get_Struct(device_val, device_ptr, device);
  ctx_ptr* pctx = ALLOC(ctx_ptr);
  CUresult result = cuCtxCreate_v2 (&pctx->ctx, UINT2NUM(flags), device->device);
  return Data_Wrap_Struct(RbCuContext, NULL, rbcu_free, pctx);
}

// CUresult cuCtxDestroy ( CUcontext ctx )
// Destroy a CUDA context.
// Parameters
// ctx
// - Context to destroy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxDestroy_v2(VALUE self, VALUE ctx_val){
  ctx_ptr* pctx;
  Data_Get_Struct(ctx_val, ctx_ptr, pctx);
  CUresult result = cuCtxDestroy_v2(pctx->ctx);
  return Qnil;
}

// CUresult cuCtxPushCurrent ( CUcontext ctx )
// Pushes a context on the current CPU thread.
// Parameters
// ctx
// - Context to push

static VALUE rb_cuCtxPushCurrent_v2(VALUE self, VALUE ctx_val){
  ctx_ptr* pctx;
  Data_Get_Struct(ctx_val, ctx_ptr, pctx);
  CUresult result = cuCtxPushCurrent_v2(pctx->ctx);
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
  ctx_ptr* pctx = ALLOC(ctx_ptr);
  CUresult result = cuCtxPopCurrent_v2(&pctx->ctx);
  return Data_Wrap_Struct(RbCuContext, NULL, rbcu_free, pctx);
}

// CUresult cuCtxSetCurrent ( CUcontext ctx )
// Binds the specified CUDA context to the calling CPU thread.
// Parameters
// ctx
// - Context to bind to the calling CPU thread
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

static VALUE rb_cuCtxSetCurrent(VALUE self, VALUE ctx_val){
  ctx_ptr* pctx;
  Data_Get_Struct(ctx_val, ctx_ptr, pctx);
  CUresult result = cuCtxSetCurrent(pctx->ctx);
  return Qnil;
}

// CUresult cuCtxGetCurrent ( CUcontext* pctx )
// Returns the CUDA context bound to the calling CPU thread.
// Parameters
// pctx
// - Returned context handle
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,

static VALUE rb_cuCtxGetCurrent(VALUE self){
  ctx_ptr* pctx = ALLOC(ctx_ptr);
  CUresult result = cuCtxGetCurrent(&pctx->ctx);
  return Data_Wrap_Struct(RbCuContext, NULL, rbcu_free, pctx);
}

// CUresult cuCtxGetDevice ( CUdevice* device )
// Returns the device ID for the current context.
// Parameters
// device
// - Returned device ID for the current context
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,

static VALUE rb_cuCtxGetDevice(VALUE self){
  device_ptr* device = ALLOC(device_ptr);
  CUresult result = cuCtxGetDevice(&device->device);
  return Data_Wrap_Struct(RbCuDevice, NULL, rbcu_free, device);
}

// CUresult cuCtxGetFlags ( unsigned int* flags )
// Returns the flags for the current context.
// Parameters
// flags
// - Pointer to store flags of current context
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,

static VALUE rb_cuCtxGetFlags(VALUE self){
  uint flags;
  CUresult result = cuCtxGetFlags(&flags);
  return UINT2NUM(flags);
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

static VALUE rb_cuCtxSetLimit(VALUE self, VALUE limit_val, VALUE limit_size){
  CUlimit limit = rb_cu_limit_from_rbsymbol(limit_val);
  CUresult result = cuCtxSetLimit(limit, NUM2ULONG(limit_size));
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

static VALUE rb_cuCtxGetLimit(VALUE self, VALUE limit_val){
  size_t pvalue;
  CUlimit limit = rb_cu_limit_from_rbsymbol(limit_val);
  CUresult result = cuCtxGetLimit(&pvalue, limit);
  return ULONG2NUM(pvalue);
}

// CUresult cuCtxGetCacheConfig ( CUfunc_cache* pconfig )
// Returns the preferred cache configuration for the current context.
// Parameters
// pconfig
// - Returned cache configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxGetCacheConfig(VALUE self){
  CUfunc_cache pconfig;
  CUresult result = cuCtxGetCacheConfig(&pconfig);
  const char* cache_name = get_func_cache_name(pconfig);
  return rb_str_new_cstr(cache_name);
}

// CUresult cuCtxSetCacheConfig ( CUfunc_cache config )
// Sets the preferred cache configuration for the current context.
// Parameters
// config
// - Requested cache configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxSetCacheConfig(VALUE self, VALUE config_val){
  CUfunc_cache config = rb_cu_func_cache_from_rbsymbol(config_val);
  CUresult result = cuCtxSetCacheConfig(config);
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
  CUsharedconfig pConfig;
  CUresult result = cuCtxGetSharedMemConfig(&pConfig);
  const char* shared_config_name = get_shared_config_name(pConfig);
  return rb_str_new_cstr(shared_config_name);
}

// CUresult cuCtxSetSharedMemConfig ( CUsharedconfig config )
// Sets the shared memory configuration for the current context.
// Parameters
// config
// - requested shared memory configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxSetSharedMemConfig(VALUE self, VALUE config_val){
  CUsharedconfig config = rb_cu_shared_config_from_rbsymbol(config_val);
  CUresult result = cuCtxSetSharedMemConfig(config);
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

static VALUE rb_cuCtxGetApiVersion(VALUE self, VALUE ctx_val){
  uint version;
  ctx_ptr* pctx;
  Data_Get_Struct(ctx_val, ctx_ptr, pctx);
  CUresult result = cuCtxGetApiVersion(pctx->ctx, &version);
  return UINT2NUM(version);
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
  int leastPriority, greatestPriority;
  CUresult result = cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority);
  return Qnil;
}

// CUresult cuCtxAttach ( CUcontext* pctx, unsigned int  flags )
// Increment a context's usage-count.
// Parameters
// pctx
// - Returned context handle of the current context
// flags
// - Context attach flags (must be 0)
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuCtxAttach(VALUE self, VALUE flags){
  ctx_ptr* pctx = ALLOC(ctx_ptr);
  CUresult result = cuCtxAttach(&pctx->ctx, NUM2UINT(flags));
  return Data_Wrap_Struct(RbCuContext, NULL, rbcu_free, pctx);
}

// CUresult cuCtxDetach ( CUcontext ctx )
// Decrement a context's usage-count.
// Parameters
// ctx
// - Context to destroy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

static VALUE rb_cuCtxDetach(VALUE self, VALUE ctx_val){
  ctx_ptr* pctx;
  Data_Get_Struct(ctx_val, ctx_ptr, pctx);
  CUresult result = cuCtxDetach(pctx->ctx);
  return Qnil;
}

// CUresult cuModuleLoad ( CUmodule* module, const char* fname )
// Loads a compute module.
// Parameters
// module
// - Returned module
// fname
// - Filename of module to load
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_PTX, CUDA_ERROR_NOT_FOUND, CUDA_ERROR_OUT_OF_MEMORY,
// CUDA_ERROR_FILE_NOT_FOUND, CUDA_ERROR_NO_BINARY_FOR_GPU, CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
// CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuModuleLoad(VALUE self, VALUE file_name){
  mod_ptr* mod = ALLOC(mod_ptr);
  const char* fname = StringValueCStr(file_name);
  CUresult result = cuModuleLoad(&mod->module, fname);
  return Data_Wrap_Struct(RbCuModule, NULL, rbcu_free, mod);
}

// CUresult cuModuleLoadData ( CUmodule* module, const void* image )
// Load a module's data.
// Parameters
// module
// - Returned module
// image
// - Module data to load
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_PTX, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU,
// CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuModuleLoadData(VALUE self, VALUE image){
  mod_ptr* mod = ALLOC(mod_ptr);
  CUresult result = cuModuleLoadData(&mod->module, (void*)image);
  return Data_Wrap_Struct(RbCuModule, NULL, rbcu_free, mod);
}

// CUresult cuModuleLoadDataEx ( CUmodule* module, const void* image, unsigned int  numOptions, CUjit_option* options, void** optionValues )
// Load a module's data with options.
// Parameters
// module
// - Returned module
// image
// - Module data to load
// numOptions
// - Number of options
// options
// - Options for JIT
// optionValues
// - Option values for JIT
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_PTX, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU,
// CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuModuleLoadDataEx(VALUE self, VALUE image, VALUE num_options, VALUE option_val, VALUE options_values){
  mod_ptr* mod = ALLOC(mod_ptr);
  CUjit_option options = rb_cu_jit_option_from_rbsymbol(option_val);
  CUresult result = cuModuleLoadDataEx (&mod->module, (void*)image, NUM2UINT(num_options), &options, (void**)options_values);
  return Data_Wrap_Struct(RbCuModule, NULL, rbcu_free, mod);
}

// CUresult cuModuleLoadFatBinary ( CUmodule* module, const void* fatCubin )
// Load a module's data.
// Parameters
// module
// - Returned module
// fatCubin
// - Fat binary to load
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_PTX, CUDA_ERROR_NOT_FOUND, CUDA_ERROR_OUT_OF_MEMORY,
// CUDA_ERROR_NO_BINARY_FOR_GPU, CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND, CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuModuleLoadFatBinary(VALUE self, VALUE fat_cu_bin){
  mod_ptr* mod = ALLOC(mod_ptr);
  CUresult result = cuModuleLoadFatBinary(&mod->module, (void*)fat_cu_bin);
  return Data_Wrap_Struct(RbCuModule, NULL, rbcu_free, mod);
}

// CUresult cuModuleUnload ( CUmodule hmod )
// Unloads a module.
// Parameters
// hmod
// - Module to unload
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuModuleUnload(VALUE self, VALUE module_val){
  mod_ptr* mod;
  Data_Get_Struct(module_val, mod_ptr, mod);
  CUresult result = cuModuleUnload(mod->module);
  return Qnil;
}

// CUresult cuModuleGetFunction ( CUfunction* hfunc, CUmodule hmod, const char* name )
// Returns a function handle.
// Parameters
// hfunc
// - Returned function handle
// hmod
// - Module to retrieve function from
// name
// - Name of function to retrieve
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND

static VALUE rb_cuModuleGetFunction(VALUE self, VALUE module_val, VALUE func_name){
  mod_ptr* hmod;
  Data_Get_Struct(module_val, mod_ptr, hmod);
  function_ptr* hfunc = ALLOC(function_ptr);
  CUresult result = cuModuleGetFunction(&hfunc->function, hmod->module, StringValueCStr(func_name));
  return Data_Wrap_Struct(RbCuFunction, NULL, rbcu_free, hfunc);
}

// CUresult cuModuleGetGlobal ( CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char* name )
// Returns a global pointer from a module.
// Parameters
// dptr
// - Returned global device pointer
// bytes
// - Returned global size in bytes
// hmod
// - Module to retrieve global from
// name
// - Name of global to retrieve
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND

static VALUE rb_cuModuleGetGlobal_v2(VALUE self, VALUE module_val, VALUE global_name){
  CUdeviceptr dptr;
  size_t bytes;
  mod_ptr* hmod;
  Data_Get_Struct(module_val, mod_ptr, hmod);
  CUresult result = cuModuleGetGlobal_v2 (&dptr, &bytes, hmod->module, StringValueCStr(global_name));
  return Qnil;
}

// CUresult cuModuleGetTexRef ( CUtexref* pTexRef, CUmodule hmod, const char* name )
// Returns a handle to a texture reference.
// Parameters
// pTexRef
// - Returned texture reference
// hmod
// - Module to retrieve texture reference from
// name
// - Name of texture reference to retrieve
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND

static VALUE rb_cuModuleGetTexRef(VALUE self, VALUE module_val, VALUE texture_name){
  texture_ptr* pTexRef = ALLOC(texture_ptr);
  mod_ptr* hmod;
  Data_Get_Struct(module_val, mod_ptr, hmod);
  CUresult result = cuModuleGetTexRef(&pTexRef->texture, hmod->module, StringValueCStr(texture_name));
  return Data_Wrap_Struct(RbCuTexture, NULL, rbcu_free, pTexRef);
}

// CUresult cuModuleGetSurfRef ( CUsurfref* pSurfRef, CUmodule hmod, const char* name )
// Returns a handle to a surface reference.
// Parameters
// pSurfRef
// - Returned surface reference
// hmod
// - Module to retrieve surface reference from
// name
// - Name of surface reference to retrieve
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_FOUND

static VALUE rb_cuModuleGetSurfRef(VALUE self, VALUE module_val, VALUE surface_name){
  surface_ptr* pSurfRef = ALLOC(surface_ptr);
  mod_ptr* hmod;
  Data_Get_Struct(module_val, mod_ptr, hmod);
  CUresult result = cuModuleGetSurfRef(&pSurfRef->surface, hmod->module, StringValueCStr(surface_name));
  return Data_Wrap_Struct(RbCuSurface, NULL, rbcu_free, pSurfRef);
}

// CUresult cuLinkCreate ( unsigned int  numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut )
// Creates a pending JIT linker invocation.
// Parameters
// numOptions
// Size of options arrays
// options
// Array of linker and compiler options
// optionValues
// Array of option values, each cast to void *
// stateOut
// On success, this will contain a CUlinkState to specify and complete this action
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuLinkCreate_v2(VALUE self, VALUE num_options, VALUE options, VALUE option_values){
  link_state_ptr* state_out = ALLOC(link_state_ptr);
  CUjit_option jit_option = rb_cu_jit_option_from_rbsymbol(options);
  CUresult result = cuLinkCreate_v2(NUM2UINT(num_options), &jit_option, (void**)option_values, &state_out->link_state);
  return Data_Wrap_Struct(RbCuLinkState, NULL, rbcu_free, state_out);
}

// CUresult cuLinkAddData ( CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int  numOptions, CUjit_option* options, void** optionValues )
// Add an input to a pending linker invocation.
// Parameters
// state
// A pending linker action.
// type
// The type of the input data.
// data
// The input data. PTX must be NULL-terminated.
// size
// The length of the input data.
// name
// An optional name for this input in log messages.
// numOptions
// Size of options.
// options
// Options to be applied only for this input (overrides options from cuLinkCreate).
// optionValues
// Array of option values, each cast to void *.
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_PTX, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU

static VALUE rb_cuLinkAddData_v2(VALUE self, VALUE state_val, VALUE jit_type, VALUE data, VALUE size, VALUE name, VALUE num_options, VALUE options, VALUE option_values){
  link_state_ptr* state_out;
  Data_Get_Struct(state_val, link_state_ptr, state_out);
  CUjit_option jit_option = rb_cu_jit_option_from_rbsymbol(options);
  CUresult result = cuLinkAddData_v2(state_out->link_state, rb_cu_jit_type_from_rbsymbol(jit_type), (void*)data, NUM2ULONG(size), StringValueCStr(name), NUM2UINT(num_options), &jit_option, (void**)option_values);
  return Qtrue;
}

// CUresult cuLinkAddFile ( CUlinkState state, CUjitInputType type, const char* path, unsigned int  numOptions, CUjit_option* options, void** optionValues )
// Add a file input to a pending linker invocation.
// Parameters
// state
// A pending linker action
// type
// The type of the input data
// path
// Path to the input file
// numOptions
// Size of options
// options
// Options to be applied only for this input (overrides options from cuLinkCreate)
// optionValues
// Array of option values, each cast to void *
// Returns
// CUDA_SUCCESS, CUDA_ERROR_FILE_NOT_FOUNDCUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_PTX, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_NO_BINARY_FOR_GPU

static VALUE rb_cuLinkAddFile_v2(VALUE self, VALUE state_val, VALUE jit_type, VALUE path, VALUE num_options, VALUE options, VALUE option_values){
  link_state_ptr* state_out;
  Data_Get_Struct(state_val, link_state_ptr, state_out);
  CUjit_option jit_option = rb_cu_jit_option_from_rbsymbol(options);
  CUresult result = cuLinkAddFile_v2(state_out->link_state, rb_cu_jit_type_from_rbsymbol(jit_type), StringValueCStr(path), NUM2UINT(num_options), &jit_option, (void**)option_values);
  return Qtrue;
}

// CUresult cuLinkComplete ( CUlinkState state, void** cubinOut, size_t* sizeOut )
// Complete a pending linker invocation.
// Parameters
// state
// A pending linker invocation
// cubinOut
// On success, this will point to the output image
// sizeOut
// Optional parameter to receive the size of the generated image
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuLinkComplete(VALUE self, VALUE state_val){
  link_state_ptr* state_out;
  Data_Get_Struct(state_val, link_state_ptr, state_out);
  void* cubin_out;
  size_t size_out;
  CUresult result = cuLinkComplete(state_out->link_state, &cubin_out, &size_out);
  return Qnil;
}

// CUresult cuLinkDestroy ( CUlinkState state )
// Destroys state for a JIT linker invocation.
// Parameters
// state
// State object for the linker invocation
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE

static VALUE rb_cuLinkDestroy(VALUE self, VALUE state_val){
  link_state_ptr* state_out;
  Data_Get_Struct(state_val, link_state_ptr, state_out);
  CUresult result = cuLinkDestroy(state_out->link_state);
  return Qnil;
}

// CUresult cuMemGetInfo ( size_t* free, size_t* total )
// Gets free and total memory.
// Parameters
// free
// - Returned free memory in bytes
// total
// - Returned total memory in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemGetInfo_v2(VALUE self){
  size_t free, total;
  CUresult result = cuMemGetInfo_v2 (&free, &total);
  return Qnil;
}

// CUresult cuMemAlloc ( CUdeviceptr* dptr, size_t bytesize )
// Allocates device memory.
// Parameters
// dptr
// - Returned device pointer
// bytesize
// - Requested allocation size in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuMemAlloc_v2(VALUE self, VALUE byte_size){
  CUdeviceptr dptr;
  CUresult result = cuMemAlloc_v2(&dptr, NUM2ULONG(byte_size));
  return ULONG2NUM(dptr);
}

// CUresult cuMemAllocPitch ( CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes )
// Allocates pitched device memory.
// Parameters
// dptr
// - Returned device pointer
// pPitch
// - Returned pitch of allocation in bytes
// WidthInBytes
// - Requested allocation width in bytes
// Height
// - Requested allocation height in rows
// ElementSizeBytes
// - Size of largest reads/writes for range
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuMemAllocPitch_v2(VALUE self, VALUE width_in_bytes, VALUE  height, VALUE element_size_bytes){
  CUdeviceptr dptr;
  size_t p_pitch;
  CUresult result = cuMemAllocPitch_v2(&dptr, &p_pitch, NUM2ULONG(width_in_bytes), NUM2ULONG(height), NUM2UINT(element_size_bytes));
  return ULONG2NUM(dptr);
}

// CUresult cuMemFree ( CUdeviceptr dptr )
// Frees device memory.
// Parameters
// dptr
// - Pointer to memory to free
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemFree_v2(VALUE self, VALUE dptr){
  CUresult result = cuMemFree_v2(NUM2ULONG(dptr));
  return Qtrue;
}

// CUresult cuMemGetAddressRange ( CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr )
// Get information on memory allocations.
// Parameters
// pbase
// - Returned base address
// psize
// - Returned size of device memory allocation
// dptr
// - Device pointer to query
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_NOT_FOUND, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemGetAddressRange_v2(VALUE self, VALUE dptr){
  CUdeviceptr p_base;
  size_t p_size;
  CUresult result = cuMemGetAddressRange_v2(&p_base, &p_size, NUM2ULONG(dptr));
  return ULONG2NUM(p_base);
}

// CUresult cuMemAllocHost ( void** pp, size_t bytesize )
// Allocates page-locked host memory.
// Parameters
// pp
// - Returned host pointer to page-locked memory
// bytesize
// - Requested allocation size in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuMemAllocHost_v2(VALUE self, VALUE byte_size){
  void* pp;
  CUresult result = cuMemAllocHost_v2 (&pp,  NUM2ULONG(byte_size));
  return (VALUE)pp;
}

// CUresult cuMemFreeHost ( void* p )
// Frees page-locked host memory.
// Parameters
// p
// - Pointer to memory to free
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemFreeHost(VALUE self, VALUE p){
  CUresult result = cuMemFreeHost((void*)p);
  return Qtrue;
}

// CUresult cuMemHostAlloc ( void** pp, size_t bytesize, unsigned int  Flags )
// Allocates page-locked host memory.
// Parameters
// pp
// - Returned host pointer to page-locked memory
// bytesize
// - Requested allocation size in bytes
// Flags
// - Flags for allocation request
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuMemHostAlloc(VALUE self, VALUE byte_size, VALUE flags){
  void* pp;
  CUresult result = cuMemHostAlloc(&pp, NUM2ULONG(byte_size), NUM2UINT(flags));
  return (VALUE)pp;
}

// CUresult cuMemHostGetDevicePointer ( CUdeviceptr* pdptr, void* p, unsigned int  Flags )
// Passes back device pointer of mapped pinned memory.
// Parameters
// pdptr
// - Returned device pointer
// p
// - Host pointer
// Flags
// - Options (must be 0)
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemHostGetDevicePointer_v2(VALUE self, VALUE p, VALUE flags){
  CUdeviceptr p_dptr;
  CUresult result = cuMemHostGetDevicePointer_v2(&p_dptr, (void*)p, NUM2UINT(flags));
  return ULONG2NUM(p_dptr);
}

// CUresult cuMemHostGetFlags ( unsigned int* pFlags, void* p )
// Passes back flags that were used for a pinned allocation.
// Parameters
// pFlags
// - Returned flags word
// p
// - Host pointer
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemHostGetFlags(VALUE self, VALUE  p){
  uint p_flags;
  CUresult result = cuMemHostGetFlags(&p_flags, (void*)p);
  return UINT2NUM(p_flags);
}

// CUresult cuMemAllocManaged ( CUdeviceptr* dptr, size_t bytesize, unsigned int  flags )
// Allocates memory that will be automatically managed by the Unified Memory system.
// Parameters
// dptr
// - Returned device pointer
// bytesize
// - Requested allocation size in bytes
// flags
// - Must be one of CU_MEM_ATTACH_GLOBAL or CU_MEM_ATTACH_HOST
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY

static VALUE rb_cuMemAllocManaged(VALUE self, VALUE  byte_size, VALUE flags){
  CUdeviceptr dptr;
  CUresult result = cuMemAllocManaged(&dptr, NUM2ULONG(byte_size), NUM2UINT(flags));
  return ULONG2NUM(dptr);
}

// CUresult cuDeviceGetByPCIBusId ( CUdevice* dev, const char* pciBusId )
// Returns a handle to a compute device.
// Parameters
// dev
// - Returned device handle
// pciBusId
// - String in one of the following forms: [domain]:[bus]:[device].[function] [domain]:[bus]:[device] [bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE


static VALUE rb_cuDeviceGetByPCIBusId(VALUE self, VALUE pci_bus_id){
  CUdevice dev;
  CUresult result = cuDeviceGetByPCIBusId(&dev, StringValueCStr(pci_bus_id));
  return INT2NUM(dev);
}

// CUresult cuDeviceGetPCIBusId ( char* pciBusId, int  len, CUdevice dev )
// Returns a PCI Bus Id string for the device.
// Parameters
// pciBusId
// - Returned identifier string for the device in the following format [domain]:[bus]:[device].[function] where domain, bus, device, and function are all hexadecimal values. pciBusId should be large enough to store 13 characters including the NULL-terminator.
// len
// - Maximum length of string to store in name
// dev
// - Device to get identifier string for
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_DEVICE

static VALUE rb_cuDeviceGetPCIBusId(VALUE self, VALUE len, VALUE dev){
  char* pci_bus_id;
  CUresult result = cuDeviceGetPCIBusId(pci_bus_id, NUM2INT(len), NUM2INT(dev));
  return rb_str_new_cstr(pci_bus_id);
}

// CUresult cuIpcGetEventHandle ( CUipcEventHandle* pHandle, CUevent event )
// Gets an interprocess handle for a previously allocated event.
// Parameters
// pHandle
// - Pointer to a user allocated CUipcEventHandle in which to return the opaque event handle
// event
// - Event allocated with CU_EVENT_INTERPROCESS and CU_EVENT_DISABLE_TIMING flags.
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_MAP_FAILED

static VALUE rb_cuIpcGetEventHandle(VALUE self, VALUE event_val){
  ipc_event_handler* handler = ALLOC(ipc_event_handler);
  cu_event* event;
  Data_Get_Struct(event_val, cu_event, event);
  CUresult result = cuIpcGetEventHandle (&handler->handle, event->event);
  return Data_Wrap_Struct(RbCuIPCEventHandler, NULL, rbcu_free, handler);
}

// CUresult cuIpcOpenEventHandle ( CUevent* phEvent, CUipcEventHandle handle )
// Opens an interprocess event handle for use in the current process.
// Parameters
// phEvent
// - Returns the imported event
// handle
// - Interprocess handle to open
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED, CUDA_ERROR_PEER_ACCESS_UNSUPPORTED, CUDA_ERROR_INVALID_HANDLE

static VALUE rb_cuIpcOpenEventHandle(VALUE self, VALUE ipc_handler_val){
  cu_event* ph_event = ALLOC(cu_event);
  ipc_event_handler* handler;
  Data_Get_Struct(ipc_handler_val, ipc_event_handler, handler);
  CUresult result = cuIpcOpenEventHandle(&ph_event->event, handler->handle);
  return Data_Wrap_Struct(RbCuEvent, NULL, rbcu_free, ph_event);
}

// CUresult cuIpcGetMemHandle ( CUipcMemHandle* pHandle, CUdeviceptr dptr )
// Gets an interprocess memory handle for an existing device memory allocation.
// Parameters
// pHandle
// - Pointer to user allocated CUipcMemHandle to return the handle in.
// dptr
// - Base pointer to previously allocated device memory
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_MAP_FAILED,

static VALUE rb_cuIpcGetMemHandle(VALUE self, VALUE dptr){
  ipc_mem_handler* handler = ALLOC(ipc_mem_handler);
  CUresult result = cuIpcGetMemHandle(&handler->handle, NUM2ULONG(dptr));
  return Data_Wrap_Struct(RbCuIPCMemHandler, NULL, rbcu_free, handler);
}

// CUresult cuIpcOpenMemHandle ( CUdeviceptr* pdptr, CUipcMemHandle handle, unsigned int  Flags )
// Opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process.
// Parameters
// pdptr
// - Returned device pointer
// handle
// - CUipcMemHandle to open
// Flags
// - Flags for this operation. Must be specified as CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_TOO_MANY_PEERS

static VALUE rb_cuIpcOpenMemHandle(VALUE self, VALUE mem_handler_val, VALUE flags){
  CUdeviceptr pdptr;
  ipc_mem_handler* handler;
  Data_Get_Struct(mem_handler_val, ipc_mem_handler, handler);
  CUresult result = cuIpcOpenMemHandle(&pdptr, handler->handle, NUM2UINT(flags));
  return ULONG2NUM(pdptr);
}

// CUresult cuIpcCloseMemHandle ( CUdeviceptr dptr )
// Close memory mapped with cuIpcOpenMemHandle.
// Parameters
// dptr
// - Device pointer returned by cuIpcOpenMemHandle
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_MAP_FAILED, CUDA_ERROR_INVALID_HANDLE,

static VALUE rb_cuIpcCloseMemHandle(VALUE self, VALUE dptr){
  CUresult result = cuIpcCloseMemHandle(NUM2ULONG(dptr));
  return Qtrue;
}

// CUresult cuMemHostRegister ( void* p, size_t bytesize, unsigned int  Flags )
// Registers an existing host memory range for use by CUDA.
// Parameters
// p
// - Host pointer to memory to page-lock
// bytesize
// - Size in bytes of the address range to page-lock
// Flags
// - Flags for allocation request
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
// CUDA_ERROR_NOT_PERMITTED, CUDA_ERROR_NOT_SUPPORTED

static VALUE rb_cuMemHostRegister_v2(VALUE self, VALUE p_val, VALUE byte_size, VALUE flags){
  CUresult result = cuMemHostRegister_v2((void*)p_val, NUM2ULONG(byte_size), NUM2UINT(flags));
  return Qtrue;
}

// CUresult cuMemHostUnregister ( void* p )
// Unregisters a memory range that was registered with cuMemHostRegister.
// Parameters
// p
// - Host pointer to memory to unregister
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,

static VALUE rb_cuMemHostUnregister(VALUE self, VALUE p_val){
  CUresult result = cuMemHostUnregister((void*)p_val);
  return Qtrue;
}

// CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount )
// Copies memory.
// Parameters
// dst
// - Destination unified virtual address space pointer
// src
// - Source unified virtual address space pointer
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy(VALUE self, VALUE dst_val, VALUE src_val, VALUE byte_count){
  CUresult result = cuMemcpy(NUM2ULONG(dst_val), NUM2ULONG(src_val), NUM2ULONG(byte_count));
  return Qtrue;
}

// CUresult cuMemcpyPeer ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount )
// Copies device memory between two contexts.
// Parameters
// dstDevice
// - Destination device pointer
// dstContext
// - Destination context
// srcDevice
// - Source device pointer
// srcContext
// - Source context
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyPeer(VALUE self, VALUE dst_device, VALUE dst_context, VALUE src_device, VALUE src_context, VALUE byte_count){
  ctx_ptr* dst_ctx;
  ctx_ptr* src_ctx;
  Data_Get_Struct(dst_context, ctx_ptr, dst_ctx);
  Data_Get_Struct(src_context, ctx_ptr, src_ctx);
  CUresult result = cuMemcpyPeer(NUM2ULONG(dst_device), dst_ctx->ctx, NUM2ULONG(src_device), src_ctx->ctx, NUM2ULONG(byte_count));
  return Qnil;
}

// CUresult cuMemcpyHtoD ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount )
// Copies memory from Host to Device.
// Parameters
// dstDevice
// - Destination device pointer
// srcHost
// - Source host pointer
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyHtoD_v2(VALUE self, VALUE dst_device, VALUE src_host, VALUE , VALUE byte_count){
  CUresult result = cuMemcpyHtoD_v2(NUM2ULONG(dst_device), (void*)src_host, NUM2ULONG(byte_count));
  return Qtrue;
}

// CUresult cuMemcpyDtoH ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount )
// Copies memory from Device to Host.
// Parameters
// dstHost
// - Destination host pointer
// srcDevice
// - Source device pointer
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyDtoH_v2(VALUE self, VALUE dst_host, VALUE src_device, VALUE byte_count){
  CUresult result = cuMemcpyDtoH_v2((void*)dst_host, NUM2ULONG(src_device), NUM2ULONG(byte_count));
  return Qtrue;
}

// CUresult cuMemcpyDtoD ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount )
// Copies memory from Device to Device.
// Parameters
// dstDevice
// - Destination device pointer
// srcDevice
// - Source device pointer
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyDtoD_v2(VALUE self, VALUE dst_device, VALUE src_device, VALUE byte_count){
  CUresult result = cuMemcpyDtoD_v2(NUM2ULONG(dst_device) , NUM2ULONG(src_device), NUM2ULONG(byte_count));
  return Qtrue;
}

// CUresult cuMemcpyDtoA ( CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount )
// Copies memory from Device to Array.
// Parameters
// dstArray
// - Destination array
// dstOffset
// - Offset in bytes of destination array
// srcDevice
// - Source device pointer
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyDtoA_v2(VALUE self, VALUE dst_array_val, VALUE dst_offset, VALUE src_device, VALUE byte_count){
  cuarray_ptr* dst_array;
  Data_Get_Struct(dst_array_val, cuarray_ptr, dst_array);
  CUresult result = cuMemcpyDtoA_v2(dst_array->array, NUM2ULONG(dst_offset), NUM2ULONG(src_device), NUM2ULONG(byte_count));
  return Qtrue;
}

// CUresult cuMemcpyAtoD ( CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount )
// Copies memory from Array to Device.
// Parameters
// dstDevice
// - Destination device pointer
// srcArray
// - Source array
// srcOffset
// - Offset in bytes of source array
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyAtoD_v2(VALUE self, VALUE dst_device, VALUE src_array_val, VALUE src_offset, VALUE byte_count){
  cuarray_ptr* src_array;
  Data_Get_Struct(src_array_val, cuarray_ptr, src_array);
  CUresult result = cuMemcpyAtoD_v2(NUM2ULONG(dst_device), src_array->array, NUM2ULONG(src_offset), NUM2ULONG(byte_count));
  return Qtrue;
}

// CUresult cuMemcpyHtoA ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount )
// Copies memory from Host to Array.
// Parameters
// dstArray
// - Destination array
// dstOffset
// - Offset in bytes of destination array
// srcHost
// - Source host pointer
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyHtoA_v2(VALUE self, VALUE dst_array_val, VALUE dst_offset, VALUE src_host, VALUE byte_count ){
  cuarray_ptr* dst_array;
  Data_Get_Struct(dst_array_val, cuarray_ptr, dst_array);
  CUresult result = cuMemcpyHtoA_v2(dst_array->array, NUM2ULONG(dst_offset), (void*)src_host, NUM2ULONG(byte_count));
  return Qtrue;
}

// CUresult cuMemcpyAtoH ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount )
// Copies memory from Array to Host.
// Parameters
// dstHost
// - Destination device pointer
// srcArray
// - Source array
// srcOffset
// - Offset in bytes of source array
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyAtoH_v2(VALUE self, VALUE dst_host, VALUE src_array_val, VALUE src_offset, VALUE byte_count){
  cuarray_ptr* src_array;
  Data_Get_Struct(src_array_val, cuarray_ptr, src_array);
  CUresult result = cuMemcpyAtoH_v2((void*)dst_host, src_array->array, NUM2ULONG(src_offset), NUM2ULONG(byte_count));
  return Qnil;
}

// CUresult cuMemcpyAtoA ( CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount )
// Copies memory from Array to Array.
// Parameters
// dstArray
// - Destination array
// dstOffset
// - Offset in bytes of destination array
// srcArray
// - Source array
// srcOffset
// - Offset in bytes of source array
// ByteCount
// - Size of memory copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyAtoA_v2(VALUE self, VALUE dst_array_val, VALUE dst_offset, VALUE src_array_val, VALUE src_offset, VALUE byte_count){
  cuarray_ptr* dst_array;
  cuarray_ptr* src_array;
  Data_Get_Struct(dst_array_val, cuarray_ptr, dst_array);
  Data_Get_Struct(src_array_val, cuarray_ptr, src_array);
  CUresult result = cuMemcpyAtoA_v2(dst_array->array, NUM2ULONG(dst_offset), src_array->array, NUM2ULONG(src_offset), NUM2ULONG(byte_count));
  return Qnil;
}

// CUresult cuMemcpy2D ( const CUDA_MEMCPY2D* pCopy )
// Copies memory for 2D arrays.
// Parameters
// pCopy
// - Parameters for the memory copy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy2D_v2(VALUE self, VALUE p_copy){
  CUresult result = cuMemcpy2D_v2((CUDA_MEMCPY2D*)p_copy);
  return Qtrue;
}

// CUresult cuMemcpy2DUnaligned ( const CUDA_MEMCPY2D* pCopy )
// Copies memory for 2D arrays.
// Parameters
// pCopy
// - Parameters for the memory copy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy2DUnaligned_v2(VALUE self, VALUE p_copy){
  CUresult result = cuMemcpy2DUnaligned_v2((CUDA_MEMCPY2D*)p_copy);
  return Qtrue;
}

// CUresult cuMemcpy3D ( const CUDA_MEMCPY3D* pCopy )
// Copies memory for 3D arrays.
// Parameters
// pCopy
// - Parameters for the memory copy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy3D_v2(VALUE self, VALUE p_copy){
  CUresult result = cuMemcpy3D_v2((CUDA_MEMCPY3D*)p_copy);
  return Qtrue;
}

// CUresult cuMemcpy3DPeer ( const CUDA_MEMCPY3D_PEER* pCopy )
// Copies memory between contexts.
// Parameters
// pCopy
// - Parameters for the memory copy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy3DPeer(VALUE self, VALUE p_copy){
  CUresult result = cuMemcpy3DPeer((CUDA_MEMCPY3D_PEER*) p_copy);
  return Qtrue;
}

// CUresult cuMemcpyAsync ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream )
// Copies memory asynchronously.
// Parameters
// dst
// - Destination unified virtual address space pointer
// src
// - Source unified virtual address space pointer
// ByteCount
// - Size of memory copy in bytes
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyAsync(VALUE self, VALUE dst, VALUE src, VALUE byte_count, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpyAsync(NUM2ULONG(dst), NUM2ULONG(src), NUM2ULONG(byte_count), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpyPeerAsync ( CUdeviceptr dstDevice, CUcontext dstContext, CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount, CUstream hStream )
// Copies device memory between two contexts asynchronously.
// Parameters
// dstDevice
// - Destination device pointer
// dstContext
// - Destination context
// srcDevice
// - Source device pointer
// srcContext
// - Source context
// ByteCount
// - Size of memory copy in bytes
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyPeerAsync(VALUE self, VALUE dst_device, VALUE dst_context_val, VALUE src_device, VALUE src_context_val, VALUE byte_count, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  ctx_ptr* dst_ctx;
  Data_Get_Struct(dst_context_val, ctx_ptr, dst_ctx);
  ctx_ptr* src_ctx;
  Data_Get_Struct(src_context_val, ctx_ptr, src_ctx);
  CUresult result = cuMemcpyPeerAsync(NUM2ULONG(dst_device), dst_ctx->ctx, NUM2ULONG(src_device), src_ctx->ctx, NUM2ULONG(byte_count), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpyHtoDAsync ( CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream )
// Copies memory from Host to Device.
// Parameters
// dstDevice
// - Destination device pointer
// srcHost
// - Source host pointer
// ByteCount
// - Size of memory copy in bytes
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyHtoDAsync_v2(VALUE self, VALUE dst_device, VALUE src_host, VALUE byte_count, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpyHtoDAsync_v2(NUM2ULONG(dst_device), (void*)src_host,  NUM2ULONG(byte_count), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpyDtoHAsync ( void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )
// Copies memory from Device to Host.
// Parameters
// dstHost
// - Destination host pointer
// srcDevice
// - Source device pointer
// ByteCount
// - Size of memory copy in bytes
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyDtoHAsync_v2(VALUE self, VALUE dst_host, VALUE src_device, VALUE byte_count, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpyDtoHAsync_v2((void*)dst_host, NUM2ULONG(src_device),  NUM2ULONG(byte_count), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpyDtoDAsync ( CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream )
// Copies memory from Device to Device.
// Parameters
// dstDevice
// - Destination device pointer
// srcDevice
// - Source device pointer
// ByteCount
// - Size of memory copy in bytes
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyDtoDAsync_v2(VALUE self, VALUE dst_device, VALUE src_device, VALUE byte_count, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpyDtoDAsync_v2(NUM2ULONG(dst_device), NUM2ULONG(src_device), NUM2ULONG(byte_count), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpyHtoAAsync ( CUarray dstArray, size_t dstOffset, const void* srcHost, size_t ByteCount, CUstream hStream )
// Copies memory from Host to Array.
// Parameters
// dstArray
// - Destination array
// dstOffset
// - Offset in bytes of destination array
// srcHost
// - Source host pointer
// ByteCount
// - Size of memory copy in bytes
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyHtoAAsync_v2(VALUE self, VALUE dst_array_val, VALUE dst_offset, VALUE src_host, VALUE  byte_count, VALUE h_stream_val){
  cuarray_ptr* dst_array;
  Data_Get_Struct(dst_array_val, cuarray_ptr, dst_array);
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpyHtoAAsync_v2(dst_array->array, NUM2ULONG(dst_offset), (void*)src_host,  NUM2ULONG(byte_count), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpyAtoHAsync ( void* dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream )
// Copies memory from Array to Host.
// Parameters
// dstHost
// - Destination pointer
// srcArray
// - Source array
// srcOffset
// - Offset in bytes of source array
// ByteCount
// - Size of memory copy in bytes
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpyAtoHAsync_v2(VALUE self, VALUE  dst_host, VALUE src_array_val, VALUE src_offset, VALUE byte_count, VALUE h_stream_val){
  cuarray_ptr* src_array;
  Data_Get_Struct(src_array_val, cuarray_ptr, src_array);
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpyAtoHAsync_v2((void*)dst_host, src_array->array, NUM2ULONG(src_offset),  NUM2ULONG(byte_count), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpy2DAsync ( const CUDA_MEMCPY2D* pCopy, CUstream hStream )
// Copies memory for 2D arrays.
// Parameters
// pCopy
// - Parameters for the memory copy
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy2DAsync_v2(VALUE self, VALUE p_copy, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpy2DAsync_v2((CUDA_MEMCPY2D*)p_copy, h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpy3DAsync ( const CUDA_MEMCPY3D* pCopy, CUstream hStream )
// Copies memory for 3D arrays.
// Parameters
// pCopy
// - Parameters for the memory copy
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy3DAsync_v2(VALUE self, VALUE p_copy, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpy3DAsync_v2 ((CUDA_MEMCPY3D*)p_copy, h_stream->stream);
  return Qtrue;
}

// CUresult cuMemcpy3DPeerAsync ( const CUDA_MEMCPY3D_PEER* pCopy, CUstream hStream )
// Copies memory between contexts asynchronously.
// Parameters
// pCopy
// - Parameters for the memory copy
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemcpy3DPeerAsync(VALUE self, VALUE p_copy, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemcpy3DPeerAsync((CUDA_MEMCPY3D_PEER*)p_copy, h_stream->stream);
  return Qtrue;
}

// CUresult cuMemsetD8 ( CUdeviceptr dstDevice, unsigned char  uc, size_t N )
// Initializes device memory.
// Parameters
// dstDevice
// - Destination device pointer
// uc
// - Value to set
// N
// - Number of elements
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD8_v2(VALUE self, VALUE dst_device, VALUE uc, VALUE n){
  CUresult result = cuMemsetD8_v2(NUM2ULONG(dst_device), NUM2CHR(uc), NUM2ULONG(n));
  return Qtrue;
}

// CUresult cuMemsetD16 ( CUdeviceptr dstDevice, unsigned short us, size_t N )
// Initializes device memory.
// Parameters
// dstDevice
// - Destination device pointer
// us
// - Value to set
// N
// - Number of elements
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD16_v2(VALUE self, VALUE dst_device, VALUE us, VALUE n){
  CUresult result = cuMemsetD16_v2(NUM2ULONG(dst_device), NUM2USHORT(us), NUM2ULONG(n));
  return Qtrue;
}

// CUresult cuMemsetD32 ( CUdeviceptr dstDevice, unsigned int  ui, size_t N )
// Initializes device memory.
// Parameters
// dstDevice
// - Destination device pointer
// ui
// - Value to set
// N
// - Number of elements
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD32_v2(VALUE self, VALUE dst_device, VALUE ui, VALUE n){
  CUresult result = cuMemsetD32_v2(NUM2ULONG(dst_device), NUM2UINT(ui), NUM2ULONG(n));
  return Qnil;
}

// CUresult cuMemsetD2D8 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height )
// Initializes device memory.
// Parameters
// dstDevice
// - Destination device pointer
// dstPitch
// - Pitch of destination device pointer
// uc
// - Value to set
// Width
// - Width of row
// Height
// - Number of rows
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD2D8_v2(VALUE self, VALUE dst_device, VALUE dst_pitch, VALUE uc, VALUE width, VALUE height){
  CUresult result = cuMemsetD2D8_v2(NUM2ULONG(dst_device), NUM2ULONG(dst_pitch), NUM2CHR(uc), NUM2ULONG(width), NUM2ULONG(height));
  return Qtrue;
}

// CUresult cuMemsetD2D16 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height )
// Initializes device memory.
// Parameters
// dstDevice
// - Destination device pointer
// dstPitch
// - Pitch of destination device pointer
// us
// - Value to set
// Width
// - Width of row
// Height
// - Number of rows
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD2D16_v2(VALUE self, VALUE dst_device, VALUE dst_pitch, VALUE us, VALUE width, VALUE height){
  CUresult result = cuMemsetD2D16_v2(NUM2ULONG(dst_device), NUM2ULONG(dst_pitch), NUM2USHORT(us), NUM2ULONG(width), NUM2ULONG(height));
  return Qtrue;
}

// CUresult cuMemsetD2D32 ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height )
// Initializes device memory.
// Parameters
// dstDevice
// - Destination device pointer
// dstPitch
// - Pitch of destination device pointer
// ui
// - Value to set
// Width
// - Width of row
// Height
// - Number of rows
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD2D32_v2(VALUE self, VALUE dst_device, VALUE dst_pitch, VALUE ui, VALUE width, VALUE height){
  CUresult result = cuMemsetD2D32_v2(NUM2ULONG(dst_device), NUM2ULONG(dst_pitch), NUM2UINT(ui), NUM2ULONG(width), NUM2ULONG(height));
  return Qtrue;
}

// CUresult cuMemsetD8Async ( CUdeviceptr dstDevice, unsigned char  uc, size_t N, CUstream hStream )
// Sets device memory.
// Parameters
// dstDevice
// - Destination device pointer
// uc
// - Value to set
// N
// - Number of elements
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD8Async(VALUE self, VALUE dst_device, VALUE uc, VALUE n, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemsetD8Async(NUM2ULONG(dst_device), NUM2CHR(uc), NUM2ULONG(n), h_stream->stream);
  return Qnil;
}

// CUresult cuMemsetD16Async ( CUdeviceptr dstDevice, unsigned short us, size_t N, CUstream hStream )
// Sets device memory.
// Parameters
// dstDevice
// - Destination device pointer
// us
// - Value to set
// N
// - Number of elements
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD16Async(VALUE self, VALUE dst_device, VALUE us, VALUE n, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemsetD16Async(NUM2ULONG(dst_device), NUM2USHORT(us), NUM2ULONG(n), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemsetD32Async ( CUdeviceptr dstDevice, unsigned int  ui, size_t N, CUstream hStream )
// Sets device memory.
// Parameters
// dstDevice
// - Destination device pointer
// ui
// - Value to set
// N
// - Number of elements
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD32Async(VALUE self, VALUE dst_device, VALUE ui, VALUE n, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemsetD32Async(NUM2ULONG(dst_device), NUM2UINT(ui), NUM2ULONG(n), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemsetD2D8Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned char  uc, size_t Width, size_t Height, CUstream hStream )
// Sets device memory.
// Parameters
// dstDevice
// - Destination device pointer
// dstPitch
// - Pitch of destination device pointer
// uc
// - Value to set
// Width
// - Width of row
// Height
// - Number of rows
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD2D8Async(VALUE self, VALUE dst_device, VALUE dst_pitch, VALUE uc, VALUE width, VALUE height, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemsetD2D8Async(NUM2ULONG(dst_device), NUM2ULONG(dst_pitch), NUM2CHR(uc), NUM2ULONG(width), NUM2ULONG(height), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemsetD2D16Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height, CUstream hStream )
// Sets device memory.
// Parameters
// dstDevice
// - Destination device pointer
// dstPitch
// - Pitch of destination device pointer
// us
// - Value to set
// Width
// - Width of row
// Height
// - Number of rows
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD2D16Async(VALUE self, VALUE dst_device, VALUE dst_pitch, VALUE us, VALUE width, VALUE height, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemsetD2D16Async(NUM2ULONG(dst_device), NUM2ULONG(dst_pitch), NUM2USHORT(us), NUM2ULONG(width), NUM2ULONG(height), h_stream->stream);
  return Qtrue;
}

// CUresult cuMemsetD2D32Async ( CUdeviceptr dstDevice, size_t dstPitch, unsigned int  ui, size_t Width, size_t Height, CUstream hStream )
// Sets device memory.
// Parameters
// dstDevice
// - Destination device pointer
// dstPitch
// - Pitch of destination device pointer
// ui
// - Value to set
// Width
// - Width of row
// Height
// - Number of rows
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuMemsetD2D32Async(VALUE self, VALUE dst_device, VALUE dst_pitch, VALUE ui, VALUE width, VALUE height, VALUE h_stream_val){
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuMemsetD2D32Async(NUM2ULONG(dst_device), NUM2ULONG(dst_pitch), NUM2UINT(ui), NUM2ULONG(width), NUM2ULONG(height), h_stream->stream);
  return Qnil;
}

static VALUE rb_cuArrayCreate_v2(VALUE self){
  // CUresult cuArrayCreate_v2 (CUarray* pHandle, const(CUDA_ARRAY_DESCRIPTOR)* pAllocateArray);
  return Qnil;
}

static VALUE rb_cuArrayGetDescriptor_v2(VALUE self){
  // CUresult cuArrayGetDescriptor_v2 (CUDA_ARRAY_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
  return Qnil;
}

static VALUE rb_cuArrayDestroy(VALUE self){
  // CUresult cuArrayDestroy (CUarray hArray);
  return Qnil;
}

static VALUE rb_cuArray3DCreate_v2(VALUE self){
  // CUresult cuArray3DCreate_v2 (CUarray* pHandle, const(CUDA_ARRAY3D_DESCRIPTOR)* pAllocateArray);
  return Qnil;
}

static VALUE rb_cuArray3DGetDescriptor_v2(VALUE self){
  // CUresult cuArray3DGetDescriptor_v2 (CUDA_ARRAY3D_DESCRIPTOR* pArrayDescriptor, CUarray hArray);
  return Qnil;
}

// CUresult cuMipmappedArrayCreate ( CUmipmappedArray* pHandle, const CUDA_ARRAY3D_DESCRIPTOR* pMipmappedArrayDesc, unsigned int  numMipmapLevels )
// Creates a CUDA mipmapped array.
// Parameters
// pHandle
// - Returned mipmapped array
// pMipmappedArrayDesc
// - mipmapped array descriptor
// numMipmapLevels
// - Number of mipmap levels
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN


static VALUE rb_cuMipmappedArrayCreate(VALUE self){
  // CUresult cuMipmappedArrayCreate (CUmipmappedArray* pHandle, const(CUDA_ARRAY3D_DESCRIPTOR)* pMipmappedArrayDesc, uint numMipmapLevels);
  return Qnil;
}

static VALUE rb_cuMipmappedArrayGetLevel(VALUE self){
  // CUresult cuMipmappedArrayGetLevel (CUarray* pLevelArray, CUmipmappedArray hMipmappedArray, uint level);
  return Qnil;
}

// CUresult cuMipmappedArrayDestroy ( CUmipmappedArray hMipmappedArray )
// Destroys a CUDA mipmapped array.
// Parameters
// hMipmappedArray
// - Mipmapped array to destroy
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ARRAY_IS_MAPPED

static VALUE rb_cuMipmappedArrayDestroy(VALUE self){
  // CUresult cuMipmappedArrayDestroy (CUmipmappedArray hMipmappedArray);
  return Qnil;
}

static VALUE rb_cuPointerGetAttribute(VALUE self){
  // CUresult cuPointerGetAttribute (void* data, CUpointer_attribute attribute, CUdeviceptr ptr);
  return Qnil;
}

static VALUE rb_cuPointerSetAttribute(VALUE self){
  // CUresult cuPointerSetAttribute (const(void)* value, CUpointer_attribute attribute, CUdeviceptr ptr);
  return Qnil;
}

static VALUE rb_cuPointerGetAttributes(VALUE self){
  // CUresult cuPointerGetAttributes (uint numAttributes, CUpointer_attribute* attributes, void** data, CUdeviceptr ptr);
  return Qnil;
}

static VALUE rb_cuStreamCreate(VALUE self){
  // CUresult cuStreamCreate (CUstream* phStream, uint Flags);
  return Qnil;
}

static VALUE rb_cuStreamCreateWithPriority(VALUE self){
  // CUresult cuStreamCreateWithPriority (CUstream* phStream, uint flags, int priority);
  return Qnil;
}

static VALUE rb_cuStreamGetPriority(VALUE self){
  // CUresult cuStreamGetPriority (CUstream hStream, int* priority);
  return Qnil;
}

static VALUE rb_cuStreamGetFlags(VALUE self){
  // CUresult cuStreamGetFlags (CUstream hStream, uint* flags);
  return Qnil;
}

static VALUE rb_cuStreamWaitEvent(VALUE self){
  // CUresult cuStreamWaitEvent (CUstream hStream, CUevent hEvent, uint Flags);
  return Qnil;
}

static VALUE rb_cuStreamAddCallback(VALUE self){
  // CUresult cuStreamAddCallback (CUstream hStream, CUstreamCallback callback, void* userData, uint flags);
  return Qnil;
}

static VALUE rb_cuStreamAttachMemAsync(VALUE self){
  // CUresult cuStreamAttachMemAsync (CUstream hStream, CUdeviceptr dptr, size_t length, uint flags);
  return Qnil;
}

static VALUE rb_cuStreamQuery(VALUE self){
  // CUresult cuStreamQuery (CUstream hStream);
  return Qnil;
}

static VALUE rb_cuStreamSynchronize(VALUE self){
  // CUresult cuStreamSynchronize (CUstream hStream);
  return Qnil;
}

static VALUE rb_cuStreamDestroy_v2(VALUE self){
  // CUresult cuStreamDestroy_v2 (CUstream hStream);
  return Qnil;
}

static VALUE rb_cuEventCreate(VALUE self){
  // CUresult cuEventCreate (CUevent* phEvent, uint Flags);
  return Qnil;
}

static VALUE rb_cuEventRecord(VALUE self){
  // CUresult cuEventRecord (CUevent hEvent, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuEventQuery(VALUE self){
  // CUresult cuEventQuery (CUevent hEvent);
  return Qnil;
}

static VALUE rb_cuEventSynchronize(VALUE self){
  // CUresult cuEventSynchronize (CUevent hEvent);
  return Qnil;
}

static VALUE rb_cuEventDestroy_v2(VALUE self){
  // CUresult cuEventDestroy_v2 (CUevent hEvent);
  return Qnil;
}

static VALUE rb_cuEventElapsedTime(VALUE self){
  // CUresult cuEventElapsedTime (float* pMilliseconds, CUevent hStart, CUevent hEnd);
  return Qnil;
}

// CUresult cuFuncGetAttribute ( int* pi, CUfunction_attribute attrib, CUfunction hfunc )
// Returns information about a function.
// Parameters
// pi
// - Returned attribute value
// attrib
// - Attribute requested
// hfunc
// - Function to query attribute of

static VALUE rb_cuFuncGetAttribute(VALUE self, VALUE function_attribute, VALUE hfunc_val){
  int pi;
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuFuncGetAttribute (&pi, rb_cu_function_attribute_from_rbsymbol(function_attribute), hfunc->function);
  return INT2NUM(pi);
}

// CUresult cuFuncSetCacheConfig ( CUfunction hfunc, CUfunc_cache config )
// Sets the preferred cache configuration for a device function.
// Parameters
// hfunc
// - Kernel to configure cache for
// config
// - Requested cache configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

static VALUE rb_cuFuncSetCacheConfig(VALUE self, VALUE hfunc_val, VALUE  config){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuFuncSetCacheConfig(hfunc->function, rb_cu_func_cache_from_rbsymbol(config));
  return Qtrue;
}

// CUresult cuFuncSetSharedMemConfig ( CUfunction hfunc, CUsharedconfig config )
// Sets the shared memory configuration for a device function.
// Parameters
// hfunc
// - kernel to be given a shared memory config
// config
// - requested shared memory configuration
// Returns
// CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT

static VALUE rb_cuFuncSetSharedMemConfig(VALUE self, VALUE hfunc_val, VALUE  config){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result =  cuFuncSetSharedMemConfig(hfunc->function, rb_cu_shared_config_from_rbsymbol(config));
  return Qtrue;
}

// IMPORTANT

// CUresult cuLaunchKernel ( CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ,
//                             unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ,
//                             unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams, void** extra )
// Launches a CUDA function.
// Parameters
// f
// - Kernel to launch
// gridDimX
// - Width of grid in blocks
// gridDimY
// - Height of grid in blocks
// gridDimZ
// - Depth of grid in blocks
// blockDimX
// - X dimension of each thread block
// blockDimY
// - Y dimension of each thread block
// blockDimZ
// - Z dimension of each thread block
// sharedMemBytes
// - Dynamic shared-memory size per thread block in bytes
// hStream
// - Stream identifier
// kernelParams
// - Array of pointers to kernel parameters
// extra
// - Extra options
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_IMAGE, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_FAILED,
// CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuLaunchKernel(VALUE self, VALUE f_val, VALUE gridDimX, VALUE gridDimY, VALUE gridDimZ, VALUE blockDimX, VALUE blockDimY, VALUE blockDimZ,
  VALUE sharedMemBytes, VALUE h_stream_val, VALUE kernel_params, VALUE extra){
  function_ptr* f;
  Data_Get_Struct(f_val, function_ptr, f);
  custream_ptr* h_stream;
  // Data_Get_Struct(h_stream_val, custream_ptr, h_stream);

  void* args[3];

  for(size_t i = 0; i < 3; i++){
    dev_ptr* ptr_arr;
    Data_Get_Struct(RARRAY_AREF(kernel_params, i), dev_ptr, ptr_arr);
    args[i] = ptr_arr->carray;
  }

  CUresult result = cuLaunchKernel(
    f->function,
    NUM2UINT(gridDimX),
    NUM2UINT(gridDimY),
    NUM2UINT(gridDimZ),
    NUM2UINT(blockDimX),
    NUM2UINT(blockDimY),
    NUM2UINT(blockDimZ),
    NUM2UINT(sharedMemBytes),
    0,
    args,
    0
  );

  return Qtrue;
}

// CUresult cuFuncSetBlockShape ( CUfunction hfunc, int  x, int  y, int  z )
// Sets the block-dimensions for the function.
// Parameters
// hfunc
// - Kernel to specify dimensions of
// x
// - X dimension
// y
// - Y dimension
// z
// - Z dimension
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuFuncSetBlockShape(VALUE self, VALUE  hfunc_val, VALUE x, VALUE y, VALUE z){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuFuncSetBlockShape(hfunc->function, NUM2INT(x), NUM2INT(y), NUM2INT(z));
  return Qtrue;
}

// CUresult cuFuncSetSharedSize ( CUfunction hfunc, unsigned int  bytes )
// Sets the dynamic shared-memory size for the function.
// Parameters
// hfunc
// - Kernel to specify dynamic shared-memory size for
// bytes
// - Dynamic shared-memory size per thread in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuFuncSetSharedSize(VALUE self, VALUE hfunc_val, VALUE bytes){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuFuncSetSharedSize(hfunc->function, NUM2UINT(bytes));
  return Qtrue;
}

// CUresult cuParamSetSize ( CUfunction hfunc, unsigned int  numbytes )
// Sets the parameter size for the function.
// Parameters
// hfunc
// - Kernel to set parameter size for
// numbytes
// - Size of parameter list in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuParamSetSize(VALUE self, VALUE hfunc_val, VALUE num_bytes){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuParamSetSize(hfunc->function, NUM2UINT(num_bytes));
  return Qtrue;
}

// CUresult cuParamSeti ( CUfunction hfunc, int  offset, unsigned int  value )
// Adds an integer parameter to the function's argument list.
// Parameters
// hfunc
// - Kernel to add parameter to
// offset
// - Offset to add parameter to argument list
// value
// - Value of parameter
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuParamSeti(VALUE self, VALUE hfunc_val, VALUE offset, VALUE param_value){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuParamSeti(hfunc->function, NUM2INT(offset), NUM2UINT(param_value));
  return Qtrue;
}

// CUresult cuParamSetf ( CUfunction hfunc, int  offset, float  value )
// Adds a floating-point parameter to the function's argument list.
// Parameters
// hfunc
// - Kernel to add parameter to
// offset
// - Offset to add parameter to argument list
// value
// - Value of parameter
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuParamSetf(VALUE self, VALUE hfunc_val, VALUE offset, VALUE param_value){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuParamSetf(hfunc->function, NUM2INT(offset), NUM2DBL(param_value));
  return Qtrue;
}

// CUresult cuParamSetv ( CUfunction hfunc, int  offset, void* ptr, unsigned int  numbytes )
// Adds arbitrary data to the function's argument list.
// Parameters
// hfunc
// - Kernel to add data to
// offset
// - Offset to add data to argument list
// ptr
// - Pointer to arbitrary data
// numbytes
// - Size of data to copy in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuParamSetv(VALUE self, VALUE hfunc_val, VALUE offset, VALUE ptr, VALUE num_bytes){
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuParamSetv(hfunc->function, NUM2INT(offset), (void*)ptr, NUM2UINT(num_bytes));
  return Qtrue;
}

// CUresult cuLaunch ( CUfunction f )
// Launches a CUDA function.
// Parameters
// f
// - Kernel to launch
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, CUDA_ERROR_LAUNCH_TIMEOUT,
// CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuLaunch(VALUE self, VALUE func_val){
  function_ptr* func;
  Data_Get_Struct(func_val, function_ptr, func);
  CUresult result = cuLaunch(func->function);
  return Qtrue;
}

// CUresult cuLaunchGrid ( CUfunction f, int  grid_width, int  grid_height )
// Launches a CUDA function.
// Parameters
// f
// - Kernel to launch
// grid_width
// - Width of grid in blocks
// grid_height
// - Height of grid in blocks
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, CUDA_ERROR_LAUNCH_TIMEOUT,
// CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuLaunchGrid(VALUE self, VALUE func_val, VALUE grid_width, VALUE grid_height){
  function_ptr* func;
  Data_Get_Struct(func_val, function_ptr, func);
  CUresult result = cuLaunchGrid(func->function, NUM2INT(grid_width), NUM2INT(grid_height));
  return Qtrue;
}

// CUresult cuLaunchGridAsync ( CUfunction f, int  grid_width, int  grid_height, CUstream hStream )
// Launches a CUDA function.
// Parameters
// f
// - Kernel to launch
// grid_width
// - Width of grid in blocks
// grid_height
// - Height of grid in blocks
// hStream
// - Stream identifier
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT,
// CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_LAUNCH_FAILED, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
// CUDA_ERROR_LAUNCH_TIMEOUT, CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, CUDA_ERROR_SHARED_OBJECT_INIT_FAILED

static VALUE rb_cuLaunchGridAsync(VALUE self, VALUE func_val, VALUE grid_width, VALUE grid_height, VALUE h_stream_val){
  function_ptr* func;
  Data_Get_Struct(func_val, function_ptr, func);
  custream_ptr* h_stream;
  Data_Get_Struct(h_stream_val, custream_ptr, h_stream);
  CUresult result = cuLaunchGridAsync(func->function, NUM2INT(grid_width), NUM2INT(grid_height), h_stream->stream);
  return Qtrue;
}

// CUresult cuParamSetTexRef ( CUfunction hfunc, int  texunit, CUtexref hTexRef )
// Adds a texture-reference to the function's argument list.
// Parameters
// hfunc
// - Kernel to add texture-reference to
// texunit
// - Texture unit (must be CU_PARAM_TR_DEFAULT)
// hTexRef
// - Texture-reference to add to argument list
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

static VALUE rb_cuParamSetTexRef(VALUE self, VALUE hfunc_val, VALUE tex_unit, VALUE tex_ref_val){
  texture_ptr* tex_ref;
  Data_Get_Struct(tex_ref_val, texture_ptr, tex_ref);
  function_ptr* hfunc;
  Data_Get_Struct(hfunc_val, function_ptr, hfunc);
  CUresult result = cuParamSetTexRef(hfunc->function, NUM2INT(tex_unit), tex_ref->texture);
  return Qtrue;
}

// CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize )
// Returns occupancy of a function.
// Parameters
// numBlocks
// - Returned occupancy
// func
// - Kernel for which occupancy is calculated
// blockSize
// - Block size the kernel is intended to be launched with
// dynamicSMemSize
// - Per-block dynamic shared memory usage intended, in bytes
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN

static VALUE rb_cuOccupancyMaxActiveBlocksPerMultiprocessor(VALUE self, VALUE func_val, VALUE block_size, VALUE dynamic_shared_mem_size){
  int num_blocks;
  function_ptr* func;
  Data_Get_Struct(func_val, function_ptr, func);
  CUresult result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, func->function, NUM2INT(block_size), NUM2ULONG(dynamic_shared_mem_size));
  return INT2NUM(num_blocks);
}

// CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ( int* numBlocks, CUfunction func, int  blockSize, size_t dynamicSMemSize, unsigned int  flags )
// Returns occupancy of a function.
// Parameters
// numBlocks
// - Returned occupancy
// func
// - Kernel for which occupancy is calculated
// blockSize
// - Block size the kernel is intended to be launched with
// dynamicSMemSize
// - Per-block dynamic shared memory usage intended, in bytes
// flags
// - Requested behavior for the occupancy calculator
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN

static VALUE rb_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(VALUE self, VALUE func_val, VALUE block_size, VALUE dynamic_shared_mem_size, VALUE flags){
  int num_blocks;
  function_ptr* func;
  Data_Get_Struct(func_val, function_ptr, func);
  CUresult result = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&num_blocks, func->function, NUM2INT(block_size) , NUM2ULONG(dynamic_shared_mem_size), NUM2UINT(flags));
  return Qnil;
}

// CUresult cuOccupancyMaxPotentialBlockSize ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit )
// Suggest a launch configuration with reasonable occupancy.
// Parameters
// minGridSize
// - Returned minimum grid size needed to achieve the maximum occupancy
// blockSize
// - Returned maximum block size that can achieve the maximum occupancy
// func
// - Kernel for which launch configuration is calculated
// blockSizeToDynamicSMemSize
// - A function that calculates how much per-block dynamic shared memory func uses based on the block size
// dynamicSMemSize
// - Dynamic shared memory usage intended, in bytes
// blockSizeLimit
// - The maximum block size func is designed to handle
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN

static VALUE rb_cuOccupancyMaxPotentialBlockSize(VALUE self, VALUE func_val, VALUE block_size_to_dynamic_shared_mem_size, VALUE dynamic_shared_mem_size, VALUE block_size_limit){
  int min_grid_size, block_size;
  function_ptr* func;
  Data_Get_Struct(func_val, function_ptr, func);
  // TODO
  // CUresult result = cuOccupancyMaxPotentialBlockSize (&min_grid_size, &block_size, func->function, CUoccupancyB2DSize block_size_to_dynamic_shared_mem_size, NUM2ULONG(dynamic_shared_mem_size), NUM2INT(block_size_limit));
  return Qnil;
}

// CUresult cuOccupancyMaxPotentialBlockSizeWithFlags ( int* minGridSize, int* blockSize, CUfunction func, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int  blockSizeLimit, unsigned int  flags )
// Suggest a launch configuration with reasonable occupancy.
// Parameters
// minGridSize
// - Returned minimum grid size needed to achieve the maximum occupancy
// blockSize
// - Returned maximum block size that can achieve the maximum occupancy
// func
// - Kernel for which launch configuration is calculated
// blockSizeToDynamicSMemSize
// - A function that calculates how much per-block dynamic shared memory func uses based on the block size
// dynamicSMemSize
// - Dynamic shared memory usage intended, in bytes
// blockSizeLimit
// - The maximum block size func is designed to handle
// flags
// - Options
// Returns
// CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN

static VALUE rb_cuOccupancyMaxPotentialBlockSizeWithFlags(VALUE self, VALUE func_val, VALUE block_size_to_dynamic_shared_mem_size, VALUE dynamic_shared_mem_size, VALUE block_size_limit, VALUE flags){
  int min_grid_size, block_size;
  function_ptr* func;
  Data_Get_Struct(func_val, function_ptr, func);
  // CUresult result = cuOccupancyMaxPotentialBlockSizeWithFlags(&min_grid_size, &block_size, func->function, CUoccupancyB2DSize block_size_to_dynamic_shared_mem_size, NUM2ULONG(dynamic_shared_mem_size), NUM2INT(block_size_limit), NUM2INT(Flags));
  return Qnil;
}

static VALUE rb_cuTexRefSetArray(VALUE self){
  // CUresult result = cuTexRefSetArray (CUtexref hTexRef, CUarray hArray, uint Flags);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmappedArray(VALUE self){
  // CUresult result = cuTexRefSetMipmappedArray (CUtexref hTexRef, CUmipmappedArray hMipmappedArray, uint Flags);
  return Qnil;
}

static VALUE rb_cuTexRefSetAddress_v2(VALUE self){
  // CUresult result = cuTexRefSetAddress_v2 (size_t* ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes);
  return Qnil;
}

static VALUE rb_cuTexRefSetAddress2D_v3(VALUE self){
  // CUresult result = cuTexRefSetAddress2D_v3 (CUtexref hTexRef, const(CUDA_ARRAY_DESCRIPTOR)* desc, CUdeviceptr dptr, size_t Pitch);
  return Qnil;
}

static VALUE rb_cuTexRefSetFormat(VALUE self){
  // CUresult result = cuTexRefSetFormat (CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents);
  return Qnil;
}

static VALUE rb_cuTexRefSetAddressMode(VALUE self){
  // CUresult result = cuTexRefSetAddressMode (CUtexref hTexRef, int dim, CUaddress_mode am);
  return Qnil;
}

static VALUE rb_cuTexRefSetFilterMode(VALUE self){
  // CUresult result = cuTexRefSetFilterMode (CUtexref hTexRef, CUfilter_mode fm);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapFilterMode(VALUE self){
  // CUresult result = cuTexRefSetMipmapFilterMode (CUtexref hTexRef, CUfilter_mode fm);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapLevelBias(VALUE self){
  // CUresult result = cuTexRefSetMipmapLevelBias (CUtexref hTexRef, float bias);
  return Qnil;
}

static VALUE rb_cuTexRefSetMipmapLevelClamp(VALUE self){
  // CUresult result = cuTexRefSetMipmapLevelClamp (CUtexref hTexRef, float minMipmapLevelClamp, float maxMipmapLevelClamp);
  return Qnil;
}

static VALUE rb_cuTexRefSetMaxAnisotropy(VALUE self){
  // CUresult result = cuTexRefSetMaxAnisotropy (CUtexref hTexRef, uint maxAniso);
  return Qnil;
}

static VALUE rb_cuTexRefSetFlags(VALUE self){
  // CUresult result = cuTexRefSetFlags (CUtexref hTexRef, uint Flags);
  return Qnil;
}

static VALUE rb_cuTexRefGetAddress_v2(VALUE self){
  // CUresult result = cuTexRefGetAddress_v2 (CUdeviceptr* pdptr, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetArray(VALUE self){
  // CUresult result = cuTexRefGetArray (CUarray* phArray, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmappedArray(VALUE self){
  // CUresult result = cuTexRefGetMipmappedArray (CUmipmappedArray* phMipmappedArray, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetAddressMode(VALUE self){
  // CUresult result = cuTexRefGetAddressMode (CUaddress_mode* pam, CUtexref hTexRef, int dim);
  return Qnil;
}

static VALUE rb_cuTexRefGetFilterMode(VALUE self){
  // CUresult result = cuTexRefGetFilterMode (CUfilter_mode* pfm, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetFormat(VALUE self){
  // CUresult result = cuTexRefGetFormat (CUarray_format* pFormat, int* pNumChannels, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapFilterMode(VALUE self){
  // CUresult result = cuTexRefGetMipmapFilterMode (CUfilter_mode* pfm, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapLevelBias(VALUE self){
  // CUresult result = cuTexRefGetMipmapLevelBias (float* pbias, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMipmapLevelClamp(VALUE self){
  // CUresult result = cuTexRefGetMipmapLevelClamp (float* pminMipmapLevelClamp, float* pmaxMipmapLevelClamp, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetMaxAnisotropy(VALUE self){
  // CUresult result = cuTexRefGetMaxAnisotropy (int* pmaxAniso, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefGetFlags(VALUE self){
  // CUresult result = cuTexRefGetFlags (uint* pFlags, CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefCreate(VALUE self){
  // CUresult result = cuTexRefCreate (CUtexref* pTexRef);
  return Qnil;
}

static VALUE rb_cuTexRefDestroy(VALUE self){
  // CUresult result = cuTexRefDestroy (CUtexref hTexRef);
  return Qnil;
}

static VALUE rb_cuSurfRefSetArray(VALUE self){
  // CUresult result = cuSurfRefSetArray (CUsurfref hSurfRef, CUarray hArray, uint Flags);
  return Qnil;
}

static VALUE rb_cuSurfRefGetArray(VALUE self){
  // CUresult result = cuSurfRefGetArray (CUarray* phArray, CUsurfref hSurfRef);
  return Qnil;
}

static VALUE rb_cuTexObjectCreate(VALUE self){
  // CUresult result = cuTexObjectCreate (CUtexObject* pTexObject, const(CUDA_RESOURCE_DESC)* pResDesc, const(CUDA_TEXTURE_DESC)* pTexDesc, const(CUDA_RESOURCE_VIEW_DESC)* pResViewDesc);
  return Qnil;
}

static VALUE rb_cuTexObjectDestroy(VALUE self){
  // CUresult result = cuTexObjectDestroy (CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuTexObjectGetResourceDesc(VALUE self){
  // CUresult result = cuTexObjectGetResourceDesc (CUDA_RESOURCE_DESC* pResDesc, CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuTexObjectGetTextureDesc(VALUE self){
  // CUresult result = cuTexObjectGetTextureDesc (CUDA_TEXTURE_DESC* pTexDesc, CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuTexObjectGetResourceViewDesc(VALUE self){
  // CUresult result = cuTexObjectGetResourceViewDesc (CUDA_RESOURCE_VIEW_DESC* pResViewDesc, CUtexObject texObject);
  return Qnil;
}

static VALUE rb_cuSurfObjectCreate(VALUE self){
  // CUresult result = cuSurfObjectCreate (CUsurfObject* pSurfObject, const(CUDA_RESOURCE_DESC)* pResDesc);
  return Qnil;
}

static VALUE rb_cuSurfObjectDestroy(VALUE self){
  // CUresult result = cuSurfObjectDestroy (CUsurfObject surfObject);
  return Qnil;
}

static VALUE rb_cuSurfObjectGetResourceDesc(VALUE self){
  // CUresult result = cuSurfObjectGetResourceDesc (CUDA_RESOURCE_DESC* pResDesc, CUsurfObject surfObject);
  return Qnil;
}

static VALUE rb_cuDeviceCanAccessPeer(VALUE self){
  // CUresult result = cuDeviceCanAccessPeer (int* canAccessPeer, CUdevice dev, CUdevice peerDev);
  return Qnil;
}

static VALUE rb_cuCtxEnablePeerAccess(VALUE self){
  // CUresult result = cuCtxEnablePeerAccess (CUcontext peerContext, uint Flags);
  return Qnil;
}

static VALUE rb_cuCtxDisablePeerAccess(VALUE self){
  // CUresult result = cuCtxDisablePeerAccess (CUcontext peerContext);
  return Qnil;
}

static VALUE rb_cuGraphicsUnregisterResource(VALUE self){
  // CUresult result = cuGraphicsUnregisterResource (CUgraphicsResource resource);
  return Qnil;
}

static VALUE rb_cuGraphicsSubResourceGetMappedArray(VALUE self){
  // CUresult result = cuGraphicsSubResourceGetMappedArray (CUarray* pArray, CUgraphicsResource resource, uint arrayIndex, uint mipLevel);
  return Qnil;
}

static VALUE rb_cuGraphicsResourceGetMappedMipmappedArray(VALUE self){
  // CUresult result = cuGraphicsResourceGetMappedMipmappedArray (CUmipmappedArray* pMipmappedArray, CUgraphicsResource resource);
  return Qnil;
}

static VALUE rb_cuGraphicsResourceGetMappedPointer_v2(VALUE self){
  // CUresult result = cuGraphicsResourceGetMappedPointer_v2 (CUdeviceptr* pDevPtr, size_t* pSize, CUgraphicsResource resource);
  return Qnil;
}

static VALUE rb_cuGraphicsResourceSetMapFlags_v2(VALUE self){
  // CUresult result = cuGraphicsResourceSetMapFlags_v2 (CUgraphicsResource resource, uint flags);
  return Qnil;
}


static VALUE rb_cuGraphicsMapResources(VALUE self){
  // CUresult result = cuGraphicsMapResources (uint count, CUgraphicsResource* resources, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuGraphicsUnmapResources(VALUE self){
  // CUresult result = cuGraphicsUnmapResources (uint count, CUgraphicsResource* resources, CUstream hStream);
  return Qnil;
}

static VALUE rb_cuGetExportTable(VALUE self){
  // CUresult result = cuGetExportTable (const(void*)* ppExportTable, const(CUuuid)* pExportTableId);
  return Qnil;
}
