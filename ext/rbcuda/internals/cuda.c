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
  CUdevice_attribute attrib = rb_cu_get_attrib_from_rbsymbol(attrib_val);
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
  CUresult result = cuDeviceGetProperties(prop, dev);
  return ULONG2NUM(dev);
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
  ctx_ptr* pctx = ALLOC(ctx_ptr);
  CUdevice dev = NUM2ULONG(device_val);
  CUresult result = cuDevicePrimaryCtxRetain(&pctx->ctx, dev);
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
  CUdevice dev = NUM2ULONG(device_val);
  CUresult result = cuDevicePrimaryCtxRelease(dev);
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
  CUresult result = cuDevicePrimaryCtxSetFlags(dev, NUM2UINT(flags));
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
  CUresult result = cuDevicePrimaryCtxGetState(dev, &flags, &active);
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
  CUresult result = cuDevicePrimaryCtxReset(dev);
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
  CUdevice dev = NUM2ULONG(device_val);
  ctx_ptr* pctx = ALLOC(ctx_ptr);
  CUresult result = cuCtxCreate_v2 (&pctx->ctx, UINT2NUM(flags), dev);
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
  CUdevice device;
  CUresult result = cuCtxGetDevice(&device);
  return ULONG2NUM(device);
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
