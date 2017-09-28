std::map<char*, size_t> CuRand_Status = {
  {"CURAND_STATUS_SUCCESS", 0}, ///< No errors
  {"CURAND_STATUS_VERSION_MISMATCH", 100}, ///< Header file and linked library version do not match
  {"CURAND_STATUS_NOT_INITIALIZED", 101}, ///< Generator not initialized
  {"CURAND_STATUS_ALLOCATION_FAILED", 102}, ///< Memory allocation failed
  {"CURAND_STATUS_TYPE_ERROR", 103}, ///< Generator is wrong type
  {"CURAND_STATUS_OUT_OF_RANGE", 104}, ///< Argument out of range
  {"CURAND_STATUS_LENGTH_NOT_MULTIPLE", 105}, ///< Length requested is not a multple of dimension
  {"CURAND_STATUS_DOUBLE_PRECISION_REQUIRED", 106}, ///< GPU does not have double precision required by MRG32k3a
  {"CURAND_STATUS_LAUNCH_FAILURE", 201}, ///< Kernel launch failure
  {"CURAND_STATUS_PREEXISTING_FAILURE", 202}, ///< Preexisting failure on library entry
  {"CURAND_STATUS_INITIALIZATION_FAILED", 203}, ///< Initialization of CUDA failed
  {"CURAND_STATUS_ARCH_MISMATCH", 204}, ///< Architecture mismatch, GPU does not support requested feature
  {"CURAND_STATUS_INTERNAL_ERROR", 999} ///< Internal library error
};

// alias curandStatus_t = curandStatus;

std::map<char*, size_t> CuRand_RngType = {
  {"CURAND_RNG_TEST", 0},
  {"CURAND_RNG_PSEUDO_DEFAULT", 100}, ///< Default pseudorandom generator
  {"CURAND_RNG_PSEUDO_XORWOW", 101}, ///< XORWOW pseudorandom generator
  {"CURAND_RNG_PSEUDO_MRG32K3A", 121}, ///< MRG32k3a pseudorandom generator
  {"CURAND_RNG_PSEUDO_MTGP32", 141}, ///< Mersenne Twister MTGP32 pseudorandom generator
  {"CURAND_RNG_PSEUDO_MT19937", 142}, ///< Mersenne Twister MT19937 pseudorandom generator
  {"CURAND_RNG_PSEUDO_PHILOX4_32_10", 161}, ///< PHILOX-4x32-10 pseudorandom generator
  {"CURAND_RNG_QUASI_DEFAULT", 200}, ///< Default quasirandom generator
  {"CURAND_RNG_QUASI_SOBOL32", 201}, ///< Sobol32 quasirandom generator
  {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL32", 202}, ///< Scrambled Sobol32 quasirandom generator
  {"CURAND_RNG_QUASI_SOBOL64", 203}, ///< Sobol64 quasirandom generator
  {"CURAND_RNG_QUASI_SCRAMBLED_SOBOL64", 204} ///< Scrambled Sobol64 quasirandom generator
};

// alias curandRngType_t = curandRngType;

std::map<char*, size_t> CuRand_Ordering = {
  {"CURAND_ORDERING_PSEUDO_BEST", 100}, ///< Best ordering for pseudorandom results
  {"CURAND_ORDERING_PSEUDO_DEFAULT", 101}, ///< Specific default 4096 thread sequence for pseudorandom results
  {"CURAND_ORDERING_PSEUDO_SEEDED", 102}, ///< Specific seeding pattern for fast lower quality pseudorandom results
  {"CURAND_ORDERING_QUASI_DEFAULT", 201} ///< Specific n-dimensional ordering for quasirandom results
};

// /*
//  * CURAND ordering of results in memory
//  */
// /** \cond UNHIDE_TYPEDEFS */
// alias curandOrdering_t = curandOrdering;

std::map<char*, size_t> CuRand_DirectionVectorSet = {
  {"CURAND_DIRECTION_VECTORS_32_JOEKUO6", 101}, ///< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
  {"CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6", 102}, ///< Specific set of 32-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
  {"CURAND_DIRECTION_VECTORS_64_JOEKUO6", 103}, ///< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
  {"CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6", 104} ///< Specific set of 64-bit direction vectors generated from polynomials recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
};

// alias curandDirectionVectorSet_t = curandDirectionVectorSet;

// alias curandDirectionVectors32_t = uint[32];

// alias curandDirectionVectors64_t = ulong[64];

// struct curandGenerator_st;

// alias curandGenerator_t = curandGenerator_st*;

// alias curandDistribution_st = double;
// alias curandDistribution_t = double*;
// struct curandDistributionShift_st;
// alias curandDistributionShift_t = curandDistributionShift_st*;

// struct curandDistributionM2Shift_st;
// alias curandDistributionM2Shift_t = curandDistributionM2Shift_st*;
// struct curandHistogramM2_st;
// alias curandHistogramM2_t = curandHistogramM2_st*;
// alias curandHistogramM2K_st = uint;
// alias curandHistogramM2K_t = uint*;
// alias curandHistogramM2V_st = double;
// alias curandHistogramM2V_t = double*;

// struct curandDiscreteDistribution_st;
// alias curandDiscreteDistribution_t = curandDiscreteDistribution_st*;

std::map<char*, size_t> CuRand_Method = {
  {"CURAND_CHOOSE_BEST", 0}, // choose best depends on args
  {"CURAND_ITR", 1},
  {"CURAND_KNUTH", 2},
  {"CURAND_HITR", 3},
  {"CURAND_M1", 4},
  {"CURAND_M2", 5},
  {"CURAND_BINARY_SEARCH", 6},
  {"CURAND_DISCRETE_GAUSS", 7},
  {"CURAND_REJECTION", 8},
  {"CURAND_DEVICE_API", 9},
  {"CURAND_FAST_REJECTION", 10},
  {"CURAND_3RD", 11},
  {"CURAND_DEFINITION", 12},
  {"CURAND_POISSON", 13}
};

// alias curandMethod_t = curandMethod;

static VALUE rb_curandCreateGenerator(VALUE self){
    return Qnil;
}

static VALUE rb_curandCreateGeneratorHost(VALUE self){
    return Qnil;
}

static VALUE rb_curandDestroyGenerator(VALUE self){
    return Qnil;
}

static VALUE rb_curandGetVersion(VALUE self){
    return Qnil;
}

static VALUE rb_curandSetStream(VALUE self){
    return Qnil;
}

static VALUE rb_curandSetPseudoRandomGeneratorSeed(VALUE self){
    return Qnil;
}

static VALUE rb_curandSetGeneratorOffset(VALUE self){
    return Qnil;
}

static VALUE rb_curandSetGeneratorOrdering(VALUE self){
    return Qnil;
}

static VALUE rb_curandSetQuasiRandomGeneratorDimensions(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerate(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateLongLong(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateUniform(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateUniformDouble(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateNormal(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateNormalDouble(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateLogNormal(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateLogNormalDouble(VALUE self){
    return Qnil;
}

static VALUE rb_curandCreatePoissonDistribution(VALUE self){
    return Qnil;
}

static VALUE rb_curandDestroyDistribution(VALUE self){
    return Qnil;
}

static VALUE rb_curandGeneratePoisson(VALUE self){
    return Qnil;
}

static VALUE rb_curandGeneratePoissonMethod(VALUE self){
    return Qnil;
}

static VALUE rb_curandGenerateSeeds(VALUE self){
    return Qnil;
}

static VALUE rb_curandGetDirectionVectors32(VALUE self){
    return Qnil;
}

static VALUE rb_curandGetScrambleConstants32(VALUE self){
    return Qnil;
}

static VALUE rb_curandGetDirectionVectors64(VALUE self){
    return Qnil;
}

static VALUE rb_curandGetScrambleConstants64(VALUE self){
    return Qnil;
}
