VALUE RbCUDA = Qnil;

VALUE CuBLAS = Qnil;

static VALUE rbcu_hello(VALUE self);

// prototypes
void Init_rbcuda();

void Init_rbcuda() {
  RbCUDA = rb_define_module("RbCUDA");


  CuBLAS = rb_define_module_under(RbCUDA, "CuBLAS");
  rb_define_singleton_method(CuBLAS, "hello",(METHOD)rbcu_hello,0);
}

static VALUE rbcu_hello(VALUE self){
  printf("Hello World!\n");
  return Qnil;
}
