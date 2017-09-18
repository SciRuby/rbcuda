#include <ruby.h>

VALUE RbCUDA = Qnil;

void Init_rbcuda();

void Init_rbcuda() {
  RbCUDA = rb_define_module("RbCUDA");

}
