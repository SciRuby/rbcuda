require_relative 'mkmf.rb'

extension_name = 'rbcuda'

LIBDIR      = RbConfig::CONFIG['libdir']
INCLUDEDIR  = RbConfig::CONFIG['includedir']

HEADER_DIRS = [
  '/opt/local/include',
  '/usr/local/include',
  INCLUDEDIR,
  '/usr/include',
]

LIB_DIRS = [
  '/opt/local/lib',
  '/usr/local/lib',
  LIBDIR,
  '/usr/lib',
]

dir_config(extension_name, HEADER_DIRS, LIB_DIRS)

create_conf_h("rbcuda_config.h")
create_makefile(extension_name)
