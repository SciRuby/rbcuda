require_relative 'mkmf.rb'

extension_name = 'rbcuda'

nmatrix_path = Gem::Specification.find_all_by_name('nmatrix').compact
abort "Cannot locate NMatrix installation" unless nmatrix_path
nmatrix_header_dir = File.join(nmatrix_path[0].require_path)

narray_path = Gem::Specification.find_by_path('narray').full_gem_path
abort "Cannot locate NArray installation" unless narray_path
narray_header_dir = File.join(narray_path)

$INSTALLFILES = [
  ['rbcuda.h'  , '$(archdir)'],
  ['rbcuda_config.h', '$(archdir)'],
]

$DEBUG = true
$CFLAGS = ["-w -Wall -Werror=return-type",$CFLAGS].join(" ")
$CXXFLAGS = ["-w -Wall -Werror=return-type",$CXXFLAGS].join(" ")
$CPPFLAGS = ["-w -Wall -Werror=return-type",$CPPFLAGS].join(" ")

LIBDIR      = RbConfig::CONFIG['libdir']
INCLUDEDIR  = RbConfig::CONFIG['includedir']

HEADER_DIRS = [
  '/opt/local/include',
  '/usr/local/include',
  INCLUDEDIR,
  '/usr/include',
  nmatrix_header_dir,
  narray_header_dir
] + (ENV['CPATH'] || '').split(':')

LIB_DIRS = [
  '/opt/local/lib',
  '/usr/local/lib',
  LIBDIR,
  '/usr/lib',
] + (ENV['LIBRARY_PATH'] || '').split(':')

dir_config(extension_name, HEADER_DIRS, LIB_DIRS)

have_library('cuda')
have_library('cudart')
have_library('cublas')
have_library('cusolver')
have_library('curand')
have_header("nmatrix_config.h")
have_header("narray.h")
abort "Cannot locate NMatrix header files : nmatrix.h" unless find_header("nmatrix.h")


basenames = %w{rbcuda}
$objs = basenames.map { |b| "#{b}.o"   }
$srcs = basenames.map { |b| "#{b}.cpp" }

create_conf_h("rbcuda_config.h")
create_makefile(extension_name)
