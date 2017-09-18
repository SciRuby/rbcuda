require "bundler/gem_tasks"
require 'rake/extensiontask'
require "rake/testtask"

Rake::ExtensionTask.new do |ext|
  ext.name = 'rbcuda'
  ext.ext_dir = 'ext/rbcuda/'
  ext.lib_dir = 'lib/'
  ext.source_pattern = '**/*.{c,cpp, h}'
end

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

task :default => :test
