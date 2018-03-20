# coding: utf-8
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "rbcuda/version"

Gem::Specification.new do |spec|
  spec.name          = "rbcuda"
  spec.version       = RbCUDA::VERSION
  spec.authors       = ["Prasun Anand"]
  spec.email         = ["prasunanand.bitsp@gmail.com"]

  spec.summary       = %q{Ruby bindings for CUDA APIs}
  spec.description   = %q{Ruby bindings for CUDA APIs}
  spec.homepage      = "https://github.com/prasunanand/rbcuda"

  # Prevent pushing this gem to RubyGems.org. To allow pushes either set the 'allowed_push_host'
  # to allow pushing to a single host or delete this section to allow pushing to any host.
  if spec.respond_to?(:metadata)
    spec.metadata["allowed_push_host"] = "TODO: Set to 'http://mygemserver.com'"
  else
    raise "RubyGems 2.0 or newer is required to protect against " \
      "public gem pushes."
  end

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions    = ['ext/rbcuda/extconf.rb']

  spec.add_development_dependency "bundler", "~> 1.15"
  spec.add_development_dependency "rake", "~> 10.0"
  spec.add_development_dependency 'rake-compiler', '~>0.8'
  spec.add_development_dependency "minitest", "~> 5.0"
  spec.add_development_dependency 'pry', '~>0.10'
  spec.add_development_dependency 'nmatrix', '~> 0.2.1'
  spec.add_development_dependency 'narray', '~> 0.6.1.2'
  spec.add_development_dependency 'rdoc', '~>4.0', '>=4.0.1'
end
