# Rbcuda

The main objective of RbCUDA would be:
1. Map all of CUDA into Ruby
2. Ready-made on-GPU linear algebra, reduction, scan using cuBLAS, cuMath, cuSolver libraries.
3. Random Numer generator using cuRand
4. Near-zero wrapping overhead.
5. CUDA profiler for Ruby.

In the near future:
6. fast-fourier transform(cuFFT)
7. Parallel Primitives and Data Structures(Thrust)
8. Image processing (NVIDIA Performance Primitives Library).

## Installation

Add this line to your application's Gemfile:

```ruby
gem 'rbcuda'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install rbcuda

## Usage

TODO: Write usage instructions here

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake test` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and tags, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/[USERNAME]/rbcuda. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## Code of Conduct

Everyone interacting in the Rbcuda project’s codebases, issue trackers, chat rooms and mailing lists is expected to follow the [code of conduct](https://github.com/prasunanand/rbcuda/blob/master/CODE_OF_CONDUCT.md).

# LICENSE

This software is distributed under the [BSD 3-Clause License](LICENSE).

Copyright © 2017, Prasun Anand