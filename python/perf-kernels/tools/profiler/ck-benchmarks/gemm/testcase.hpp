#pragma once

#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"

using real = ck::half_t;

struct TestCase {
  struct Config {
    size_t m{1024};
    size_t n{1024};
    size_t k{1024};
    size_t kbatch{1};
    bool transA{false};
    bool transB{false};
    int logLevel{1};
    int coldNumIters{5};
    int numRepeat{50};
    bool flushCache{false};
    int rotatingCount{1};
  };
  static void launchKernel(real *, real *, real *, const Config &);
};
