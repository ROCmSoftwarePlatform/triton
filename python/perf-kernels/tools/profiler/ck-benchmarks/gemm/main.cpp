#include "CLI/CLI.hpp"
#include "testcase.hpp"
#include <iostream>
#include <random>

#define checkHIPErrors(err) __checkHIPErrors(err, __FILE__, __LINE__)
void __checkHIPErrors(hipError_t err, const char *file, const int line) {
  if (hipSuccess != err) {
    const char *errorStr = hipGetErrorString(err);

    std::cout << "checkHIPErrors() Driver API error = " << err << "\""
              << errorStr << "\""
              << " from file <" << file << "> line " << line << std::endl;
    throw std::runtime_error("failed to process a hip command");
  }
}

void init(std::vector<real> &mat, size_t dim0, size_t dim1) {
  std::random_device randomeDev;
  std::default_random_engine randomeEngine(randomeDev());
  std::uniform_real_distribution<float> uniformDist(-5.0, 5.0);

  std::array<float, 256> randomNumbers;
  for (size_t i = 0; i < randomNumbers.size(); ++i) {
    randomNumbers[i] = static_cast<real>(uniformDist(randomeDev));
  }
  static size_t startIndex = 0;
  startIndex += 4;
  startIndex = startIndex > randomNumbers.size() ? 0 : startIndex;

#pragma omp paralle for collapse(2)
  for (size_t j = 0; j < dim0; ++j) {
    for (size_t i = 0; i < dim1; ++i) {
      const size_t index = j * dim1 + i;
      const size_t randomNumberIndex =
          (startIndex + index) % randomNumbers.size();
      mat[index] = randomNumbers[randomNumberIndex];
    }
  }
}

void run(const TestCase::Config &config) {
  const size_t sizeA = config.m * config.k;
  const size_t sizeB = config.k * config.n;
  const size_t sizeC = config.m * config.n;

  std::vector<real> hostA(sizeA);
  std::vector<real> hostB(sizeB);
  std::vector<real> hostC(sizeC);

  init(hostA, config.m, config.k);
  init(hostB, config.k, config.n);
  init(hostC, config.m, config.n);

  real *devA{nullptr};
  real *devB{nullptr};
  real *devC{nullptr};

  checkHIPErrors(hipMalloc((void **)&devA, sizeA * sizeof(real)));
  checkHIPErrors(hipMalloc((void **)&devB, sizeB * sizeof(real)));
  checkHIPErrors(hipMalloc((void **)&devC, sizeC * sizeof(real)));

  checkHIPErrors(hipMemcpy(devA, hostA.data(), sizeA * sizeof(real),
                           hipMemcpyKind::hipMemcpyHostToDevice));
  checkHIPErrors(hipMemcpy(devB, hostB.data(), sizeB * sizeof(real),
                           hipMemcpyKind::hipMemcpyHostToDevice));
  checkHIPErrors(hipMemcpy(devC, hostC.data(), sizeC * sizeof(real),
                           hipMemcpyKind::hipMemcpyHostToDevice));

  TestCase::launchKernel(devA, devB, devC, config);

  checkHIPErrors(hipFree(devA));
  checkHIPErrors(hipFree(devB));
  checkHIPErrors(hipFree(devC));
}

int main(int argc, char *argv[]) {
  CLI::App app{"ck gemm examples"};
  TestCase::Config config{};

  app.add_option("-m", config.m, "M size");
  app.add_option("-n", config.n, "N size");
  app.add_option("-k", config.k, "K size");
  app.add_option("--kbatch", config.kbatch, "kbatch (for split-k)");
  app.add_flag("--trans-a", config.transA, "transpose A");
  app.add_flag("--trans-b", config.transB, "transpose B");
  app.add_option("--log-level", config.logLevel, "CK's log level");
  app.add_option("--cold-num-iters", config.coldNumIters,
                 "num cold iterations");
  app.add_option("--num-repeat", config.numRepeat, "num repeats");
  app.add_option("--rotating-count", config.rotatingCount, "rotating count");
  app.add_flag("--flush-cache", config.flushCache, "flush cache");
  CLI11_PARSE(app, argc, argv);

  run(config);
  return 0;
}
