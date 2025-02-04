// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <hip/hip_runtime.h>
#include <cstdio>

// Producer-Consumer Synchronization using Full and Empty LDS Barriers
// 512 Producer threads write to a shared buffer in LDS and 512 Consumer threads
// accumulate values from the shared buffer, over N/512 iterations.
// Barrier:
// - Synchronization is done using arrive/wait.
// - Barriers are initialized to an expected-count of wave arrivals
// - State: phase(binary) and arrival-count are used to synchronize waves.
//
// Arrive: Decrement arrival-count for each wave arrival.
//         when arrival-count becomes 0, barrier phase is flipped
// Wait:   Waiting waves maintain a local copy of the barrier phase and
//         spin-wait until the phase is flipped by ariving waves.
//         Waiting waves then flip the local copy of the barrier phase

const int BLOCK_SIZE = 1024;

__launch_bounds__(BLOCK_SIZE) __global__
    void kernel(float* a, int N, float* result) {
  using LFP = __attribute__((address_space(3))) float*;
  using LF = __attribute__((address_space(3))) float;
  using LU32 = __attribute__((address_space(3))) uint32_t;

  // Barrier state variables
  // Full barrier
  __shared__ volatile LU32 bufferFullCount;
  __shared__ volatile LU32 fullPhase;
  // Empty barrier
  __shared__ volatile LU32 bufferEmptyCount;
  __shared__ volatile LU32 emptyPhase;

  int numProducers = BLOCK_SIZE / 2;
  int numConsumers = numProducers;

  // buffer in LDS that is shared between producers and consumers
  __shared__ LF data[BLOCK_SIZE / 2];
  // consumer threads accumulate values from data[] into acc
  __shared__ volatile LF acc;

  int threadId = threadIdx.x;
  int iters = N / numProducers;
  const uint32_t threadsPerWave = 64;
  const uint32_t ONE = 1;

  

  // Barrier init
  // Set barrier expectedCount to be 1 less than the number of waves
  // The ds_dec_rtn_u32 op will set the barrier count to expectedCount when
  // pre-decrement expectedCount is 0
  const uint32_t expectedProducerCount = (numProducers / threadsPerWave) - 1;
  const uint32_t expectedConsumerCount = expectedProducerCount;
  if (threadId == 0) {
    // Measure the cost of ds_dec_rtn_u32 + wait
    // (~ 60 cycles)
    // temp1 = 5;
    // uint32_t temp2;
    // long start1 = clock64();
    // asm volatile("ds_dec_rtn_u32 %0, %1 %2\n"
    //              : "=v"(temp2)
    //              : "v"(&temp1), "v"(5)
    //              : "memory");
    // asm volatile("s_waitcnt lgkmcnt(0) \n" ::: "memory");
    // long x = clock64() - start1;
    // printf("dec cost = %d %ld\n", temp2, x);

    bufferFullCount = expectedProducerCount;
    bufferEmptyCount = expectedConsumerCount;
    acc = 0.0;
    emptyPhase = 0;
    fullPhase = 0;
  }
  long updatecost1 = 0;
  long updatecost2 = 0;
  long workloadtime = 0;

  __syncthreads();

  if (threadId < numProducers) {
    // Producer does not wait for consumer for the first iteration
    int producerCurrentEmptyPhase = 1;
    for (int i = 0; i < iters; i++) {
      //========Producer wait() ===========
      while (producerCurrentEmptyPhase == emptyPhase) {
        asm volatile("s_sleep 10 \n" ::);
      }
      asm volatile("s_wakeup \n" ::);
      producerCurrentEmptyPhase = producerCurrentEmptyPhase ^ 1;
      //=========Producer wait() end =========

      // Producer Workload
      //(~720 cycles)
      // global -> lds
      data[threadId] = a[i * numProducers + threadId];
      // Producer Workload end

      //==========Producer arrive()=============
      // (~140 cycles total)
      // LDS countdown from expectedProducerCount -> -1
      // Note: ds_dec op will set bufferFullCount to expectedProducerCount
      // when pre-decrement bufferFullCount is 0
      uint32_t preDecrement;
      // One thread per wave does the countdown
      if (threadId % threadsPerWave == 0) {
        // (dec + s_wait = ~60 cycles)
        asm volatile("ds_dec_rtn_u32 %0, %1 %2\n"
                     : "=v"(preDecrement)
                     : "v"(&bufferFullCount), "v"(expectedProducerCount)
                     : "memory");
        // Ensure that the bufferFullCount LDS dec op has completed
        // before using the return value (preDecrement).
        asm volatile("s_waitcnt lgkmcnt(0) \n" ::: "memory");
        // Flip the phase when the last wave arrives
        if (preDecrement == 0) {
          // (~ 24 cycles)
          asm volatile("ds_xor_b32 %0, %1 \n" ::"v"(&fullPhase), "v"(ONE));
        }
      }
      //==========Producer arrive() end=============
    }
  } else {
    int consumerCurrentFullPhase = 0;
    for (int i = 0; i < iters; i++) {
      //============Consumer wait()============
      while (true) {
        if (consumerCurrentFullPhase ^ fullPhase) {
          // wake up any waves that are sleeping in the workgroup
          asm volatile("s_wakeup \n" ::);
          break;
        }
        asm volatile("s_sleep 10 \n" ::);
      }
      consumerCurrentFullPhase = consumerCurrentFullPhase ^ 1;
      //============Consumer wait() end============

      // Consumer workload
      // read lds value and accumulate into an lds location (acc)
      float increment = data[(threadId - numProducers)];
      __builtin_amdgcn_ds_faddf((LFP)&acc, increment, 0, 0, false);
      // Consumer workload end

      //============Consumer arrive()============
      // LDS countdown from expectedConsumerCount -> -1
      // Note: ds_dec op will set bufferEmptyCount to expectedConsumerCount
      // when pre-decrement bufferEmptyCount is 0
      uint32_t preDecrement;
      // One thread per wave does the countdown
      if (threadId % threadsPerWave == 0) {
        // (dec + s_wait = ~60 cycles)
        asm volatile("ds_dec_rtn_u32 %0, %1 %2\n"
                     : "=v"(preDecrement)
                     : "v"(&bufferEmptyCount), "v"(expectedConsumerCount)
                     : "memory");
        // Ensure that the bufferFullCount LDS dec op has completed
        // before using the return value (preDecrement).
        asm volatile("s_waitcnt lgkmcnt(0) \n" ::: "memory");
        if (preDecrement == 0) {
          // (~ 24 cycles)
          asm volatile("ds_xor_b32 %0, %1 \n" ::"v"(&emptyPhase), "v"(ONE));
        }
      }
      //============Consumer arrive() end============
    }
  }
  // Write out the workload result
  __syncthreads();
  if (threadId == BLOCK_SIZE - 1) {
    *result = acc;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Invalid args. Usage: ./barrier <N> , where N mod 512 == 0\n");
    exit(-2);
  }
  int N = atoi(argv[1]); // problem size
  float *a, *result;
  int err;

  for (int i = 0; i < 100; ++i) {
    float expected = 0.0f;
    err = hipMalloc(&a, sizeof(*a) * N);
    err = hipMalloc(&result, sizeof(*result));
    for (int i = 0; i < N; ++i) {
      a[i] = i * 1.0f;
      expected += a[i];
    }
    kernel<<<304, BLOCK_SIZE>>>(a, N, result);
    err = hipDeviceSynchronize();
    err = hipFree(a);
    err = hipFree(result);
    if (*result != expected) {
      printf("Mismatch\n");
      printf("result = %f\n", *result);
      printf("expected = %f\n", expected);
      exit(-1);
    }
  }
  printf("Pass\n");
}
