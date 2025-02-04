Producer-Consumer Synchronization using Full and Empty LDS Barriers

barrier.cpp contains barrier code + test producer-consumer code

Barrier:
 Barrier:
 - Synchronization is done using arrive/wait.
 - Barriers are initialized to an expected-count of wave arrivals
 - State: phase(binary) and arrival-count are used to synchronize waves.

 Arrive: Decrement arrival-count for each wave arrival.
         when arrival-count becomes 0, barrier phase is flipped
         
 Wait:   Waiting waves maintain a local copy of the barrier phase and
         spin-wait until the phase is flipped by ariving waves.
         Waiting waves then flip the local copy of the barrier phase

Build
-----

hipcc -O3 -gline-tables-only -save-temps --offload-arch=gfx942 barrier.cpp -o barier

Run (with multiple of 512 inputs)
-----------
./barrier 512
Pass

./barrier 1024
Pass

./barrier 1536
Pass
