#ifndef PEAK_STACK_WORKER_HPP_
#define PEAK_STACK_WORKER_HPP_

#include <cstddef>
#include <cstdint>

// Host-only bookkeeping; none of these bytes are charged to the algorithm.
struct PeakStackSample {
  void (*operation)(void *);
  void *context;
  std::uintptr_t stack_low;
  std::uintptr_t stack_high;
  unsigned char pattern;
  std::size_t written_bytes;
  bool valid;
};

extern void *(*peak_stack_worker_entry)(void *);

#endif
