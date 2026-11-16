// Compile this harness translation unit with -mno-red-zone. Library kernels
// retain their ordinary compilation flags in a separate translation unit.
#include "peak_stack_worker.hpp"

#if !defined(__x86_64__) || !defined(__GNUC__)
#error This stack painter requires the GNU x86-64 Linux ABI.
#endif

static inline __attribute__((noinline)) void *paint_and_measure(void *argument) {
  PeakStackSample *sample = static_cast<PeakStackSample *>(argument);
  std::uintptr_t origin;
  asm volatile("mov %%rsp, %0" : "=r"(origin) : : "memory");
  if (origin <= sample->stack_low || origin >= sample->stack_high) {
    sample->valid = false;
    return nullptr;
  }

  // The painter has no red zone and makes no calls until the target operation.
  // Its live frame is at/above origin. Volatile byte stores prevent memset
  // replacement, so no painter call frame can contaminate the measured region.
  volatile unsigned char *bytes =
      reinterpret_cast<volatile unsigned char *>(sample->stack_low);
  const std::size_t size = origin - sample->stack_low;
  for (std::size_t index = 0; index < size; ++index) {
    bytes[index] = sample->pattern;
  }
  asm volatile("" : : : "memory");
  sample->operation(sample->context);
  asm volatile("" : : : "memory");

  // Scan before making any other call. The operation includes all callees and
  // their red zones; pthread startup/teardown and our own frame are excluded.
  std::size_t first_written = size;
  for (std::size_t index = 0; index < size; ++index) {
    if (bytes[index] != sample->pattern) {
      first_written = index;
      break;
    }
  }
  std::uintptr_t returned;
  asm volatile("mov %%rsp, %0" : "=r"(returned) : : "memory");
  sample->written_bytes = size - first_written;
  sample->valid = returned == origin && first_written != 0;
  return nullptr;
}

void *(*peak_stack_worker_entry)(void *) = &paint_and_measure;
