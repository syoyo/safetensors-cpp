// NOTE: If you compile example-c in header-only mode(Makefile define `***_IMPLEMENTATION`), this example-c.c must be compiled with C++ compiler.
// If you compile example-c by linking libsafetensors-c(CMake build. `***_NO_IMPLEMENTATION`), you can use C compile only.
#if !defined(SAFETENSORS_C_NO_IMPLEMENTATION)
#define SAFETENSORS_C_IMPLEMENTATION
#endif
#include "safetensors-c.h"

#include <stdio.h>

int main(int argc, char **argv) {

  const char *filename = "test/safetensors_abuse_attempt_1.safetensors";

  if (argc > 1) {
    filename = argv[1];
  }

  char *warn = NULL;
  char *err = NULL;
  safetensors_c_safetensors_t safetensors;
  safetensors_c_status_t status = safetensors_c_load_from_file(filename, &safetensors, &warn, &err);

  if (warn) {
    free(warn);
  }

  if (err) {
    free(err);
  }

  return EXIT_SUCCESS;
}

