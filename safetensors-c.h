// SPDX-License-Identifier: MIT
// Copyright 2023 - Present, Syoyo Fujita.
// C binding to safetensors.hh
#pragma once

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAFETENSORS_C_MAX_DIM (8)

typedef enum safetensors_c_dtype {
  SAFETENSORS_C_BOOL,
  SAFETENSORS_C_UINT8,
  SAFETENSORS_C_INT8,
  SAFETENSORS_C_UINT16,
  SAFETENSORS_C_INT16,
  SAFETENSORS_C_FLOAT16,
  SAFETENSORS_C_BFLOAT16,
  SAFETENSORS_C_UINT32,
  SAFETENSORS_C_INT32,
  SAFETENSORS_C_FLOAT32,
  SAFETENSORS_C_FLOAT64,
  SAFETENSORS_C_UINT64,
  SAFETENSORS_C_INT64,
} safetensors_c_dtype_t;

// TODO: Use error code used in official rust implementation
typedef enum safetensors_c_status {
  SAFETENSORS_C_SUCCESS = 0,
  SAFETENSORS_C_FILE_NOT_FOUND = -1,
  SAFETENSORS_C_FILE_READ_FAILURE = -2,
  SAFETENSORS_C_FILE_WRITE_FAILURE = -3,
  SAFETENSORS_C_CORRUPTED_DATA = -4,
  SAFETENSORS_C_INVALID_SAFETENSORS = -5,
  SAFETENSORS_C_MALLOC_ERROR= -6
} safetensors_c_status_t;

typedef struct safetensors_c_safetensors {
  void *tensors;   // opaque pointer to tensor dict.
  void *metadata;  // opaque pointer to metadata dict.
  int mmaped;
  unsigned char *data;
  size_t nbytes;
} safetensors_c_safetensors_t;

typedef struct safetensors_c_tensor {
  safetensors_c_dtype_t dtype;
  size_t shape[SAFETENSORS_C_MAX_DIM];
  uint32_t ndim;  // up to SAFETENSORS_C_MAX_DIM;

  size_t data_offsets[2];  // [BEGIN, END]

} safetensors_c_tensor_t;

void safetensors_c_init(safetensors_c_safetensors_t *st);

//
// Load safetensors from a file.
//
// @param[in] filename Filename(Assume UTF-8)
// @param[out] warn Warning message when failed. Must free the pointer after
// using it. Pass NULL if you don't need it.
// @param[out] err Error message when failed. Must free the pointer after using
// it. Pass NULL if you don't need it.
// @return `SAFETENSORS_C_SUCCESS` upon success.
//
safetensors_c_status_t safetensors_c_load_from_file(
    const char *filename, safetensors_c_safetensors_t *st, char *const *warn,
    char *const *err);

//
// Load safetensors from a memory
//
// @param[in] addr Buffer address
// @param[in] nbytes Buffer size
// @param[in] filename Filename(Assume UTF-8). Optional. Can be NULL.
// @param[out] warn Warning message when failed. Must free the pointer after
// using it. Pass NULL if you don't need it.
// @param[out] err Error message when failed. Must free the pointer after using
// it. Pass NULL if you don't need it.
// @return `SAFETENSORS_C_SUCCESS` upon success.
//
safetensors_c_status_t safetensors_c_load_from_memory(
    const char *filename, safetensors_c_safetensors_t *st, char *const*warn,
    char *const *err);

// mmap version
// Still need to call `safetensors_c_safetensors_free` API to free JSON data in
// safetensors struct.
//
safetensors_c_status_t safetensors_c_mmap_from_file(
    const char *filename, safetensors_c_safetensors_t *st, char *const*warn,
    char *const*err);
safetensors_c_status_t safetensors_c_mmap_from_memory(
    const char *filename, safetensors_c_safetensors_t *st, char *const*warn,
    char *const*err);

//
// Fee memory of safetensors struct.
// No need to call this API for mmaped safetensors data.
// @return 1 upon success, 0 failed.
//
int safetensors_c_safetensors_free(safetensors_c_safetensors_t *st);

///
/// Check if metadata item with `key` exists.
/// @return 1 if the metadata has item with `key`. 0 not.
int safetensors_c_has_metadata(const safetensors_c_safetensors_t *st,
                               const char *key);

///
/// Get metadata item for `key`. Return NULL when no corresponding metadata item
/// exists for `key`.
///
const char *safetensors_c_get_metadata(const safetensors_c_safetensors_t *st,
                                       const char *key);

// TODO: Write API

#ifdef __cplusplus
}  // extern "C"
#endif

#if defined(SAFETENSORS_C_IMPLEMENTATION)

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "safetensors.hh"

void safetensors_c_init(safetensors_c_safetensors_t *st) {
  if (!st) {
    return;
  }

  st->dtype = SAFETENSORS_C_FLOAT32;
  memset(st->shape, 0, sizeof(size_t) * SAFETENSORS_C_MAX_DIM);
  st->ndim = 0;
  st->mmaped = 0;
  st->data = 0;
  st->nbytes = 0;
}

safetensors_c_status_t safetensors_c_load_from_file(
    const char *filename, safetensors_c_safetensors_t *st, char *const *warn,
    char *const *err) {

  safetensors_c_init(st);

  std::string err;
  const char *err_msg = malloc(err.size());
  if (!err_msg) {
    return SAFETENSORS_C_MALLOC_ERROR;
  }

}

int safetensors_c_free(safetensors_c_safetensors_t *st )
{
  if (!st) {
    return 0;
  }

  if (st->mmaped) {
    return 1;
  }

  if (st->data) {
    free(st->data);
    st->data = NULL;
  }

  st->nbytes = 0;

}

#endif

