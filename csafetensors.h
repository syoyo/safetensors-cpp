/* SPDX-License-Identifier: MIT */
/* Copyright 2023 - Present, Syoyo Fujita. */
/* Pure C11 implementation of safetensors loader */
#ifndef CSAFETENSORS_H_
#define CSAFETENSORS_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum dimension of tensor shape */
#define CSAFETENSORS_MAX_DIM 8

/* Maximum number of tensors */
#define CSAFETENSORS_MAX_TENSORS 65536

/* Maximum number of metadata entries */
#define CSAFETENSORS_MAX_METADATA 1024

/* Maximum key/value string length */
#define CSAFETENSORS_MAX_STRING_LEN 4096

/* Data types */
typedef enum csafetensors_dtype {
    CSAFETENSORS_DTYPE_BOOL = 0,
    CSAFETENSORS_DTYPE_UINT8,
    CSAFETENSORS_DTYPE_INT8,
    CSAFETENSORS_DTYPE_INT16,
    CSAFETENSORS_DTYPE_UINT16,
    CSAFETENSORS_DTYPE_FLOAT16,
    CSAFETENSORS_DTYPE_BFLOAT16,
    CSAFETENSORS_DTYPE_INT32,
    CSAFETENSORS_DTYPE_UINT32,
    CSAFETENSORS_DTYPE_FLOAT32,
    CSAFETENSORS_DTYPE_FLOAT64,
    CSAFETENSORS_DTYPE_INT64,
    CSAFETENSORS_DTYPE_UINT64
} csafetensors_dtype_t;

/* Error codes */
typedef enum csafetensors_error {
    CSAFETENSORS_SUCCESS = 0,
    CSAFETENSORS_ERROR_INVALID_ARGUMENT,
    CSAFETENSORS_ERROR_FILE_NOT_FOUND,
    CSAFETENSORS_ERROR_FILE_READ,
    CSAFETENSORS_ERROR_INVALID_HEADER,
    CSAFETENSORS_ERROR_JSON_PARSE,
    CSAFETENSORS_ERROR_MEMORY_ALLOCATION,
    CSAFETENSORS_ERROR_MMAP_FAILED,
    CSAFETENSORS_ERROR_INVALID_TENSOR,
    CSAFETENSORS_ERROR_BUFFER_TOO_SMALL
} csafetensors_error_t;

/* Tensor structure */
typedef struct csafetensors_tensor {
    char name[CSAFETENSORS_MAX_STRING_LEN];
    csafetensors_dtype_t dtype;
    size_t shape[CSAFETENSORS_MAX_DIM];
    size_t n_dims;
    size_t data_offset_begin;
    size_t data_offset_end;
} csafetensors_tensor_t;

/* Metadata key-value pair */
typedef struct csafetensors_metadata {
    char key[CSAFETENSORS_MAX_STRING_LEN];
    char value[CSAFETENSORS_MAX_STRING_LEN];
} csafetensors_metadata_t;

/* Main safetensors structure */
typedef struct csafetensors {
    /* Tensors (ordered by insertion order in JSON) */
    csafetensors_tensor_t *tensors;
    size_t n_tensors;

    /* Metadata */
    csafetensors_metadata_t *metadata;
    size_t n_metadata;

    /* Header size (JSON size) */
    size_t header_size;

    /* Data storage (when not mmaped) */
    uint8_t *storage;
    size_t storage_size;

    /* mmap info */
    bool mmaped;
    const uint8_t *mmap_addr;
    size_t mmap_size;
    const uint8_t *databuffer_addr;
    size_t databuffer_size;

    /* Internal handles for mmap cleanup */
    void *_internal_file;
    void *_internal_mmap;

    /* Error message buffer */
    char error_msg[1024];
} csafetensors_t;

/**
 * Initialize a csafetensors structure.
 * Must be called before using the structure.
 *
 * @param st Pointer to csafetensors structure
 */
void csafetensors_init(csafetensors_t *st);

/**
 * Free resources associated with a csafetensors structure.
 *
 * @param st Pointer to csafetensors structure
 */
void csafetensors_free(csafetensors_t *st);

/**
 * Load safetensors from file (copies data to memory).
 *
 * @param filename Path to safetensors file (UTF-8 encoded)
 * @param st Output safetensors structure
 * @return Error code
 */
csafetensors_error_t csafetensors_load_from_file(const char *filename, csafetensors_t *st);

/**
 * Load safetensors from memory buffer (copies data).
 *
 * @param data Pointer to safetensors data
 * @param size Size of data in bytes
 * @param st Output safetensors structure
 * @return Error code
 */
csafetensors_error_t csafetensors_load_from_memory(const uint8_t *data, size_t size, csafetensors_t *st);

/**
 * Load safetensors using memory mapping (zero-copy).
 * The file must remain open during the lifetime of the csafetensors structure.
 *
 * @param filename Path to safetensors file (UTF-8 encoded)
 * @param st Output safetensors structure
 * @return Error code
 */
csafetensors_error_t csafetensors_mmap_from_file(const char *filename, csafetensors_t *st);

/**
 * Load safetensors from already mmaped memory (zero-copy).
 * The caller must keep the memory valid during the lifetime of the csafetensors structure.
 *
 * @param data Pointer to mmaped safetensors data
 * @param size Size of data in bytes
 * @param st Output safetensors structure
 * @return Error code
 */
csafetensors_error_t csafetensors_mmap_from_memory(const uint8_t *data, size_t size, csafetensors_t *st);

/**
 * Get tensor by name.
 *
 * @param st Pointer to csafetensors structure
 * @param name Tensor name
 * @return Pointer to tensor, or NULL if not found
 */
const csafetensors_tensor_t *csafetensors_get_tensor(const csafetensors_t *st, const char *name);

/**
 * Get tensor by index.
 *
 * @param st Pointer to csafetensors structure
 * @param index Tensor index
 * @return Pointer to tensor, or NULL if index out of range
 */
const csafetensors_tensor_t *csafetensors_get_tensor_by_index(const csafetensors_t *st, size_t index);

/**
 * Get pointer to tensor data.
 *
 * @param st Pointer to csafetensors structure
 * @param tensor Pointer to tensor
 * @return Pointer to tensor data, or NULL on error
 */
const uint8_t *csafetensors_get_tensor_data(const csafetensors_t *st, const csafetensors_tensor_t *tensor);

/**
 * Get metadata value by key.
 *
 * @param st Pointer to csafetensors structure
 * @param key Metadata key
 * @return Pointer to value string, or NULL if not found
 */
const char *csafetensors_get_metadata(const csafetensors_t *st, const char *key);

/* Utility functions */

/**
 * Get dtype size in bytes.
 */
size_t csafetensors_dtype_size(csafetensors_dtype_t dtype);

/**
 * Get dtype name string.
 */
const char *csafetensors_dtype_name(csafetensors_dtype_t dtype);

/**
 * Get total number of elements in tensor shape.
 * Returns 1 for scalar (empty shape), 0 for empty tensor (shape contains 0).
 */
size_t csafetensors_shape_size(const csafetensors_tensor_t *tensor);

/**
 * Validate data offsets of all tensors.
 *
 * @param st Pointer to csafetensors structure
 * @return true if valid, false otherwise (error_msg will be set)
 */
bool csafetensors_validate(const csafetensors_t *st);

/* Float conversion utilities */

/**
 * Convert bfloat16 to float32.
 */
float csafetensors_bf16_to_f32(uint16_t x);

/**
 * Convert float32 to bfloat16.
 */
uint16_t csafetensors_f32_to_bf16(float x);

/**
 * Convert float16 (IEEE 754 half) to float32.
 */
float csafetensors_f16_to_f32(uint16_t x);

/**
 * Convert float32 to float16 (IEEE 754 half).
 */
uint16_t csafetensors_f32_to_f16(float x);

/**
 * Get last error message.
 */
const char *csafetensors_get_error(const csafetensors_t *st);

#ifdef __cplusplus
}
#endif

#endif /* CSAFETENSORS_H_ */
