/* Unit tests for csafetensors C11 implementation */
/* SPDX-License-Identifier: MIT */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../../csafetensors.h"

/* Simple test framework */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("  Running %s... ", #name); \
    fflush(stdout); \
    tests_run++; \
    test_##name(); \
    printf("\n"); \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("FAILED at line %d: %s", __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAILED at line %d: %s != %s", __LINE__, #a, #b); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
    if (strcmp((a), (b)) != 0) { \
        printf("FAILED at line %d: '%s' != '%s'", __LINE__, (a), (b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (fabs((double)(a) - (double)(b)) > (eps)) { \
        printf("FAILED at line %d: %f != %f (eps=%f)", __LINE__, (double)(a), (double)(b), (eps)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define PASS() do { tests_passed++; printf("OK"); } while(0)

/* Helper to create safetensors data in memory */
static uint8_t *create_safetensors_data(const char *header_json, const uint8_t *tensor_data,
                                         size_t tensor_size, size_t *out_size) {
    size_t header_len = strlen(header_json);
    size_t total_size = 8 + header_len + tensor_size;

    uint8_t *data = (uint8_t *)malloc(total_size);
    if (!data) return NULL;

    /* Write header size (little-endian 64-bit) */
    uint64_t hsize = (uint64_t)header_len;
    memcpy(data, &hsize, 8);

    /* Write header JSON */
    memcpy(data + 8, header_json, header_len);

    /* Write tensor data */
    if (tensor_data && tensor_size > 0) {
        memcpy(data + 8 + header_len, tensor_data, tensor_size);
    }

    *out_size = total_size;
    return data;
}

/* Helper to write safetensors file */
static int write_safetensors_file(const char *filename, const char *header_json,
                                   const uint8_t *tensor_data, size_t tensor_size) {
    size_t data_size;
    uint8_t *data = create_safetensors_data(header_json, tensor_data, tensor_size, &data_size);
    if (!data) return -1;

    FILE *f = fopen(filename, "wb");
    if (!f) {
        free(data);
        return -1;
    }

    size_t written = fwrite(data, 1, data_size, f);
    fclose(f);
    free(data);

    return (written == data_size) ? 0 : -1;
}

/* ============================================================================
 * Test: Initialization and cleanup
 * ============================================================================ */

TEST(init_free) {
    csafetensors_t st;
    csafetensors_init(&st);

    ASSERT_EQ(st.n_tensors, 0);
    ASSERT_EQ(st.n_metadata, 0);
    ASSERT_EQ(st.tensors, NULL);
    ASSERT_EQ(st.metadata, NULL);
    ASSERT_EQ(st.storage, NULL);
    ASSERT_EQ(st.mmaped, false);

    csafetensors_free(&st);

    /* Should be safe to call free multiple times */
    csafetensors_free(&st);

    PASS();
}

/* ============================================================================
 * Test: dtype utilities
 * ============================================================================ */

TEST(dtype_size) {
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_BOOL), 1);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_UINT8), 1);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_INT8), 1);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_UINT16), 2);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_INT16), 2);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_FLOAT16), 2);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_BFLOAT16), 2);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_UINT32), 4);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_INT32), 4);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_FLOAT32), 4);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_UINT64), 8);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_INT64), 8);
    ASSERT_EQ(csafetensors_dtype_size(CSAFETENSORS_DTYPE_FLOAT64), 8);

    PASS();
}

TEST(dtype_name) {
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_BOOL), "BOOL");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_UINT8), "U8");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_INT8), "I8");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_FLOAT16), "F16");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_BFLOAT16), "BF16");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_FLOAT32), "F32");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_FLOAT64), "F64");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_INT32), "I32");
    ASSERT_STR_EQ(csafetensors_dtype_name(CSAFETENSORS_DTYPE_INT64), "I64");

    PASS();
}

/* ============================================================================
 * Test: Float conversion utilities
 * ============================================================================ */

TEST(bf16_conversion) {
    /* Test round-trip for common values */
    float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, 100.0f, -100.0f, 0.001f};
    int n = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < n; i++) {
        float orig = test_values[i];
        uint16_t bf16 = csafetensors_f32_to_bf16(orig);
        float back = csafetensors_bf16_to_f32(bf16);
        /* BF16 has ~2 decimal digits of precision */
        ASSERT_NEAR(back, orig, fabs(orig) * 0.01f + 0.001f);
    }

    /* Test special values */
    uint16_t zero_bf16 = csafetensors_f32_to_bf16(0.0f);
    ASSERT_NEAR(csafetensors_bf16_to_f32(zero_bf16), 0.0f, 1e-10);

    PASS();
}

TEST(fp16_conversion) {
    /* Test round-trip for common values */
    float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, 100.0f, -100.0f, 0.001f};
    int n = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < n; i++) {
        float orig = test_values[i];
        uint16_t fp16 = csafetensors_f32_to_f16(orig);
        float back = csafetensors_f16_to_f32(fp16);
        /* FP16 has ~3 decimal digits of precision */
        ASSERT_NEAR(back, orig, fabs(orig) * 0.001f + 0.0001f);
    }

    /* Test special values */
    uint16_t zero_fp16 = csafetensors_f32_to_f16(0.0f);
    ASSERT_NEAR(csafetensors_f16_to_f32(zero_fp16), 0.0f, 1e-10);

    PASS();
}

/* ============================================================================
 * Test: Load from memory - basic
 * ============================================================================ */

TEST(load_simple_tensor) {
    const char *header = "{\"test\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}";
    float tensor_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, (uint8_t *)tensor_data,
                                             sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    ASSERT_EQ(st.n_tensors, 1);
    ASSERT_EQ(st.n_metadata, 0);

    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "test");
    ASSERT(t != NULL);
    ASSERT_STR_EQ(t->name, "test");
    ASSERT_EQ(t->dtype, CSAFETENSORS_DTYPE_FLOAT32);
    ASSERT_EQ(t->n_dims, 2);
    ASSERT_EQ(t->shape[0], 2);
    ASSERT_EQ(t->shape[1], 3);
    ASSERT_EQ(csafetensors_shape_size(t), 6);

    const uint8_t *tdata = csafetensors_get_tensor_data(&st, t);
    ASSERT(tdata != NULL);

    const float *fdata = (const float *)tdata;
    ASSERT_NEAR(fdata[0], 1.0f, 1e-6);
    ASSERT_NEAR(fdata[5], 6.0f, 1e-6);

    ASSERT(csafetensors_validate(&st));

    csafetensors_free(&st);
    free(data);

    PASS();
}

TEST(load_multiple_tensors) {
    const char *header = "{"
        "\"weight\":{\"dtype\":\"F32\",\"shape\":[3,4],\"data_offsets\":[0,48]},"
        "\"bias\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[48,64]}"
        "}";

    float tensor_data[16];
    for (int i = 0; i < 16; i++) tensor_data[i] = (float)i;

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, (uint8_t *)tensor_data,
                                             sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    ASSERT_EQ(st.n_tensors, 2);

    const csafetensors_tensor_t *weight = csafetensors_get_tensor(&st, "weight");
    ASSERT(weight != NULL);
    ASSERT_EQ(weight->n_dims, 2);
    ASSERT_EQ(weight->shape[0], 3);
    ASSERT_EQ(weight->shape[1], 4);

    const csafetensors_tensor_t *bias = csafetensors_get_tensor(&st, "bias");
    ASSERT(bias != NULL);
    ASSERT_EQ(bias->n_dims, 1);
    ASSERT_EQ(bias->shape[0], 4);

    /* Test get by index */
    const csafetensors_tensor_t *t0 = csafetensors_get_tensor_by_index(&st, 0);
    const csafetensors_tensor_t *t1 = csafetensors_get_tensor_by_index(&st, 1);
    ASSERT(t0 != NULL);
    ASSERT(t1 != NULL);

    ASSERT(csafetensors_validate(&st));

    csafetensors_free(&st);
    free(data);

    PASS();
}

TEST(load_with_metadata) {
    const char *header = "{"
        "\"__metadata__\":{\"format\":\"pt\",\"model\":\"test\"},"
        "\"tensor\":{\"dtype\":\"I32\",\"shape\":[4],\"data_offsets\":[0,16]}"
        "}";

    int32_t tensor_data[] = {100, 200, 300, 400};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, (uint8_t *)tensor_data,
                                             sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    ASSERT_EQ(st.n_tensors, 1);
    ASSERT_EQ(st.n_metadata, 2);

    const char *format = csafetensors_get_metadata(&st, "format");
    ASSERT(format != NULL);
    ASSERT_STR_EQ(format, "pt");

    const char *model = csafetensors_get_metadata(&st, "model");
    ASSERT(model != NULL);
    ASSERT_STR_EQ(model, "test");

    /* Non-existent key should return NULL */
    ASSERT(csafetensors_get_metadata(&st, "nonexistent") == NULL);

    csafetensors_free(&st);
    free(data);

    PASS();
}

/* ============================================================================
 * Test: Different dtypes
 * ============================================================================ */

TEST(load_different_dtypes) {
    /* Test all supported dtypes */
    struct {
        const char *dtype_str;
        csafetensors_dtype_t dtype;
        size_t size;
    } dtype_tests[] = {
        {"BOOL", CSAFETENSORS_DTYPE_BOOL, 1},
        {"U8", CSAFETENSORS_DTYPE_UINT8, 1},
        {"I8", CSAFETENSORS_DTYPE_INT8, 1},
        {"U16", CSAFETENSORS_DTYPE_UINT16, 2},
        {"I16", CSAFETENSORS_DTYPE_INT16, 2},
        {"F16", CSAFETENSORS_DTYPE_FLOAT16, 2},
        {"BF16", CSAFETENSORS_DTYPE_BFLOAT16, 2},
        {"U32", CSAFETENSORS_DTYPE_UINT32, 4},
        {"I32", CSAFETENSORS_DTYPE_INT32, 4},
        {"F32", CSAFETENSORS_DTYPE_FLOAT32, 4},
        {"U64", CSAFETENSORS_DTYPE_UINT64, 8},
        {"I64", CSAFETENSORS_DTYPE_INT64, 8},
        {"F64", CSAFETENSORS_DTYPE_FLOAT64, 8},
    };
    int n = sizeof(dtype_tests) / sizeof(dtype_tests[0]);

    for (int i = 0; i < n; i++) {
        char header[256];
        size_t data_bytes = dtype_tests[i].size * 4;  /* 4 elements */
        snprintf(header, sizeof(header),
                 "{\"t\":{\"dtype\":\"%s\",\"shape\":[4],\"data_offsets\":[0,%zu]}}",
                 dtype_tests[i].dtype_str, data_bytes);

        uint8_t *tensor_data = (uint8_t *)calloc(1, data_bytes);
        ASSERT(tensor_data != NULL);

        size_t total_size;
        uint8_t *data = create_safetensors_data(header, tensor_data, data_bytes, &total_size);
        ASSERT(data != NULL);

        csafetensors_t st;
        csafetensors_error_t err = csafetensors_load_from_memory(data, total_size, &st);
        ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

        const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "t");
        ASSERT(t != NULL);
        ASSERT_EQ(t->dtype, dtype_tests[i].dtype);

        csafetensors_free(&st);
        free(data);
        free(tensor_data);
    }

    PASS();
}

/* ============================================================================
 * Test: Shape handling
 * ============================================================================ */

TEST(shape_size_scalar) {
    /* Scalar tensor (empty shape) */
    const char *header = "{\"scalar\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[0,4]}}";
    float tensor_data[] = {42.0f};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, (uint8_t *)tensor_data,
                                             sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "scalar");
    ASSERT(t != NULL);
    ASSERT_EQ(t->n_dims, 0);
    ASSERT_EQ(csafetensors_shape_size(t), 1);  /* Scalar has 1 element */

    csafetensors_free(&st);
    free(data);

    PASS();
}

TEST(shape_size_empty_tensor) {
    /* Empty tensor (one dimension is 0) - no data_offsets needed */
    const char *header = "{\"empty\":{\"dtype\":\"F32\",\"shape\":[0,10]}}";

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, NULL, 0, &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "empty");
    ASSERT(t != NULL);
    ASSERT_EQ(csafetensors_shape_size(t), 0);  /* Empty tensor has 0 elements */

    csafetensors_free(&st);
    free(data);

    PASS();
}

TEST(shape_high_dimensional) {
    /* High dimensional tensor */
    const char *header = "{\"hd\":{\"dtype\":\"U8\",\"shape\":[2,3,4,5],\"data_offsets\":[0,120]}}";
    uint8_t tensor_data[120];
    memset(tensor_data, 0, sizeof(tensor_data));

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, tensor_data, sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "hd");
    ASSERT(t != NULL);
    ASSERT_EQ(t->n_dims, 4);
    ASSERT_EQ(csafetensors_shape_size(t), 2 * 3 * 4 * 5);

    csafetensors_free(&st);
    free(data);

    PASS();
}

/* ============================================================================
 * Test: Error handling
 * ============================================================================ */

TEST(error_invalid_argument) {
    csafetensors_t st;

    /* NULL data */
    csafetensors_error_t err = csafetensors_load_from_memory(NULL, 100, &st);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_INVALID_ARGUMENT);

    /* NULL st */
    uint8_t dummy[100];
    err = csafetensors_load_from_memory(dummy, 100, NULL);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_INVALID_ARGUMENT);

    /* Too small */
    err = csafetensors_load_from_memory(dummy, 8, &st);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_INVALID_ARGUMENT);

    PASS();
}

TEST(error_invalid_json) {
    /* Invalid JSON */
    const char *bad_json = "{invalid json}";

    size_t data_size;
    uint8_t *data = create_safetensors_data(bad_json, NULL, 0, &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_JSON_PARSE);

    free(data);

    PASS();
}

TEST(error_missing_dtype) {
    const char *header = "{\"test\":{\"shape\":[4],\"data_offsets\":[0,16]}}";
    uint8_t tensor_data[16] = {0};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, tensor_data, sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_JSON_PARSE);

    free(data);

    PASS();
}

TEST(error_unknown_dtype) {
    const char *header = "{\"test\":{\"dtype\":\"UNKNOWN\",\"shape\":[4],\"data_offsets\":[0,16]}}";
    uint8_t tensor_data[16] = {0};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, tensor_data, sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_JSON_PARSE);

    free(data);

    PASS();
}

TEST(error_header_too_large) {
    /* Header size claims to be larger than data */
    uint8_t data[32];
    uint64_t fake_header_size = 1000000;
    memcpy(data, &fake_header_size, 8);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, 32, &st);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_JSON_PARSE);

    PASS();
}

/* ============================================================================
 * Test: Validation
 * ============================================================================ */

TEST(validate_offset_mismatch) {
    /* Data offsets don't match tensor size */
    const char *header = "{\"test\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}";
    /* Should be 16 bytes for 4 floats, but we say 8 */
    uint8_t tensor_data[16] = {0};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, tensor_data, sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);  /* Parsing succeeds */

    /* Validation should fail */
    ASSERT(!csafetensors_validate(&st));

    csafetensors_free(&st);
    free(data);

    PASS();
}

/* ============================================================================
 * Test: File I/O
 * ============================================================================ */

TEST(load_from_file) {
    const char *filename = "/tmp/test_csafetensors_unit.safetensors";
    const char *header = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}";
    float tensor_data[] = {1.0f, 2.0f, 3.0f, 4.0f};

    int ret = write_safetensors_file(filename, header, (uint8_t *)tensor_data, sizeof(tensor_data));
    ASSERT_EQ(ret, 0);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_file(filename, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    ASSERT_EQ(st.n_tensors, 1);
    ASSERT(!st.mmaped);

    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "tensor");
    ASSERT(t != NULL);

    const float *fdata = (const float *)csafetensors_get_tensor_data(&st, t);
    ASSERT(fdata != NULL);
    ASSERT_NEAR(fdata[0], 1.0f, 1e-6);
    ASSERT_NEAR(fdata[3], 4.0f, 1e-6);

    csafetensors_free(&st);
    remove(filename);

    PASS();
}

TEST(load_file_not_found) {
    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_file("/nonexistent/path/file.safetensors", &st);
    ASSERT_EQ(err, CSAFETENSORS_ERROR_FILE_NOT_FOUND);

    PASS();
}

/* ============================================================================
 * Test: Memory mapping
 * ============================================================================ */

TEST(mmap_from_file) {
    const char *filename = "/tmp/test_csafetensors_mmap.safetensors";
    const char *header = "{\"tensor\":{\"dtype\":\"I32\",\"shape\":[3],\"data_offsets\":[0,12]}}";
    int32_t tensor_data[] = {100, 200, 300};

    int ret = write_safetensors_file(filename, header, (uint8_t *)tensor_data, sizeof(tensor_data));
    ASSERT_EQ(ret, 0);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_mmap_from_file(filename, &st);

    /* mmap might not be supported on all platforms */
    if (err == CSAFETENSORS_ERROR_MMAP_FAILED) {
        remove(filename);
        printf("(mmap not supported, skipping) ");
        PASS();
        return;
    }

    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);
    ASSERT(st.mmaped);
    ASSERT(st.mmap_addr != NULL);
    ASSERT(st.databuffer_addr != NULL);

    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "tensor");
    ASSERT(t != NULL);

    const int32_t *idata = (const int32_t *)csafetensors_get_tensor_data(&st, t);
    ASSERT(idata != NULL);
    ASSERT_EQ(idata[0], 100);
    ASSERT_EQ(idata[1], 200);
    ASSERT_EQ(idata[2], 300);

    csafetensors_free(&st);
    remove(filename);

    PASS();
}

TEST(mmap_from_memory) {
    const char *header = "{\"test\":{\"dtype\":\"F64\",\"shape\":[2],\"data_offsets\":[0,16]}}";
    double tensor_data[] = {3.14159, 2.71828};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, (uint8_t *)tensor_data,
                                             sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_mmap_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    ASSERT(st.mmaped);
    ASSERT_EQ(st.storage, NULL);  /* No storage copy when mmaped */

    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "test");
    ASSERT(t != NULL);
    ASSERT_EQ(t->dtype, CSAFETENSORS_DTYPE_FLOAT64);

    const double *ddata = (const double *)csafetensors_get_tensor_data(&st, t);
    ASSERT(ddata != NULL);
    ASSERT_NEAR(ddata[0], 3.14159, 1e-10);
    ASSERT_NEAR(ddata[1], 2.71828, 1e-10);

    csafetensors_free(&st);
    free(data);

    PASS();
}

/* ============================================================================
 * Test: JSON edge cases
 * ============================================================================ */

TEST(json_unicode_escape) {
    /* Test tensor name with unicode */
    const char *header = "{\"test\\u0041\\u0042\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}";
    uint8_t tensor_data[] = {1, 2};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, tensor_data, sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    /* \u0041\u0042 = "AB" */
    const csafetensors_tensor_t *t = csafetensors_get_tensor(&st, "testAB");
    ASSERT(t != NULL);

    csafetensors_free(&st);
    free(data);

    PASS();
}

TEST(json_escaped_chars) {
    /* Test various JSON escape sequences in metadata */
    const char *header = "{"
        "\"__metadata__\":{\"desc\":\"line1\\nline2\\ttab\"},"
        "\"t\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}"
        "}";
    uint8_t tensor_data[] = {0};

    size_t data_size;
    uint8_t *data = create_safetensors_data(header, tensor_data, sizeof(tensor_data), &data_size);
    ASSERT(data != NULL);

    csafetensors_t st;
    csafetensors_error_t err = csafetensors_load_from_memory(data, data_size, &st);
    ASSERT_EQ(err, CSAFETENSORS_SUCCESS);

    const char *desc = csafetensors_get_metadata(&st, "desc");
    ASSERT(desc != NULL);
    ASSERT(strstr(desc, "\n") != NULL);
    ASSERT(strstr(desc, "\t") != NULL);

    csafetensors_free(&st);
    free(data);

    PASS();
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("Running csafetensors unit tests...\n\n");

    printf("Initialization tests:\n");
    RUN_TEST(init_free);

    printf("\nDtype utility tests:\n");
    RUN_TEST(dtype_size);
    RUN_TEST(dtype_name);

    printf("\nFloat conversion tests:\n");
    RUN_TEST(bf16_conversion);
    RUN_TEST(fp16_conversion);

    printf("\nBasic loading tests:\n");
    RUN_TEST(load_simple_tensor);
    RUN_TEST(load_multiple_tensors);
    RUN_TEST(load_with_metadata);
    RUN_TEST(load_different_dtypes);

    printf("\nShape handling tests:\n");
    RUN_TEST(shape_size_scalar);
    RUN_TEST(shape_size_empty_tensor);
    RUN_TEST(shape_high_dimensional);

    printf("\nError handling tests:\n");
    RUN_TEST(error_invalid_argument);
    RUN_TEST(error_invalid_json);
    RUN_TEST(error_missing_dtype);
    RUN_TEST(error_unknown_dtype);
    RUN_TEST(error_header_too_large);

    printf("\nValidation tests:\n");
    RUN_TEST(validate_offset_mismatch);

    printf("\nFile I/O tests:\n");
    RUN_TEST(load_from_file);
    RUN_TEST(load_file_not_found);

    printf("\nMemory mapping tests:\n");
    RUN_TEST(mmap_from_file);
    RUN_TEST(mmap_from_memory);

    printf("\nJSON edge cases:\n");
    RUN_TEST(json_unicode_escape);
    RUN_TEST(json_escaped_chars);

    printf("\n========================================\n");
    printf("Tests run:    %d\n", tests_run);
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("========================================\n");

    return (tests_failed > 0) ? 1 : 0;
}
