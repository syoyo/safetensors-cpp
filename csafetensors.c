/* SPDX-License-Identifier: MIT */
/* Copyright 2023 - Present, Syoyo Fujita. */
/* Pure C11 implementation of safetensors loader */
/* Includes embedded minijson parser */

/* Enable POSIX features for fileno() on some systems */
#if !defined(_WIN32) && !defined(_WIN64)
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "csafetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <math.h>

/* Platform detection */
#if defined(_WIN32) || defined(_WIN64)
#define CSAFETENSORS_WINDOWS 1
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#else
#define CSAFETENSORS_POSIX 1
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
#define CSAFETENSORS_HAS_MMAP 1
#include <sys/mman.h>
#endif
#endif

/* Max header (JSON) size: 100 MB */
#define CSAFETENSORS_MAX_JSON_SIZE (100ULL * 1024ULL * 1024ULL)

/* ============================================================================
 * Internal JSON Parser (minijson-like for C11)
 * ============================================================================ */

typedef enum {
    JSON_TYPE_NULL,
    JSON_TYPE_BOOL,
    JSON_TYPE_NUMBER,
    JSON_TYPE_STRING,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT
} json_type_t;

typedef struct json_value json_value_t;

typedef struct json_pair {
    char *key;
    json_value_t *value;
} json_pair_t;

typedef struct json_array {
    json_value_t **items;
    size_t count;
    size_t capacity;
} json_array_t;

typedef struct json_object {
    json_pair_t *pairs;
    size_t count;
    size_t capacity;
} json_object_t;

struct json_value {
    json_type_t type;
    union {
        bool boolean;
        double number;
        char *string;
        json_array_t array;
        json_object_t object;
    } data;
};

typedef struct {
    const char *input;
    const char *pos;
    const char *end;
    char error[256];
} json_parser_t;

/* Forward declarations */
static json_value_t *json_parse_value(json_parser_t *p);
static void json_free_value(json_value_t *v);

static void json_skip_whitespace(json_parser_t *p) {
    while (p->pos < p->end && (*p->pos == ' ' || *p->pos == '\t' || *p->pos == '\n' || *p->pos == '\r')) {
        p->pos++;
    }
}

static bool json_match(json_parser_t *p, const char *str) {
    size_t len = strlen(str);
    if ((size_t)(p->end - p->pos) < len) return false;
    if (memcmp(p->pos, str, len) == 0) {
        p->pos += len;
        return true;
    }
    return false;
}

static json_value_t *json_alloc_value(json_type_t type) {
    json_value_t *v = (json_value_t *)calloc(1, sizeof(json_value_t));
    if (v) v->type = type;
    return v;
}

static int json_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool json_parse_unicode_escape(json_parser_t *p, char **out, size_t *out_cap, size_t *out_len) {
    if (p->end - p->pos < 4) return false;

    int codepoint = 0;
    for (int i = 0; i < 4; i++) {
        int digit = json_hex_digit(p->pos[i]);
        if (digit < 0) return false;
        codepoint = (codepoint << 4) | digit;
    }
    p->pos += 4;

    /* Handle surrogate pairs */
    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
        if (p->end - p->pos < 6) return false;
        if (p->pos[0] != '\\' || p->pos[1] != 'u') return false;
        p->pos += 2;

        int low = 0;
        for (int i = 0; i < 4; i++) {
            int digit = json_hex_digit(p->pos[i]);
            if (digit < 0) return false;
            low = (low << 4) | digit;
        }
        p->pos += 4;

        if (low < 0xDC00 || low > 0xDFFF) return false;
        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
    }

    /* Encode as UTF-8 */
    char utf8[4];
    size_t utf8_len = 0;

    if (codepoint < 0x80) {
        utf8[0] = (char)codepoint;
        utf8_len = 1;
    } else if (codepoint < 0x800) {
        utf8[0] = (char)(0xC0 | (codepoint >> 6));
        utf8[1] = (char)(0x80 | (codepoint & 0x3F));
        utf8_len = 2;
    } else if (codepoint < 0x10000) {
        utf8[0] = (char)(0xE0 | (codepoint >> 12));
        utf8[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        utf8[2] = (char)(0x80 | (codepoint & 0x3F));
        utf8_len = 3;
    } else {
        utf8[0] = (char)(0xF0 | (codepoint >> 18));
        utf8[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        utf8[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        utf8[3] = (char)(0x80 | (codepoint & 0x3F));
        utf8_len = 4;
    }

    /* Append to output */
    while (*out_len + utf8_len >= *out_cap) {
        *out_cap *= 2;
        *out = (char *)realloc(*out, *out_cap);
        if (!*out) return false;
    }
    memcpy(*out + *out_len, utf8, utf8_len);
    *out_len += utf8_len;

    return true;
}

static json_value_t *json_parse_string(json_parser_t *p) {
    if (*p->pos != '"') {
        snprintf(p->error, sizeof(p->error), "Expected '\"' at position %zu", (size_t)(p->pos - p->input));
        return NULL;
    }
    p->pos++;

    size_t cap = 64;
    size_t len = 0;
    char *str = (char *)malloc(cap);
    if (!str) {
        snprintf(p->error, sizeof(p->error), "Memory allocation failed");
        return NULL;
    }

    while (p->pos < p->end && *p->pos != '"') {
        char c = *p->pos;

        /* Check for control characters */
        if ((unsigned char)c < 0x20) {
            snprintf(p->error, sizeof(p->error), "Invalid control character in string");
            free(str);
            return NULL;
        }

        if (c == '\\') {
            p->pos++;
            if (p->pos >= p->end) {
                snprintf(p->error, sizeof(p->error), "Unexpected end of string");
                free(str);
                return NULL;
            }

            switch (*p->pos) {
                case '"':  c = '"'; break;
                case '\\': c = '\\'; break;
                case '/':  c = '/'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u':
                    p->pos++;
                    if (!json_parse_unicode_escape(p, &str, &cap, &len)) {
                        snprintf(p->error, sizeof(p->error), "Invalid unicode escape");
                        free(str);
                        return NULL;
                    }
                    continue;
                default:
                    snprintf(p->error, sizeof(p->error), "Invalid escape character");
                    free(str);
                    return NULL;
            }
            p->pos++;
        } else {
            p->pos++;
        }

        if (len + 1 >= cap) {
            cap *= 2;
            str = (char *)realloc(str, cap);
            if (!str) {
                snprintf(p->error, sizeof(p->error), "Memory allocation failed");
                return NULL;
            }
        }
        str[len++] = c;
    }

    if (p->pos >= p->end || *p->pos != '"') {
        snprintf(p->error, sizeof(p->error), "Unterminated string");
        free(str);
        return NULL;
    }
    p->pos++;

    str[len] = '\0';

    json_value_t *v = json_alloc_value(JSON_TYPE_STRING);
    if (!v) {
        free(str);
        snprintf(p->error, sizeof(p->error), "Memory allocation failed");
        return NULL;
    }
    v->data.string = str;
    return v;
}

static json_value_t *json_parse_number(json_parser_t *p) {
    const char *start = p->pos;

    if (*p->pos == '-') p->pos++;

    if (p->pos >= p->end) {
        snprintf(p->error, sizeof(p->error), "Unexpected end of number");
        return NULL;
    }

    if (*p->pos == '0') {
        p->pos++;
    } else if (*p->pos >= '1' && *p->pos <= '9') {
        while (p->pos < p->end && *p->pos >= '0' && *p->pos <= '9') p->pos++;
    } else {
        snprintf(p->error, sizeof(p->error), "Invalid number");
        return NULL;
    }

    if (p->pos < p->end && *p->pos == '.') {
        p->pos++;
        if (p->pos >= p->end || *p->pos < '0' || *p->pos > '9') {
            snprintf(p->error, sizeof(p->error), "Invalid number");
            return NULL;
        }
        while (p->pos < p->end && *p->pos >= '0' && *p->pos <= '9') p->pos++;
    }

    if (p->pos < p->end && (*p->pos == 'e' || *p->pos == 'E')) {
        p->pos++;
        if (p->pos < p->end && (*p->pos == '+' || *p->pos == '-')) p->pos++;
        if (p->pos >= p->end || *p->pos < '0' || *p->pos > '9') {
            snprintf(p->error, sizeof(p->error), "Invalid number exponent");
            return NULL;
        }
        while (p->pos < p->end && *p->pos >= '0' && *p->pos <= '9') p->pos++;
    }

    size_t len = (size_t)(p->pos - start);
    char *numstr = (char *)malloc(len + 1);
    if (!numstr) {
        snprintf(p->error, sizeof(p->error), "Memory allocation failed");
        return NULL;
    }
    memcpy(numstr, start, len);
    numstr[len] = '\0';

    double val = strtod(numstr, NULL);
    free(numstr);

    json_value_t *v = json_alloc_value(JSON_TYPE_NUMBER);
    if (!v) {
        snprintf(p->error, sizeof(p->error), "Memory allocation failed");
        return NULL;
    }
    v->data.number = val;
    return v;
}

static json_value_t *json_parse_array(json_parser_t *p) {
    if (*p->pos != '[') {
        snprintf(p->error, sizeof(p->error), "Expected '[' at position %zu", (size_t)(p->pos - p->input));
        return NULL;
    }
    p->pos++;

    json_value_t *v = json_alloc_value(JSON_TYPE_ARRAY);
    if (!v) {
        snprintf(p->error, sizeof(p->error), "Memory allocation failed");
        return NULL;
    }
    v->data.array.items = NULL;
    v->data.array.count = 0;
    v->data.array.capacity = 0;

    json_skip_whitespace(p);

    if (p->pos < p->end && *p->pos == ']') {
        p->pos++;
        return v;
    }

    for (;;) {
        json_value_t *item = json_parse_value(p);
        if (!item) {
            json_free_value(v);
            return NULL;
        }

        if (v->data.array.count >= v->data.array.capacity) {
            size_t new_cap = v->data.array.capacity == 0 ? 8 : v->data.array.capacity * 2;
            json_value_t **new_items = (json_value_t **)realloc(v->data.array.items, new_cap * sizeof(json_value_t *));
            if (!new_items) {
                json_free_value(item);
                json_free_value(v);
                snprintf(p->error, sizeof(p->error), "Memory allocation failed");
                return NULL;
            }
            v->data.array.items = new_items;
            v->data.array.capacity = new_cap;
        }
        v->data.array.items[v->data.array.count++] = item;

        json_skip_whitespace(p);

        if (p->pos >= p->end) {
            snprintf(p->error, sizeof(p->error), "Unexpected end of array");
            json_free_value(v);
            return NULL;
        }

        if (*p->pos == ']') {
            p->pos++;
            return v;
        }

        if (*p->pos != ',') {
            snprintf(p->error, sizeof(p->error), "Expected ',' or ']' in array");
            json_free_value(v);
            return NULL;
        }
        p->pos++;
        json_skip_whitespace(p);
    }
}

static json_value_t *json_parse_object(json_parser_t *p) {
    if (*p->pos != '{') {
        snprintf(p->error, sizeof(p->error), "Expected '{' at position %zu", (size_t)(p->pos - p->input));
        return NULL;
    }
    p->pos++;

    json_value_t *v = json_alloc_value(JSON_TYPE_OBJECT);
    if (!v) {
        snprintf(p->error, sizeof(p->error), "Memory allocation failed");
        return NULL;
    }
    v->data.object.pairs = NULL;
    v->data.object.count = 0;
    v->data.object.capacity = 0;

    json_skip_whitespace(p);

    if (p->pos < p->end && *p->pos == '}') {
        p->pos++;
        return v;
    }

    for (;;) {
        if (p->pos >= p->end || *p->pos != '"') {
            snprintf(p->error, sizeof(p->error), "Expected string key in object");
            json_free_value(v);
            return NULL;
        }

        json_value_t *key_val = json_parse_string(p);
        if (!key_val) {
            json_free_value(v);
            return NULL;
        }
        char *key = key_val->data.string;
        key_val->data.string = NULL;
        json_free_value(key_val);

        json_skip_whitespace(p);

        if (p->pos >= p->end || *p->pos != ':') {
            free(key);
            snprintf(p->error, sizeof(p->error), "Expected ':' after key");
            json_free_value(v);
            return NULL;
        }
        p->pos++;

        json_value_t *val = json_parse_value(p);
        if (!val) {
            free(key);
            json_free_value(v);
            return NULL;
        }

        /* Check for duplicate keys */
        for (size_t i = 0; i < v->data.object.count; i++) {
            if (strcmp(v->data.object.pairs[i].key, key) == 0) {
                free(key);
                json_free_value(val);
                snprintf(p->error, sizeof(p->error), "Duplicate key in object");
                json_free_value(v);
                return NULL;
            }
        }

        if (v->data.object.count >= v->data.object.capacity) {
            size_t new_cap = v->data.object.capacity == 0 ? 8 : v->data.object.capacity * 2;
            json_pair_t *new_pairs = (json_pair_t *)realloc(v->data.object.pairs, new_cap * sizeof(json_pair_t));
            if (!new_pairs) {
                free(key);
                json_free_value(val);
                json_free_value(v);
                snprintf(p->error, sizeof(p->error), "Memory allocation failed");
                return NULL;
            }
            v->data.object.pairs = new_pairs;
            v->data.object.capacity = new_cap;
        }
        v->data.object.pairs[v->data.object.count].key = key;
        v->data.object.pairs[v->data.object.count].value = val;
        v->data.object.count++;

        json_skip_whitespace(p);

        if (p->pos >= p->end) {
            snprintf(p->error, sizeof(p->error), "Unexpected end of object");
            json_free_value(v);
            return NULL;
        }

        if (*p->pos == '}') {
            p->pos++;
            return v;
        }

        if (*p->pos != ',') {
            snprintf(p->error, sizeof(p->error), "Expected ',' or '}' in object");
            json_free_value(v);
            return NULL;
        }
        p->pos++;
        json_skip_whitespace(p);
    }
}

static json_value_t *json_parse_value(json_parser_t *p) {
    json_skip_whitespace(p);

    if (p->pos >= p->end) {
        snprintf(p->error, sizeof(p->error), "Unexpected end of input");
        return NULL;
    }

    switch (*p->pos) {
        case 'n':
            if (json_match(p, "null")) {
                return json_alloc_value(JSON_TYPE_NULL);
            }
            break;
        case 't':
            if (json_match(p, "true")) {
                json_value_t *v = json_alloc_value(JSON_TYPE_BOOL);
                if (v) v->data.boolean = true;
                return v;
            }
            break;
        case 'f':
            if (json_match(p, "false")) {
                json_value_t *v = json_alloc_value(JSON_TYPE_BOOL);
                if (v) v->data.boolean = false;
                return v;
            }
            break;
        case '"':
            return json_parse_string(p);
        case '[':
            return json_parse_array(p);
        case '{':
            return json_parse_object(p);
        default:
            if (*p->pos == '-' || (*p->pos >= '0' && *p->pos <= '9')) {
                return json_parse_number(p);
            }
    }

    snprintf(p->error, sizeof(p->error), "Invalid JSON value at position %zu", (size_t)(p->pos - p->input));
    return NULL;
}

static void json_free_value(json_value_t *v) {
    if (!v) return;

    switch (v->type) {
        case JSON_TYPE_STRING:
            free(v->data.string);
            break;
        case JSON_TYPE_ARRAY:
            for (size_t i = 0; i < v->data.array.count; i++) {
                json_free_value(v->data.array.items[i]);
            }
            free(v->data.array.items);
            break;
        case JSON_TYPE_OBJECT:
            for (size_t i = 0; i < v->data.object.count; i++) {
                free(v->data.object.pairs[i].key);
                json_free_value(v->data.object.pairs[i].value);
            }
            free(v->data.object.pairs);
            break;
        default:
            break;
    }
    free(v);
}

static json_value_t *json_object_get(const json_value_t *obj, const char *key) {
    if (!obj || obj->type != JSON_TYPE_OBJECT) return NULL;
    for (size_t i = 0; i < obj->data.object.count; i++) {
        if (strcmp(obj->data.object.pairs[i].key, key) == 0) {
            return obj->data.object.pairs[i].value;
        }
    }
    return NULL;
}

/* ============================================================================
 * Float conversion utilities
 * ============================================================================ */

float csafetensors_bf16_to_f32(uint16_t x) {
    union {
        uint32_t u32;
        float f32;
    } cvt;
    cvt.u32 = ((uint32_t)x) << 16;
    return cvt.f32;
}

uint16_t csafetensors_f32_to_bf16(float x) {
    union {
        uint32_t u32;
        uint16_t u16[2];
        float f32;
    } cvt;
    cvt.f32 = x;

    /* Handle Inf/NaN */
    if ((~cvt.u32 & 0x7f800000) == 0) {
        if ((cvt.u32 & 0xffff) != 0) {
            cvt.u32 |= 0x10000;  /* Preserve signaling NaN */
        }
    } else {
        /* Round to nearest even */
        cvt.u32 += (0x7fff + (cvt.u16[1] & 1));
    }

    return cvt.u16[1];
}

float csafetensors_f16_to_f32(uint16_t x) {
    union {
        uint32_t u;
        float f;
    } o;

    uint32_t shifted_exp = 0x7c00 << 13;

    o.u = (x & 0x7fff) << 13;
    uint32_t exp = shifted_exp & o.u;
    o.u += (127 - 15) << 23;

    if (exp == shifted_exp) {
        o.u += (128 - 16) << 23;
    } else if (exp == 0) {
        union { uint32_t u; float f; } magic;
        magic.u = 113 << 23;
        o.u += 1 << 23;
        o.f -= magic.f;
    }

    o.u |= (x & 0x8000) << 16;
    return o.f;
}

uint16_t csafetensors_f32_to_f16(float x) {
    union {
        uint32_t u;
        float f;
    } f32;
    f32.f = x;

    uint32_t sign = (f32.u >> 16) & 0x8000;
    uint32_t exp = (f32.u >> 23) & 0xff;
    uint32_t mant = f32.u & 0x7fffff;

    uint16_t result;

    if (exp == 0) {
        result = (uint16_t)sign;
    } else if (exp == 255) {
        result = (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
    } else {
        int newexp = (int)exp - 127 + 15;
        if (newexp >= 31) {
            result = (uint16_t)(sign | 0x7c00);
        } else if (newexp <= 0) {
            if ((14 - newexp) <= 24) {
                uint32_t m = mant | 0x800000;
                result = (uint16_t)(sign | (m >> (14 - newexp)));
                if ((m >> (13 - newexp)) & 1) result++;
            } else {
                result = (uint16_t)sign;
            }
        } else {
            result = (uint16_t)(sign | ((uint32_t)newexp << 10) | (mant >> 13));
            if (mant & 0x1000) result++;
        }
    }

    return result;
}

/* ============================================================================
 * Internal structures for mmap
 * ============================================================================ */

typedef struct {
    FILE *fp;
    size_t size;
} internal_file_t;

typedef struct {
    uint8_t *addr;
    size_t size;
#ifdef CSAFETENSORS_WINDOWS
    HANDLE hMapping;
#endif
} internal_mmap_t;

/* ============================================================================
 * Helper functions
 * ============================================================================ */

static bool parse_dtype_string(const char *s, csafetensors_dtype_t *dtype) {
    if (strcmp(s, "BOOL") == 0) { *dtype = CSAFETENSORS_DTYPE_BOOL; return true; }
    if (strcmp(s, "U8") == 0) { *dtype = CSAFETENSORS_DTYPE_UINT8; return true; }
    if (strcmp(s, "I8") == 0) { *dtype = CSAFETENSORS_DTYPE_INT8; return true; }
    if (strcmp(s, "U16") == 0) { *dtype = CSAFETENSORS_DTYPE_UINT16; return true; }
    if (strcmp(s, "I16") == 0) { *dtype = CSAFETENSORS_DTYPE_INT16; return true; }
    if (strcmp(s, "U32") == 0) { *dtype = CSAFETENSORS_DTYPE_UINT32; return true; }
    if (strcmp(s, "I32") == 0) { *dtype = CSAFETENSORS_DTYPE_INT32; return true; }
    if (strcmp(s, "U64") == 0) { *dtype = CSAFETENSORS_DTYPE_UINT64; return true; }
    if (strcmp(s, "I64") == 0) { *dtype = CSAFETENSORS_DTYPE_INT64; return true; }
    if (strcmp(s, "F16") == 0) { *dtype = CSAFETENSORS_DTYPE_FLOAT16; return true; }
    if (strcmp(s, "BF16") == 0) { *dtype = CSAFETENSORS_DTYPE_BFLOAT16; return true; }
    if (strcmp(s, "F32") == 0) { *dtype = CSAFETENSORS_DTYPE_FLOAT32; return true; }
    if (strcmp(s, "F64") == 0) { *dtype = CSAFETENSORS_DTYPE_FLOAT64; return true; }
    return false;
}

static bool parse_tensor(const char *name, const json_value_t *obj, csafetensors_tensor_t *tensor, char *err, size_t err_size) {
    if (!obj || obj->type != JSON_TYPE_OBJECT) {
        snprintf(err, err_size, "Tensor '%s' is not a JSON object", name);
        return false;
    }

    memset(tensor, 0, sizeof(*tensor));
    strncpy(tensor->name, name, CSAFETENSORS_MAX_STRING_LEN - 1);

    /* Parse dtype */
    json_value_t *dtype_val = json_object_get(obj, "dtype");
    if (!dtype_val || dtype_val->type != JSON_TYPE_STRING) {
        snprintf(err, err_size, "Tensor '%s' missing 'dtype' string", name);
        return false;
    }
    if (!parse_dtype_string(dtype_val->data.string, &tensor->dtype)) {
        snprintf(err, err_size, "Unknown dtype '%s' in tensor '%s'", dtype_val->data.string, name);
        return false;
    }

    /* Parse shape */
    json_value_t *shape_val = json_object_get(obj, "shape");
    if (!shape_val || shape_val->type != JSON_TYPE_ARRAY) {
        snprintf(err, err_size, "Tensor '%s' missing 'shape' array", name);
        return false;
    }

    tensor->n_dims = shape_val->data.array.count;
    if (tensor->n_dims > CSAFETENSORS_MAX_DIM) {
        snprintf(err, err_size, "Tensor '%s' has too many dimensions (%zu)", name, tensor->n_dims);
        return false;
    }

    bool is_empty = false;
    for (size_t i = 0; i < tensor->n_dims; i++) {
        json_value_t *dim = shape_val->data.array.items[i];
        if (!dim || dim->type != JSON_TYPE_NUMBER) {
            snprintf(err, err_size, "Invalid shape dimension in tensor '%s'", name);
            return false;
        }
        tensor->shape[i] = (size_t)dim->data.number;
        if (tensor->shape[i] == 0) is_empty = true;
    }

    /* Parse data_offsets (optional for empty tensors) */
    json_value_t *offsets_val = json_object_get(obj, "data_offsets");
    if (is_empty) {
        if (offsets_val != NULL) {
            snprintf(err, err_size, "Empty tensor '%s' should not have data_offsets", name);
            return false;
        }
        tensor->data_offset_begin = 0;
        tensor->data_offset_end = 0;
    } else {
        if (!offsets_val || offsets_val->type != JSON_TYPE_ARRAY || offsets_val->data.array.count != 2) {
            snprintf(err, err_size, "Tensor '%s' missing valid 'data_offsets' array", name);
            return false;
        }

        json_value_t *begin = offsets_val->data.array.items[0];
        json_value_t *end = offsets_val->data.array.items[1];
        if (!begin || begin->type != JSON_TYPE_NUMBER || !end || end->type != JSON_TYPE_NUMBER) {
            snprintf(err, err_size, "Invalid data_offsets in tensor '%s'", name);
            return false;
        }

        tensor->data_offset_begin = (size_t)begin->data.number;
        tensor->data_offset_end = (size_t)end->data.number;
    }

    return true;
}

static bool parse_safetensors_header(const uint8_t *data, size_t size, csafetensors_t *st) {
    if (size < 16) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Data too short (< 16 bytes)");
        return false;
    }

    /* Read header size (8 bytes, little-endian) */
    uint64_t header_size = 0;
    memcpy(&header_size, data, sizeof(uint64_t));

    if (header_size < 2) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Header size too small");
        return false;
    }

    if (header_size > CSAFETENSORS_MAX_JSON_SIZE) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Header size exceeds limit");
        return false;
    }

    if (8 + header_size > size) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Header size exceeds data size");
        return false;
    }

    st->header_size = (size_t)header_size;

    /* Parse JSON header */
    json_parser_t parser;
    parser.input = (const char *)(data + 8);
    parser.pos = parser.input;
    parser.end = parser.input + header_size;
    parser.error[0] = '\0';

    json_value_t *root = json_parse_value(&parser);
    if (!root) {
        snprintf(st->error_msg, sizeof(st->error_msg), "JSON parse error: %s", parser.error);
        return false;
    }

    if (root->type != JSON_TYPE_OBJECT) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Root JSON must be an object");
        json_free_value(root);
        return false;
    }

    /* Count tensors and metadata */
    size_t n_tensors = 0;
    size_t n_metadata = 0;

    for (size_t i = 0; i < root->data.object.count; i++) {
        const char *key = root->data.object.pairs[i].key;
        if (strcmp(key, "__metadata__") == 0) {
            json_value_t *meta = root->data.object.pairs[i].value;
            if (meta && meta->type == JSON_TYPE_OBJECT) {
                n_metadata = meta->data.object.count;
            }
        } else {
            n_tensors++;
        }
    }

    /* Allocate arrays */
    if (n_tensors > 0) {
        st->tensors = (csafetensors_tensor_t *)calloc(n_tensors, sizeof(csafetensors_tensor_t));
        if (!st->tensors) {
            snprintf(st->error_msg, sizeof(st->error_msg), "Memory allocation failed");
            json_free_value(root);
            return false;
        }
    }

    if (n_metadata > 0) {
        st->metadata = (csafetensors_metadata_t *)calloc(n_metadata, sizeof(csafetensors_metadata_t));
        if (!st->metadata) {
            snprintf(st->error_msg, sizeof(st->error_msg), "Memory allocation failed");
            free(st->tensors);
            st->tensors = NULL;
            json_free_value(root);
            return false;
        }
    }

    /* Parse tensors and metadata */
    size_t tensor_idx = 0;
    size_t meta_idx = 0;

    for (size_t i = 0; i < root->data.object.count; i++) {
        const char *key = root->data.object.pairs[i].key;
        json_value_t *val = root->data.object.pairs[i].value;

        if (strcmp(key, "__metadata__") == 0) {
            if (val && val->type == JSON_TYPE_OBJECT) {
                for (size_t j = 0; j < val->data.object.count && meta_idx < n_metadata; j++) {
                    const char *mk = val->data.object.pairs[j].key;
                    json_value_t *mv = val->data.object.pairs[j].value;
                    if (mv && mv->type == JSON_TYPE_STRING) {
                        strncpy(st->metadata[meta_idx].key, mk, CSAFETENSORS_MAX_STRING_LEN - 1);
                        strncpy(st->metadata[meta_idx].value, mv->data.string, CSAFETENSORS_MAX_STRING_LEN - 1);
                        meta_idx++;
                    }
                }
            }
        } else {
            if (tensor_idx >= n_tensors) continue;

            if (!parse_tensor(key, val, &st->tensors[tensor_idx], st->error_msg, sizeof(st->error_msg))) {
                free(st->tensors);
                free(st->metadata);
                st->tensors = NULL;
                st->metadata = NULL;
                json_free_value(root);
                return false;
            }
            tensor_idx++;
        }
    }

    st->n_tensors = tensor_idx;
    st->n_metadata = meta_idx;

    json_free_value(root);
    return true;
}

/* ============================================================================
 * Public API implementation
 * ============================================================================ */

void csafetensors_init(csafetensors_t *st) {
    if (!st) return;
    memset(st, 0, sizeof(*st));
}

void csafetensors_free(csafetensors_t *st) {
    if (!st) return;

    free(st->tensors);
    free(st->metadata);
    free(st->storage);

    if (st->_internal_mmap) {
        internal_mmap_t *m = (internal_mmap_t *)st->_internal_mmap;
#ifdef CSAFETENSORS_HAS_MMAP
        if (m->addr) munmap(m->addr, m->size);
#elif defined(CSAFETENSORS_WINDOWS)
        if (m->addr) UnmapViewOfFile(m->addr);
        if (m->hMapping) CloseHandle(m->hMapping);
#endif
        free(m);
    }

    if (st->_internal_file) {
        internal_file_t *f = (internal_file_t *)st->_internal_file;
        if (f->fp) fclose(f->fp);
        free(f);
    }

    memset(st, 0, sizeof(*st));
}

csafetensors_error_t csafetensors_load_from_file(const char *filename, csafetensors_t *st) {
    if (!filename || !st) return CSAFETENSORS_ERROR_INVALID_ARGUMENT;

    csafetensors_init(st);

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Cannot open file: %s", filename);
        return CSAFETENSORS_ERROR_FILE_NOT_FOUND;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size < 0) {
        fclose(fp);
        snprintf(st->error_msg, sizeof(st->error_msg), "Failed to get file size");
        return CSAFETENSORS_ERROR_FILE_READ;
    }

    uint8_t *data = (uint8_t *)malloc((size_t)file_size);
    if (!data) {
        fclose(fp);
        snprintf(st->error_msg, sizeof(st->error_msg), "Memory allocation failed");
        return CSAFETENSORS_ERROR_MEMORY_ALLOCATION;
    }

    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        free(data);
        fclose(fp);
        snprintf(st->error_msg, sizeof(st->error_msg), "Failed to read file");
        return CSAFETENSORS_ERROR_FILE_READ;
    }
    fclose(fp);

    csafetensors_error_t err = csafetensors_load_from_memory(data, (size_t)file_size, st);
    free(data);
    return err;
}

csafetensors_error_t csafetensors_load_from_memory(const uint8_t *data, size_t size, csafetensors_t *st) {
    if (!data || !st || size < 16) return CSAFETENSORS_ERROR_INVALID_ARGUMENT;

    csafetensors_init(st);

    if (!parse_safetensors_header(data, size, st)) {
        return CSAFETENSORS_ERROR_JSON_PARSE;
    }

    size_t data_offset = 8 + st->header_size;
    size_t data_size = size - data_offset;

    st->storage = (uint8_t *)malloc(data_size);
    if (!st->storage && data_size > 0) {
        csafetensors_free(st);
        snprintf(st->error_msg, sizeof(st->error_msg), "Memory allocation failed");
        return CSAFETENSORS_ERROR_MEMORY_ALLOCATION;
    }

    if (data_size > 0) {
        memcpy(st->storage, data + data_offset, data_size);
    }
    st->storage_size = data_size;
    st->mmaped = false;

    return CSAFETENSORS_SUCCESS;
}

csafetensors_error_t csafetensors_mmap_from_file(const char *filename, csafetensors_t *st) {
    if (!filename || !st) return CSAFETENSORS_ERROR_INVALID_ARGUMENT;

    csafetensors_init(st);

#if defined(CSAFETENSORS_HAS_MMAP)
    /* POSIX mmap */
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Cannot open file: %s", filename);
        return CSAFETENSORS_ERROR_FILE_NOT_FOUND;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size < 0) {
        fclose(fp);
        snprintf(st->error_msg, sizeof(st->error_msg), "Failed to get file size");
        return CSAFETENSORS_ERROR_FILE_READ;
    }

    int fd = fileno(fp);
    void *addr = mmap(NULL, (size_t)file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        fclose(fp);
        snprintf(st->error_msg, sizeof(st->error_msg), "mmap failed: %s", strerror(errno));
        return CSAFETENSORS_ERROR_MMAP_FAILED;
    }

    csafetensors_error_t err = csafetensors_mmap_from_memory((const uint8_t *)addr, (size_t)file_size, st);
    if (err != CSAFETENSORS_SUCCESS) {
        munmap(addr, (size_t)file_size);
        fclose(fp);
        return err;
    }

    /* Store handles for cleanup */
    internal_file_t *f = (internal_file_t *)malloc(sizeof(internal_file_t));
    internal_mmap_t *m = (internal_mmap_t *)malloc(sizeof(internal_mmap_t));
    if (!f || !m) {
        free(f);
        free(m);
        munmap(addr, (size_t)file_size);
        fclose(fp);
        csafetensors_free(st);
        snprintf(st->error_msg, sizeof(st->error_msg), "Memory allocation failed");
        return CSAFETENSORS_ERROR_MEMORY_ALLOCATION;
    }

    f->fp = fp;
    f->size = (size_t)file_size;
    m->addr = (uint8_t *)addr;
    m->size = (size_t)file_size;

    st->_internal_file = f;
    st->_internal_mmap = m;
    st->mmap_addr = (const uint8_t *)addr;
    st->mmap_size = (size_t)file_size;
    st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
    st->databuffer_size = st->mmap_size - 8 - st->header_size;

    return CSAFETENSORS_SUCCESS;

#elif defined(CSAFETENSORS_WINDOWS)
    /* Windows memory mapping */
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        snprintf(st->error_msg, sizeof(st->error_msg), "Cannot open file: %s", filename);
        return CSAFETENSORS_ERROR_FILE_NOT_FOUND;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        CloseHandle(hFile);
        snprintf(st->error_msg, sizeof(st->error_msg), "Failed to get file size");
        return CSAFETENSORS_ERROR_FILE_READ;
    }

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        CloseHandle(hFile);
        snprintf(st->error_msg, sizeof(st->error_msg), "CreateFileMapping failed");
        return CSAFETENSORS_ERROR_MMAP_FAILED;
    }

    void *addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (addr == NULL) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        snprintf(st->error_msg, sizeof(st->error_msg), "MapViewOfFile failed");
        return CSAFETENSORS_ERROR_MMAP_FAILED;
    }

    csafetensors_error_t err = csafetensors_mmap_from_memory((const uint8_t *)addr, (size_t)file_size.QuadPart, st);
    if (err != CSAFETENSORS_SUCCESS) {
        UnmapViewOfFile(addr);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return err;
    }

    /* Store handles for cleanup */
    internal_mmap_t *m = (internal_mmap_t *)malloc(sizeof(internal_mmap_t));
    if (!m) {
        UnmapViewOfFile(addr);
        CloseHandle(hMapping);
        CloseHandle(hFile);
        csafetensors_free(st);
        snprintf(st->error_msg, sizeof(st->error_msg), "Memory allocation failed");
        return CSAFETENSORS_ERROR_MEMORY_ALLOCATION;
    }

    m->addr = (uint8_t *)addr;
    m->size = (size_t)file_size.QuadPart;
    m->hMapping = hMapping;

    /* Note: We close hFile here as it's no longer needed after CreateFileMapping */
    CloseHandle(hFile);

    st->_internal_mmap = m;
    st->mmap_addr = (const uint8_t *)addr;
    st->mmap_size = (size_t)file_size.QuadPart;
    st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
    st->databuffer_size = st->mmap_size - 8 - st->header_size;

    return CSAFETENSORS_SUCCESS;

#else
    snprintf(st->error_msg, sizeof(st->error_msg), "mmap not supported on this platform");
    return CSAFETENSORS_ERROR_MMAP_FAILED;
#endif
}

csafetensors_error_t csafetensors_mmap_from_memory(const uint8_t *data, size_t size, csafetensors_t *st) {
    if (!data || !st || size < 16) return CSAFETENSORS_ERROR_INVALID_ARGUMENT;

    csafetensors_init(st);

    if (!parse_safetensors_header(data, size, st)) {
        return CSAFETENSORS_ERROR_JSON_PARSE;
    }

    st->mmaped = true;
    st->mmap_addr = data;
    st->mmap_size = size;
    st->databuffer_addr = data + 8 + st->header_size;
    st->databuffer_size = size - 8 - st->header_size;

    return CSAFETENSORS_SUCCESS;
}

const csafetensors_tensor_t *csafetensors_get_tensor(const csafetensors_t *st, const char *name) {
    if (!st || !name) return NULL;
    for (size_t i = 0; i < st->n_tensors; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) {
            return &st->tensors[i];
        }
    }
    return NULL;
}

const csafetensors_tensor_t *csafetensors_get_tensor_by_index(const csafetensors_t *st, size_t index) {
    if (!st || index >= st->n_tensors) return NULL;
    return &st->tensors[index];
}

const uint8_t *csafetensors_get_tensor_data(const csafetensors_t *st, const csafetensors_tensor_t *tensor) {
    if (!st || !tensor) return NULL;

    const uint8_t *base;
    size_t max_size;

    if (st->mmaped) {
        base = st->databuffer_addr;
        max_size = st->databuffer_size;
    } else {
        base = st->storage;
        max_size = st->storage_size;
    }

    if (!base) return NULL;
    if (tensor->data_offset_begin > max_size) return NULL;

    return base + tensor->data_offset_begin;
}

const char *csafetensors_get_metadata(const csafetensors_t *st, const char *key) {
    if (!st || !key) return NULL;
    for (size_t i = 0; i < st->n_metadata; i++) {
        if (strcmp(st->metadata[i].key, key) == 0) {
            return st->metadata[i].value;
        }
    }
    return NULL;
}

size_t csafetensors_dtype_size(csafetensors_dtype_t dtype) {
    switch (dtype) {
        case CSAFETENSORS_DTYPE_BOOL:    return 1;
        case CSAFETENSORS_DTYPE_UINT8:   return 1;
        case CSAFETENSORS_DTYPE_INT8:    return 1;
        case CSAFETENSORS_DTYPE_UINT16:  return 2;
        case CSAFETENSORS_DTYPE_INT16:   return 2;
        case CSAFETENSORS_DTYPE_FLOAT16: return 2;
        case CSAFETENSORS_DTYPE_BFLOAT16:return 2;
        case CSAFETENSORS_DTYPE_UINT32:  return 4;
        case CSAFETENSORS_DTYPE_INT32:   return 4;
        case CSAFETENSORS_DTYPE_FLOAT32: return 4;
        case CSAFETENSORS_DTYPE_UINT64:  return 8;
        case CSAFETENSORS_DTYPE_INT64:   return 8;
        case CSAFETENSORS_DTYPE_FLOAT64: return 8;
        default: return 0;
    }
}

const char *csafetensors_dtype_name(csafetensors_dtype_t dtype) {
    switch (dtype) {
        case CSAFETENSORS_DTYPE_BOOL:    return "BOOL";
        case CSAFETENSORS_DTYPE_UINT8:   return "U8";
        case CSAFETENSORS_DTYPE_INT8:    return "I8";
        case CSAFETENSORS_DTYPE_UINT16:  return "U16";
        case CSAFETENSORS_DTYPE_INT16:   return "I16";
        case CSAFETENSORS_DTYPE_FLOAT16: return "F16";
        case CSAFETENSORS_DTYPE_BFLOAT16:return "BF16";
        case CSAFETENSORS_DTYPE_UINT32:  return "U32";
        case CSAFETENSORS_DTYPE_INT32:   return "I32";
        case CSAFETENSORS_DTYPE_FLOAT32: return "F32";
        case CSAFETENSORS_DTYPE_UINT64:  return "U64";
        case CSAFETENSORS_DTYPE_INT64:   return "I64";
        case CSAFETENSORS_DTYPE_FLOAT64: return "F64";
        default: return "???";
    }
}

size_t csafetensors_shape_size(const csafetensors_tensor_t *tensor) {
    if (!tensor) return 0;
    if (tensor->n_dims == 0) return 1; /* scalar */

    size_t size = 1;
    for (size_t i = 0; i < tensor->n_dims; i++) {
        if (tensor->shape[i] == 0) return 0; /* empty tensor */
        size *= tensor->shape[i];
    }
    return size;
}

bool csafetensors_validate(const csafetensors_t *st) {
    if (!st) return false;

    size_t data_size;
    if (st->mmaped) {
        data_size = st->databuffer_size;
    } else {
        data_size = st->storage_size;
    }

    for (size_t i = 0; i < st->n_tensors; i++) {
        const csafetensors_tensor_t *t = &st->tensors[i];

        if (t->data_offset_begin > t->data_offset_end) {
            snprintf((char *)st->error_msg, sizeof(st->error_msg),
                     "Tensor '%s': begin offset > end offset", t->name);
            return false;
        }

        size_t tensor_size = csafetensors_dtype_size(t->dtype) * csafetensors_shape_size(t);
        if (tensor_size == 0) continue; /* empty tensor is OK */

        if (t->data_offset_end > data_size) {
            snprintf((char *)st->error_msg, sizeof(st->error_msg),
                     "Tensor '%s': data offset exceeds buffer size", t->name);
            return false;
        }

        size_t stored_size = t->data_offset_end - t->data_offset_begin;
        if (tensor_size != stored_size) {
            snprintf((char *)st->error_msg, sizeof(st->error_msg),
                     "Tensor '%s': size mismatch (expected %zu, got %zu)", t->name, tensor_size, stored_size);
            return false;
        }
    }

    return true;
}

const char *csafetensors_get_error(const csafetensors_t *st) {
    if (!st) return "";
    return st->error_msg;
}
