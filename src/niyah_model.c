#include "niyah/niyah.h"
#include "niyah_io.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t k_magic[8] = {'N','I','Y','A','H','B','G','1'};

typedef struct model_header {
    uint8_t magic[8];
    uint32_t version;
    uint32_t alphabet;
    uint64_t bytes_seen;
} model_header;

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9E3779B97F4A7C15ULL;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

void niyah_model_init(niyah_model *model) {
    if (model) memset(model, 0, sizeof(*model));
}

niyah_status niyah_model_train_bytes(niyah_model *model, const uint8_t *data, size_t size) {
    size_t i;
    if (!model || (!data && size != 0)) return NIYAH_ERR_INVALID_ARGUMENT;
    for (i = 0; i < size; ++i) {
        model->unigram[data[i]]++;
        if (i > 0) model->bigram[data[i - 1]][data[i]]++;
    }
    model->bytes_seen += (uint64_t)size;
    return NIYAH_OK;
}

niyah_status niyah_model_train_file(niyah_model *model, const char *path) {
    uint8_t *data = NULL;
    size_t size = 0;
    niyah_status s = niyah_read_file(path, &data, &size);
    if (s != NIYAH_OK) return s;
    s = niyah_model_train_bytes(model, data, size);
    free(data);
    return s;
}

niyah_status niyah_model_save(const niyah_model *model, const char *path) {
    FILE *f;
    model_header h;
    size_t n;
    if (!model || !path) return NIYAH_ERR_INVALID_ARGUMENT;
    memcpy(h.magic, k_magic, sizeof(k_magic));
    h.version = NIYAH_MODEL_VERSION;
    h.alphabet = NIYAH_ALPHABET_SIZE;
    h.bytes_seen = model->bytes_seen;

    f = fopen(path, "wb");
    if (!f) return NIYAH_ERR_IO;
    n = fwrite(&h, 1, sizeof(h), f);
    if (n != sizeof(h)) { fclose(f); return NIYAH_ERR_IO; }
    n = fwrite(model->unigram, sizeof(uint64_t), NIYAH_ALPHABET_SIZE, f);
    if (n != NIYAH_ALPHABET_SIZE) { fclose(f); return NIYAH_ERR_IO; }
    n = fwrite(model->bigram, sizeof(uint64_t), NIYAH_ALPHABET_SIZE * NIYAH_ALPHABET_SIZE, f);
    if (n != NIYAH_ALPHABET_SIZE * NIYAH_ALPHABET_SIZE) { fclose(f); return NIYAH_ERR_IO; }
    return fclose(f) == 0 ? NIYAH_OK : NIYAH_ERR_IO;
}

niyah_status niyah_model_load(niyah_model *model, const char *path) {
    FILE *f;
    model_header h;
    size_t n;
    if (!model || !path) return NIYAH_ERR_INVALID_ARGUMENT;
    f = fopen(path, "rb");
    if (!f) return NIYAH_ERR_IO;
    n = fread(&h, 1, sizeof(h), f);
    if (n != sizeof(h) || memcmp(h.magic, k_magic, sizeof(k_magic)) != 0 ||
        h.version != NIYAH_MODEL_VERSION || h.alphabet != NIYAH_ALPHABET_SIZE) {
        fclose(f);
        return NIYAH_ERR_FORMAT;
    }
    niyah_model_init(model);
    n = fread(model->unigram, sizeof(uint64_t), NIYAH_ALPHABET_SIZE, f);
    if (n != NIYAH_ALPHABET_SIZE) { fclose(f); return NIYAH_ERR_FORMAT; }
    n = fread(model->bigram, sizeof(uint64_t), NIYAH_ALPHABET_SIZE * NIYAH_ALPHABET_SIZE, f);
    if (n != NIYAH_ALPHABET_SIZE * NIYAH_ALPHABET_SIZE) { fclose(f); return NIYAH_ERR_FORMAT; }
    model->bytes_seen = h.bytes_seen;
    if (fgetc(f) != EOF) { fclose(f); return NIYAH_ERR_FORMAT; }
    fclose(f);
    return NIYAH_OK;
}

niyah_status niyah_model_eval_bytes(const niyah_model *model, const uint8_t *data, size_t size,
                                    double *out_bits_per_byte, double *out_perplexity) {
    size_t i;
    double nll_bits = 0.0;
    if (!model || !data || size < 2 || !out_bits_per_byte || !out_perplexity)
        return NIYAH_ERR_INVALID_ARGUMENT;

    for (i = 1; i < size; ++i) {
        uint8_t prev = data[i - 1];
        uint8_t cur = data[i];
        uint64_t row_total = 0;
        unsigned j;
        double p;
        for (j = 0; j < NIYAH_ALPHABET_SIZE; ++j) row_total += model->bigram[prev][j];
        p = ((double)model->bigram[prev][cur] + 1.0) /
            ((double)row_total + (double)NIYAH_ALPHABET_SIZE);
        nll_bits += -log(p) / log(2.0);
    }
    *out_bits_per_byte = nll_bits / (double)(size - 1);
    *out_perplexity = pow(2.0, *out_bits_per_byte);
    return NIYAH_OK;
}

niyah_status niyah_model_eval_file(const niyah_model *model, const char *path,
                                   double *out_bits_per_byte, double *out_perplexity) {
    uint8_t *data = NULL;
    size_t size = 0;
    niyah_status s = niyah_read_file(path, &data, &size);
    if (s != NIYAH_OK) return s;
    if (size < 2) { free(data); return NIYAH_ERR_INVALID_ARGUMENT; }
    s = niyah_model_eval_bytes(model, data, size, out_bits_per_byte, out_perplexity);
    free(data);
    return s;
}

niyah_status niyah_model_generate(const niyah_model *model,
                                  const uint8_t *prompt, size_t prompt_size,
                                  size_t tokens, uint64_t seed,
                                  uint8_t *out, size_t out_capacity,
                                  size_t *out_size) {
    size_t k;
    uint8_t prev;
    uint64_t state = seed;
    if (!model || !out || !out_size || tokens > out_capacity) return NIYAH_ERR_INVALID_ARGUMENT;
    if (prompt_size > 0 && !prompt) return NIYAH_ERR_INVALID_ARGUMENT;

    prev = prompt_size ? prompt[prompt_size - 1] : (uint8_t)' ';
    for (k = 0; k < tokens; ++k) {
        uint64_t total = NIYAH_ALPHABET_SIZE;
        uint64_t r;
        uint64_t acc = 0;
        unsigned j;
        for (j = 0; j < NIYAH_ALPHABET_SIZE; ++j) total += model->bigram[prev][j];
        r = rng_next(&state) % total;
        for (j = 0; j < NIYAH_ALPHABET_SIZE; ++j) {
            acc += model->bigram[prev][j] + 1u;
            if (r < acc) {
                out[k] = (uint8_t)j;
                prev = (uint8_t)j;
                break;
            }
        }
    }
    *out_size = tokens;
    return NIYAH_OK;
}

const char *niyah_status_string(niyah_status status) {
    switch (status) {
        case NIYAH_OK: return "ok";
        case NIYAH_ERR_INVALID_ARGUMENT: return "invalid_argument";
        case NIYAH_ERR_IO: return "io_error";
        case NIYAH_ERR_FORMAT: return "format_error";
        case NIYAH_ERR_NOMEM: return "out_of_memory";
        default: return "unknown_error";
    }
}
