#ifndef NIYAH_NIYAH_H
#define NIYAH_NIYAH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_MODEL_VERSION 1u
#define NIYAH_ALPHABET_SIZE 256u

typedef enum niyah_status {
    NIYAH_OK = 0,
    NIYAH_ERR_INVALID_ARGUMENT = 1,
    NIYAH_ERR_IO = 2,
    NIYAH_ERR_FORMAT = 3,
    NIYAH_ERR_NOMEM = 4
} niyah_status;

typedef struct niyah_model {
    uint64_t unigram[NIYAH_ALPHABET_SIZE];
    uint64_t bigram[NIYAH_ALPHABET_SIZE][NIYAH_ALPHABET_SIZE];
    uint64_t bytes_seen;
} niyah_model;

void niyah_model_init(niyah_model *model);
niyah_status niyah_model_train_bytes(niyah_model *model, const uint8_t *data, size_t size);
niyah_status niyah_model_train_file(niyah_model *model, const char *path);
niyah_status niyah_model_save(const niyah_model *model, const char *path);
niyah_status niyah_model_load(niyah_model *model, const char *path);
niyah_status niyah_model_eval_bytes(const niyah_model *model, const uint8_t *data, size_t size,
                                    double *out_bits_per_byte, double *out_perplexity);
niyah_status niyah_model_eval_file(const niyah_model *model, const char *path,
                                   double *out_bits_per_byte, double *out_perplexity);
niyah_status niyah_model_generate(const niyah_model *model,
                                  const uint8_t *prompt, size_t prompt_size,
                                  size_t tokens, uint64_t seed,
                                  uint8_t *out, size_t out_capacity,
                                  size_t *out_size);

const char *niyah_status_string(niyah_status status);

#ifdef __cplusplus
}
#endif

#endif
