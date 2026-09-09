#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "niyah_mini_model.h"
#include "niyah_mini_train.h"

#define SEQ 64

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr,
            "usage: %s CONFIG WEIGHTS EVAL_BIN\n",
            argv[0]);
        return 2;
    }

    NiyahMiniModel model;
    NiyahMiniTrainCache cache;

    memset(&model, 0, sizeof(model));
    memset(&cache, 0, sizeof(cache));

    if (niyah_mini_model_load(
            &model,
            argv[1],
            argv[2]
        ) != NIYAH_OK) {
        fprintf(stderr, "MODEL_LOAD_FAILED\n");
        return 3;
    }

    const int V = model.config.n_vocab;

    if (model.config.n_ctx < SEQ) {
        fprintf(stderr, "CTX_TOO_SMALL\n");
        return 4;
    }

    if (niyah_mini_cache_allocate(
            &cache,
            &model.config,
            SEQ
        ) != NIYAH_OK) {
        fprintf(stderr, "CACHE_ALLOC_FAILED\n");
        return 5;
    }

    float *logits = malloc(
        (size_t)SEQ * V * sizeof(float)
    );

    float *dlogits = malloc(
        (size_t)SEQ * V * sizeof(float)
    );

    if (!logits || !dlogits) {
        fprintf(stderr, "ALLOC_FAILED\n");
        return 6;
    }

    FILE *f = fopen(argv[3], "rb");

    if (!f) {
        perror("fopen");
        return 7;
    }

    int32_t batch[SEQ + 1];

    long windows = 0;
    double sum = 0.0;
    float min_loss = 1e30f;
    float max_loss = -1e30f;

    while (1) {
        size_t got = fread(
            batch,
            sizeof(int32_t),
            SEQ + 1,
            f
        );

        if (got == 0)
            break;

        if (got != SEQ + 1) {
            fprintf(stderr,
                "PARTIAL_WINDOW=%zu\n",
                got);
            return 8;
        }

        for (int i = 0; i < SEQ + 1; ++i) {
            if (batch[i] < 0 || batch[i] >= V) {
                fprintf(stderr,
                    "BAD_TOKEN=%d vocab=%d\n",
                    batch[i], V);
                return 9;
            }
        }

        if (niyah_mini_train_forward(
                &model,
                &cache,
                batch,
                SEQ,
                logits
            ) != NIYAH_OK) {
            fprintf(stderr,
                "FORWARD_FAILED window=%ld\n",
                windows);
            return 10;
        }

        float loss = niyah_mini_loss_and_dlogits(
            logits,
            batch + 1,
            SEQ,
            V,
            dlogits
        );

        if (!isfinite(loss)) {
            fprintf(stderr,
                "NONFINITE_LOSS window=%ld\n",
                windows);
            return 11;
        }

        sum += loss;

        if (loss < min_loss)
            min_loss = loss;

        if (loss > max_loss)
            max_loss = loss;

        windows++;
    }

    fclose(f);

    if (windows == 0) {
        fprintf(stderr, "NO_EVAL_WINDOWS\n");
        return 12;
    }

    double avg = sum / (double)windows;

    printf("EVAL_COMPLETE\n");
    printf("windows=%ld\n", windows);
    printf("avg_loss=%.6f\n", avg);
    printf("perplexity=%.6f\n", exp(avg));
    printf("min_loss=%.6f\n", min_loss);
    printf("max_loss=%.6f\n", max_loss);

    free(logits);
    free(dlogits);

    niyah_mini_cache_free(&cache);
    niyah_mini_model_free(&model);

    return 0;
}
