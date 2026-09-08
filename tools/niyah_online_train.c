#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#include "niyah_mini_config.h"
#include "niyah_mini_model.h"
#include "niyah_mini_train.h"

#define SEQ 64

static int save_model(
    NiyahMiniModel *model,
    const char *out_dir
) {
    char cfg[1024];
    char weights[1024];

    mkdir(out_dir, 0755);

    snprintf(cfg, sizeof(cfg),
             "%s/config.json", out_dir);

    snprintf(weights, sizeof(weights),
             "%s/weights.f32.bin", out_dir);

    if (niyah_mini_model_save(
            model, cfg, weights
        ) != NIYAH_OK) {
        return 0;
    }

    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s CONFIG WEIGHTS OUT_DIR "
            "[lr] [checkpoint_steps] [max_norm]\n",
            argv[0]);
        return 2;
    }

    const char *config_path = argv[1];
    const char *weights_path = argv[2];
    const char *out_dir = argv[3];

    float lr = argc >= 5
        ? strtof(argv[4], NULL)
        : 1e-4f;

    long checkpoint_steps = argc >= 6
        ? atol(argv[5])
        : 100;

    float max_norm = 1.0f;
    if (argc >= 7) {
        char *end = NULL;
        float parsed = strtof(argv[6], &end);
        if (end != argv[6] &&
            *end == '\0' &&
            isfinite(parsed) &&
            parsed >= 0.0f) {
            max_norm = parsed;
        }
    }

    if (lr <= 0.0f) lr = 1e-4f;
    if (checkpoint_steps <= 0)
        checkpoint_steps = 100;

    NiyahMiniModel model;
    NiyahMiniGrads grads;
    NiyahMiniOptimizerState opt;
    NiyahMiniTrainCache cache;

    memset(&model, 0, sizeof(model));
    memset(&grads, 0, sizeof(grads));
    memset(&opt, 0, sizeof(opt));
    memset(&cache, 0, sizeof(cache));

    if (niyah_mini_model_load(
            &model,
            config_path,
            weights_path
        ) != NIYAH_OK) {
        fprintf(stderr, "MODEL_LOAD_FAILED\n");
        return 3;
    }

    const NiyahMiniConfig *cfg = &model.config;

    if (cfg->n_ctx < SEQ) {
        fprintf(stderr,
            "MODEL_CTX_TOO_SMALL=%d\n",
            cfg->n_ctx);
        return 4;
    }

    const int V = cfg->n_vocab;

    if (niyah_mini_grads_allocate(
            &grads, cfg
        ) != NIYAH_OK) {
        fprintf(stderr, "GRADS_ALLOC_FAILED\n");
        return 5;
    }

    /*
     * Optimizer moments start fresh on this online run.
     * Model weights themselves resume from checkpoint.
     */
    if (niyah_mini_optim_init(
            &opt, cfg
        ) != NIYAH_OK) {
        fprintf(stderr, "OPT_INIT_FAILED\n");
        return 6;
    }

    if (niyah_mini_cache_allocate(
            &cache, cfg, SEQ
        ) != NIYAH_OK) {
        fprintf(stderr, "CACHE_ALLOC_FAILED\n");
        return 7;
    }

    float *logits = malloc(
        (size_t)SEQ * (size_t)V * sizeof(float)
    );

    float *dlogits = malloc(
        (size_t)SEQ * (size_t)V * sizeof(float)
    );

    if (!logits || !dlogits) {
        fprintf(stderr, "LOGITS_ALLOC_FAILED\n");
        return 8;
    }

    int32_t batch[SEQ + 1];

    long step = 0;
    double loss_sum = 0.0;
    float first_loss = -1.0f;
    float last_loss = -1.0f;

    printf(
        "ONLINE_TRAINER_READY\n"
        "layers=%d dim=%d vocab=%d ctx=%d\n"
        "lr=%.8g checkpoint_steps=%ld max_norm=%.8g\n",
        cfg->n_layers,
        cfg->n_dim,
        cfg->n_vocab,
        cfg->n_ctx,
        lr,
        checkpoint_steps,
        max_norm
    );
    fflush(stdout);

    while (1) {
        size_t got = fread(
            batch,
            sizeof(int32_t),
            SEQ + 1,
            stdin
        );

        if (got == 0)
            break;

        if (got != SEQ + 1) {
            fprintf(stderr,
                "PARTIAL_BATCH=%zu\n", got);
            break;
        }

        for (int i = 0; i < SEQ + 1; ++i) {
            if (batch[i] < 0 ||
                batch[i] >= V) {
                fprintf(stderr,
                    "BAD_TOKEN=%d vocab=%d\n",
                    batch[i], V);
                return 9;
            }
        }

        niyah_mini_grads_zero(
            &grads, cfg
        );

        if (niyah_mini_train_forward(
                &model,
                &cache,
                batch,
                SEQ,
                logits
            ) != NIYAH_OK) {
            fprintf(stderr,
                "FORWARD_FAILED step=%ld\n",
                step);
            return 10;
        }

        float loss =
            niyah_mini_loss_and_dlogits(
                logits,
                batch + 1,
                SEQ,
                V,
                dlogits
            );

        if (step == 0)
            first_loss = loss;

        last_loss = loss;

        if (niyah_mini_train_backward(
                &model,
                &grads,
                &cache,
                batch,
                dlogits
            ) != NIYAH_OK) {
            fprintf(stderr,
                "BACKWARD_FAILED step=%ld\n",
                step);
            return 11;
        }

        float grad_norm =
            niyah_mini_clip_grads(
                &grads,
                &model.config,
                max_norm
            );

        if (max_norm > 0.0f &&
            grad_norm > max_norm) {
            printf(
                "clipped_norm=%.6f\n",
                max_norm
            );
            fflush(stdout);
        }

        if (niyah_mini_step_adamw(
                &model,
                &grads,
                &opt,
                cfg,
                lr,
                0.9f,
                0.999f,
                1e-8f,
                0.01f
            ) != NIYAH_OK) {
            fprintf(stderr,
                "ADAMW_FAILED step=%ld\n",
                step);
            return 12;
        }

        step++;
        loss_sum += loss;

        if (step == 1 || step % 25 == 0) {
            printf(
                "step=%ld loss=%.6f "
                "grad_norm=%.6f\n",
                step,
                loss,
                grad_norm
            );
            fflush(stdout);
        }

        if (step % checkpoint_steps == 0) {
            if (!save_model(
                    &model,
                    out_dir)) {
                fprintf(stderr,
                    "CHECKPOINT_FAILED\n");
                return 13;
            }

            printf(
                "CHECKPOINT step=%ld\n",
                step
            );
            fflush(stdout);
        }
    }

    if (ferror(stdin)) {
        fprintf(stderr, "STDIN_READ_ERROR\n");
        return 14;
    }

    if (!save_model(&model, out_dir)) {
        fprintf(stderr, "FINAL_SAVE_FAILED\n");
        return 15;
    }

    printf(
        "ONLINE_TRAINING_COMPLETE\n"
        "steps=%ld\n"
        "first_loss=%.6f\n"
        "last_loss=%.6f\n"
        "avg_loss=%.6f\n",
        step,
        first_loss,
        last_loss,
        step > 0
            ? loss_sum / (double)step
            : 0.0
    );

    free(logits);
    free(dlogits);

    niyah_mini_cache_free(&cache);
    niyah_mini_optim_free(&opt);
    niyah_mini_grads_free(&grads);
    niyah_mini_model_free(&model);

    return 0;
}
