#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>

#include "niyah_mini_config.h"
#include "niyah_mini_model.h"
#include "niyah_mini_train.h"

#define SEQ 64

static int ensure_dir(const char *path)
{
    struct stat st;

    if (mkdir(path, 0755) == 0)
        return 1;

    if (errno != EEXIST)
        return 0;

    if (stat(path, &st) != 0)
        return 0;

    return S_ISDIR(st.st_mode);
}

static int save_model(
    NiyahMiniModel *model,
    const char *out_dir
) {
    char cfg[1024];
    char weights[1024];

    if (!ensure_dir(out_dir))
        return 0;

    if (snprintf(cfg, sizeof(cfg),
                 "%s/config.json", out_dir) >= (int)sizeof(cfg)) {
        return 0;
    }

    if (snprintf(weights, sizeof(weights),
                 "%s/weights.f32.bin", out_dir) >= (int)sizeof(weights)) {
        return 0;
    }

    if (niyah_mini_model_save(
            model, cfg, weights
        ) != NIYAH_OK) {
        return 0;
    }

    return 1;
}

static int save_checkpoint(
    NiyahMiniModel *model,
    const char *out_dir,
    long optimizer_step
) {
    char checkpoint_dir[1024];

    if (!ensure_dir(out_dir))
        return 0;

    if (snprintf(
            checkpoint_dir,
            sizeof(checkpoint_dir),
            "%s/checkpoint-%08ld",
            out_dir,
            optimizer_step
        ) >= (int)sizeof(checkpoint_dir)) {
        return 0;
    }

    /*
     * Checkpoints are immutable evidence. Refuse to overwrite an existing
     * checkpoint directory, even if a previous run used the same step.
     */
    if (mkdir(checkpoint_dir, 0755) != 0)
        return 0;

    if (!save_model(model, checkpoint_dir))
        return 0;

    /* Preserve the historical live-head behavior for existing tooling. */
    if (!save_model(model, out_dir))
        return 0;

    return 1;
}

static void grads_add(
    NiyahMiniGrads *dst,
    const NiyahMiniGrads *src
) {
    size_t n = dst->memory_size / sizeof(float);
    float *d = (float *)dst->memory_block;
    const float *s = (const float *)src->memory_block;

    for (size_t i = 0; i < n; ++i)
        d[i] += s[i];
}

static void grads_scale(
    NiyahMiniGrads *grads,
    float scale
) {
    size_t n = grads->memory_size / sizeof(float);
    float *g = (float *)grads->memory_block;

    for (size_t i = 0; i < n; ++i)
        g[i] *= scale;
}

static NiyahStatus apply_accumulated_step(
    NiyahMiniModel *model,
    NiyahMiniGrads *grads,
    NiyahMiniOptimizerState *opt,
    const NiyahMiniConfig *cfg,
    long accumulated_windows,
    float max_norm,
    float lr,
    float *grad_norm_out
) {
    if (accumulated_windows <= 0)
        return NIYAH_ERR_INVALID_ARG;

    grads_scale(
        grads,
        1.0f / (float)accumulated_windows
    );

    float grad_norm = niyah_mini_clip_grads(
        grads,
        cfg,
        max_norm
    );

    if (grad_norm_out)
        *grad_norm_out = grad_norm;

    return niyah_mini_step_adamw(
        model,
        grads,
        opt,
        cfg,
        lr,
        0.9f,
        0.999f,
        1e-8f,
        0.01f
    );
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s CONFIG WEIGHTS OUT_DIR "
            "[lr] [checkpoint_steps] [max_norm] [micro_batch]\n",
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

    long micro_batch = 1;
    if (argc >= 8) {
        char *end = NULL;
        long parsed = strtol(argv[7], &end, 10);
        if (end != argv[7] &&
            *end == '\0' &&
            parsed > 0) {
            micro_batch = parsed;
        } else {
            fprintf(stderr, "BAD_MICRO_BATCH=%s\n", argv[7]);
            return 3;
        }
    }

    if (lr <= 0.0f)
        lr = 1e-4f;

    if (checkpoint_steps <= 0)
        checkpoint_steps = 100;

    NiyahMiniModel model;
    NiyahMiniGrads grads;
    NiyahMiniGrads accum_grads;
    NiyahMiniOptimizerState opt;
    NiyahMiniTrainCache cache;

    memset(&model, 0, sizeof(model));
    memset(&grads, 0, sizeof(grads));
    memset(&accum_grads, 0, sizeof(accum_grads));
    memset(&opt, 0, sizeof(opt));
    memset(&cache, 0, sizeof(cache));

    if (niyah_mini_model_load(
            &model,
            config_path,
            weights_path
        ) != NIYAH_OK) {
        fprintf(stderr, "MODEL_LOAD_FAILED\n");
        return 4;
    }

    const NiyahMiniConfig *cfg = &model.config;

    if (cfg->n_ctx < SEQ) {
        fprintf(stderr,
            "MODEL_CTX_TOO_SMALL=%d\n",
            cfg->n_ctx);
        return 5;
    }

    const int V = cfg->n_vocab;

    if (niyah_mini_grads_allocate(
            &grads, cfg
        ) != NIYAH_OK) {
        fprintf(stderr, "GRADS_ALLOC_FAILED\n");
        return 6;
    }

    if (micro_batch > 1) {
        if (niyah_mini_grads_allocate(
                &accum_grads, cfg
            ) != NIYAH_OK) {
            fprintf(stderr, "ACCUM_GRADS_ALLOC_FAILED\n");
            return 7;
        }

        if (accum_grads.memory_size != grads.memory_size) {
            fprintf(stderr, "ACCUM_GRADS_LAYOUT_MISMATCH\n");
            return 8;
        }

        niyah_mini_grads_zero(
            &accum_grads, cfg
        );
    }

    /*
     * Optimizer moments start fresh on this online run.
     * Model weights themselves resume from checkpoint.
     */
    if (niyah_mini_optim_init(
            &opt, cfg
        ) != NIYAH_OK) {
        fprintf(stderr, "OPT_INIT_FAILED\n");
        return 9;
    }

    if (niyah_mini_cache_allocate(
            &cache, cfg, SEQ
        ) != NIYAH_OK) {
        fprintf(stderr, "CACHE_ALLOC_FAILED\n");
        return 10;
    }

    float *logits = malloc(
        (size_t)SEQ * (size_t)V * sizeof(float)
    );

    float *dlogits = malloc(
        (size_t)SEQ * (size_t)V * sizeof(float)
    );

    if (!logits || !dlogits) {
        fprintf(stderr, "LOGITS_ALLOC_FAILED\n");
        return 11;
    }

    int32_t batch[SEQ + 1];

    long step = 0;
    long optimizer_steps = 0;
    long accumulated_windows = 0;
    double loss_sum = 0.0;
    float first_loss = -1.0f;
    float last_loss = -1.0f;

    printf(
        "ONLINE_TRAINER_READY\n"
        "layers=%d dim=%d vocab=%d ctx=%d\n"
        "lr=%.8g checkpoint_steps=%ld max_norm=%.8g micro_batch=%ld\n",
        cfg->n_layers,
        cfg->n_dim,
        cfg->n_vocab,
        cfg->n_ctx,
        lr,
        checkpoint_steps,
        max_norm,
        micro_batch
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

            free(logits);
            free(dlogits);
            niyah_mini_cache_free(&cache);
            niyah_mini_optim_free(&opt);
            if (micro_batch > 1)
                niyah_mini_grads_free(&accum_grads);
            niyah_mini_grads_free(&grads);
            niyah_mini_model_free(&model);
            return 22;
        }

        for (int i = 0; i < SEQ + 1; ++i) {
            if (batch[i] < 0 ||
                batch[i] >= V) {
                fprintf(stderr,
                    "BAD_TOKEN=%d vocab=%d\n",
                    batch[i], V);
                return 12;
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
            return 13;
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
            return 14;
        }

        float grad_norm;

        if (micro_batch == 1) {
            /*
             * Preserve the pre-accumulation execution order exactly:
             * zero -> forward -> backward -> clip -> AdamW.
             */
            grad_norm = niyah_mini_clip_grads(
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
                return 15;
            }

            optimizer_steps++;
        } else {
            /* Raw per-window norm only; max_norm=0 performs no rescale. */
            grad_norm = niyah_mini_clip_grads(
                &grads,
                &model.config,
                0.0f
            );

            grads_add(
                &accum_grads,
                &grads
            );

            accumulated_windows++;

            if (accumulated_windows == micro_batch) {
                float accumulated_norm = 0.0f;

                if (apply_accumulated_step(
                        &model,
                        &accum_grads,
                        &opt,
                        cfg,
                        accumulated_windows,
                        max_norm,
                        lr,
                        &accumulated_norm
                    ) != NIYAH_OK) {
                    fprintf(stderr,
                        "ADAMW_FAILED optimizer_step=%ld\n",
                        optimizer_steps);
                    return 16;
                }

                if (max_norm > 0.0f &&
                    accumulated_norm > max_norm) {
                    printf(
                        "clipped_accum_norm=%.6f windows=%ld\n",
                        max_norm,
                        accumulated_windows
                    );
                    fflush(stdout);
                }

                optimizer_steps++;
                accumulated_windows = 0;

                niyah_mini_grads_zero(
                    &accum_grads, cfg
                );
            }
        }

        step++;
        loss_sum += loss;

        if (step == 1 || step % 25 == 0) {
            printf(
                "step=%ld loss=%.6f "
                "grad_norm=%.6f optimizer_steps=%ld\n",
                step,
                loss,
                grad_norm,
                optimizer_steps
            );
            fflush(stdout);
        }

        if (optimizer_steps > 0 &&
            accumulated_windows == 0 &&
            optimizer_steps % checkpoint_steps == 0) {
            if (!save_checkpoint(
                    &model,
                    out_dir,
                    optimizer_steps)) {
                fprintf(stderr,
                    "CHECKPOINT_FAILED optimizer_step=%ld\n",
                    optimizer_steps);
                return 17;
            }

            printf(
                "CHECKPOINT optimizer_step=%ld windows=%ld\n",
                optimizer_steps,
                step
            );
            fflush(stdout);
        }
    }

    if (ferror(stdin)) {
        fprintf(stderr, "STDIN_READ_ERROR\n");
        return 18;
    }

    /*
     * Do not discard a final short accumulation group. Average by the
     * actual group size, clip once, then take one final optimizer step.
     */
    if (micro_batch > 1 &&
        accumulated_windows > 0) {
        float accumulated_norm = 0.0f;

        if (apply_accumulated_step(
                &model,
                &accum_grads,
                &opt,
                cfg,
                accumulated_windows,
                max_norm,
                lr,
                &accumulated_norm
            ) != NIYAH_OK) {
            fprintf(stderr,
                "ADAMW_FAILED optimizer_step=%ld\n",
                optimizer_steps);
            return 19;
        }

        if (max_norm > 0.0f &&
            accumulated_norm > max_norm) {
            printf(
                "clipped_accum_norm=%.6f windows=%ld\n",
                max_norm,
                accumulated_windows
            );
            fflush(stdout);
        }

        optimizer_steps++;
        accumulated_windows = 0;

        niyah_mini_grads_zero(
            &accum_grads, cfg
        );

        if (optimizer_steps % checkpoint_steps == 0) {
            if (!save_checkpoint(
                    &model,
                    out_dir,
                    optimizer_steps)) {
                fprintf(stderr,
                    "CHECKPOINT_FAILED optimizer_step=%ld\n",
                    optimizer_steps);
                return 20;
            }

            printf(
                "CHECKPOINT optimizer_step=%ld windows=%ld\n",
                optimizer_steps,
                step
            );
            fflush(stdout);
        }
    }

    if (!save_model(&model, out_dir)) {
        fprintf(stderr, "FINAL_SAVE_FAILED\n");
        return 21;
    }

    printf(
        "ONLINE_TRAINING_COMPLETE\n"
        "steps=%ld\n"
        "optimizer_steps=%ld\n"
        "micro_batch=%ld\n"
        "first_loss=%.6f\n"
        "last_loss=%.6f\n"
        "avg_loss=%.6f\n",
        step,
        optimizer_steps,
        micro_batch,
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

    if (micro_batch > 1)
        niyah_mini_grads_free(&accum_grads);

    niyah_mini_grads_free(&grads);
    niyah_mini_model_free(&model);

    return 0;
}
