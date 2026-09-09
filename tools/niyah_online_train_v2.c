#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "niyah_mini_config.h"
#include "niyah_mini_model.h"
#include "niyah_mini_train.h"
#include "niyah_window_format.h"

static int ensure_dir(const char *path)
{
    struct stat st;
    if (mkdir(path, 0755) == 0) return 1;
    if (errno != EEXIST) return 0;
    if (stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
}

static int size_mul_ok_local(size_t a, size_t b, size_t *out)
{
    if (!out) return 0;
    if (a != 0U && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static int save_model(NiyahMiniModel *model, const char *out_dir)
{
    char cfg[1024], weights[1024];

    if (!ensure_dir(out_dir)) return 0;
    if (snprintf(cfg, sizeof(cfg), "%s/config.json", out_dir) >= (int)sizeof(cfg)) return 0;
    if (snprintf(weights, sizeof(weights), "%s/weights.f32.bin", out_dir) >= (int)sizeof(weights)) return 0;

    return niyah_mini_model_save(model, cfg, weights) == NIYAH_OK;
}

static int save_checkpoint(NiyahMiniModel *model, const char *out_dir, long optimizer_step)
{
    char dir[1024];

    if (!ensure_dir(out_dir)) return 0;
    if (snprintf(dir, sizeof(dir), "%s/checkpoint-%08ld", out_dir, optimizer_step) >= (int)sizeof(dir))
        return 0;
    if (mkdir(dir, 0755) != 0) return 0;
    if (!save_model(model, dir)) return 0;
    return save_model(model, out_dir);
}

static void grads_add(NiyahMiniGrads *dst, const NiyahMiniGrads *src)
{
    size_t i, n = dst->memory_size / sizeof(float);
    float *d = (float *)dst->memory_block;
    const float *s = (const float *)src->memory_block;

    for (i = 0U; i < n; ++i) d[i] += s[i];
}

static void grads_scale(NiyahMiniGrads *g, float scale)
{
    size_t i, n = g->memory_size / sizeof(float);
    float *p = (float *)g->memory_block;

    for (i = 0U; i < n; ++i) p[i] *= scale;
}

static NiyahStatus apply_accumulated_step(
    NiyahMiniModel *model,
    NiyahMiniGrads *grads,
    NiyahMiniOptimizerState *opt,
    const NiyahMiniConfig *cfg,
    long windows,
    float max_norm,
    float lr,
    float *norm_out)
{
    float norm;

    if (windows <= 0) return NIYAH_ERR_INVALID_ARG;

    grads_scale(grads, 1.0f / (float)windows);
    norm = niyah_mini_clip_grads(grads, cfg, max_norm);

    if (norm_out) *norm_out = norm;

    return niyah_mini_step_adamw(
        model, grads, opt, cfg,
        lr, 0.9f, 0.999f, 1e-8f, 0.01f);
}

int main(int argc, char **argv)
{
    const char *config_path;
    const char *weights_path;
    const char *out_dir;
    const char *format;

    float lr = 1e-4f;
    float max_norm = 1.0f;
    long checkpoint_steps = 100;
    long micro_batch = 1;

    NiyahMiniModel model = {0};
    NiyahMiniGrads grads = {0};
    NiyahMiniGrads accum = {0};
    NiyahMiniOptimizerState opt = {0};
    NiyahMiniTrainCache cache = {0};
    const NiyahMiniConfig *cfg;

    NiyahWindowHeader wh = {0};
    int headered;
    int32_t seq;
    int32_t vocab;

    size_t batch_count;
    size_t batch_bytes;
    size_t logits_count;
    size_t logits_bytes;

    int32_t *batch = NULL;
    float *logits = NULL;
    float *dlogits = NULL;

    long step = 0;
    long optimizer_steps = 0;
    long accumulated_windows = 0;
    double loss_sum = 0.0;
    float first_loss = -1.0f;
    float last_loss = -1.0f;
    int rc = 1;

    if (argc < 4) {
        fprintf(stderr,
            "usage: %s CONFIG WEIGHTS OUT_DIR [lr] [checkpoint_steps] "
            "[max_norm] [micro_batch] [legacy64|niyahw1]\n",
            argv[0]);
        return 2;
    }

    config_path = argv[1];
    weights_path = argv[2];
    out_dir = argv[3];

    if (argc >= 5) lr = strtof(argv[4], NULL);
    if (argc >= 6) checkpoint_steps = atol(argv[5]);
    if (argc >= 7) max_norm = strtof(argv[6], NULL);
    if (argc >= 8) micro_batch = atol(argv[7]);

    format = argc >= 9 ? argv[8] : "legacy64";

    if (!isfinite(lr) || lr <= 0.0f) lr = 1e-4f;
    if (checkpoint_steps <= 0) checkpoint_steps = 100;
    if (!isfinite(max_norm) || max_norm < 0.0f) max_norm = 1.0f;

    if (micro_batch <= 0) {
        fprintf(stderr, "BAD_MICRO_BATCH\n");
        return 3;
    }

    if (niyah_mini_model_load(&model, config_path, weights_path) != NIYAH_OK) {
        fprintf(stderr, "MODEL_LOAD_FAILED\n");
        return 4;
    }

    cfg = &model.config;
    seq = cfg->n_ctx;
    vocab = cfg->n_vocab;

    if (seq <= 0 || seq > NIYAH_MAX_SEQ_LEN || vocab <= 0) {
        fprintf(stderr, "BAD_MODEL_SHAPE ctx=%d vocab=%d\n", seq, vocab);
        goto cleanup;
    }

    headered = strcmp(format, "niyahw1") == 0;

    if (!headered && strcmp(format, "legacy64") != 0) {
        fprintf(stderr, "BAD_WINDOW_FORMAT=%s\n", format);
        goto cleanup;
    }

    if (!headered && seq != 64) {
        fprintf(stderr, "LEGACY64_REQUIRES_CTX64 actual=%d\n", seq);
        goto cleanup;
    }

    if (headered) {
        if (!niyah_window_header_read(stdin, &wh)) {
            fprintf(stderr, "WINDOW_HEADER_INVALID\n");
            goto cleanup;
        }

        if (wh.version != NIYAH_WINDOW_VERSION ||
            wh.sequence_length != (uint32_t)seq ||
            wh.vocabulary_size != (uint32_t)vocab ||
            wh.window_count == 0U) {
            fprintf(stderr,
                "WINDOW_HEADER_MISMATCH version=%u seq=%u vocab=%u windows=%llu "
                "model_seq=%d model_vocab=%d\n",
                wh.version,
                wh.sequence_length,
                wh.vocabulary_size,
                (unsigned long long)wh.window_count,
                seq,
                vocab);
            goto cleanup;
        }
    }

    batch_count = (size_t)seq + 1U;

    if (!size_mul_ok_local(batch_count, sizeof(int32_t), &batch_bytes) ||
        !size_mul_ok_local((size_t)seq, (size_t)vocab, &logits_count) ||
        !size_mul_ok_local(logits_count, sizeof(float), &logits_bytes)) {
        fprintf(stderr, "BUFFER_SIZE_OVERFLOW\n");
        goto cleanup;
    }

    batch = (int32_t *)malloc(batch_bytes);
    logits = (float *)malloc(logits_bytes);
    dlogits = (float *)malloc(logits_bytes);

    if (!batch || !logits || !dlogits) {
        fprintf(stderr, "TRAIN_BUFFER_ALLOC_FAILED\n");
        goto cleanup;
    }

    if (niyah_mini_grads_allocate(&grads, cfg) != NIYAH_OK ||
        niyah_mini_optim_init(&opt, cfg) != NIYAH_OK ||
        niyah_mini_cache_allocate(&cache, cfg, seq) != NIYAH_OK) {
        fprintf(stderr, "TRAIN_STATE_ALLOC_FAILED\n");
        goto cleanup;
    }

    if (micro_batch > 1) {
        if (niyah_mini_grads_allocate(&accum, cfg) != NIYAH_OK ||
            accum.memory_size != grads.memory_size) {
            fprintf(stderr, "ACCUM_GRADS_ALLOC_FAILED\n");
            goto cleanup;
        }
        niyah_mini_grads_zero(&accum, cfg);
    }

    printf("ONLINE_TRAINER_V2_READY\n");
    printf("layers=%d dim=%d vocab=%d ctx=%d format=%s\n",
        cfg->n_layers, cfg->n_dim, vocab, seq, format);
    printf("lr=%.8g checkpoint_steps=%ld max_norm=%.8g micro_batch=%ld\n",
        lr, checkpoint_steps, max_norm, micro_batch);

    if (headered)
        printf("declared_windows=%llu\n", (unsigned long long)wh.window_count);

    fflush(stdout);

    for (;;) {
        size_t got;
        size_t i;
        float loss;
        float grad_norm;

        if (headered && (uint64_t)step >= wh.window_count)
            break;

        got = fread(batch, sizeof(int32_t), batch_count, stdin);

        if (got == 0U) {
            if (headered) {
                fprintf(stderr,
                    "WINDOW_STREAM_TRUNCATED at=%ld expected=%llu\n",
                    step,
                    (unsigned long long)wh.window_count);
                goto cleanup;
            }
            break;
        }

        if (got != batch_count) {
            fprintf(stderr,
                "PARTIAL_WINDOW got=%zu expected=%zu\n",
                got,
                batch_count);
            goto cleanup;
        }

        for (i = 0U; i < batch_count; ++i) {
            if (batch[i] < 0 || batch[i] >= vocab) {
                fprintf(stderr,
                    "BAD_TOKEN window=%ld offset=%zu token=%d vocab=%d\n",
                    step, i, batch[i], vocab);
                goto cleanup;
            }
        }

        niyah_mini_grads_zero(&grads, cfg);

        if (niyah_mini_train_forward(&model, &cache, batch, seq, logits) != NIYAH_OK) {
            fprintf(stderr, "FORWARD_FAILED step=%ld\n", step);
            goto cleanup;
        }

        loss = niyah_mini_loss_and_dlogits(
            logits,
            batch + 1,
            seq,
            vocab,
            dlogits);

        if (!isfinite(loss)) {
            fprintf(stderr, "NONFINITE_LOSS step=%ld\n", step);
            goto cleanup;
        }

        if (step == 0) first_loss = loss;
        last_loss = loss;

        if (niyah_mini_train_backward(
                &model,
                &grads,
                &cache,
                batch,
                dlogits) != NIYAH_OK) {
            fprintf(stderr, "BACKWARD_FAILED step=%ld\n", step);
            goto cleanup;
        }

        if (micro_batch == 1) {
            grad_norm = niyah_mini_clip_grads(&grads, cfg, max_norm);

            if (niyah_mini_step_adamw(
                    &model,
                    &grads,
                    &opt,
                    cfg,
                    lr,
                    0.9f,
                    0.999f,
                    1e-8f,
                    0.01f) != NIYAH_OK) {
                fprintf(stderr, "ADAMW_FAILED step=%ld\n", step);
                goto cleanup;
            }

            optimizer_steps++;
        } else {
            grad_norm = niyah_mini_clip_grads(&grads, cfg, 0.0f);
            grads_add(&accum, &grads);
            accumulated_windows++;

            if (accumulated_windows == micro_batch) {
                float accumulated_norm = 0.0f;

                if (apply_accumulated_step(
                        &model,
                        &accum,
                        &opt,
                        cfg,
                        accumulated_windows,
                        max_norm,
                        lr,
                        &accumulated_norm) != NIYAH_OK) {
                    fprintf(stderr,
                        "ADAMW_FAILED optimizer_step=%ld\n",
                        optimizer_steps);
                    goto cleanup;
                }

                optimizer_steps++;
                accumulated_windows = 0;
                niyah_mini_grads_zero(&accum, cfg);
            }
        }

        step++;
        loss_sum += loss;

        if (step == 1 || step % 25 == 0) {
            printf(
                "step=%ld loss=%.6f grad_norm=%.6f optimizer_steps=%ld\n",
                step,
                loss,
                grad_norm,
                optimizer_steps);
            fflush(stdout);
        }

        if (optimizer_steps > 0 &&
            accumulated_windows == 0 &&
            optimizer_steps % checkpoint_steps == 0) {
            if (!save_checkpoint(&model, out_dir, optimizer_steps)) {
                fprintf(stderr,
                    "CHECKPOINT_FAILED optimizer_step=%ld\n",
                    optimizer_steps);
                goto cleanup;
            }

            printf(
                "CHECKPOINT optimizer_step=%ld windows=%ld\n",
                optimizer_steps,
                step);
            fflush(stdout);
        }
    }

    if (ferror(stdin)) {
        fprintf(stderr, "STDIN_READ_ERROR\n");
        goto cleanup;
    }

    if (headered) {
        unsigned char extra;

        if ((uint64_t)step != wh.window_count) {
            fprintf(stderr,
                "WINDOW_COUNT_MISMATCH got=%ld expected=%llu\n",
                step,
                (unsigned long long)wh.window_count);
            goto cleanup;
        }

        if (fread(&extra, 1U, 1U, stdin) != 0U) {
            fprintf(stderr, "WINDOW_STREAM_HAS_TRAILING_BYTES\n");
            goto cleanup;
        }

        if (ferror(stdin)) {
            fprintf(stderr, "STDIN_READ_ERROR_AFTER_WINDOWS\n");
            goto cleanup;
        }
    }

    if (micro_batch > 1 && accumulated_windows > 0) {
        float accumulated_norm = 0.0f;

        if (apply_accumulated_step(
                &model,
                &accum,
                &opt,
                cfg,
                accumulated_windows,
                max_norm,
                lr,
                &accumulated_norm) != NIYAH_OK) {
            fprintf(stderr, "FINAL_ACCUM_ADAMW_FAILED\n");
            goto cleanup;
        }

        optimizer_steps++;
        accumulated_windows = 0;
        niyah_mini_grads_zero(&accum, cfg);
    }

    if (!save_model(&model, out_dir)) {
        fprintf(stderr, "FINAL_SAVE_FAILED\n");
        goto cleanup;
    }

    printf("ONLINE_TRAINING_COMPLETE\n");
    printf("steps=%ld\n", step);
    printf("optimizer_steps=%ld\n", optimizer_steps);
    printf("micro_batch=%ld\n", micro_batch);
    printf("first_loss=%.6f\n", first_loss);
    printf("last_loss=%.6f\n", last_loss);
    printf("avg_loss=%.6f\n", step > 0 ? loss_sum / (double)step : 0.0);

    rc = 0;

cleanup:
    free(batch);
    free(logits);
    free(dlogits);
    niyah_mini_cache_free(&cache);
    niyah_mini_optim_free(&opt);

    if (accum.memory_block)
        niyah_mini_grads_free(&accum);

    if (grads.memory_block)
        niyah_mini_grads_free(&grads);

    niyah_mini_model_free(&model);
    return rc;
}
