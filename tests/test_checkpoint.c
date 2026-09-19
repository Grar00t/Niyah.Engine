#include "niyah/checkpoint.h"
#include "niyah/tokenizer.h"
#include "niyah/optimizer.h"
#include "niyah/transformer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *test_fopen(const char *path, const char *mode)
{
#if defined(_MSC_VER)
    FILE *file = NULL;
    if (fopen_s(&file, path, mode) != 0) {
        return NULL;
    }
    return file;
#else
    return fopen(path, mode);
#endif
}

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        failures += 1; \
    } \
} while (0)

typedef struct TestSections {
    size_t header_offset[5];
    size_t payload_offset[5];
    uint64_t payload_bytes[5];
    size_t footer_offset;
} TestSections;

static uint32_t load_u32_le(const unsigned char *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t load_u64_le(const unsigned char *p)
{
    uint64_t value = UINT64_C(0);
    size_t i;
    for (i = 0U; i < 8U; ++i) {
        value |= ((uint64_t)p[i]) << (i * 8U);
    }
    return value;
}

static void store_u32_le(unsigned char *p, uint32_t value)
{
    p[0] = (unsigned char)(value & UINT32_C(0xff));
    p[1] = (unsigned char)((value >> 8) & UINT32_C(0xff));
    p[2] = (unsigned char)((value >> 16) & UINT32_C(0xff));
    p[3] = (unsigned char)((value >> 24) & UINT32_C(0xff));
}

static void store_u64_le(unsigned char *p, uint64_t value)
{
    size_t i;
    for (i = 0U; i < 8U; ++i) {
        p[i] = (unsigned char)((value >> (i * 8U)) & UINT64_C(0xff));
    }
}

static uint32_t float_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t crc32_reference(const unsigned char *data, size_t size)
{
    uint32_t crc = UINT32_C(0xffffffff);
    size_t i;
    for (i = 0U; i < size; ++i) {
        unsigned bit;
        crc ^= (uint32_t)data[i];
        for (bit = 0U; bit < 8U; ++bit) {
            crc = (crc & UINT32_C(1)) != 0U
                ? (crc >> 1) ^ UINT32_C(0xedb88320)
                : crc >> 1;
        }
    }
    return crc ^ UINT32_C(0xffffffff);
}

static NiyahModelConfig test_model_config(int tied)
{
    NiyahModelConfig c;
    memset(&c, 0, sizeof(c));
    c.vocab_size = 8U;
    c.context_length = 4U;
    c.embedding_dim = 4U;
    c.n_layers = 1U;
    c.n_heads = 2U;
    c.n_kv_heads = 1U;
    c.ffn_hidden_dim = 8U;
    c.rms_norm_eps = 1.0e-5f;
    c.tie_word_embeddings = tied;
    return c;
}

static NiyahAdamWConfig test_optimizer_config(void)
{
    NiyahAdamWConfig c;
    c.learning_rate = 5.0e-3f;
    c.beta1 = 0.9f;
    c.beta2 = 0.999f;
    c.epsilon = 1.0e-8f;
    c.weight_decay = 1.0e-2f;
    c.max_grad_norm = 1.0f;
    return c;
}

static int config_equal(const NiyahModelConfig *a, const NiyahModelConfig *b)
{
    return a->vocab_size == b->vocab_size &&
           a->context_length == b->context_length &&
           a->embedding_dim == b->embedding_dim &&
           a->n_layers == b->n_layers &&
           a->n_heads == b->n_heads &&
           a->n_kv_heads == b->n_kv_heads &&
           a->ffn_hidden_dim == b->ffn_hidden_dim &&
           float_bits(a->rms_norm_eps) == float_bits(b->rms_norm_eps) &&
           (a->tie_word_embeddings != 0) == (b->tie_word_embeddings != 0);
}

static int layout_equal(const NiyahModelLayout *a, const NiyahModelLayout *b)
{
    return a->token_embedding == b->token_embedding &&
           a->layers == b->layers &&
           a->layer_stride == b->layer_stride &&
           a->final_norm == b->final_norm &&
           a->lm_head == b->lm_head &&
           a->total_floats == b->total_floats &&
           a->head_dim == b->head_dim &&
           a->kv_dim == b->kv_dim;
}

static int optimizer_config_equal(const NiyahAdamWConfig *a,
                                  const NiyahAdamWConfig *b)
{
    return float_bits(a->learning_rate) == float_bits(b->learning_rate) &&
           float_bits(a->beta1) == float_bits(b->beta1) &&
           float_bits(a->beta2) == float_bits(b->beta2) &&
           float_bits(a->epsilon) == float_bits(b->epsilon) &&
           float_bits(a->weight_decay) == float_bits(b->weight_decay) &&
           float_bits(a->max_grad_norm) == float_bits(b->max_grad_norm);
}

static int write_bytes(const char *path, const unsigned char *data, size_t size)
{
    FILE *f = test_fopen(path, "wb");
    int ok;
    if (f == NULL) return 0;
    ok = fwrite(data, 1U, size, f) == size;
    if (fclose(f) != 0) ok = 0;
    return ok;
}

static unsigned char *read_bytes(const char *path, size_t *out_size)
{
    FILE *f = test_fopen(path, "rb");
    long end;
    unsigned char *data;
    if (f == NULL) return NULL;
    if (fseek(f, 0L, SEEK_END) != 0) { fclose(f); return NULL; }
    end = ftell(f);
    if (end < 0 || fseek(f, 0L, SEEK_SET) != 0) { fclose(f); return NULL; }
    data = (unsigned char *)malloc((size_t)end == 0U ? 1U : (size_t)end);
    if (data == NULL) { fclose(f); return NULL; }
    if (fread(data, 1U, (size_t)end, f) != (size_t)end) {
        free(data); fclose(f); return NULL;
    }
    if (fclose(f) != 0) { free(data); return NULL; }
    *out_size = (size_t)end;
    return data;
}

static int locate_sections(const unsigned char *data, size_t size, TestSections *out)
{
    size_t cursor = 68U;
    uint32_t count;
    uint32_t i;
    memset(out, 0, sizeof(*out));
    if (size < 76U) return 0;
    count = load_u32_le(data + 16U);
    if (count > 64U) return 0;
    for (i = 0U; i < count; ++i) {
        uint32_t id;
        uint64_t payload;
        if (cursor > size || size - cursor < 16U) return 0;
        id = load_u32_le(data + cursor);
        payload = load_u64_le(data + cursor + 8U);
        if (id <= 4U) {
            out->header_offset[id] = cursor;
            out->payload_offset[id] = cursor + 16U;
            out->payload_bytes[id] = payload;
        }
        cursor += 16U;
        if (payload > (uint64_t)(SIZE_MAX - cursor)) return 0;
        cursor += (size_t)payload;
        if (cursor > size) return 0;
    }
    if (size - cursor < 8U) return 0;
    out->footer_offset = cursor;
    return 1;
}

static void test_locate_sections_rejects_short_buffer(void)
{
    static const unsigned char short_buffer[40] = {0};
    TestSections s;
    size_t i;

    memset(&s, 0xff, sizeof(s));
    CHECK(locate_sections(short_buffer, sizeof(short_buffer), &s) == 0);
    CHECK(s.footer_offset == 0U);
    for (i = 0U; i < 5U; ++i) {
        CHECK(s.header_offset[i] == 0U);
        CHECK(s.payload_offset[i] == 0U);
        CHECK(s.payload_bytes[i] == UINT64_C(0));
    }
}

static void rewrite_crc(unsigned char *data, size_t size, size_t footer_offset)
{
    CHECK(size >= footer_offset + 8U);
    store_u32_le(data + footer_offset, UINT32_C(1));
    store_u32_le(data + footer_offset + 4U, crc32_reference(data, footer_offset));
}

static int create_training_state(int tied,
                                 NiyahModel *model,
                                 NiyahAdamWState *state,
                                 NiyahAdamWConfig *config)
{
    NiyahModelConfig mc = test_model_config(tied);
    NiyahModelGradients g;
    size_t i;
    unsigned step;
    memset(model, 0, sizeof(*model));
    memset(state, 0, sizeof(*state));
    *config = test_optimizer_config();
    if (niyah_model_create(model, &mc) != NIYAH_OK) return 0;
    if (niyah_model_reset_parameters(model, UINT64_C(20260916)) != NIYAH_OK) {
        niyah_model_destroy(model); return 0;
    }
    if (niyah_adamw_state_create(state, model) != NIYAH_OK) {
        niyah_model_destroy(model); return 0;
    }
    g.count = model->weight_count;
    g.values = (float *)calloc(g.count, sizeof(float));
    if (g.values == NULL) {
        niyah_adamw_state_destroy(state); niyah_model_destroy(model); return 0;
    }
    for (step = 0U; step < 3U; ++step) {
        for (i = 0U; i < g.count; ++i) {
            const int centered = (int)((i + (size_t)step) % 11U) - 5;
            g.values[i] = (float)centered * 0.01f;
        }
        if (niyah_adamw_step(model, &g, state, config) != NIYAH_OK) {
            free(g.values); niyah_adamw_state_destroy(state); niyah_model_destroy(model); return 0;
        }
    }
    free(g.values);
    return 1;
}

static void destroy_loaded(NiyahModel *model, NiyahAdamWState *state)
{
    niyah_adamw_state_destroy(state);
    niyah_model_destroy(model);
}

static void check_forward_equal(const NiyahModel *a, const NiyahModel *b)
{
    const uint32_t tokens[3] = {1U, 2U, 3U};
    size_t workspace_count = 0U;
    size_t logits_count = 3U * (size_t)a->config.vocab_size;
    float *wa;
    float *wb;
    float *la;
    float *lb;
    CHECK(niyah_transformer_workspace_floats(&a->config, 3U, &workspace_count) == NIYAH_OK);
    wa = (float *)calloc(workspace_count, sizeof(float));
    wb = (float *)calloc(workspace_count, sizeof(float));
    la = (float *)calloc(logits_count, sizeof(float));
    lb = (float *)calloc(logits_count, sizeof(float));
    CHECK(wa != NULL && wb != NULL && la != NULL && lb != NULL);
    if (wa != NULL && wb != NULL && la != NULL && lb != NULL) {
        CHECK(niyah_transformer_forward(a, tokens, 3U, la, logits_count, wa, workspace_count) == NIYAH_OK);
        CHECK(niyah_transformer_forward(b, tokens, 3U, lb, logits_count, wb, workspace_count) == NIYAH_OK);
        CHECK(memcmp(la, lb, logits_count * sizeof(float)) == 0);
    }
    free(lb); free(la); free(wb); free(wa);
}

static void roundtrip_case(int tied, const char *path)
{
    NiyahModel original;
    NiyahAdamWState original_state;
    NiyahAdamWConfig original_config;
    NiyahModel loaded;
    NiyahAdamWState loaded_state;
    NiyahAdamWConfig loaded_config;
    size_t bytes;

    memset(&loaded, 0, sizeof(loaded));
    memset(&loaded_state, 0, sizeof(loaded_state));
    memset(&loaded_config, 0, sizeof(loaded_config));
    CHECK(create_training_state(tied, &original, &original_state, &original_config));
    CHECK(niyah_checkpoint_save(path, &original, &original_state, &original_config) == NIYAH_OK);
    CHECK(niyah_checkpoint_load(path, &loaded, &loaded_state, &loaded_config) == NIYAH_OK);
    if (loaded.weights != NULL) {
        bytes = original.weight_count * sizeof(float);
        CHECK(config_equal(&original.config, &loaded.config));
        CHECK(layout_equal(&original.layout, &loaded.layout));
        CHECK(original.weight_count == loaded.weight_count);
        CHECK(memcmp(original.weights, loaded.weights, bytes) == 0);
        CHECK(memcmp(original_state.m, loaded_state.m, bytes) == 0);
        CHECK(memcmp(original_state.v, loaded_state.v, bytes) == 0);
        CHECK(original_state.step == loaded_state.step);
        CHECK(optimizer_config_equal(&original_config, &loaded_config));
        CHECK(loaded_state.bound_model == &loaded);
        CHECK(loaded_state.bound_weights == loaded.weights);
        CHECK(config_equal(&loaded_state.model_config, &loaded.config));
        CHECK(layout_equal(&loaded_state.model_layout, &loaded.layout));
        check_forward_equal(&original, &loaded);
    }
    destroy_loaded(&loaded, &loaded_state);
    niyah_adamw_state_destroy(&original_state);
    niyah_model_destroy(&original);
    (void)remove(path);
}

static void test_resume_equivalence(void)
{
    const char *path = "niyah_checkpoint_resume.bin";
    NiyahModel a;
    NiyahAdamWState sa;
    NiyahAdamWConfig ca;
    NiyahModel b;
    NiyahAdamWState sb;
    NiyahAdamWConfig cb;
    NiyahModelGradients ga;
    NiyahModelGradients gb;
    size_t bytes;
    size_t i;

    memset(&b, 0, sizeof(b)); memset(&sb, 0, sizeof(sb)); memset(&cb, 0, sizeof(cb));
    CHECK(create_training_state(0, &a, &sa, &ca));
    CHECK(niyah_checkpoint_save(path, &a, &sa, &ca) == NIYAH_OK);
    CHECK(niyah_checkpoint_load(path, &b, &sb, &cb) == NIYAH_OK);
    ga.count = a.weight_count; gb.count = b.weight_count;
    ga.values = (float *)calloc(ga.count, sizeof(float));
    gb.values = (float *)calloc(gb.count, sizeof(float));
    CHECK(ga.values != NULL && gb.values != NULL);
    if (ga.values != NULL && gb.values != NULL) {
        for (i = 0U; i < ga.count; ++i) {
            ga.values[i] = (float)((int)(i % 9U) - 4) * 0.007f;
            gb.values[i] = ga.values[i];
        }
        CHECK(niyah_adamw_step(&a, &ga, &sa, &ca) == NIYAH_OK);
        CHECK(niyah_adamw_step(&b, &gb, &sb, &cb) == NIYAH_OK);
        bytes = a.weight_count * sizeof(float);
        CHECK(memcmp(a.weights, b.weights, bytes) == 0);
        CHECK(memcmp(sa.m, sb.m, bytes) == 0);
        CHECK(memcmp(sa.v, sb.v, bytes) == 0);
        CHECK(sa.step == sb.step);
    }
    free(gb.values); free(ga.values);
    destroy_loaded(&b, &sb);
    niyah_adamw_state_destroy(&sa); niyah_model_destroy(&a);
    (void)remove(path);
}

static int outputs_empty(const NiyahModel *model, const NiyahAdamWState *state)
{
    return model->weights == NULL && model->weight_count == 0U &&
           state->m == NULL && state->v == NULL && state->count == 0U &&
           state->bound_model == NULL && state->bound_weights == NULL;
}

static void expect_load_failure(const char *path, NiyahStatus expected)
{
    NiyahModel model;
    NiyahAdamWState state;
    NiyahAdamWConfig config;
    NiyahAdamWConfig sentinel;
    NiyahStatus status;
    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&config, 0x5a, sizeof(config));
    sentinel = config;
    status = niyah_checkpoint_load(path, &model, &state, &config);
    CHECK(status == expected);
    CHECK(outputs_empty(&model, &state));
    CHECK(memcmp(&config, &sentinel, sizeof(config)) == 0);
}

static void expect_any_load_failure(const char *path)
{
    NiyahModel model;
    NiyahAdamWState state;
    NiyahAdamWConfig config;
    NiyahAdamWConfig sentinel;
    NiyahStatus status;
    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&config, 0xa5, sizeof(config));
    sentinel = config;
    status = niyah_checkpoint_load(path, &model, &state, &config);
    CHECK(status != NIYAH_OK);
    CHECK(outputs_empty(&model, &state));
    CHECK(memcmp(&config, &sentinel, sizeof(config)) == 0);
}

static void test_format_and_malformed(void)
{
    const char *valid_path = "niyah_checkpoint_valid.bin";
    const char *mut_path = "niyah_checkpoint_mut.bin";
    NiyahModel model;
    NiyahAdamWState state;
    NiyahAdamWConfig config;
    unsigned char *valid;
    size_t valid_size = 0U;
    TestSections s;

    CHECK(create_training_state(0, &model, &state, &config));
    CHECK(niyah_checkpoint_save(valid_path, &model, &state, &config) == NIYAH_OK);
    valid = read_bytes(valid_path, &valid_size);
    CHECK(valid != NULL);
    if (valid == NULL) goto cleanup;
    CHECK(valid_size > 76U);
    CHECK(memcmp(valid, "NIYAHCKP", 8U) == 0);
    CHECK(valid[8] == 0x01U && valid[9] == 0x00U && valid[10] == 0x00U && valid[11] == 0x00U);
    CHECK(crc32_reference((const unsigned char *)"123456789", 9U) == UINT32_C(0xcbf43926));
    {
        const int sections_located = locate_sections(valid, valid_size, &s);
        CHECK(sections_located);
        /* a zeroed TestSections would make the offset arithmetic below underflow */
        if (!sections_located) goto cleanup;
    }
    CHECK(s.footer_offset + 8U == valid_size);
    CHECK(load_u32_le(valid + s.footer_offset) == UINT32_C(1));
    CHECK(load_u32_le(valid + s.footer_offset + 4U) == crc32_reference(valid, s.footer_offset));

#define MUTATE_COPY() \
    unsigned char *data = (unsigned char *)malloc(valid_size); \
    CHECK(data != NULL); \
    if (data == NULL) break; \
    memcpy(data, valid, valid_size)

    do { MUTATE_COPY(); data[0] ^= 0x01U; CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + 8U, UINT32_C(2)); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_UNSUPPORTED_VERSION); } while (0);
    CHECK(write_bytes(mut_path, valid, 20U)); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA);
    CHECK(write_bytes(mut_path, valid, 68U + 7U)); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA);
    CHECK(write_bytes(mut_path, valid, s.payload_offset[1] + (size_t)s.payload_bytes[1] - 1U)); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA);
    CHECK(write_bytes(mut_path, valid, valid_size - 1U)); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA);

    do { MUTATE_COPY(); data[s.payload_offset[1]] ^= 0x80U; CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { unsigned char *data = (unsigned char *)malloc(valid_size + 1U); CHECK(data != NULL); if (data == NULL) break; memcpy(data, valid, valid_size); data[valid_size] = 0x42U; CHECK(write_bytes(mut_path, data, valid_size + 1U)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + s.header_offset[2], UINT32_C(1)); rewrite_crc(data, valid_size, s.footer_offset); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + s.header_offset[4], UINT32_C(99)); store_u32_le(data + s.header_offset[4] + 4U, UINT32_C(0)); rewrite_crc(data, valid_size, s.footer_offset); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u64_le(data + s.header_offset[1] + 8U, s.payload_bytes[1] - UINT64_C(4)); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u64_le(data + s.header_offset[1] + 8U, UINT64_MAX); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u64_le(data + 60U, load_u64_le(data + 60U) + UINT64_C(1)); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + s.header_offset[4], UINT32_C(99)); store_u32_le(data + s.header_offset[4] + 4U, UINT32_C(1)); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_UNSUPPORTED_VERSION); } while (0);

    do {
        const size_t extra = 16U + 3U;
        unsigned char *data = (unsigned char *)malloc(valid_size + extra);
        size_t new_footer = s.footer_offset + extra;
        NiyahModel lm; NiyahAdamWState ls; NiyahAdamWConfig lc;
        CHECK(data != NULL); if (data == NULL) break;
        memcpy(data, valid, s.footer_offset);
        store_u32_le(data + 16U, UINT32_C(5));
        store_u32_le(data + s.footer_offset + 0U, UINT32_C(99));
        store_u32_le(data + s.footer_offset + 4U, UINT32_C(0));
        store_u64_le(data + s.footer_offset + 8U, UINT64_C(3));
        data[s.footer_offset + 16U] = 1U; data[s.footer_offset + 17U] = 2U; data[s.footer_offset + 18U] = 3U;
        memcpy(data + new_footer, valid + s.footer_offset, 8U);
        rewrite_crc(data, valid_size + extra, new_footer);
        CHECK(write_bytes(mut_path, data, valid_size + extra));
        memset(&lm, 0, sizeof(lm)); memset(&ls, 0, sizeof(ls)); memset(&lc, 0, sizeof(lc));
        CHECK(niyah_checkpoint_load(mut_path, &lm, &ls, &lc) == NIYAH_OK);
        if (lm.weights != NULL) destroy_loaded(&lm, &ls);
        free(data);
    } while (0);

    do { MUTATE_COPY(); store_u32_le(data + s.payload_offset[4], UINT32_C(0)); rewrite_crc(data, valid_size, s.footer_offset); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + s.payload_offset[1], UINT32_C(0x7fc00000)); rewrite_crc(data, valid_size, s.footer_offset); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + s.payload_offset[2], UINT32_C(0x7f800000)); rewrite_crc(data, valid_size, s.footer_offset); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + s.payload_offset[3], UINT32_C(0x7fc00000)); rewrite_crc(data, valid_size, s.footer_offset); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);
    do { MUTATE_COPY(); store_u32_le(data + s.payload_offset[3], UINT32_C(0xbf800000)); rewrite_crc(data, valid_size, s.footer_offset); CHECK(write_bytes(mut_path, data, valid_size)); free(data); expect_load_failure(mut_path, NIYAH_ERR_CORRUPT_DATA); } while (0);

    {
        size_t prefix;
        for (prefix = 0U; prefix < valid_size; prefix += 17U) {
            CHECK(write_bytes(mut_path, valid, prefix));
            expect_any_load_failure(mut_path);
        }
        for (prefix = valid_size > 32U ? valid_size - 32U : 0U; prefix < valid_size; ++prefix) {
            CHECK(write_bytes(mut_path, valid, prefix));
            expect_any_load_failure(mut_path);
        }
    }
#undef MUTATE_COPY
cleanup:
    free(valid);
    niyah_adamw_state_destroy(&state); niyah_model_destroy(&model);
    (void)remove(mut_path); (void)remove(valid_path);
}

static int file_equals(const char *path, const unsigned char *expected, size_t size)
{
    size_t actual_size = 0U;
    unsigned char *actual = read_bytes(path, &actual_size);
    int ok = actual != NULL && actual_size == size && memcmp(actual, expected, size) == 0;
    free(actual);
    return ok;
}

static void reset_sentinel(const char *path)
{
    static const unsigned char sentinel[4] = {'K','E','E','P'};
    CHECK(write_bytes(path, sentinel, sizeof(sentinel)));
}

static void test_save_preflight_no_truncate(void)
{
    const char *path = "niyah_checkpoint_preflight.bin";
    static const unsigned char sentinel[4] = {'K','E','E','P'};
    NiyahModel model;
    NiyahAdamWState state;
    NiyahAdamWConfig config;
    float saved;
    const NiyahModel *saved_bound;

    CHECK(create_training_state(0, &model, &state, &config));

    reset_sentinel(path); saved = model.weights[0]; model.weights[0] = NAN;
    CHECK(niyah_checkpoint_save(path, &model, &state, &config) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(file_equals(path, sentinel, sizeof(sentinel))); model.weights[0] = saved;

    reset_sentinel(path); saved = state.m[0]; state.m[0] = NAN;
    CHECK(niyah_checkpoint_save(path, &model, &state, &config) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(file_equals(path, sentinel, sizeof(sentinel))); state.m[0] = saved;

    reset_sentinel(path); saved = state.v[0]; state.v[0] = -1.0f;
    CHECK(niyah_checkpoint_save(path, &model, &state, &config) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(file_equals(path, sentinel, sizeof(sentinel))); state.v[0] = saved;

    reset_sentinel(path); saved = config.learning_rate; config.learning_rate = 0.0f;
    CHECK(niyah_checkpoint_save(path, &model, &state, &config) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(file_equals(path, sentinel, sizeof(sentinel))); config.learning_rate = saved;

    reset_sentinel(path); saved_bound = state.bound_model; state.bound_model = NULL;
    CHECK(niyah_checkpoint_save(path, &model, &state, &config) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(file_equals(path, sentinel, sizeof(sentinel))); state.bound_model = saved_bound;

    (void)remove(path);
    state.m[0] = NAN;
    CHECK(niyah_checkpoint_save(path, &model, &state, &config) == NIYAH_ERR_INVALID_CONFIG);
    {
        FILE *f = test_fopen(path, "rb");
        CHECK(f == NULL);
        if (f != NULL) fclose(f);
    }
    niyah_adamw_state_destroy(&state); niyah_model_destroy(&model); (void)remove(path);
}

static void test_public_adamw_validator(void)
{
    NiyahAdamWConfig c = test_optimizer_config();
    CHECK(niyah_adamw_config_validate(&c) == NIYAH_OK);
    c.learning_rate = NAN; CHECK(niyah_adamw_config_validate(&c) == NIYAH_ERR_INVALID_CONFIG);
    c = test_optimizer_config(); c.beta1 = 1.0f; CHECK(niyah_adamw_config_validate(&c) == NIYAH_ERR_INVALID_CONFIG);
    c = test_optimizer_config(); c.beta2 = -1.0f; CHECK(niyah_adamw_config_validate(&c) == NIYAH_ERR_INVALID_CONFIG);
    c = test_optimizer_config(); c.epsilon = INFINITY; CHECK(niyah_adamw_config_validate(&c) == NIYAH_ERR_INVALID_CONFIG);
    c = test_optimizer_config(); c.weight_decay = -0.1f; CHECK(niyah_adamw_config_validate(&c) == NIYAH_ERR_INVALID_CONFIG);
    c = test_optimizer_config(); c.max_grad_norm = 0.0f; CHECK(niyah_adamw_config_validate(&c) == NIYAH_ERR_INVALID_CONFIG);
}


static void test_tokenizer_identity_binding(void)
{
    static const uint8_t corpus_a[] = "aaaaaaaaaaaaaaaaaaaaaaaa";
    static const uint8_t corpus_b[] = "zzzzzzzzzzzzzzzzzzzzzzzz";
    const char *v2_path = "niyah_checkpoint_tokenizer_v2.bin";
    const char *v1_path = "niyah_checkpoint_tokenizer_v1.bin";

    NiyahTokenizerTrainConfig tc;
    NiyahTokenizer *tokenizer_a = NULL;
    NiyahTokenizer *tokenizer_b = NULL;
    uint8_t identity_a[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];
    uint8_t identity_b[NIYAH_TOKENIZER_IDENTITY_SHA256_SIZE];

    NiyahModelConfig mc;
    NiyahModel model;
    NiyahAdamWState state;
    NiyahAdamWConfig config;

    NiyahModel loaded;
    NiyahAdamWState loaded_state;
    NiyahAdamWConfig loaded_config;
    NiyahStatus status;
    size_t bytes;

    memset(&model, 0, sizeof(model));
    memset(&state, 0, sizeof(state));
    memset(&loaded, 0, sizeof(loaded));
    memset(&loaded_state, 0, sizeof(loaded_state));
    memset(&loaded_config, 0, sizeof(loaded_config));

    tc.target_vocab_size = 260U;
    tc.min_pair_frequency = 1U;

    CHECK(niyah_tokenizer_train(
              corpus_a, sizeof(corpus_a) - 1U,
              &tc, &tokenizer_a) == NIYAH_OK);
    CHECK(niyah_tokenizer_train(
              corpus_b, sizeof(corpus_b) - 1U,
              &tc, &tokenizer_b) == NIYAH_OK);
    if (tokenizer_a == NULL || tokenizer_b == NULL) {
        goto cleanup;
    }

    CHECK(niyah_tokenizer_vocab_size(tokenizer_a) == 260U);
    CHECK(niyah_tokenizer_vocab_size(tokenizer_b) == 260U);
    CHECK(niyah_tokenizer_identity_sha256(
              tokenizer_a, identity_a) == NIYAH_OK);
    CHECK(niyah_tokenizer_identity_sha256(
              tokenizer_b, identity_b) == NIYAH_OK);
    CHECK(memcmp(identity_a, identity_b, sizeof(identity_a)) != 0);

    mc = test_model_config(1);
    mc.vocab_size = (uint32_t)niyah_tokenizer_vocab_size(tokenizer_a);
    config = test_optimizer_config();

    CHECK(niyah_model_create(&model, &mc) == NIYAH_OK);
    if (model.weights == NULL) {
        goto cleanup;
    }
    CHECK(niyah_model_reset_parameters(
              &model, UINT64_C(20260916)) == NIYAH_OK);
    CHECK(niyah_adamw_state_create(&state, &model) == NIYAH_OK);
    if (state.m == NULL || state.v == NULL) {
        goto cleanup;
    }

    CHECK(niyah_checkpoint_save_with_tokenizer(
              v2_path, &model, &state, &config,
              tokenizer_a) == NIYAH_OK);

    status = niyah_checkpoint_load_with_tokenizer(
        v2_path, tokenizer_a,
        &loaded, &loaded_state, &loaded_config);
    CHECK(status == NIYAH_OK);
    if (status == NIYAH_OK) {
        bytes = model.weight_count * sizeof(float);
        CHECK(config_equal(&model.config, &loaded.config));
        CHECK(model.weight_count == loaded.weight_count);
        CHECK(memcmp(model.weights, loaded.weights, bytes) == 0);
        CHECK(memcmp(state.m, loaded_state.m, bytes) == 0);
        CHECK(memcmp(state.v, loaded_state.v, bytes) == 0);
        CHECK(optimizer_config_equal(&config, &loaded_config));
    }
    destroy_loaded(&loaded, &loaded_state);
    memset(&loaded, 0, sizeof(loaded));
    memset(&loaded_state, 0, sizeof(loaded_state));
    memset(&loaded_config, 0, sizeof(loaded_config));

    status = niyah_checkpoint_load_with_tokenizer(
        v2_path, tokenizer_b,
        &loaded, &loaded_state, &loaded_config);
    CHECK(status == NIYAH_ERR_INVALID_CONFIG);
    CHECK(outputs_empty(&loaded, &loaded_state));

    memset(&loaded_config, 0, sizeof(loaded_config));
    status = niyah_checkpoint_load(
        v2_path, &loaded, &loaded_state, &loaded_config);
    CHECK(status == NIYAH_ERR_UNSUPPORTED_VERSION);
    CHECK(outputs_empty(&loaded, &loaded_state));

    CHECK(niyah_checkpoint_save(
              v1_path, &model, &state, &config) == NIYAH_OK);

    memset(&loaded_config, 0, sizeof(loaded_config));
    status = niyah_checkpoint_load_with_tokenizer(
        v1_path, tokenizer_a,
        &loaded, &loaded_state, &loaded_config);
    CHECK(status == NIYAH_ERR_UNSUPPORTED_VERSION);
    CHECK(outputs_empty(&loaded, &loaded_state));

cleanup:
    destroy_loaded(&loaded, &loaded_state);
    niyah_adamw_state_destroy(&state);
    niyah_model_destroy(&model);
    niyah_tokenizer_destroy(tokenizer_b);
    niyah_tokenizer_destroy(tokenizer_a);
    (void)remove(v2_path);
    (void)remove(v1_path);
}

int main(void)
{
    test_locate_sections_rejects_short_buffer();
    test_public_adamw_validator();
    roundtrip_case(1, "niyah_checkpoint_tied.bin");
    roundtrip_case(0, "niyah_checkpoint_untied.bin");
    test_resume_equivalence();
    test_tokenizer_identity_binding();
    test_format_and_malformed();
    test_save_preflight_no_truncate();

    if (failures != 0) {
        fprintf(stderr, "niyah_checkpoint_test: %d failure(s)\n", failures);
        return 1;
    }
    puts("NIYAH_CHECKPOINT_P6_B=PASS");
    return 0;
}
