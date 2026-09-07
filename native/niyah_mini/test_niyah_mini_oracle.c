/* ==========================================================================
 * test_niyah_mini_oracle.c -- constructed-weight oracles for forward_one.
 *
 * The other mini tests use init_small / init_xavier weights. Those prove the
 * forward pass allocates, runs and returns finite numbers. They cannot prove
 * any single stage computes the right value, because nobody knows what the
 * right value is for arbitrary weights.
 *
 * Here the weights are chosen so the network collapses to a function with a
 * closed form. Every assertion below is an exact expected value computed
 * from the definition in double precision, not a tolerance on noise.
 *
 * The reference expressions are written from the mathematical definitions,
 * deliberately not copied from niyah_mini_model.c, so that a wrong formula
 * in either place disagrees with the other.
 *
 * WHAT THIS FILE DOES NOT TEST
 *   - Whether the model produces meaningful text. It cannot; there are no
 *     trained weights here. This validates arithmetic, nothing more.
 *   - Multi-layer interaction: n_layers is 1 throughout, so that a failure
 *     names a stage rather than a depth.
 * ========================================================================== */

#undef NDEBUG

#include "niyah_mini_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks = 0;
static int g_failures = 0;

static void check(int cond, const char *what)
{
    ++g_checks;
    if (!cond) {
        ++g_failures;
        printf("    FAIL  %s\n", what);
    }
}

/* Relative tolerance, floored at 1.0 so near-zero expectations do not demand
 * absurd precision. The engine accumulates in double and stores float, so
 * 1e-4 relative is comfortably above representation error and far below any
 * real formula mistake. */
static void check_close(double got, double want, const char *what)
{
    double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
    ++g_checks;
    if (!(fabs(got - want) <= 1e-4 * scale)) {
        ++g_failures;
        printf("    FAIL  %s: got %.9g want %.9g\n", what, got, want);
    }
}

static void fill(float *dst, int n, float v)
{
    int i;
    for (i = 0; i < n; ++i) dst[i] = v;
}

/* Row-major identity, matching matvec's w[r * cols + c] indexing. */
static void identity(float *dst, int rows, int cols)
{
    int r, c;
    for (r = 0; r < rows; ++r)
        for (c = 0; c < cols; ++c)
            dst[(size_t)r * (size_t)cols + (size_t)c] = (r == c) ? 1.0f : 0.0f;
}

/* RMSNorm scale factor, from the definition: 1 / sqrt(mean(x^2) + eps).
 * The mean divides by n, not by n - 1, and eps sits inside the sqrt. Both
 * choices are asserted by every expected value in this file. */
static double rms_inv(const double *x, int n, double eps)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < n; ++i) sum += x[i] * x[i];
    return 1.0 / sqrt(sum / (double)n + eps);
}

static void oracle_config(NiyahMiniConfig *cfg, int dim, int heads,
                          int kv_heads, int ff, int vocab)
{
    niyah_mini_config_init(cfg, NIYAH_MINI_TINY);
    cfg->n_layers = 1;
    cfg->n_dim = dim;
    cfg->n_heads = heads;
    cfg->n_kv_heads = kv_heads;
    cfg->n_ff = ff;
    cfg->n_vocab = vocab;
    cfg->n_ctx = 8;
    cfg->rope_theta = 10000.0f;
    cfg->norm_eps = 1e-5f;
    cfg->activation = NIYAH_MINI_ACT_SILU;
    cfg->pos_encoding = NIYAH_MINI_POS_ROPE;
    cfg->tie_word_embeddings = true;
}

/* ---------------------------------------------------------------------------
 * Shared weight construction for the uniform-attention oracle.
 *
 * Wq = Wk = 0, left zero by the memset. Every score is then 0, so softmax is
 * exactly uniform -- no dependence on RoPE, on the score scale, or on the
 * max-subtraction trick. Wv and Wo are identity, so the head output passes
 * through untouched. The FFN gate and up projections stay zero, which makes
 * silu_mul return exactly 0 regardless of ffn_down, so the FFN branch
 * contributes nothing and cannot mask an attention error.
 *
 * Assumes dim = ff = vocab = 4.
 * --------------------------------------------------------------------------- */
static void set_uniform_weights(NiyahMiniModel *model)
{
    NiyahMiniLayerWeights *w = &model->weights.layers[0];
    memset(model->weights.memory_block, 0, model->weights.memory_size);
    identity(model->weights.embedding, 4, 4);
    fill(w->attn_norm, 4, 1.0f);
    identity(w->wv, 4, 4);
    identity(w->wo, 4, 4);
    fill(w->ffn_norm, 4, 1.0f);
    identity(w->ffn_down, 4, 4);
    fill(model->weights.final_norm, 4, 1.0f);
}

/* With an identity embedding a token is a unit basis vector, so its RMSNorm
 * factor is the same for every token: sum of squares is exactly 1. */
static double unit_token_inv(double eps)
{
    double one[4] = { 1.0, 0.0, 0.0, 0.0 };
    return rms_inv(one, 4, eps);
}

/* ---------------------------------------------------------------------------
 * uniform_attention
 *
 * At position p the head output is the mean of the cached value vectors, and
 * each cached value is inv * e_{token}. So
 *
 *     x[j] = [j == token_p] + inv * count_j / (p + 1)
 *
 * where count_j is how many of tokens 0..p equal j. The (p + 1) divisor is
 * the causal mask: attending one position too far changes it.
 * --------------------------------------------------------------------------- */
static void test_uniform_attention(void)
{
    NiyahMiniConfig cfg;
    NiyahMiniModel model;
    NiyahMiniForwardState state;
    const int32_t tokens[4] = { 0, 1, 2, 1 };
    double counts[4] = { 0.0, 0.0, 0.0, 0.0 };
    float logits[4];
    double eps, inv, inv2, xv[4];
    int p, j;

    printf("  uniform_attention\n");
    oracle_config(&cfg, 4, 2, 2, 4, 4);
    eps = (double)cfg.norm_eps;
    inv = unit_token_inv(eps);

    check(niyah_mini_model_init(&model, &cfg) == NIYAH_OK, "model init");
    check(niyah_mini_forward_state_init(&state, &cfg, cfg.n_ctx) == NIYAH_OK,
          "forward state init");
    set_uniform_weights(&model);
    niyah_mini_reset_kv_cache(&model);

    for (p = 0; p < 4; ++p) {
        counts[tokens[p]] += 1.0;
        check(niyah_mini_forward_token(&model, &state, tokens[p],
                                       (int32_t)p, logits) == NIYAH_OK,
              "forward_token");
        for (j = 0; j < 4; ++j)
            xv[j] = (j == tokens[p] ? 1.0 : 0.0)
                  + inv * counts[j] / (double)(p + 1);
        inv2 = rms_inv(xv, 4, eps);
        for (j = 0; j < 4; ++j) {
            char label[64];
            snprintf(label, sizeof(label), "logit[%d] at position %d", j, p);
            check_close((double)logits[j], xv[j] * inv2, label);
        }
    }

    niyah_mini_forward_state_free(&state);
    niyah_mini_model_free(&model);
}

/* ---------------------------------------------------------------------------
 * cache_reset
 *
 * uniform_attention walks positions forward only, so a cache that never
 * clears would still pass it. Run a sequence, reset, replay position 0 and
 * require the original logits back.
 * --------------------------------------------------------------------------- */
static void test_cache_reset(void)
{
    NiyahMiniConfig cfg;
    NiyahMiniModel model;
    NiyahMiniForwardState state;
    const int32_t tokens[3] = { 0, 1, 2 };
    float first[4], again[4];
    int p, j;

    printf("  cache_reset\n");
    oracle_config(&cfg, 4, 2, 2, 4, 4);
    check(niyah_mini_model_init(&model, &cfg) == NIYAH_OK, "model init");
    check(niyah_mini_forward_state_init(&state, &cfg, cfg.n_ctx) == NIYAH_OK,
          "forward state init");
    set_uniform_weights(&model);

    niyah_mini_reset_kv_cache(&model);
    check(niyah_mini_forward_token(&model, &state, tokens[0], 0, first)
          == NIYAH_OK, "first pass position 0");
    for (p = 1; p < 3; ++p) {
        float scratch[4];
        check(niyah_mini_forward_token(&model, &state, tokens[p],
                                       (int32_t)p, scratch) == NIYAH_OK,
              "first pass tail");
    }

    niyah_mini_reset_kv_cache(&model);
    check(model.kv_cache_seq_len == 0, "reset clears kv_cache_seq_len");
    check(niyah_mini_forward_token(&model, &state, tokens[0], 0, again)
          == NIYAH_OK, "replay position 0");
    for (j = 0; j < 4; ++j) {
        char label[64];
        snprintf(label, sizeof(label), "replayed logit[%d] matches", j);
        check_close((double)again[j], (double)first[j], label);
    }

    niyah_mini_forward_state_free(&state);
    niyah_mini_model_free(&model);
}

/* ---------------------------------------------------------------------------
 * swiglu_exact
 *
 * Wv = 0 makes the attention branch contribute exactly zero, so x entering
 * the FFN is the raw embedding. With identity gate/up/down, only the token's
 * own coordinate is non-zero, and it becomes g * sigmoid(g) * g with
 * g = inv. Every other coordinate is 0 * 0.5 * 0 = 0.
 * --------------------------------------------------------------------------- */
static void test_swiglu_exact(void)
{
    NiyahMiniConfig cfg;
    NiyahMiniModel model;
    NiyahMiniForwardState state;
    NiyahMiniLayerWeights *w;
    const int32_t token = 2;
    float logits[4];
    double eps, inv, sig, act, inv2, xv[4];
    int j;

    printf("  swiglu_exact\n");
    oracle_config(&cfg, 4, 2, 2, 4, 4);
    eps = (double)cfg.norm_eps;
    check(niyah_mini_model_init(&model, &cfg) == NIYAH_OK, "model init");
    check(niyah_mini_forward_state_init(&state, &cfg, cfg.n_ctx) == NIYAH_OK,
          "forward state init");

    w = &model.weights.layers[0];
    memset(model.weights.memory_block, 0, model.weights.memory_size);
    identity(model.weights.embedding, 4, 4);
    fill(w->attn_norm, 4, 1.0f);
    /* Wq, Wk, Wv all remain zero: no attention contribution at all. */
    identity(w->wo, 4, 4);
    fill(w->ffn_norm, 4, 1.0f);
    identity(w->ffn_gate, 4, 4);
    identity(w->ffn_up, 4, 4);
    identity(w->ffn_down, 4, 4);
    fill(model.weights.final_norm, 4, 1.0f);

    niyah_mini_reset_kv_cache(&model);
    check(niyah_mini_forward_token(&model, &state, token, 0, logits)
          == NIYAH_OK, "forward_token");

    inv = unit_token_inv(eps);
    sig = 1.0 / (1.0 + exp(-inv));
    act = inv * sig * inv;
    for (j = 0; j < 4; ++j)
        xv[j] = (j == token) ? (1.0 + act) : 0.0;
    inv2 = rms_inv(xv, 4, eps);
    for (j = 0; j < 4; ++j) {
        char label[64];
        snprintf(label, sizeof(label), "swiglu logit[%d]", j);
        check_close((double)logits[j], xv[j] * inv2, label);
    }

    /* Guard against the activation silently vanishing, which would make the
     * assertions above pass for the wrong reason. */
    check(act > 0.1, "activation is non-trivial");

    niyah_mini_forward_state_free(&state);
    niyah_mini_model_free(&model);
}

/* ---------------------------------------------------------------------------
 * rope_relative
 *
 * RoPE's defining property is that the score between a query at position p
 * and a key at position t depends only on p - t. Asserting that is stronger
 * than recomputing the angle formula, which would only confirm the code
 * agrees with itself.
 *
 * Feed the same token at every position with identity Wq/Wk. Then
 *
 *     probs[p] / probs[p - 1] = exp(s(0) - s(1))
 *
 * which is independent of p. An implementation that used absolute position
 * additively, or indexed the position off by one, breaks this.
 *
 * Wv and Wo stay zero so the residual stream stays clean; attn_probs is
 * still populated. It holds the last head of the last layer, and there is
 * exactly one of each here.
 * --------------------------------------------------------------------------- */
static void test_rope_relative(void)
{
    NiyahMiniConfig cfg;
    NiyahMiniModel model;
    NiyahMiniForwardState state;
    NiyahMiniLayerWeights *w;
    const int32_t token = 1;
    float logits[4];
    double ratio[4] = { 0.0, 0.0, 0.0, 0.0 };
    double sum;
    int p, t;

    printf("  rope_relative\n");
    oracle_config(&cfg, 4, 1, 1, 4, 4);
    check(niyah_mini_model_init(&model, &cfg) == NIYAH_OK, "model init");
    check(niyah_mini_forward_state_init(&state, &cfg, cfg.n_ctx) == NIYAH_OK,
          "forward state init");

    w = &model.weights.layers[0];
    memset(model.weights.memory_block, 0, model.weights.memory_size);
    identity(model.weights.embedding, 4, 4);
    fill(w->attn_norm, 4, 1.0f);
    identity(w->wq, 4, 4);
    identity(w->wk, 4, 4);
    /* Wv, Wo, and the whole FFN stay zero. */
    fill(w->ffn_norm, 4, 1.0f);
    fill(model.weights.final_norm, 4, 1.0f);

    niyah_mini_reset_kv_cache(&model);
    for (p = 0; p < 4; ++p) {
        check(niyah_mini_forward_token(&model, &state, token,
                                       (int32_t)p, logits) == NIYAH_OK,
              "forward_token");
        sum = 0.0;
        for (t = 0; t <= p; ++t) sum += (double)state.attn_probs[t];
        {
            char label[64];
            snprintf(label, sizeof(label),
                     "attention mass sums to 1 at position %d", p);
            check_close(sum, 1.0, label);
        }
        if (p > 0) {
            check(state.attn_probs[p] > state.attn_probs[p - 1],
                  "self-attention score exceeds offset-one score");
            ratio[p] = (double)state.attn_probs[p]
                     / (double)state.attn_probs[p - 1];
        }
    }

    /* The invariant. Same relative offset, different absolute positions. */
    check(ratio[1] > 0.0, "offset-one ratio is positive");
    check_close(ratio[2], ratio[1], "offset-one ratio invariant at p=2");
    check_close(ratio[3], ratio[1], "offset-one ratio invariant at p=3");

    niyah_mini_forward_state_free(&state);
    niyah_mini_model_free(&model);
}

/* ---------------------------------------------------------------------------
 * rope_split_half_layout
 *
 * This tests the actual coordinate pairing, not merely RoPE's relative-
 * position property.
 *
 * For head_dim = 4, split-half RoPE pairs:
 *
 *     (0, 2)
 *     (1, 3)
 *
 * NOT:
 *
 *     (0, 1)
 *     (2, 3)
 *
 * Wq is identity and token 0 embeds as [1,2,3,4], so state.q after
 * forward_token exposes the rotated vector directly.
 * --------------------------------------------------------------------------- */
static void test_rope_split_half_layout(void)
{
    NiyahMiniConfig cfg;
    NiyahMiniModel model;
    NiyahMiniForwardState state;
    NiyahMiniLayerWeights *w;

    float warmup_logits[4];
    float logits[4];

    double x[4] = {
        1.0,
        2.0,
        3.0,
        4.0
    };

    double expected[4];
    double inv;
    double angle0;
    double angle1;
    double c0;
    double s0;
    double c1;
    double s1;

    int j;

    printf("  rope_split_half_layout\n");

    oracle_config(
        &cfg,
        4,  /* dim */
        1,  /* heads */
        1,  /* kv heads */
        4,  /* ff */
        4   /* vocab */
    );

    check(
        niyah_mini_model_init(
            &model,
            &cfg
        ) == NIYAH_OK,
        "rope layout model init"
    );

    check(
        niyah_mini_forward_state_init(
            &state,
            &cfg,
            cfg.n_ctx
        ) == NIYAH_OK,
        "rope layout state init"
    );

    w = &model.weights.layers[0];

    memset(
        model.weights.memory_block,
        0,
        model.weights.memory_size
    );

    /*
     * token 0 = [1,2,3,4]
     * token 1 remains zero and is used only to populate position 0.
     */
    model.weights.embedding[0] = 1.0f;
    model.weights.embedding[1] = 2.0f;
    model.weights.embedding[2] = 3.0f;
    model.weights.embedding[3] = 4.0f;

    fill(
        w->attn_norm,
        4,
        1.0f
    );

    identity(
        w->wq,
        4,
        4
    );

    fill(
        w->ffn_norm,
        4,
        1.0f
    );

    fill(
        model.weights.final_norm,
        4,
        1.0f
    );

    niyah_mini_reset_kv_cache(&model);

    /*
     * Populate position 0 first so the target is a normal sequential
     * position-1 forward pass.
     */
    check(
        niyah_mini_forward_token(
            &model,
            &state,
            1,
            0,
            warmup_logits
        ) == NIYAH_OK,
        "rope layout warmup"
    );

    check(
        niyah_mini_forward_token(
            &model,
            &state,
            0,
            1,
            logits
        ) == NIYAH_OK,
        "rope layout target"
    );

    inv = rms_inv(
        x,
        4,
        (double)cfg.norm_eps
    );

    /*
     * split-half:
     *
     * j=0 pairs coordinates 0 and 2
     * j=1 pairs coordinates 1 and 3
     */
    angle0 =
        1.0 /
        pow(
            (double)cfg.rope_theta,
            0.0 / 4.0
        );

    angle1 =
        1.0 /
        pow(
            (double)cfg.rope_theta,
            2.0 / 4.0
        );

    c0 = cos(angle0);
    s0 = sin(angle0);

    c1 = cos(angle1);
    s1 = sin(angle1);

    expected[0] =
        inv * (x[0] * c0 - x[2] * s0);

    expected[2] =
        inv * (x[2] * c0 + x[0] * s0);

    expected[1] =
        inv * (x[1] * c1 - x[3] * s1);

    expected[3] =
        inv * (x[3] * c1 + x[1] * s1);

    for (j = 0; j < 4; ++j) {
        char label[80];

        snprintf(
            label,
            sizeof(label),
            "split-half RoPE q[%d]",
            j
        );

        check_close(
            (double)state.q[j],
            expected[j],
            label
        );
    }

    niyah_mini_forward_state_free(&state);
    niyah_mini_model_free(&model);
}


/* ---------------------------------------------------------------------------
 * gqa_contiguous_mapping
 *
 * Configuration:
 *
 *     4 query heads
 *     2 KV heads
 *     head_dim = 2
 *
 * Correct contiguous GQA mapping:
 *
 *     Q0 Q1 -> KV0
 *     Q2 Q3 -> KV1
 *
 * At position 0 the attention probability is exactly 1, so each query
 * head's attention output must be exactly the selected V head.
 *
 * V head 0 = inv * [1,2]
 * V head 1 = inv * [3,4]
 *
 * Therefore:
 *
 *     attn_out =
 *       [1,2, 1,2, 3,4, 3,4] * inv
 *
 * The old modulo mapping would instead produce:
 *
 *       [1,2, 3,4, 1,2, 3,4] * inv
 *
 * so this oracle distinguishes the two exactly.
 * --------------------------------------------------------------------------- */
static void test_gqa_contiguous_mapping(void)
{
    NiyahMiniConfig cfg;
    NiyahMiniModel model;
    NiyahMiniForwardState state;
    NiyahMiniLayerWeights *w;

    float logits[8];

    double basis[8] = {
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    };

    double expected[8];
    double inv;

    int j;

    printf("  gqa_contiguous_mapping\n");

    oracle_config(
        &cfg,
        8,  /* dim */
        4,  /* query heads */
        2,  /* KV heads */
        8,  /* ff */
        8   /* vocab */
    );

    check(
        niyah_mini_model_init(
            &model,
            &cfg
        ) == NIYAH_OK,
        "GQA model init"
    );

    check(
        niyah_mini_forward_state_init(
            &state,
            &cfg,
            cfg.n_ctx
        ) == NIYAH_OK,
        "GQA state init"
    );

    w = &model.weights.layers[0];

    memset(
        model.weights.memory_block,
        0,
        model.weights.memory_size
    );

    /*
     * token 0 is e0.
     * After RMSNorm its first coordinate is inv.
     */
    model.weights.embedding[0] = 1.0f;

    fill(
        w->attn_norm,
        8,
        1.0f
    );

    /*
     * Wq and Wk remain zero.
     *
     * Wv has 4 output rows:
     *   rows 0,1 = KV head 0
     *   rows 2,3 = KV head 1
     *
     * Only input column 0 is non-zero.
     */
    w->wv[0 * 8 + 0] = 1.0f;
    w->wv[1 * 8 + 0] = 2.0f;

    w->wv[2 * 8 + 0] = 3.0f;
    w->wv[3 * 8 + 0] = 4.0f;

    niyah_mini_reset_kv_cache(&model);

    check(
        niyah_mini_forward_token(
            &model,
            &state,
            0,
            0,
            logits
        ) == NIYAH_OK,
        "GQA forward_token"
    );

    inv = rms_inv(
        basis,
        8,
        (double)cfg.norm_eps
    );

    expected[0] = 1.0 * inv;
    expected[1] = 2.0 * inv;

    expected[2] = 1.0 * inv;
    expected[3] = 2.0 * inv;

    expected[4] = 3.0 * inv;
    expected[5] = 4.0 * inv;

    expected[6] = 3.0 * inv;
    expected[7] = 4.0 * inv;

    for (j = 0; j < 8; ++j) {
        char label[80];

        snprintf(
            label,
            sizeof(label),
            "contiguous GQA attn_out[%d]",
            j
        );

        check_close(
            (double)state.attn_out[j],
            expected[j],
            label
        );
    }

    niyah_mini_forward_state_free(&state);
    niyah_mini_model_free(&model);
}


int main(void)
{
    printf("niyah_mini constructed-weight oracles\n");
    test_uniform_attention();
    test_cache_reset();
    test_swiglu_exact();
    test_rope_relative();
    test_rope_split_half_layout();
    test_gqa_contiguous_mapping();

    if (g_failures != 0) {
        printf("\nFAILED %d of %d checks\n", g_failures, g_checks);
        return 1;
    }
    printf("\nok: %d checks passed\n", g_checks);
    return 0;
}
