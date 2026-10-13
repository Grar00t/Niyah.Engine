#ifndef NIYAH_SHARED_GUARD_H
#define NIYAH_SHARED_GUARD_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#define NIYAH_GUARD_RING 32U

typedef enum {
    NIYAH_STOP_NONE=0, NIYAH_STOP_EOS, NIYAH_STOP_TOKEN_BUDGET,
    NIYAH_STOP_SINGLE_TOKEN_LOOP, NIYAH_STOP_MULTI_TOKEN_LOOP,
    NIYAH_STOP_ENTROPY_FLOOR, NIYAH_STOP_TIMEOUT, NIYAH_STOP_CANCELLED,
    NIYAH_STOP_NUMERIC_ERROR, NIYAH_STOP_BACKEND_ERROR,
    NIYAH_STOP_INVALID_CONFIG, NIYAH_STOP_CHARACTER_LOOP,
    NIYAH_STOP_INVALID_UTF8
} NiyahStopReason;

typedef struct {
    size_t max_tokens;
    unsigned same_limit, max_period, cycles;
    float entropy_floor;
    uint32_t eos_token;
    int stop_on_eos;
    uint64_t deadline_ms; /* Absolute monotonic time; 0 disables deadline. */
} NiyahGuardConfig;

typedef struct {
    NiyahGuardConfig config;
    NiyahStopReason reason;
    uint32_t ring[NIYAH_GUARD_RING], last;
    size_t generated;
    unsigned next, used, same_run, detected_period;
} NiyahGuard;

static inline NiyahStopReason niyah_guard_init(NiyahGuard *g,
                                              const NiyahGuardConfig *c) {
    if (!g) return NIYAH_STOP_INVALID_CONFIG;
    memset(g,0,sizeof(*g));
    if (!c || !c->max_tokens || c->same_limit<2U || c->max_period<2U ||
        c->max_period>8U || c->cycles<2U || c->cycles>16U ||
        c->max_period*c->cycles>NIYAH_GUARD_RING ||
        !isfinite(c->entropy_floor) || c->entropy_floor<0.0f) {
        g->reason=NIYAH_STOP_INVALID_CONFIG; return g->reason;
    }
    g->config=*c; return NIYAH_STOP_NONE;
}

static inline NiyahStopReason niyah_guard_poll(NiyahGuard *g,
                                              uint64_t now_ms, int cancelled) {
    if (!g) return NIYAH_STOP_INVALID_CONFIG;
    if (g->reason!=NIYAH_STOP_NONE) return g->reason;
    if (cancelled) g->reason=NIYAH_STOP_CANCELLED;
    else if (g->config.deadline_ms && now_ms>=g->config.deadline_ms)
        g->reason=NIYAH_STOP_TIMEOUT;
    else if (g->generated>=g->config.max_tokens)
        g->reason=NIYAH_STOP_TOKEN_BUDGET;
    return g->reason;
}

static inline NiyahStopReason niyah_guard_entropy(NiyahGuard *g,float entropy) {
    if (!g) return NIYAH_STOP_INVALID_CONFIG;
    if (g->reason!=NIYAH_STOP_NONE) return g->reason;
    if (!isfinite(entropy) || entropy<0.0f) g->reason=NIYAH_STOP_NUMERIC_ERROR;
    else if (entropy<g->config.entropy_floor) g->reason=NIYAH_STOP_ENTROPY_FLOOR;
    return g->reason;
}

static inline uint32_t niyah_guard_back(const NiyahGuard *g,unsigned back) {
    return g->ring[(g->next+NIYAH_GUARD_RING-1U-back)%NIYAH_GUARD_RING];
}

/* Observe GENERATED tokens only. Never feed prompt/history into this state. */
static inline NiyahStopReason niyah_guard_token(NiyahGuard *g,uint32_t token) {
    unsigned period,j;
    if (!g) return NIYAH_STOP_INVALID_CONFIG;
    if (g->reason!=NIYAH_STOP_NONE) return g->reason;
    if (g->generated>=g->config.max_tokens) {
        g->reason=NIYAH_STOP_TOKEN_BUDGET;return g->reason;
    }
    g->same_run=(g->generated && token==g->last)?g->same_run+1U:1U;
    g->last=token;g->generated++;
    g->ring[g->next]=token;g->next=(g->next+1U)%NIYAH_GUARD_RING;
    if (g->used<NIYAH_GUARD_RING)g->used++;
    if (g->config.stop_on_eos && token==g->config.eos_token)
        g->reason=NIYAH_STOP_EOS;
    else if (g->same_run>=g->config.same_limit)
        g->reason=NIYAH_STOP_SINGLE_TOKEN_LOOP;
    else for (period=2U;period<=g->config.max_period;period++) {
        unsigned span=period*g->config.cycles;
        if (g->used<span)continue;
        for (j=period;j<span;j++)
            if (niyah_guard_back(g,j)!=niyah_guard_back(g,j%period))break;
        if (j==span) {
            g->detected_period=period;g->reason=NIYAH_STOP_MULTI_TOKEN_LOOP;break;
        }
    }
    if (g->reason==NIYAH_STOP_NONE && g->generated>=g->config.max_tokens)
        g->reason=NIYAH_STOP_TOKEN_BUDGET;
    return g->reason;
}

typedef struct {
    uint32_t cp, minimum, last;
    unsigned need, run, limit;
    NiyahStopReason reason;
} NiyahTextGuard;

static inline int niyah_guard_letter(uint32_t cp) {
    return (cp>=65U && cp<=90U)||(cp>=97U && cp<=122U)||
           (cp>=0x621U && cp<=0x63aU)||(cp>=0x641U && cp<=0x64aU)||
           (cp>=0x66eU && cp<=0x66fU)||(cp>=0x671U && cp<=0x6d3U);
}
static inline void niyah_text_guard_init(NiyahTextGuard *s,unsigned limit) {
    memset(s,0,sizeof(*s));s->limit=limit;
    if (limit<2U)s->reason=NIYAH_STOP_INVALID_CONFIG;
}
/* Feed exact tokenizer-decoded bytes, including fragments across tokens. */
static inline NiyahStopReason niyah_text_guard_feed(NiyahTextGuard *s,
                                                   const uint8_t *data,size_t n) {
    size_t i;
    if (!s || (!data && n))return NIYAH_STOP_INVALID_CONFIG;
    if (s->reason!=NIYAH_STOP_NONE)return s->reason;
    for (i=0;i<n;i++) {
        unsigned b=data[i];
        if (!s->need) {
            if (b<0x80U)s->cp=b;
            else if (b>=0xc2U && b<=0xdfU){s->cp=b&31U;s->need=1U;s->minimum=0x80U;continue;}
            else if (b>=0xe0U && b<=0xefU){s->cp=b&15U;s->need=2U;s->minimum=0x800U;continue;}
            else if (b>=0xf0U && b<=0xf4U){s->cp=b&7U;s->need=3U;s->minimum=0x10000U;continue;}
            else {s->reason=NIYAH_STOP_INVALID_UTF8;break;}
        } else {
            if ((b&0xc0U)!=0x80U){s->reason=NIYAH_STOP_INVALID_UTF8;break;}
            s->cp=(s->cp<<6U)|(b&63U);s->need--;
            if (s->need)continue;
            if (s->cp<s->minimum || s->cp>0x10ffffU ||
                (s->cp>=0xd800U && s->cp<=0xdfffU)) {
                s->reason=NIYAH_STOP_INVALID_UTF8;break;
            }
        }
        if (niyah_guard_letter(s->cp))
            s->run=(s->run && s->cp==s->last)?s->run+1U:1U;
        else s->run=0U;
        s->last=s->cp;
        if (s->run>=s->limit){s->reason=NIYAH_STOP_CHARACTER_LOOP;break;}
    }
    return s->reason;
}
static inline NiyahStopReason niyah_text_guard_finish(NiyahTextGuard *s) {
    if (!s)return NIYAH_STOP_INVALID_CONFIG;
    if (s->reason==NIYAH_STOP_NONE && s->need)s->reason=NIYAH_STOP_INVALID_UTF8;
    return s->reason;
}

/* Conservative commit fence, NOT a session store or semantic validator. */
static inline int niyah_session_can_commit(NiyahStopReason reason,
             int execution_ok,int utf8_valid,int quality_approved) {
    return reason==NIYAH_STOP_EOS && execution_ok==1 && utf8_valid==1 &&
           quality_approved==1;
}
static inline const char *niyah_stop_name(NiyahStopReason r) {
    switch(r) {
    case NIYAH_STOP_NONE:return "NONE";
    case NIYAH_STOP_EOS:return "EOS";
    case NIYAH_STOP_TOKEN_BUDGET:return "TOKEN_BUDGET";
    case NIYAH_STOP_SINGLE_TOKEN_LOOP:return "SINGLE_TOKEN_LOOP";
    case NIYAH_STOP_MULTI_TOKEN_LOOP:return "MULTI_TOKEN_LOOP";
    case NIYAH_STOP_ENTROPY_FLOOR:return "ENTROPY_FLOOR";
    case NIYAH_STOP_TIMEOUT:return "TIMEOUT";
    case NIYAH_STOP_CANCELLED:return "CANCELLED";
    case NIYAH_STOP_NUMERIC_ERROR:return "NUMERIC_ERROR";
    case NIYAH_STOP_BACKEND_ERROR:return "BACKEND_ERROR";
    case NIYAH_STOP_CHARACTER_LOOP:return "CHARACTER_LOOP";
    case NIYAH_STOP_INVALID_UTF8:return "INVALID_UTF8";
    case NIYAH_STOP_INVALID_CONFIG:return "INVALID_CONFIG";
    default:return "UNKNOWN";
    }
}
#endif
