/* Deterministic decoder TEST DOUBLE. Does not evaluate model weights or GPU. */
#include "niyah/generate.h"
#include "niyah_cuda_generate_repetition.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); return 1; } } while (0)
#define RUN(x) do { CHECK((x)()==0); ++tests; printf("PASS %s\n",#x); } while (0)
static size_t decode_size=4U, fail_position=SIZE_MAX;
static int logits_mode=0, old_cuda_calls=0, reset_fail=0;
static float dummy_weight=1.0f;
static const float raw[4]={3.0f,2.0f,1.0f,0.0f};
static NiyahModel model;
static NiyahGenerationConfig cfg;
static NiyahKVCache cache;
static NiyahCudaModelState cm;
static NiyahCudaDecodeState cd;
static NiyahGenerationResult result;
static uint32_t output[16];
static float ws[40];
static uint32_t prompt[2]={0U,2U};
NiyahStatus niyah_model_config_validate(const NiyahModelConfig *c) { return c && c->vocab_size && c->context_length ? NIYAH_OK : NIYAH_ERR_INVALID_CONFIG; }
void niyah_kv_cache_reset(NiyahKVCache *c) { c->next_position=0U; }
size_t niyah_kv_cache_position(const NiyahKVCache *c) { return c->next_position; }
NiyahStatus niyah_decode_workspace_floats(const NiyahModelConfig *c,size_t *out) { if(!c||!out) return NIYAH_ERR_INVALID_ARGUMENT; *out=decode_size; return NIYAH_OK; }
static void fill_logits(float *l) {
    memcpy(l,raw,sizeof(raw));
    if(logits_mode==1) l[0]=NAN;
    if(logits_mode==2) { l[0]=10000.0f; l[1]=l[2]=l[3]=-10000.0f; }
    if(logits_mode==3) l[0]=l[1]=l[2]=l[3]=0.0f;
}
NiyahStatus niyah_transformer_decode_token(const NiyahModel *m,NiyahKVCache *c,uint32_t t,float *l,size_t n,float *w,size_t wc) {
    (void)w;(void)wc;
    if(t>=m->config.vocab_size || n<4U) return NIYAH_ERR_INVALID_ARGUMENT;
    if(c->next_position==fail_position) return NIYAH_ERR_IO;
    fill_logits(l); c->next_position++; return NIYAH_OK;
}
int niyah_cuda_decode_state_reset(NiyahCudaDecodeState *d) { if(reset_fail)return 1; d->next_position=0U; return 0; }
int niyah_cuda_decode_token(const NiyahCudaModelState *m,NiyahCudaDecodeState *d,uint32_t t,float *l,size_t n) {
    if(t>=m->config.vocab_size || n<4U || d->next_position==fail_position) return 1;
    fill_logits(l); d->next_position++; return 0;
}
NiyahStatus niyah_cuda_generate(const NiyahCudaModelState *m,NiyahCudaDecodeState *d,const uint32_t *p,size_t pc,const NiyahGenerationConfig *c,uint32_t *o,size_t oc,NiyahGenerationResult *r,float *w,size_t wc) {
    (void)m;(void)d;(void)p;(void)pc;(void)c;(void)oc;(void)w;(void)wc;
    ++old_cuda_calls; o[0]=123U; r->generated_tokens=1U; return NIYAH_OK;
}
static void setup(void) {
    memset(&model,0,sizeof(model)); memset(&cfg,0,sizeof(cfg));
    memset(&cache,0,sizeof(cache)); memset(&cm,0,sizeof(cm)); memset(&cd,0,sizeof(cd));
    memset(&result,0,sizeof(result)); memset(output,0x55,sizeof(output));
    for(size_t i=0;i<40U;i++)ws[i]=77.0f;
    model.weights=&dummy_weight; model.config.vocab_size=4U; model.config.context_length=32U;
    cm.config=model.config; cfg.max_new_tokens=6U; cfg.sampler.seed=42U;
    decode_size=4U; fail_position=SIZE_MAX; logits_mode=0; old_cuda_calls=0; reset_fail=0;
}
static NiyahStatus cpu(const NiyahRepetitionConfig *r,size_t wc) {
    return niyah_generate_with_repetition(&model,&cache,prompt,2U,&cfg,r,output,16U,&result,ws,wc);
}
static NiyahStatus gpu(const NiyahRepetitionConfig *r,size_t wc) {
    return niyah_cuda_generate_with_repetition(&cm,&cd,prompt,2U,&cfg,r,output,16U,&result,ws,wc);
}
static int window1_generated_only(void) {
    NiyahRepetitionConfig r={4.0f,1U}; uint32_t want[]={0,1,0,1,0,1}; setup();
    CHECK(cpu(&r,12U)==NIYAH_OK); CHECK(result.generated_tokens==6U); CHECK(memcmp(output,want,sizeof(want))==0);
    CHECK(cache.next_position==8U); CHECK(memcmp(ws+4U,raw,sizeof(raw))==0); CHECK(ws[12]==77.0f); return 0;
}
static int window2_expiry(void) {
    NiyahRepetitionConfig r={4.0f,2U}; uint32_t want[]={0,1,2,0,1,2}; setup();
    CHECK(cpu(&r,12U)==NIYAH_OK); CHECK(memcmp(output,want,sizeof(want))==0); return 0;
}
static int huge_window_no_overflow(void) {
    NiyahRepetitionConfig r={4.0f,SIZE_MAX}; setup();
    CHECK(cpu(&r,12U)==NIYAH_OK); CHECK(output[0]==0U && output[1]==1U && output[2]==2U); return 0;
}
static int disabled_exact_generation(void) {
    uint32_t old[16]; NiyahGenerationResult oldr; NiyahRepetitionConfig r={1.0f,64U}; size_t legacy=0U,disabled=0U;
    setup(); cfg.sampler.temperature=0.7f;
    CHECK(niyah_generation_workspace_floats(&model.config,&legacy)==NIYAH_OK);
    CHECK(niyah_generation_workspace_floats_with_repetition(&model.config,&r,&disabled)==NIYAH_OK); CHECK(disabled==legacy);
    CHECK(niyah_generate(&model,&cache,prompt,2U,&cfg,old,16U,&oldr,ws,legacy)==NIYAH_OK);
    cache.next_position=0U;
    CHECK(cpu(&r,legacy)==NIYAH_OK); CHECK(result.generated_tokens==oldr.generated_tokens);
    CHECK(memcmp(old,output,oldr.generated_tokens*sizeof(uint32_t))==0); return 0;
}
static int disabled_rng_state(void) {
    NiyahSampler a,b; NiyahSamplerConfig sc={0.7f,998877U}; float scratch[4]={77,77,77,77}; uint32_t history[]={0,1,0};
    CHECK(niyah_sampler_init(&a,&sc)==NIYAH_OK); CHECK(niyah_sampler_init(&b,&sc)==NIYAH_OK);
    for(size_t i=0;i<1000U;i++) { uint32_t ta=99,tb=98;
        CHECK(niyah_sampler_sample(&a,raw,4U,&ta)==NIYAH_OK);
        CHECK(niyah_sampler_sample_with_history(&b,raw,4U,history,3U,1.0f,scratch,4U,&tb)==NIYAH_OK);
        CHECK(ta==tb && a.state==b.state);
    }
    CHECK(scratch[0]==77.0f && scratch[3]==77.0f); return 0;
}
static int signs_unique_history_and_logits(void) {
    float values[]={8,-2,0,3}; float scratch[4]; const uint32_t history[]={0,0,1,1,2}; NiyahSampler s; NiyahSamplerConfig sc={0,1}; uint32_t t;
    CHECK(niyah_sampler_init(&s,&sc)==NIYAH_OK);
    CHECK(niyah_sampler_sample_with_history(&s,values,4U,history,5U,2.0f,scratch,4U,&t)==NIYAH_OK);
    CHECK(scratch[0]==4.0f && scratch[1]==-4.0f && scratch[2]==0.0f && scratch[3]==3.0f);
    CHECK(values[0]==8.0f && values[1]==-2.0f); CHECK(t==0U); return 0;
}
static int scratch_overlap_rejected(void) {
    NiyahSampler s; NiyahSamplerConfig sc={1,4}; uint32_t t=77,h=0; float a[8]={3,2,1,0}; uint64_t before;
    CHECK(niyah_sampler_init(&s,&sc)==NIYAH_OK); before=s.state;
    CHECK(niyah_sampler_sample_with_history(&s,a,4U,&h,1U,2.0f,a+1U,4U,&t)==NIYAH_ERR_INVALID_ARGUMENT);
    CHECK(s.state==before && t==77U); return 0;
}
static int sampling_overflow_no_rng(void) {
    NiyahSampler s; NiyahSamplerConfig sc={1,4}; uint32_t t=77,h=0; float a[2]={-FLT_MAX,0},scratch[2];
    CHECK(niyah_sampler_init(&s,&sc)==NIYAH_OK);
    CHECK(niyah_sampler_sample_with_history(&s,a,2U,&h,1U,FLT_MAX,scratch,2U,&t)==NIYAH_ERR_INVALID_CONFIG);
    CHECK(s.state==4U && t==77U); return 0;
}
static int cpu_short_workspace_no_write(void) {
    NiyahRepetitionConfig r={2.0f,64U}; setup(); cache.next_position=0U;
    CHECK(cpu(&r,11U)==NIYAH_ERR_BUFFER_TOO_SMALL); CHECK(cache.next_position==0U && result.generated_tokens==0U);
    for(size_t i=0;i<40U;i++)CHECK(ws[i]==77.0f);
    CHECK(output[0]==UINT32_C(0x55555555)); return 0;
}
static int checked_add_and_bytes(void) {
    NiyahRepetitionConfig r={2.0f,64U}; size_t n=7U; setup(); decode_size=SIZE_MAX-5U;
    CHECK(niyah_generation_workspace_floats_with_repetition(&model.config,&r,&n)==NIYAH_ERR_OVERFLOW); CHECK(n==0U);
    decode_size=SIZE_MAX/sizeof(float);
    CHECK(niyah_generation_workspace_floats_with_repetition(&model.config,&r,&n)==NIYAH_ERR_OVERFLOW); CHECK(n==0U); return 0;
}
static int invalid_repetition(void) {
    float invalid[]={NAN,INFINITY,-INFINITY,-1,0,0.9f}; NiyahRepetitionConfig r={2,64}; size_t n;
    for(size_t i=0;i<sizeof(invalid)/sizeof(invalid[0]);i++) { setup();r.penalty=invalid[i];
        CHECK(cpu(&r,12U)==NIYAH_ERR_INVALID_CONFIG); CHECK(result.generated_tokens==0U);
        CHECK(niyah_generation_workspace_floats_with_repetition(&model.config,&r,&n)==NIYAH_ERR_INVALID_CONFIG);
    }
    setup();r.penalty=2;r.window=0; CHECK(cpu(&r,12U)==NIYAH_ERR_INVALID_CONFIG);
    CHECK(cpu(NULL,12U)==NIYAH_ERR_INVALID_ARGUMENT); return 0;
}
static int eos_not_decoded(void) {
    NiyahRepetitionConfig r={2,64}; setup();cfg.stop_on_eos=1;cfg.eos_token=0;
    CHECK(cpu(&r,12U)==NIYAH_OK); CHECK(result.generated_tokens==1U && result.stopped_on_eos==1);
    CHECK(cache.next_position==2U); return 0;
}
static int cpu_prefill_and_decode_fail_reset(void) {
    NiyahRepetitionConfig r={2,64};
    for(size_t i=0;i<5;i++){setup();fail_position=i;CHECK(cpu(&r,12U)==NIYAH_ERR_IO);
        CHECK(cache.next_position==0U && result.prompt_tokens==0U && result.generated_tokens==0U);}
    return 0;
}
static int nonfinite_raw_and_stop_guards(void) {
    NiyahRepetitionConfig r={2,64};setup();logits_mode=1;
    CHECK(cpu(&r,12U)==NIYAH_ERR_INVALID_CONFIG);CHECK(cache.next_position==0U);
    setup();logits_mode=2;cfg.max_new_tokens=10U;CHECK(cpu(&r,12U)==NIYAH_OK);CHECK(result.generated_tokens==8U);
    setup();logits_mode=3;cfg.max_new_tokens=10U;CHECK(cpu(&r,12U)==NIYAH_OK);CHECK(result.generated_tokens==8U);return 0;
}
static int cuda_enabled_control_parity(void) {
    NiyahRepetitionConfig r={4,2};uint32_t want[]={0,1,2,0,1,2};size_t n=0;
    setup();CHECK(niyah_cuda_generation_workspace_floats_with_repetition(&model.config,&r,&n)==NIYAH_OK);CHECK(n==8U);
    CHECK(gpu(&r,n)==NIYAH_OK);CHECK(old_cuda_calls==0);CHECK(memcmp(output,want,sizeof(want))==0);
    CHECK(cd.next_position==8U);CHECK(memcmp(ws,raw,sizeof(raw))==0);CHECK(ws[8]==77.0f);return 0;
}
static int cuda_disabled_legacy_delegate(void) {
    NiyahRepetitionConfig r={1,64};size_t n;setup();
    CHECK(niyah_cuda_generation_workspace_floats_with_repetition(&model.config,&r,&n)==NIYAH_OK);CHECK(n==4U);
    CHECK(gpu(&r,n)==NIYAH_OK);CHECK(old_cuda_calls==1 && output[0]==123U);return 0;
}
static int cuda_short_workspace_no_write(void) {
    NiyahRepetitionConfig r={2,64};setup();CHECK(gpu(&r,7U)==NIYAH_ERR_BUFFER_TOO_SMALL);
    for(size_t i=0;i<40U;i++)CHECK(ws[i]==77.0f);
    CHECK(cd.next_position==0U && result.generated_tokens==0U);return 0;
}
static int cuda_fault_reset_and_eos(void) {
    NiyahRepetitionConfig r={2,64};
    for(size_t i=0;i<5U;i++){setup();fail_position=i;CHECK(gpu(&r,8U)==NIYAH_ERR_IO);
        CHECK(cd.next_position==0U && result.generated_tokens==0U);}
    setup();cfg.stop_on_eos=1;cfg.eos_token=0;CHECK(gpu(&r,8U)==NIYAH_OK);
    CHECK(result.generated_tokens==1U && result.stopped_on_eos==1 && cd.next_position==2U);
    setup();reset_fail=1;CHECK(gpu(&r,7U)==NIYAH_ERR_IO);return 0;
}
static int cuda_invalid_and_guards(void) {
    NiyahRepetitionConfig r={NAN,64};setup();CHECK(gpu(&r,8U)==NIYAH_ERR_INVALID_CONFIG);
    r.penalty=2;setup();logits_mode=1;CHECK(gpu(&r,8U)==NIYAH_ERR_INVALID_CONFIG);CHECK(cd.next_position==0);
    setup();logits_mode=2;cfg.max_new_tokens=10U;CHECK(gpu(&r,8U)==NIYAH_OK);CHECK(result.generated_tokens==8U);
    setup();logits_mode=3;cfg.max_new_tokens=10;CHECK(gpu(&r,8U)==NIYAH_OK);CHECK(result.generated_tokens==8U);return 0;
}
static int shared_cpu_cycle_report(void) {
    NiyahRepetitionConfig r={4.0f,1U}; NiyahGenerationStopReport report;
    setup();cfg.max_new_tokens=12U;
    CHECK(niyah_generate_with_repetition_ex(&model,&cache,prompt,2U,&cfg,
        &r,output,16U,&result,ws,12U,&report)==NIYAH_OK);
    CHECK(result.generated_tokens==8U);
    CHECK(report.reason==NIYAH_STOP_MULTI_TOKEN_LOOP && report.detected_period==2U);
    CHECK(cache.next_position==9U);return 0;
}
static int shared_cuda_cycle_report(void) {
    NiyahRepetitionConfig r={4.0f,1U}; NiyahGenerationStopReport report;
    setup();cfg.max_new_tokens=12U;
    CHECK(niyah_cuda_generate_with_repetition_ex(&cm,&cd,prompt,2U,&cfg,
        &r,output,16U,&result,ws,8U,&report)==NIYAH_OK);
    CHECK(result.generated_tokens==8U);
    CHECK(report.reason==NIYAH_STOP_MULTI_TOKEN_LOOP && report.detected_period==2U);
    CHECK(cd.next_position==9U);return 0;
}
static int shared_budget_preserves_final_decode(void) {
    NiyahRepetitionConfig r={4.0f,1U}; NiyahGenerationStopReport report;
    setup();
    CHECK(niyah_generate_with_repetition_ex(&model,&cache,prompt,2U,&cfg,
        &r,output,16U,&result,ws,12U,&report)==NIYAH_OK);
    CHECK(report.reason==NIYAH_STOP_TOKEN_BUDGET && cache.next_position==8U);
    setup();
    CHECK(niyah_cuda_generate_with_repetition_ex(&cm,&cd,prompt,2U,&cfg,
        &r,output,16U,&result,ws,8U,&report)==NIYAH_OK);
    CHECK(report.reason==NIYAH_STOP_TOKEN_BUDGET && cd.next_position==8U);return 0;
}
static int shared_eos_and_numeric_report(void) {
    NiyahRepetitionConfig r={2.0f,64U}; NiyahGenerationStopReport report;
    setup();cfg.stop_on_eos=1;cfg.eos_token=0;
    CHECK(niyah_generate_with_repetition_ex(&model,&cache,prompt,2U,&cfg,
        &r,output,16U,&result,ws,12U,&report)==NIYAH_OK);
    CHECK(report.reason==NIYAH_STOP_EOS && cache.next_position==2U);
    setup();logits_mode=1;
    CHECK(niyah_generate_with_repetition_ex(&model,&cache,prompt,2U,&cfg,
        &r,output,16U,&result,ws,12U,&report)==NIYAH_ERR_INVALID_CONFIG);
    CHECK(report.reason==NIYAH_STOP_NUMERIC_ERROR && cache.next_position==0U);return 0;
}

int main(void) {
    int tests=0;
    RUN(shared_cpu_cycle_report);RUN(shared_cuda_cycle_report);
    RUN(shared_budget_preserves_final_decode);RUN(shared_eos_and_numeric_report);
    RUN(window1_generated_only);RUN(window2_expiry);RUN(huge_window_no_overflow);
    RUN(disabled_exact_generation);RUN(disabled_rng_state);RUN(signs_unique_history_and_logits);
    RUN(scratch_overlap_rejected);RUN(sampling_overflow_no_rng);RUN(cpu_short_workspace_no_write);
    RUN(checked_add_and_bytes);RUN(invalid_repetition);RUN(eos_not_decoded);
    RUN(cpu_prefill_and_decode_fail_reset);RUN(nonfinite_raw_and_stop_guards);
    RUN(cuda_enabled_control_parity);RUN(cuda_disabled_legacy_delegate);
    RUN(cuda_short_workspace_no_write);RUN(cuda_fault_reset_and_eos);RUN(cuda_invalid_and_guards);
    printf("ISOLATED_CONTROL_TESTS=%d/%d PASS; DECODER=TEST_DOUBLE; GPU=NOT_RUN\n",tests,tests);
    return 0;
}

/* NIYAH_SHARED_GUARD_INTEGRATION_V1: legacy CUDA double adapter only. */
NiyahStatus niyah_cuda_generate_ex(
 const NiyahCudaModelState *m,NiyahCudaDecodeState *d,const uint32_t *p,
 size_t pc,const NiyahGenerationConfig *c,uint32_t *o,size_t oc,
 NiyahGenerationResult *r,float *w,size_t wc,NiyahGenerationStopReport *report)
{
    if (report != NULL) memset(report,0,sizeof(*report));
    return niyah_cuda_generate(m,d,p,pc,c,o,oc,r,w,wc);
}
