#include "niyah/generate.h"
#ifdef NIYAH_REPETITION_TEST_CUDA
#include "niyah_cuda_generate_repetition.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); return 1; } } while (0)

int main(void)
{
    NiyahModelConfig mc = {0};
    NiyahModel model = {0};
    NiyahKVCache a = {0}, b = {0};
    NiyahGenerationConfig gc = {0};
    NiyahGenerationResult ra = {0}, rb = {0};
    NiyahRepetitionConfig disabled = {1.0f,64U}, enabled = {2.0f,2U};
    NiyahSampler sampler;
    const uint32_t prompt[] = {1U,2U,3U};
    uint32_t old[6] = {0}, current[6] = {0}, manual[6] = {0};
    float logits[16], scratch[16];
    size_t legacy_count = 0U, new_count = 0U, disabled_count = 0U, decode_count = 0U;
    float *wa, *wb;
    size_t i;

    mc.vocab_size=16U; mc.context_length=32U; mc.embedding_dim=8U;
    mc.n_layers=2U; mc.n_heads=4U; mc.n_kv_heads=2U;
    mc.ffn_hidden_dim=16U; mc.rms_norm_eps=1.0e-5f;
    CHECK(niyah_model_create(&model,&mc)==NIYAH_OK);
    CHECK(niyah_model_reset_parameters(&model,987654321U)==NIYAH_OK);
    CHECK(niyah_kv_cache_create(&a,&mc)==NIYAH_OK);
    CHECK(niyah_kv_cache_create(&b,&mc)==NIYAH_OK);
    CHECK(niyah_generation_workspace_floats(&mc,&legacy_count)==NIYAH_OK);
    CHECK(niyah_decode_workspace_floats(&mc,&decode_count)==NIYAH_OK);
    CHECK(niyah_generation_workspace_floats_with_repetition(&mc,&disabled,&disabled_count)==NIYAH_OK);
    CHECK(disabled_count==legacy_count);
    CHECK(niyah_generation_workspace_floats_with_repetition(&mc,&enabled,&new_count)==NIYAH_OK);
    CHECK(new_count==legacy_count+16U);
    wa=(float *)calloc(new_count+1U,sizeof(float));
    wb=(float *)calloc(new_count+1U,sizeof(float));
    CHECK(wa!=NULL && wb!=NULL);
    gc.max_new_tokens=6U; gc.sampler.temperature=0.7f; gc.sampler.seed=998877U;

    CHECK(niyah_generate(&model,&a,prompt,3U,&gc,old,6U,&ra,wa,legacy_count)==NIYAH_OK);
    CHECK(niyah_generate_with_repetition(&model,&b,prompt,3U,&gc,&disabled,current,6U,&rb,wb,disabled_count)==NIYAH_OK);
    CHECK(ra.generated_tokens==rb.generated_tokens && ra.prompt_tokens==rb.prompt_tokens);
    CHECK(memcmp(old,current,ra.generated_tokens*sizeof(*old))==0);
    CHECK(a.next_position==b.next_position);
    CHECK(memcmp(a.keys,b.keys,a.values_per_tensor*sizeof(float))==0);
    CHECK(memcmp(a.values,b.values,a.values_per_tensor*sizeof(float))==0);

    niyah_kv_cache_reset(&a); niyah_kv_cache_reset(&b);
    wa[new_count]=12345.0f;
    CHECK(niyah_generate_with_repetition(&model,&a,prompt,3U,&gc,&enabled,current,6U,&ra,wa,new_count)==NIYAH_OK);
    CHECK(ra.generated_tokens==6U && wa[new_count]==12345.0f);
    CHECK(niyah_sampler_init(&sampler,&gc.sampler)==NIYAH_OK);
    for(i=0U;i<3U;i++)
        CHECK(niyah_transformer_decode_token(&model,&b,prompt[i],logits,16U,wb,decode_count)==NIYAH_OK);
    for(i=0U;i<6U;i++) {
        const size_t n=i<enabled.window?i:enabled.window;
        CHECK(niyah_sampler_sample_with_history(&sampler,logits,16U,manual+i-n,n,
            enabled.penalty,scratch,16U,&manual[i])==NIYAH_OK);
        CHECK(manual[i]==current[i]);
        CHECK(niyah_transformer_decode_token(&model,&b,manual[i],logits,16U,wb,decode_count)==NIYAH_OK);
    }
    CHECK(memcmp(wa+decode_count,logits,sizeof(logits))==0);
    CHECK(a.next_position==b.next_position);

    niyah_kv_cache_reset(&a);
    wa[new_count-1U]=6789.0f;
    CHECK(niyah_generate_with_repetition(&model,&a,prompt,3U,&gc,&enabled,current,6U,&ra,wa,new_count-1U)==NIYAH_ERR_BUFFER_TOO_SMALL);
    CHECK(a.next_position==0U && ra.generated_tokens==0U && wa[new_count-1U]==6789.0f);
    {
        const uint32_t bad[]={1U,99U};
        CHECK(niyah_generate_with_repetition(&model,&a,bad,2U,&gc,&enabled,current,6U,&ra,wa,new_count)==NIYAH_ERR_INVALID_ARGUMENT);
        CHECK(a.next_position==0U && ra.prompt_tokens==0U && ra.generated_tokens==0U);
    }

#ifdef NIYAH_REPETITION_TEST_CUDA
    {
        NiyahCudaModelState cm={0};
        NiyahCudaDecodeState ca={0},cb={0};
        float h1[32]={0},h2[32]={0};
        size_t nc=0U;
        CHECK(niyah_cuda_model_state_create(&cm,&model)==0);
        CHECK(niyah_cuda_decode_state_create(&ca,&cm)==0);
        CHECK(niyah_cuda_decode_state_create(&cb,&cm)==0);
        gc.sampler.temperature=0.0f;
        CHECK(niyah_cuda_generate(&cm,&ca,prompt,3U,&gc,old,6U,&ra,h1,16U)==NIYAH_OK);
        CHECK(niyah_cuda_generate_with_repetition(&cm,&cb,prompt,3U,&gc,&disabled,current,6U,&rb,h2,16U)==NIYAH_OK);
        CHECK(ra.generated_tokens==rb.generated_tokens);
        CHECK(memcmp(old,current,ra.generated_tokens*sizeof(*old))==0);
        CHECK(niyah_cuda_decode_state_reset(&cb)==0);
        niyah_kv_cache_reset(&a);
        CHECK(niyah_cuda_generation_workspace_floats_with_repetition(&mc,&enabled,&nc)==NIYAH_OK && nc==32U);
        CHECK(niyah_cuda_generate_with_repetition(&cm,&cb,prompt,3U,&gc,&enabled,current,6U,&rb,h2,nc)==NIYAH_OK);
        CHECK(niyah_generate_with_repetition(&model,&a,prompt,3U,&gc,&enabled,old,6U,&ra,wa,new_count)==NIYAH_OK);
        CHECK(ra.generated_tokens==rb.generated_tokens);
        CHECK(memcmp(old,current,ra.generated_tokens*sizeof(*old))==0);
        niyah_cuda_decode_state_destroy(&cb); niyah_cuda_decode_state_destroy(&ca);
        niyah_cuda_model_state_destroy(&cm);
        puts("REAL_CUDA_REPETITION_TINY_MODEL=PASS");
    }
#endif
    free(wb);free(wa);niyah_kv_cache_destroy(&b);niyah_kv_cache_destroy(&a);niyah_model_destroy(&model);
    puts("REAL_CPU_REPETITION_TINY_MODEL=PASS");
    return 0;
}
