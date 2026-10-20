/* Real native decoder regression. No trained/user checkpoint and no mocked logits.
 * With NIYAH_CONFIDENCE_TEST_CUDA, this creates actual CUDA model/cache states.
 */
#include "niyah/generate.h"
#ifdef NIYAH_CONFIDENCE_TEST_CUDA
#include "niyah_cuda_generate_repetition.h"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr,"FAIL kind=%u penalty=%.1f line=%d: %s\n",kind,(double)penalty,__LINE__,#x); \
    goto done; } } while(0)

static int exercise(unsigned kind,float penalty)
{
    NiyahModelConfig mc={0}; NiyahModel model={0};
    NiyahGenerationConfig gc={0}; NiyahGenerationResult result={0};
    NiyahGenerationStopReport stop={0}; NiyahRepetitionConfig rp={penalty,64U};
    const uint32_t prompt=0U; uint32_t out[24]; float *workspace=NULL;
    size_t n=0U,i; int passed=0; NiyahStatus status;
#ifdef NIYAH_CONFIDENCE_TEST_CUDA
    NiyahCudaModelState device={0}; NiyahCudaDecodeState cache={0};
#else
    NiyahKVCache cache={0};
#endif
    uint32_t next[8]={1U,2U,3U,7U,5U,6U,7U,7U};
    mc.vocab_size=8U;mc.context_length=32U;mc.embedding_dim=8U;
    mc.n_layers=1U;mc.n_heads=2U;mc.n_kv_heads=1U;
    mc.ffn_hidden_dim=16U;mc.rms_norm_eps=1e-5f;
    REQUIRE(niyah_model_create(&model,&mc)==NIYAH_OK);
    /* Identity token embeddings, zero residual projections, known next-token map. */
    for(i=0U;i<8U;i++) {
        model.weights[model.layout.token_embedding+i*8U+i]=1.0f;
        model.weights[model.layout.final_norm+i]=1.0f;
    }
    if(kind==1U)next[0]=7U;           /* Confident EOS on first step. */
    if(kind==2U)next[1]=1U;           /* Actual single-token loop. */
    if(kind==3U)next[2]=1U;           /* Actual two-token loop. */
    for(i=0U;i<8U;i++)model.weights[model.layout.lm_head+next[i]*8U+i]=10.0f;
    if(kind==5U)model.weights[model.layout.lm_head]=NAN;
    if(kind==6U)model.weights[model.layout.lm_head]=INFINITY;
    gc.max_new_tokens=kind==4U?1U:24U;
    gc.eos_token=7U;gc.stop_on_eos=1;gc.sampler.temperature=0.0f;gc.sampler.seed=42U;
    memset(out,0x55,sizeof(out));
#ifdef NIYAH_CONFIDENCE_TEST_CUDA
    REQUIRE(niyah_cuda_model_state_create(&device,&model)==0);
    REQUIRE(niyah_cuda_decode_state_create(&cache,&device)==0);
    REQUIRE(niyah_cuda_generation_workspace_floats_with_repetition(&mc,&rp,&n)==NIYAH_OK);
#else
    REQUIRE(niyah_kv_cache_create(&cache,&mc)==NIYAH_OK);
    REQUIRE(niyah_generation_workspace_floats_with_repetition(&mc,&rp,&n)==NIYAH_OK);
#endif
    workspace=(float*)calloc(n+1U,sizeof(float));REQUIRE(workspace!=NULL);
    workspace[n]=12345.0f;
#ifdef NIYAH_CONFIDENCE_TEST_CUDA
    status=niyah_cuda_generate_with_repetition_ex(&device,&cache,&prompt,1U,&gc,&rp,
        out,24U,&result,workspace,n,&stop);
#else
    status=niyah_generate_with_repetition_ex(&model,&cache,&prompt,1U,&gc,&rp,
        out,24U,&result,workspace,n,&stop);
#endif
    REQUIRE(workspace[n]==12345.0f);
    if(kind>=5U) {
        REQUIRE(status!=NIYAH_OK);
        REQUIRE(stop.reason==NIYAH_STOP_NUMERIC_ERROR);
        REQUIRE(result.generated_tokens==0U && result.prompt_tokens==0U);
        REQUIRE(cache.next_position==0U && out[0]==UINT32_C(0x55555555));
    } else {
        REQUIRE(status==NIYAH_OK);
        if(kind==0U) {
            const uint32_t expected[4]={1U,2U,3U,7U};
            if(result.generated_tokens!=4U)fprintf(stderr,"OBSERVED reason=%s tokens=%zu\n",niyah_stop_name(stop.reason),result.generated_tokens);
            REQUIRE(result.generated_tokens==4U && memcmp(out,expected,sizeof(expected))==0);
            REQUIRE(stop.reason==NIYAH_STOP_EOS && result.stopped_on_eos==1);
        } else if(kind==1U) {
            REQUIRE(result.generated_tokens==1U && out[0]==7U);
            REQUIRE(stop.reason==NIYAH_STOP_EOS && cache.next_position==1U);
        } else if(kind==2U) {
            REQUIRE(result.generated_tokens==8U && stop.reason==NIYAH_STOP_SINGLE_TOKEN_LOOP);
            for(i=0U;i<8U;i++)REQUIRE(out[i]==1U);
        } else if(kind==3U) {
            REQUIRE(result.generated_tokens==8U && stop.reason==NIYAH_STOP_MULTI_TOKEN_LOOP && stop.detected_period==2U);
            for(i=0U;i<8U;i++)REQUIRE(out[i]==1U+(uint32_t)(i%2U));
        } else {
            REQUIRE(result.generated_tokens==1U && out[0]==1U);
            REQUIRE(stop.reason==NIYAH_STOP_TOKEN_BUDGET);
        }
    }
    passed=1;
done:
    free(workspace);
#ifdef NIYAH_CONFIDENCE_TEST_CUDA
    niyah_cuda_decode_state_destroy(&cache);niyah_cuda_model_state_destroy(&device);
#else
    niyah_kv_cache_destroy(&cache);
#endif
    niyah_model_destroy(&model);
    return passed;
}
int main(void)
{
    unsigned kind,p,passed=0U;
    const float penalties[2]={1.0f,1.1f};
    for(p=0U;p<2U;p++)for(kind=0U;kind<7U;kind++) {
        if(!exercise(kind,penalties[p]))return 1;
        passed++;
    }
    printf("CONFIDENT_GENERATION_CASES=%u/14 PASS\n",passed);
    return 0;
}
