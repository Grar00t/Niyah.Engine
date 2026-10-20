#include "niyah/shared_guard.h"
#include <stdio.h>
static unsigned passed,failed;
#define CHECK(name,expr) do{if(expr){passed++;printf("%s=PASS\n",name);}else{failed++;printf("%s=FAIL line=%d\n",name,__LINE__);}}while(0)
static NiyahGuardConfig config(void){NiyahGuardConfig c;memset(&c,0,sizeof(c));c.max_tokens=64U;c.same_limit=8U;c.max_period=8U;c.cycles=4U;c.entropy_floor=0.05f;c.stop_on_eos=1;c.eos_token=257U;return c;}
int main(void){
 NiyahGuard g;NiyahGuardConfig c=config();NiyahTextGuard text;unsigned i,p;
 CHECK("init",niyah_guard_init(&g,&c)==NIYAH_STOP_NONE);
 for(i=0;i<8U;i++)(void)niyah_guard_token(&g,7U);
 CHECK("single",g.reason==NIYAH_STOP_SINGLE_TOKEN_LOOP && g.generated==8U);
 CHECK("terminal_sticky",niyah_guard_token(&g,9U)==NIYAH_STOP_SINGLE_TOKEN_LOOP && g.generated==8U);
 for(p=2U;p<=8U;p++){
  (void)niyah_guard_init(&g,&c);for(i=0;i<p*4U;i++)(void)niyah_guard_token(&g,i%p);
  CHECK("period_2_through_8",g.reason==NIYAH_STOP_MULTI_TOKEN_LOOP && g.detected_period==p);
 }
 (void)niyah_guard_init(&g,&c);for(i=0;i<40U;i++)(void)niyah_guard_token(&g,1000U+i);
 for(i=0;i<12U;i++)(void)niyah_guard_token(&g,i%3U);
 CHECK("prefix_ring_wrap",g.reason==NIYAH_STOP_MULTI_TOKEN_LOOP && g.detected_period==3U);
 (void)niyah_guard_init(&g,&c);for(i=0;i<63U;i++)(void)niyah_guard_token(&g,i);
 CHECK("diverse",g.reason==NIYAH_STOP_NONE);
 CHECK("budget",niyah_guard_token(&g,63U)==NIYAH_STOP_TOKEN_BUDGET && g.generated==64U);
 (void)niyah_guard_init(&g,&c);CHECK("eos",niyah_guard_token(&g,257U)==NIYAH_STOP_EOS);
 c.max_tokens=1U;(void)niyah_guard_init(&g,&c);CHECK("eos_budget_priority",niyah_guard_token(&g,257U)==NIYAH_STOP_EOS);c=config();
 (void)niyah_guard_init(&g,&c);CHECK("wide_entropy",niyah_guard_entropy(&g,8.294050f)==NIYAH_STOP_NONE);
 for(i=0;i<8U;i++){(void)niyah_guard_token(&g,1U);}
 CHECK("wide_greedy_loop",g.reason==NIYAH_STOP_SINGLE_TOKEN_LOOP);
 (void)niyah_guard_init(&g,&c);CHECK("entropy_floor",niyah_guard_entropy(&g,0.01f)==NIYAH_STOP_ENTROPY_FLOOR);
 (void)niyah_guard_init(&g,&c);CHECK("nan",niyah_guard_entropy(&g,NAN)==NIYAH_STOP_NUMERIC_ERROR);
 (void)niyah_guard_init(&g,&c);CHECK("inf",niyah_guard_entropy(&g,INFINITY)==NIYAH_STOP_NUMERIC_ERROR);
 c.deadline_ms=100U;(void)niyah_guard_init(&g,&c);CHECK("before_deadline",niyah_guard_poll(&g,99U,0)==NIYAH_STOP_NONE);
 CHECK("timeout",niyah_guard_poll(&g,100U,0)==NIYAH_STOP_TIMEOUT);
 (void)niyah_guard_init(&g,&c);CHECK("cancel_priority",niyah_guard_poll(&g,100U,1)==NIYAH_STOP_CANCELLED);
 c=config();c.max_period=9U;CHECK("bad_config",niyah_guard_init(&g,&c)==NIYAH_STOP_INVALID_CONFIG);
 niyah_text_guard_init(&text,16U);for(i=0;i<16U;i++){const uint8_t a=0xd9U,b=0x87U;(void)niyah_text_guard_feed(&text,&a,1U);(void)niyah_text_guard_feed(&text,&b,1U);}
 CHECK("heh_split_utf8",text.reason==NIYAH_STOP_CHARACTER_LOOP);
 niyah_text_guard_init(&text,16U);{const uint8_t bytes[]={0xd9U,0x87U,0xd9U,0x88U,0x20U,0xd9U,0x85U};(void)niyah_text_guard_feed(&text,bytes,sizeof(bytes));}
 CHECK("normal_arabic",niyah_text_guard_finish(&text)==NIYAH_STOP_NONE);
 niyah_text_guard_init(&text,16U);for(i=0;i<20U;i++){const uint8_t b='1';(void)niyah_text_guard_feed(&text,&b,1U);}
 CHECK("digits",text.reason==NIYAH_STOP_NONE);
 niyah_text_guard_init(&text,16U);{const uint8_t b=0xd9U;(void)niyah_text_guard_feed(&text,&b,1U);}
 CHECK("incomplete_utf8",niyah_text_guard_finish(&text)==NIYAH_STOP_INVALID_UTF8);
 niyah_text_guard_init(&text,16U);{const uint8_t b[]={0xc0U,0x80U};(void)niyah_text_guard_feed(&text,b,sizeof(b));}
 CHECK("overlong_utf8",text.reason==NIYAH_STOP_INVALID_UTF8);
 niyah_text_guard_init(&text,16U);{const uint8_t b[]={0xedU,0xa0U,0x80U};(void)niyah_text_guard_feed(&text,b,sizeof(b));}
 CHECK("surrogate",text.reason==NIYAH_STOP_INVALID_UTF8);
 for(i=0;i<=12U;i++)CHECK("session_fence",niyah_session_can_commit((NiyahStopReason)i,1,1,1)==(i==1U));
 CHECK("quality_not_autoapproved",!niyah_session_can_commit(NIYAH_STOP_EOS,1,1,0));
 CHECK("execution_failed",!niyah_session_can_commit(NIYAH_STOP_EOS,0,1,1));
 CHECK("invalid_text",!niyah_session_can_commit(NIYAH_STOP_EOS,1,0,1));
 CHECK("reason_name",strcmp(niyah_stop_name(NIYAH_STOP_MULTI_TOKEN_LOOP),"MULTI_TOKEN_LOOP")==0);
 printf("GUARD_CHECKS=%u passed=%u failed=%u\n",passed+failed,passed,failed);return failed?1:0;
}
