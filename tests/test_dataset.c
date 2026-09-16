#include "niyah/dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures=0;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr,"CHECK failed at %s:%d: %s\n",__FILE__,__LINE__,#x); \
    failures++; } } while (0)

static FILE *tfopen(const char *p,const char *m)
{
#if defined(_MSC_VER)
    FILE *f=NULL; if (fopen_s(&f,p,m)!=0) return NULL; return f;
#else
    return fopen(p,m);
#endif
}

static unsigned char *read_all(const char *p,size_t *n)
{
    FILE *f=tfopen(p,"rb"); long end; unsigned char *d;
    *n=0U; if(!f) return NULL;
    if(fseek(f,0L,SEEK_END)!=0){fclose(f);return NULL;}
    end=ftell(f);
    if(end<0L||fseek(f,0L,SEEK_SET)!=0){fclose(f);return NULL;}
    d=(unsigned char*)malloc(end==0L?1U:(size_t)end);
    if(!d){fclose(f);return NULL;}
    if((size_t)end!=0U&&fread(d,1U,(size_t)end,f)!=(size_t)end){free(d);fclose(f);return NULL;}
    if(fclose(f)!=0){free(d);return NULL;}
    *n=(size_t)end; return d;
}

static int write_all(const char *p,const unsigned char *d,size_t n)
{
    FILE *f=tfopen(p,"wb"); if(!f) return 0;
    if(n&&fwrite(d,1U,n,f)!=n){fclose(f);return 0;}
    return fclose(f)==0;
}

static void test_permutation(void)
{
    NiyahDatasetCursor c; unsigned seen[17]; size_t i,idx=0U;
    memset(&c,0,sizeof(c)); memset(seen,0,sizeof(seen));
    CHECK(niyah_dataset_cursor_init(&c,17U,UINT64_C(7))==NIYAH_OK);
    for(i=0U;i<17U;++i){
        CHECK(niyah_dataset_cursor_next(&c,&idx)==NIYAH_OK);
        CHECK(idx<17U);
        if(idx<17U){CHECK(seen[idx]==0U);seen[idx]++;}
    }
    CHECK(c.epoch==UINT64_C(0)); CHECK(c.position==17U);
    CHECK(niyah_dataset_cursor_next(&c,&idx)==NIYAH_OK);
    CHECK(c.epoch==UINT64_C(1)); CHECK(c.position==1U);
    niyah_dataset_cursor_destroy(&c);
}

static void test_determinism(void)
{
    NiyahDatasetCursor a,b,c; size_t i,ai,bi,ci; int diff=0;
    memset(&a,0,sizeof(a));memset(&b,0,sizeof(b));memset(&c,0,sizeof(c));
    CHECK(niyah_dataset_cursor_init(&a,23U,UINT64_C(1234))==NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&b,23U,UINT64_C(1234))==NIYAH_OK);
    CHECK(niyah_dataset_cursor_init(&c,23U,UINT64_C(4321))==NIYAH_OK);
    for(i=0U;i<69U;++i){
        CHECK(niyah_dataset_cursor_next(&a,&ai)==NIYAH_OK);
        CHECK(niyah_dataset_cursor_next(&b,&bi)==NIYAH_OK);
        CHECK(niyah_dataset_cursor_next(&c,&ci)==NIYAH_OK);
        CHECK(ai==bi); if(ai!=ci) diff=1;
    }
    CHECK(diff);
    niyah_dataset_cursor_destroy(&c);niyah_dataset_cursor_destroy(&b);niyah_dataset_cursor_destroy(&a);
}

static void test_resume(void)
{
    const char *p="niyah_dataset_cursor_v1.bin";
    NiyahDatasetCursor a,b; size_t i,ai,bi;
    memset(&a,0,sizeof(a));memset(&b,0,sizeof(b));(void)remove(p);
    CHECK(niyah_dataset_cursor_init(&a,19U,UINT64_C(0x12345678))==NIYAH_OK);
    for(i=0U;i<27U;++i) CHECK(niyah_dataset_cursor_next(&a,&ai)==NIYAH_OK);
    CHECK(a.epoch==UINT64_C(1));CHECK(a.position==8U);
    CHECK(niyah_dataset_cursor_save(&a,p)==NIYAH_OK);
    CHECK(niyah_dataset_cursor_load(p,&b)==NIYAH_OK);
    CHECK(a.sample_count==b.sample_count);CHECK(a.position==b.position);
    CHECK(a.seed==b.seed);CHECK(a.epoch==b.epoch);
    for(i=0U;i<50U;++i){
        CHECK(niyah_dataset_cursor_next(&a,&ai)==NIYAH_OK);
        CHECK(niyah_dataset_cursor_next(&b,&bi)==NIYAH_OK);
        CHECK(ai==bi);CHECK(a.epoch==b.epoch);CHECK(a.position==b.position);
    }
    niyah_dataset_cursor_destroy(&b);niyah_dataset_cursor_destroy(&a);(void)remove(p);
}

static void test_rejection(void)
{
    const char *p="niyah_dataset_cursor_v1.bin",*bad="niyah_dataset_cursor_bad.bin";
    NiyahDatasetCursor c,l; unsigned char *d; size_t n=0U;
    memset(&c,0,sizeof(c));memset(&l,0,sizeof(l));(void)remove(p);(void)remove(bad);
    CHECK(niyah_dataset_cursor_init(&c,11U,UINT64_C(99))==NIYAH_OK);
    CHECK(niyah_dataset_cursor_save(&c,p)==NIYAH_OK);
    d=read_all(p,&n);CHECK(d!=NULL);CHECK(n==56U);
    if(d&&n==56U){
        d[55U]^=1U;CHECK(write_all(bad,d,n));
        CHECK(niyah_dataset_cursor_load(bad,&l)==NIYAH_ERR_CORRUPT_DATA);CHECK(l.order==NULL);
        d[55U]^=1U;d[8U]=2U;CHECK(write_all(bad,d,n));
        CHECK(niyah_dataset_cursor_load(bad,&l)==NIYAH_ERR_UNSUPPORTED_VERSION);CHECK(l.order==NULL);
        d[8U]=1U;CHECK(write_all(bad,d,n-1U));
        CHECK(niyah_dataset_cursor_load(bad,&l)==NIYAH_ERR_CORRUPT_DATA);CHECK(l.order==NULL);
    }
    free(d);niyah_dataset_cursor_destroy(&c);(void)remove(p);(void)remove(bad);
}

int main(void)
{
    test_permutation();
    test_determinism();
    test_resume();
    test_rejection();
    if(failures){fprintf(stderr,"niyah_dataset_test: %d failure(s)\n",failures);return 1;}
    puts("NIYAH_DATASET_LIFECYCLE_P6_D=PASS");
    return 0;
}
