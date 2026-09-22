#include "niyah/checkpoint.h"
#include "niyah/dataset.h"
#include "niyah/eval.h"
#include "niyah/sha256.h"
#include "niyah/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  niyah prepare --input <text> --output <shard>\n"
        "  niyah eval --shard <shard> [--checkpoint <ckpt>]\n"
        "  niyah run --checkpoint <ckpt> --prompt <text> [--max-tokens N]\n"
        "  niyah inspect --shard <shard>\n");
}

static const char *arg_value(int argc, char **argv, const char *name) {
    for (int i=2; i+1<argc; ++i) if (strcmp(argv[i],name)==0) return argv[i+1];
    return NULL;
}

static int parse_u64(const char *s, uint64_t *out) {
    if(!s || !out || *s=='\0') return 0;
    char *end=NULL;
    unsigned long long v=strtoull(s,&end,10);
    if(!end || *end!='\0') return 0;
    *out=(uint64_t)v;
    return 1;
}

static niyah_status read_all(const char *path, uint8_t **out, size_t *len) {
    if(!path || !out || !len) return NIYAH_ERR_INVALID;
    FILE *f=fopen(path,"rb"); if(!f) return NIYAH_ERR_IO;
    if(fseek(f,0,SEEK_END)!=0) { fclose(f); return NIYAH_ERR_IO; }
    long n=ftell(f); if(n<0) { fclose(f); return NIYAH_ERR_IO; }
    if(fseek(f,0,SEEK_SET)!=0) { fclose(f); return NIYAH_ERR_IO; }
    uint8_t *buf=NULL;
    if(n>0) {
        buf=(uint8_t*)malloc((size_t)n);
        if(!buf) { fclose(f); return NIYAH_ERR_NOMEM; }
        if(fread(buf,1,(size_t)n,f)!=(size_t)n) { free(buf); fclose(f); return NIYAH_ERR_IO; }
    }
    fclose(f); *out=buf; *len=(size_t)n; return NIYAH_OK;
}

static int cmd_prepare(int argc, char **argv) {
    const char *input=arg_value(argc,argv,"--input");
    const char *output=arg_value(argc,argv,"--output");
    if(!input || !output) { usage(); return 2; }
    uint8_t *bytes=NULL; size_t byte_count=0u;
    niyah_status st=read_all(input,&bytes,&byte_count);
    if(st!=NIYAH_OK) { fprintf(stderr,"error=%s\n",niyah_status_string(st)); return 1; }
    niyah_tokenizer tok; st=niyah_tokenizer_init_byte(&tok);
    uint32_t *tokens=NULL;
    if(st==NIYAH_OK && byte_count!=0u) {
        tokens=(uint32_t*)malloc(byte_count*sizeof(uint32_t));
        if(!tokens) st=NIYAH_ERR_NOMEM;
    }
    size_t token_count=0u;
    if(st==NIYAH_OK) st=niyah_tokenizer_encode(&tok,bytes,byte_count,tokens,byte_count,&token_count);
    niyah_dataset ds; niyah_dataset_init(&ds);
    if(st==NIYAH_OK) st=niyah_dataset_from_tokens(tokens,(uint64_t)token_count,&ds);
    if(st==NIYAH_OK) st=niyah_dataset_save(&ds,output);
    if(st==NIYAH_OK) {
        char hash[65]; niyah_sha256_hex(ds.content_sha256,hash);
        printf("status=ok\nbytes=%zu\ntokens=%zu\nsha256=%s\n",byte_count,token_count,hash);
    } else fprintf(stderr,"error=%s\n",niyah_status_string(st));
    niyah_dataset_free(&ds); free(tokens); free(bytes);
    return st==NIYAH_OK ? 0 : 1;
}

static int cmd_eval(int argc, char **argv) {
    const char *shard=arg_value(argc,argv,"--shard");
    const char *ckpt_path=arg_value(argc,argv,"--checkpoint");
    if(!shard) { usage(); return 2; }
    niyah_dataset ds; niyah_dataset_init(&ds);
    niyah_status st=niyah_dataset_load(shard,&ds);
    niyah_eval_result r;
    if(st==NIYAH_OK && ckpt_path) {
        niyah_checkpoint ckpt; niyah_checkpoint_init(&ckpt);
        st=niyah_checkpoint_load(ckpt_path,&ckpt);
        if(st==NIYAH_OK) st=niyah_checkpoint_require_dataset(&ckpt,ds.content_sha256);
        if(st==NIYAH_OK) st=niyah_eval_model(&ckpt.model,&ds,&r);
    } else if(st==NIYAH_OK) {
        st=niyah_eval_add1_bigram_baseline(&ds,&r);
    }
    if(st==NIYAH_OK) {
        printf("status=ok\nmode=%s\ntransitions=%llu\nnll=%.12f\navg_nll=%.12f\nperplexity=%.12f\n",
            ckpt_path?"checkpoint":"add1_bigram_baseline",
            (unsigned long long)r.transitions,r.nll,r.avg_nll,r.perplexity);
    } else fprintf(stderr,"error=%s\n",niyah_status_string(st));
    niyah_dataset_free(&ds); return st==NIYAH_OK ? 0 : 1;
}

static int cmd_inspect(int argc, char **argv) {
    const char *shard=arg_value(argc,argv,"--shard");
    if(!shard) { usage(); return 2; }
    niyah_dataset ds; niyah_dataset_init(&ds);
    niyah_status st=niyah_dataset_load(shard,&ds);
    if(st==NIYAH_OK) {
        char hash[65]; niyah_sha256_hex(ds.content_sha256,hash);
        printf("status=ok\ntokens=%llu\nsha256=%s\n",(unsigned long long)ds.token_count,hash);
    } else fprintf(stderr,"error=%s\n",niyah_status_string(st));
    niyah_dataset_free(&ds); return st==NIYAH_OK ? 0 : 1;
}

static int cmd_run(int argc, char **argv) {
    const char *ckpt_path=arg_value(argc,argv,"--checkpoint");
    const char *prompt=arg_value(argc,argv,"--prompt");
    const char *max_s=arg_value(argc,argv,"--max-tokens");
    uint64_t max_tokens=64u;
    if(max_s && !parse_u64(max_s,&max_tokens)) { fprintf(stderr,"invalid --max-tokens\n"); return 2; }
    if(!ckpt_path || !prompt || *prompt=='\0') { usage(); return 2; }
    niyah_checkpoint ckpt; niyah_checkpoint_init(&ckpt);
    niyah_status st=niyah_checkpoint_load(ckpt_path,&ckpt);
    if(st!=NIYAH_OK) { fprintf(stderr,"error=%s\n",niyah_status_string(st)); return 1; }
    const uint8_t *p=(const uint8_t*)prompt;
    size_t n=strlen(prompt);
    uint32_t prev=(uint32_t)p[n-1u];
    fwrite(prompt,1,n,stdout);
    for(uint64_t i=0u;i<max_tokens;++i) {
        uint32_t next=niyah_model_greedy_next(&ckpt.model,prev);
        uint8_t b=(uint8_t)next;
        fwrite(&b,1,1u,stdout);
        prev=next;
    }
    fputc('\n',stdout);
    return 0;
}

int main(int argc, char **argv) {
    if(argc<2) { usage(); return 2; }
    if(strcmp(argv[1],"prepare")==0) return cmd_prepare(argc,argv);
    if(strcmp(argv[1],"eval")==0) return cmd_eval(argc,argv);
    if(strcmp(argv[1],"run")==0) return cmd_run(argc,argv);
    if(strcmp(argv[1],"inspect")==0) return cmd_inspect(argc,argv);
    usage(); return 2;
}
