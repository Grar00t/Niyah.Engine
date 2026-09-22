#include "niyah/checkpoint.h"
#include "niyah/dataset.h"
#include "niyah/train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  niyah-train new --shard <shard> --checkpoint-out <ckpt> --updates N\n"
        "  niyah-train resume --shard <shard> --checkpoint <ckpt> --checkpoint-out <ckpt> --updates N\n");
}

static const char *arg_value(int argc, char **argv, const char *name) {
    for(int i=2;i+1<argc;++i) if(strcmp(argv[i],name)==0) return argv[i+1];
    return NULL;
}

static int parse_u64(const char *s, uint64_t *out) {
    if(!s || !out || *s=='\0') return 0;
    char *end=NULL; unsigned long long v=strtoull(s,&end,10);
    if(!end || *end!='\0') return 0;
    *out=(uint64_t)v;
    return 1;
}

int main(int argc, char **argv) {
    if(argc<2) { usage(); return 2; }
    int is_new=strcmp(argv[1],"new")==0;
    int is_resume=strcmp(argv[1],"resume")==0;
    if(!is_new && !is_resume) { usage(); return 2; }
    const char *shard=arg_value(argc,argv,"--shard");
    const char *out_path=arg_value(argc,argv,"--checkpoint-out");
    const char *updates_s=arg_value(argc,argv,"--updates");
    const char *in_path=is_resume?arg_value(argc,argv,"--checkpoint"):NULL;
    uint64_t updates=0u;
    if(!shard || !out_path || !updates_s || !parse_u64(updates_s,&updates) || (is_resume && !in_path)) {
        usage(); return 2;
    }
    niyah_dataset ds; niyah_dataset_init(&ds);
    niyah_status st=niyah_dataset_load(shard,&ds);
    niyah_checkpoint ckpt; niyah_checkpoint_init(&ckpt);
    niyah_train_result r;
    if(st==NIYAH_OK && is_new) st=niyah_train_new(&ds,updates,&ckpt,&r);
    if(st==NIYAH_OK && is_resume) {
        st=niyah_checkpoint_load(in_path,&ckpt);
        if(st==NIYAH_OK) st=niyah_train_resume(&ds,&ckpt,updates,&r);
    }
    if(st==NIYAH_OK) st=niyah_checkpoint_save(&ckpt,out_path);
    if(st==NIYAH_OK) {
        printf("status=ok\nmode=%s\nupdates_requested=%llu\nupdates_applied=%llu\ncursor_before=%llu\ncursor_after=%llu\ntransitions_seen=%llu\n",
            is_new?"new":"resume",
            (unsigned long long)r.updates_requested,
            (unsigned long long)r.updates_applied,
            (unsigned long long)r.cursor_before,
            (unsigned long long)r.cursor_after,
            (unsigned long long)ckpt.model.transitions_seen);
    } else fprintf(stderr,"error=%s\n",niyah_status_string(st));
    niyah_dataset_free(&ds); return st==NIYAH_OK ? 0 : 1;
}
