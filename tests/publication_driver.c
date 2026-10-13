#define _CRT_SECURE_NO_WARNINGS
#include "niyah_pair_internal.h"
#include "niyah_publish_internal.h"
#include <string.h>
#include <stdlib.h>
static const char *fault;
static int inject(const char *point) { return fault && strcmp(point, fault) == 0; }
int main(int argc, char **argv)
{
    NiyahStatus status = NIYAH_ERR_INVALID_ARGUMENT;
    if (argc >= 2 && strcmp(argv[1], "validate") == 0 && argc == 4) {
        int legacy = 0;
        status = niyah_pair_validate(argv[2], argv[3], &legacy);
        printf("legacy=%d\n", legacy);
    } else if (argc == 8 && strcmp(argv[1], "save") == 0) {
        NiyahTokenizer *tok = NULL;
        NiyahModel model = {0};
        NiyahAdamWState state = {0};
        NiyahAdamWConfig config;
        NiyahDatasetCursor cursor = {0};
        status = niyah_tokenizer_load(argv[2], &tok);
        if (status == NIYAH_OK) status = niyah_checkpoint_load_with_tokenizer(argv[3], tok, &model, &state, &config);
        if (status == NIYAH_OK) status = niyah_dataset_cursor_load(argv[4], &cursor);
        fault = argv[7]; niyah_publish_set_fault(inject);
        if (status == NIYAH_OK) status = niyah_pair_save(argv[5], argv[6], &model, &state, &config, tok, &cursor);
        niyah_dataset_cursor_destroy(&cursor); niyah_adamw_state_destroy(&state); niyah_model_destroy(&model); niyah_tokenizer_destroy(tok);
    } else if (argc == 9 && strcmp(argv[1], "serialize") == 0) {
        NiyahTokenizer *tok = NULL;
        NiyahDatasetShard shard = {0};
        NiyahModel model = {0};
        NiyahAdamWState state = {0};
        NiyahAdamWConfig config;
        NiyahDatasetCursor cursor = {0};
        status = niyah_tokenizer_load(argv[3], &tok);
        if (status == NIYAH_OK) status = niyah_dataset_shard_load(argv[4], tok, &shard);
        if (status == NIYAH_OK) status = niyah_checkpoint_load_with_tokenizer(argv[5], tok, &model, &state, &config);
        if (status == NIYAH_OK) status = niyah_dataset_cursor_load(argv[6], &cursor);
        fault = argv[8]; niyah_publish_set_fault(inject);
        if (status == NIYAH_OK) {
            if (strcmp(argv[2], "tokenizer") == 0) status = niyah_tokenizer_save(tok, argv[7]);
            else if (strcmp(argv[2], "shard") == 0) status = niyah_dataset_shard_save(&shard, tok, argv[7]);
            else if (strcmp(argv[2], "cursor") == 0) status = niyah_dataset_cursor_save(&cursor, argv[7]);
            else if (strcmp(argv[2], "checkpoint") == 0) status = niyah_checkpoint_save_with_tokenizer(argv[7], &model, &state, &config, tok);
            else if (strcmp(argv[2], "segment") == 0) {
                NiyahModelConfig segmented = model.config;
                niyah_adamw_state_destroy(&state); niyah_model_destroy(&model);
                segmented.n_segments = 2U;
                status = niyah_model_create(&model, &segmented);
                if (status == NIYAH_OK) status = niyah_model_reset_parameters(&model, 42U);
                if (status == NIYAH_OK) status = niyah_adamw_state_create(&state, &model);
                if (status == NIYAH_OK) status = niyah_checkpoint_save_with_tokenizer(argv[7], &model, &state, &config, tok);
            } else status = NIYAH_ERR_INVALID_ARGUMENT;
        }
        niyah_dataset_shard_destroy(&shard); niyah_dataset_cursor_destroy(&cursor); niyah_adamw_state_destroy(&state); niyah_model_destroy(&model); niyah_tokenizer_destroy(tok);
    } else if (argc == 5 && (strcmp(argv[1], "atomic") == 0 || strcmp(argv[1], "publish") == 0)) {
        NiyahPublication stage = {0};
        FILE *file;
        fault = argv[4]; niyah_publish_set_fault(inject);
        if (strcmp(argv[1], "atomic") == 0) {
            file = niyah_atomic_open(argv[2], &stage);
            if (file) {
                size_t count = strlen(argv[3]);
                int partial = strcmp(fault, "partial_write") == 0;
                if (partial) count /= 2U;
                status = fwrite(argv[3], 1U, count, file) == count ? NIYAH_OK : NIYAH_ERR_IO;
                if (partial) {
                    printf("partial_staged_bytes=%zu\n", count);
                    status = NIYAH_ERR_IO;
                }
                status = niyah_atomic_finish(file, &stage, argv[2], status);
            }
        } else {
            status = niyah_publish_begin(argv[2], &stage);
            if (status == NIYAH_OK) {
                file = fopen(stage.path, "wb");
                if (!file) status = NIYAH_ERR_IO;
                else {
                    if (fwrite(argv[3], 1U, strlen(argv[3]), file) != strlen(argv[3])) status = NIYAH_ERR_IO;
                    if (fclose(file) != 0) status = NIYAH_ERR_IO;
                }
                if (status == NIYAH_OK) status = niyah_publish_commit(&stage, argv[2], 0);
                if (niyah_publish_cleanup(&stage) != NIYAH_OK) status = NIYAH_ERR_IO;
            }
        }
    }
    printf("status=%d\n", (int)status);
    return status == NIYAH_OK ? 0 : 1;
}
