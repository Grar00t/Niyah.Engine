#define _CRT_SECURE_NO_WARNINGS
#include "niyah_pair_internal.h"
#include "niyah_publish_internal.h"
#include <stdlib.h>
#include <string.h>
static const unsigned char pending_magic[8] = {'N','I','Y','A','H','P','I','1'};
static const unsigned char complete_magic[8] = {'N','I','Y','A','H','P','R','1'};
static char *sidecar(const char *path, const char *suffix)
{
    size_t a, b;
    char *out;
    if (!path) return NULL;
    a = strlen(path); b = strlen(suffix);
    if (a > SIZE_MAX - b - 1U) return NULL;
    out = (char *)malloc(a + b + 1U);
    if (out) { memcpy(out, path, a); memcpy(out+a, suffix, b+1U); }
    return out;
}
static NiyahStatus publish_bytes(const char *path, const unsigned char *bytes, size_t count)
{
    NiyahPublication stage = {0};
    FILE *file;
    NiyahStatus status = niyah_publish_begin(path, &stage);
    if (status != NIYAH_OK) return status;
    file = niyah_file_open(stage.path, "wb");
    if (!file) status = NIYAH_ERR_IO;
    else {
        if (fwrite(bytes, 1U, count, file) != count) status = NIYAH_ERR_IO;
        if (fflush(file) != 0) status = NIYAH_ERR_IO;
        if (fclose(file) != 0) status = NIYAH_ERR_IO;
    }
    if (status == NIYAH_OK) status = niyah_publish_commit(&stage, path, 0);
    if (niyah_publish_cleanup(&stage) != NIYAH_OK) status = NIYAH_ERR_IO;
    return status;
}
static NiyahStatus read_exact(const char *path, unsigned char *bytes, size_t count)
{
    FILE *file = niyah_file_open(path, "rb");
    NiyahStatus status = NIYAH_OK;
    if (!file) return NIYAH_ERR_IO;
    if (fread(bytes, 1U, count, file) != count || fgetc(file) != EOF || ferror(file)) status = NIYAH_ERR_INVALID_CONFIG;
    if (fclose(file) != 0) status = NIYAH_ERR_IO;
    return status;
}
NiyahStatus niyah_pair_validate(const char *checkpoint, const char *cursor_path, int *legacy)
{
    char *pending = sidecar(checkpoint, ".pending"), *complete = sidecar(checkpoint, ".complete");
    unsigned char bytes[72], digest[32], intent[8];
    NiyahDatasetCursor cursor = {0};
    NiyahStatus status = NIYAH_ERR_IO;
    int p, c;
    if (!pending || !complete || !cursor_path || !legacy) goto done;
    *legacy = 0;
    p = niyah_publish_path_state(pending); c = niyah_publish_path_state(complete);
    if (p < 0 || c < 0) goto done;
    if (p == 0 && c == 0) { *legacy = 1; status = NIYAH_OK; goto done; }
    if (p != 0) {
        status = read_exact(pending, intent, sizeof(intent));
        if (status != NIYAH_OK || memcmp(intent, pending_magic, sizeof(intent)) != 0) { status = NIYAH_ERR_INVALID_CONFIG; goto done; }
    }
    if (c == 0) { status = NIYAH_ERR_INVALID_CONFIG; goto done; }
    status = read_exact(complete, bytes, sizeof(bytes));
    if (status != NIYAH_OK) goto done;
    if (memcmp(bytes, complete_magic, 8U) != 0) { status = NIYAH_ERR_INVALID_CONFIG; goto done; }
    status = niyah_checkpoint_identity_sha256(checkpoint, digest);
    if (status != NIYAH_OK) goto done;
    if (memcmp(bytes+8U, digest, 32U) != 0) { status = NIYAH_ERR_INVALID_CONFIG; goto done; }
    status = niyah_dataset_cursor_load(cursor_path, &cursor);
    if (status != NIYAH_OK) goto done;
    if (!cursor.has_checkpoint_identity || memcmp(cursor.checkpoint_identity, digest, 32U) != 0) { status = NIYAH_ERR_INVALID_CONFIG; goto done; }
    status = niyah_checkpoint_identity_sha256(cursor_path, digest);
    if (status == NIYAH_OK && memcmp(bytes+40U, digest, 32U) != 0) status = NIYAH_ERR_INVALID_CONFIG;
done:
    niyah_dataset_cursor_destroy(&cursor); free(pending); free(complete); return status;
}
NiyahStatus niyah_pair_save(const char *checkpoint, const char *cursor_path,
    const NiyahModel *model, const NiyahAdamWState *state,
    const NiyahAdamWConfig *config, const NiyahTokenizer *tokenizer,
    const NiyahDatasetCursor *cursor)
{
    char *pending = sidecar(checkpoint, ".pending"), *complete = sidecar(checkpoint, ".complete");
    NiyahPublication ck = {0}, cur = {0};
    NiyahDatasetCursor bound, loaded_cursor = {0};
    NiyahModel loaded_model = {0};
    NiyahAdamWState loaded_state = {0};
    NiyahAdamWConfig loaded_config;
    unsigned char bytes[72];
    NiyahStatus status = NIYAH_ERR_INVALID_ARGUMENT;
    int legacy = 0;
    if (!pending || !complete || !cursor || !cursor_path) goto done;
    if (niyah_publish_path_state(checkpoint) != 0 || niyah_publish_path_state(cursor_path) != 0 ||
        niyah_publish_path_state(pending) != 0 || niyah_publish_path_state(complete) != 0) goto done;
    status = publish_bytes(pending, pending_magic, sizeof(pending_magic));
    if (status != NIYAH_OK) goto done;
    status = niyah_publish_begin(checkpoint, &ck);
    if (status != NIYAH_OK) goto done;
    status = niyah_publish_begin(cursor_path, &cur);
    if (status != NIYAH_OK) goto done;
#define PAIR_FAULT(point) do { if (niyah_publish_fault(point)) { status = NIYAH_ERR_IO; goto done; } } while (0)
    PAIR_FAULT("checkpoint_save");
    status = niyah_checkpoint_save_with_tokenizer(ck.path, model, state, config, tokenizer);
    if (status != NIYAH_OK) goto done;
    PAIR_FAULT("checkpoint_identity");
    memcpy(bytes, complete_magic, 8U);
    status = niyah_checkpoint_identity_sha256(ck.path, bytes+8U);
    if (status != NIYAH_OK) goto done;
    bound = *cursor;
    PAIR_FAULT("cursor_bind");
    status = niyah_dataset_cursor_bind_checkpoint_identity(&bound, bytes+8U);
    if (status != NIYAH_OK) goto done;
    PAIR_FAULT("cursor_save");
    status = niyah_dataset_cursor_save(&bound, cur.path);
    if (status != NIYAH_OK) goto done;
    status = niyah_checkpoint_identity_sha256(cur.path, bytes+40U);
    if (status != NIYAH_OK) goto done;
    /* Read through existing readers before exposing either final artifact. */
    status = niyah_checkpoint_load_with_tokenizer(ck.path, tokenizer, &loaded_model, &loaded_state, &loaded_config);
    if (status != NIYAH_OK) goto done;
    status = niyah_dataset_cursor_load(cur.path, &loaded_cursor);
    if (status != NIYAH_OK) goto done;
    if (loaded_state.step != state->step || !loaded_cursor.has_checkpoint_identity ||
        memcmp(loaded_cursor.checkpoint_identity, bytes+8U, 32U) != 0 ||
        memcmp(loaded_cursor.dataset_identity, cursor->dataset_identity, 32U) != 0) { status = NIYAH_ERR_INVALID_CONFIG; goto done; }
    status = niyah_publish_commit(&ck, checkpoint, 0);
    if (status != NIYAH_OK) goto done;
    PAIR_FAULT("after_checkpoint");
    status = niyah_publish_commit(&cur, cursor_path, 0);
    if (status != NIYAH_OK) goto done;
    PAIR_FAULT("after_cursor");
    PAIR_FAULT("before_complete");
    /* Rehash published artifacts, not just temporary names. */
    status = niyah_checkpoint_identity_sha256(checkpoint, bytes+8U);
    if (status == NIYAH_OK) status = niyah_checkpoint_identity_sha256(cursor_path, bytes+40U);
    if (status != NIYAH_OK) goto done;
    status = publish_bytes(complete, bytes, sizeof(bytes));
    if (status == NIYAH_OK) status = niyah_pair_validate(checkpoint, cursor_path, &legacy);
done:
    if (niyah_publish_cleanup(&ck) != NIYAH_OK) status = NIYAH_ERR_IO;
    if (niyah_publish_cleanup(&cur) != NIYAH_OK) status = NIYAH_ERR_IO;
    niyah_dataset_cursor_destroy(&loaded_cursor); niyah_adamw_state_destroy(&loaded_state); niyah_model_destroy(&loaded_model);
    free(pending); free(complete); return status;
}
