#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "niyah_mini_config.h"
#include "niyah_mini_model.h"

#define main niyah_online_train_main
#include "../../tools/niyah_online_train.c"
#undef main

static void path_join(char *dst, size_t cap, const char *dir, const char *name)
{
    int n = snprintf(dst, cap, "%s/%s", dir, name);
    assert(n > 0 && (size_t)n < cap);
}

int main(void)
{
    char root[160];
    int n = snprintf(root, sizeof(root), "niyah_online_train_partial_%ld",
                     (long)getpid());
    assert(n > 0 && (size_t)n < sizeof(root));
    assert(mkdir(root, 0700) == 0);

    char config_path[256];
    char weights_path[256];
    char out_dir[256];
    path_join(config_path, sizeof(config_path), root, "config.json");
    path_join(weights_path, sizeof(weights_path), root, "weights.f32.bin");
    path_join(out_dir, sizeof(out_dir), root, "output");

    NiyahMiniConfig config;
    niyah_mini_config_init(&config, NIYAH_MINI_TINY);
    config.n_layers = 1;
    config.n_dim = 16;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.n_ff = 32;
    config.n_vocab = 32;
    config.n_ctx = 64;
    config.tie_word_embeddings = true;
    assert(niyah_mini_config_validate(&config) == NIYAH_OK);

    NiyahMiniModel model;
    memset(&model, 0, sizeof(model));
    assert(niyah_mini_model_init(&model, &config) == NIYAH_OK);
    assert(niyah_mini_model_save(&model, config_path, weights_path) == NIYAH_OK);
    niyah_mini_model_free(&model);

    FILE *input = tmpfile();
    assert(input != NULL);
    int32_t one_token = 1;
    assert(fwrite(&one_token, sizeof(one_token), 1, input) == 1);
    assert(fflush(input) == 0);
    rewind(input);

    int saved_stdin = dup(STDIN_FILENO);
    assert(saved_stdin >= 0);
    assert(dup2(fileno(input), STDIN_FILENO) >= 0);
    clearerr(stdin);

    char *argv[] = {
        (char *)"niyah-online-train",
        config_path,
        weights_path,
        out_dir,
        NULL
    };
    int rc = niyah_online_train_main(4, argv);

    assert(dup2(saved_stdin, STDIN_FILENO) >= 0);
    close(saved_stdin);
    fclose(input);
    clearerr(stdin);

    assert(rc == 22);

    struct stat st;
    errno = 0;
    assert(stat(out_dir, &st) != 0);
    assert(errno == ENOENT);

    assert(unlink(config_path) == 0);
    assert(unlink(weights_path) == 0);
    assert(rmdir(root) == 0);

    return 0;
}
