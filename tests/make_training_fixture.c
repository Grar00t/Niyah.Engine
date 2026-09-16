#include "niyah/dataset.h"
#include "niyah/tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    static const uint8_t corpus[] =
        "abababababababab cdcdcdcdcdcd efefefefefef "
        "abab cdcd efef abab cdcd efef";
    static const uint8_t text[] =
        "abababab cdcdcdcd efefefef abab cdcd efef";
    NiyahTokenizerTrainConfig config;
    NiyahTokenizer *tokenizer = NULL;
    NiyahDatasetShard shard;
    NiyahStatus status;
    int i;

    if (argc < 4) return 2;
    for (i = 1; i < argc; ++i) (void)remove(argv[i]);
    memset(&shard, 0, sizeof(shard));

    config.target_vocab_size = 266U;
    config.min_pair_frequency = 2U;
    status = niyah_tokenizer_train(
        corpus, sizeof(corpus) - 1U, &config, &tokenizer);
    if (status != NIYAH_OK) return 3;
    status = niyah_tokenizer_save(tokenizer, argv[1]);
    if (status != NIYAH_OK) {
        niyah_tokenizer_destroy(tokenizer);
        return 4;
    }
    status = niyah_dataset_shard_build_text(
        tokenizer, text, sizeof(text) - 1U, 4U, &shard);
    if (status != NIYAH_OK) {
        niyah_tokenizer_destroy(tokenizer);
        return 5;
    }
    status = niyah_dataset_shard_save(&shard, tokenizer, argv[2]);
    if (status == NIYAH_OK) {
        size_t swap_i;
        size_t swap_j;
        int changed = 0;
        for (swap_i = 1U; swap_i + 1U < shard.token_count && changed == 0; ++swap_i) {
            for (swap_j = swap_i + 1U; swap_j + 1U < shard.token_count; ++swap_j) {
                if (shard.tokens[swap_i] != shard.tokens[swap_j]) {
                    uint32_t tmp = shard.tokens[swap_i];
                    shard.tokens[swap_i] = shard.tokens[swap_j];
                    shard.tokens[swap_j] = tmp;
                    changed = 1;
                    break;
                }
            }
        }
        if (changed == 0) {
            status = NIYAH_ERR_INVALID_CONFIG;
        } else {
            status = niyah_dataset_shard_save(&shard, tokenizer, argv[3]);
        }
    }
    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
    return status == NIYAH_OK ? 0 : 6;
}
