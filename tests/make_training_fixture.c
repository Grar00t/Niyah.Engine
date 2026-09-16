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

    if (argc < 3) return 2;
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
    niyah_dataset_shard_destroy(&shard);
    niyah_tokenizer_destroy(tokenizer);
    return status == NIYAH_OK ? 0 : 6;
}
