#include "niyah/dataset.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #x); \
    failures++; } } while (0)

int main(void)
{
    uint32_t tokens_a[] = {NIYAH_TOKEN_BOS, 10U, 11U, NIYAH_TOKEN_EOS};
    uint32_t tokens_b[] = {NIYAH_TOKEN_BOS, 12U, 13U, NIYAH_TOKEN_EOS};
    uint32_t tokens_c[] = {NIYAH_TOKEN_BOS, 12U, 14U, NIYAH_TOKEN_EOS};
    NiyahDatasetShard ab[2];
    NiyahDatasetShard ab_same[2];
    NiyahDatasetShard ba[2];
    NiyahDatasetShard ac[2];
    NiyahDatasetShard mixed[2];
    uint8_t id_ab[NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE];
    uint8_t id_ab_same[NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE];
    uint8_t id_ba[NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE];
    uint8_t id_ac[NIYAH_DATASET_COLLECTION_IDENTITY_SHA256_SIZE];
    size_t i;

    memset(ab, 0, sizeof(ab));
    for (i = 0U; i < NIYAH_DATASET_TOKENIZER_IDENTITY_SIZE; ++i) {
        ab[0].tokenizer_identity[i] = (uint8_t)(i + 1U);
        ab[1].tokenizer_identity[i] = (uint8_t)(i + 1U);
    }
    ab[0].tokens = tokens_a;
    ab[0].token_count = 4U;
    ab[0].sequence_length = 2U;
    ab[0].sample_count = 2U;
    ab[1].tokens = tokens_b;
    ab[1].token_count = 4U;
    ab[1].sequence_length = 2U;
    ab[1].sample_count = 2U;

    memcpy(ab_same, ab, sizeof(ab));
    ba[0] = ab[1];
    ba[1] = ab[0];
    ac[0] = ab[0];
    ac[1] = ab[1];
    ac[1].tokens = tokens_c;
    memcpy(mixed, ab, sizeof(ab));
    mixed[1].tokenizer_identity[0] ^= UINT8_C(1);

    CHECK(niyah_dataset_collection_identity_sha256(ab, 2U, id_ab) == NIYAH_OK);
    CHECK(niyah_dataset_collection_identity_sha256(
        ab_same, 2U, id_ab_same) == NIYAH_OK);
    CHECK(memcmp(id_ab, id_ab_same, sizeof(id_ab)) == 0);

    CHECK(niyah_dataset_collection_identity_sha256(ba, 2U, id_ba) == NIYAH_OK);
    CHECK(memcmp(id_ab, id_ba, sizeof(id_ab)) != 0);

    CHECK(niyah_dataset_collection_identity_sha256(ac, 2U, id_ac) == NIYAH_OK);
    CHECK(memcmp(id_ab, id_ac, sizeof(id_ab)) != 0);

    CHECK(niyah_dataset_collection_identity_sha256(
        mixed, 2U, id_ac) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(niyah_dataset_collection_identity_sha256(
        ab, 0U, id_ac) == NIYAH_ERR_INVALID_CONFIG);
    CHECK(niyah_dataset_collection_identity_sha256(
        NULL, 2U, id_ac) == NIYAH_ERR_INVALID_ARGUMENT);

    if (failures != 0) {
        fprintf(stderr, "niyah_dataset_collection_test: %d failure(s)\n", failures);
        return 1;
    }
    puts("NIYAH_DATASET_COLLECTION_IDENTITY_P6_KC=PASS");
    return 0;
}
