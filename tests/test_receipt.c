#include "niyah/receipt.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>


static int failures = 0;


#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf( \
            stderr, \
            "CHECK failed at %s:%d: %s\n", \
            __FILE__, \
            __LINE__, \
            #expr); \
        failures += 1; \
    } \
} while (0)


static NiyahCoordinationDigest digest(
    uint8_t seed)
{
    NiyahCoordinationDigest d;
    size_t i;

    for (i = 0U;
         i < NIYAH_COORDINATION_DIGEST_BYTES;
         ++i) {
        d.bytes[i] =
            (uint8_t)(
                seed +
                (uint8_t)i);
    }

    return d;
}


static NiyahReceiptV1 base_receipt(void)
{
    NiyahReceiptV1 r;

    memset(&r, 0, sizeof(r));

    r.version =
        NIYAH_RECEIPT_V1_VERSION;

    r.task_id = digest(1U);

    r.producer_lobe =
        NIYAH_LOBE_COMPUTE;

    r.project_id =
        UINT32_C(0x4e495941);

    r.scope =
        NIYAH_SCOPE_ENGINE_CORE |
        NIYAH_SCOPE_CHECKPOINT;

    r.base_revision = digest(33U);
    r.result_revision = digest(65U);

    r.parent_receipt_sha256 =
        digest(97U);

    r.delta_digest =
        digest(129U);

    r.evidence_root =
        digest(161U);

    r.state_before =
        NIYAH_RECEIPT_STATE_UNVERIFIED;

    r.state_after =
        NIYAH_RECEIPT_STATE_VERIFIED;

    r.conflict_count = 0U;

    r.next_action =
        NIYAH_COORDINATION_PROMOTE;

    r.flags = 0U;

    return r;
}


static void test_deterministic_encoding_and_hash(void)
{
    NiyahReceiptV1 a =
        base_receipt();

    NiyahReceiptV1 b =
        base_receipt();

    uint8_t bytes_a[
        NIYAH_RECEIPT_V1_CANONICAL_BYTES];

    uint8_t bytes_b[
        NIYAH_RECEIPT_V1_CANONICAL_BYTES];

    NiyahCoordinationDigest hash_a;
    NiyahCoordinationDigest hash_b;

    CHECK(
        niyah_receipt_v1_encode(
            &a,
            bytes_a) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        niyah_receipt_v1_encode(
            &b,
            bytes_b) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        memcmp(
            bytes_a,
            bytes_b,
            sizeof(bytes_a)) == 0);

    CHECK(
        niyah_receipt_v1_sha256(
            &a,
            &hash_a) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        niyah_receipt_v1_sha256(
            &b,
            &hash_b) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        niyah_coordination_digest_equal(
            &hash_a,
            &hash_b));

    puts("DETERMINISTIC_RECEIPT=PASS");
}


static void test_delta_changes_identity(void)
{
    NiyahReceiptV1 a =
        base_receipt();

    NiyahReceiptV1 b = a;

    NiyahCoordinationDigest hash_a;
    NiyahCoordinationDigest hash_b;

    b.delta_digest.bytes[11] ^=
        UINT8_C(0x40);

    CHECK(
        niyah_receipt_v1_sha256(
            &a,
            &hash_a) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        niyah_receipt_v1_sha256(
            &b,
            &hash_b) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        !niyah_coordination_digest_equal(
            &hash_a,
            &hash_b));

    puts("DELTA_SENSITIVE=PASS");
}


static void test_parent_changes_identity(void)
{
    NiyahReceiptV1 a =
        base_receipt();

    NiyahReceiptV1 b = a;

    NiyahCoordinationDigest hash_a;
    NiyahCoordinationDigest hash_b;

    b.parent_receipt_sha256.bytes[3] ^=
        UINT8_C(0x01);

    CHECK(
        niyah_receipt_v1_sha256(
            &a,
            &hash_a) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        niyah_receipt_v1_sha256(
            &b,
            &hash_b) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        !niyah_coordination_digest_equal(
            &hash_a,
            &hash_b));

    puts("PARENT_CHAIN_SENSITIVE=PASS");
}


static void test_genesis_parent_rule(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    memset(
        &r.parent_receipt_sha256,
        0,
        sizeof(r.parent_receipt_sha256));

    CHECK(
        niyah_receipt_v1_validate(&r) ==
        NIYAH_RECEIPT_INVALID_PARENT);

    r.flags =
        NIYAH_RECEIPT_FLAG_GENESIS;

    CHECK(
        niyah_receipt_v1_validate(&r) ==
        NIYAH_RECEIPT_OK);

    r.parent_receipt_sha256 =
        digest(201U);

    CHECK(
        niyah_receipt_v1_validate(&r) ==
        NIYAH_RECEIPT_INVALID_PARENT);

    puts("GENESIS_PARENT_RULE=PASS");
}


static void test_conflict_is_explicit(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    r.conflict_count = 1U;

    CHECK(
        niyah_receipt_v1_validate(&r) ==
        NIYAH_RECEIPT_INVALID_CONFLICT_STATE);

    r.state_after =
        NIYAH_RECEIPT_STATE_CONFLICT;

    CHECK(
        niyah_receipt_v1_validate(&r) ==
        NIYAH_RECEIPT_OK);

    r.conflict_count = 0U;

    CHECK(
        niyah_receipt_v1_validate(&r) ==
        NIYAH_RECEIPT_INVALID_CONFLICT_STATE);

    puts("EXPLICIT_CONFLICT_STATE=PASS");
}


static void test_bounded_representation(void)
{
    /*
     * The native struct may contain implementation padding.
     * The portable representation is fixed at exactly 232 bytes.
     */
    CHECK(
        NIYAH_RECEIPT_V1_CANONICAL_BYTES ==
        232U);

    CHECK(
        sizeof(NiyahReceiptV1) <= 320U);

    puts("BOUNDED_CONTEXT=PASS");
}


static void test_canonical_magic_and_version(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    uint8_t bytes[
        NIYAH_RECEIPT_V1_CANONICAL_BYTES];

    CHECK(
        niyah_receipt_v1_encode(
            &r,
            bytes) ==
        NIYAH_RECEIPT_OK);

    CHECK(bytes[0] == (uint8_t)'N');
    CHECK(bytes[1] == (uint8_t)'R');
    CHECK(bytes[2] == (uint8_t)'C');
    CHECK(bytes[3] == (uint8_t)'1');

    CHECK(bytes[4] == 0U);
    CHECK(bytes[5] == 0U);
    CHECK(bytes[6] == 0U);
    CHECK(bytes[7] == 1U);

    puts("CANONICAL_ENCODING=PASS");
}


int main(void)
{
    test_deterministic_encoding_and_hash();
    test_delta_changes_identity();
    test_parent_changes_identity();
    test_genesis_parent_rule();
    test_conflict_is_explicit();
    test_bounded_representation();
    test_canonical_magic_and_version();

    if (failures != 0) {
        fprintf(
            stderr,
            "receipt_failures=%d\n",
            failures);

        return 1;
    }

    puts("NIYAH_CONTEXT_RECEIPT_V1=PASS");
    puts("CANONICAL_BYTES=232");

    return 0;
}
