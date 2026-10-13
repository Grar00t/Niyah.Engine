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



static void test_coordination_binding_happy_path(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER,
            &gate) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_PASS);

    puts("RECEIPT_COORDINATION_BINDING=PASS");
}


static void test_coordination_binding_stale_base(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    observed.bytes[5] ^=
        UINT8_C(0x80);

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER,
            &gate) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_STALE_BASE);

    puts("RECEIPT_STALE_BASE=REJECTED");
}


static void test_coordination_binding_chain(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_PASS;

    expected_parent.bytes[17] ^=
        UINT8_C(0x01);

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER,
            &gate) ==
        NIYAH_RECEIPT_CHAIN_MISMATCH);

    CHECK(
        gate ==
        NIYAH_COORDINATION_INVALID_ENVELOPE);

    puts("RECEIPT_CHAIN_MISMATCH=REJECTED");
}


static void test_coordination_binding_action(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_PASS;

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_VERIFY,
            NIYAH_LOBE_VERIFIER,
            &gate) ==
        NIYAH_RECEIPT_ACTION_MISMATCH);

    CHECK(
        gate ==
        NIYAH_COORDINATION_INVALID_ENVELOPE);

    puts("RECEIPT_ACTION_MISMATCH=REJECTED");
}


static void test_coordination_binding_unverified_promotion(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    r.state_after =
        NIYAH_RECEIPT_STATE_UNVERIFIED;

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER,
            &gate) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_PROMOTION_WITHOUT_PROOF);

    puts("UNVERIFIED_PROMOTION=REJECTED");
}


static void test_coordination_binding_conflict(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    r.state_after =
        NIYAH_RECEIPT_STATE_CONFLICT;

    r.conflict_count = 2U;

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER,
            &gate) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_CONTRADICTION);

    puts("RECEIPT_CONTRADICTION=REJECTED");
}


static void test_coordination_binding_self_verification(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_COMPUTE,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_COMPUTE,
            &gate) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_SELF_VERIFICATION);

    puts("RECEIPT_SELF_VERIFICATION=REJECTED");
}


static void test_coordination_binding_scope(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            NIYAH_SCOPE_ENGINE_CORE,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER,
            &gate) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_WRONG_SCOPE);

    puts("RECEIPT_SCOPE_VIOLATION=REJECTED");
}


static void test_coordination_binding_writer(void)
{
    NiyahReceiptV1 r =
        base_receipt();

    NiyahCoordinationDigest observed =
        r.base_revision;

    NiyahCoordinationDigest expected_parent =
        r.parent_receipt_sha256;

    NiyahCoordinationGate gate =
        NIYAH_COORDINATION_INVALID_ENVELOPE;

    r.next_action =
        NIYAH_COORDINATION_EXECUTE;

    r.state_after =
        NIYAH_RECEIPT_STATE_UNVERIFIED;

    CHECK(
        niyah_receipt_v1_coordination_gate(
            &r,
            &observed,
            &expected_parent,
            r.project_id,
            r.scope,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_CHRONICLE,
            NIYAH_COORDINATION_EXECUTE,
            NIYAH_LOBE_COMPUTE,
            &gate) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_WRITER_VIOLATION);

    puts("RECEIPT_WRITER_VIOLATION=REJECTED");
}



/* Independently assembled big-endian protocol bytes and Python hashlib SHA-256
 * known answers. These constants are not generated by niyah_receipt_v1_encode. */
static void test_canonical_known_answers(void)
{
    static const uint8_t expected_bytes[232] = {
        0x4e, 0x52, 0x43, 0x31, 0x00, 0x00, 0x00, 0x01, 0x01, 0x02, 0x03, 0x04,
        0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
        0x1d, 0x1e, 0x1f, 0x20, 0x00, 0x00, 0x00, 0x01, 0x4e, 0x49, 0x59, 0x41,
        0x00, 0x00, 0x00, 0x09, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30, 0x31, 0x32, 0x33, 0x34,
        0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f, 0x40,
        0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c,
        0x4d, 0x4e, 0x4f, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60, 0x61, 0x62, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f, 0x70,
        0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c,
        0x7d, 0x7e, 0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
        0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94,
        0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f, 0xa0,
        0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xab, 0xac,
        0xad, 0xae, 0xaf, 0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8,
        0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03,
        0x00, 0x00, 0x00, 0x00,
    };
    static const uint8_t expected_hash[32] = {
        0xde, 0x7f, 0x74, 0x56, 0x9b, 0x3c, 0xd8, 0xb2, 0x34, 0xf9, 0xf3, 0x9a,
        0x72, 0x78, 0x54, 0x25, 0x31, 0x65, 0x82, 0x7b, 0x56, 0x5b, 0x0f, 0x26,
        0x88, 0xcd, 0xaf, 0xaa, 0x9e, 0x08, 0xa4, 0xcc,
    };
    static const uint8_t genesis_hash[32] = {
        0x8a, 0xa8, 0xc6, 0xc4, 0xbd, 0x8d, 0x7c, 0xf9, 0x2f, 0xf6, 0xb3, 0xf2,
        0xfd, 0x85, 0x59, 0xb1, 0x2a, 0xa6, 0xfb, 0x70, 0x53, 0xcb, 0x98, 0x9e,
        0x4b, 0xaa, 0x23, 0xee, 0xbe, 0x7f, 0x66, 0xc6,
    };
    NiyahReceiptV1 r = base_receipt();
    NiyahCoordinationDigest actual;
    uint8_t bytes[232];
    CHECK(niyah_receipt_v1_encode(&r, bytes) == NIYAH_RECEIPT_OK);
    CHECK(memcmp(bytes, expected_bytes, sizeof(bytes)) == 0);
    CHECK(niyah_receipt_v1_sha256(&r, &actual) == NIYAH_RECEIPT_OK);
    CHECK(memcmp(actual.bytes, expected_hash, sizeof(expected_hash)) == 0);
    r.flags = NIYAH_RECEIPT_FLAG_GENESIS;
    memset(&r.parent_receipt_sha256, 0, sizeof(r.parent_receipt_sha256));
    CHECK(niyah_receipt_v1_sha256(&r, &actual) == NIYAH_RECEIPT_OK);
    CHECK(memcmp(actual.bytes, genesis_hash, sizeof(genesis_hash)) == 0);
}

static void test_invalid_fields_and_encoding_preservation(void)
{
    NiyahReceiptV1 r = base_receipt();
    uint8_t bytes[232], before[232];
    memset(bytes, 0xa5, sizeof(bytes));
    memcpy(before, bytes, sizeof(bytes));
#define REJECT_FIELD(field, value, expected) do { \
    r.field = (value); \
    CHECK(niyah_receipt_v1_encode(&r, bytes) == (expected)); \
    CHECK(memcmp(bytes, before, sizeof(bytes)) == 0); \
    r = base_receipt(); \
} while (0)
    REJECT_FIELD(version, 0U, NIYAH_RECEIPT_INVALID_VERSION);
    REJECT_FIELD(version, UINT32_MAX, NIYAH_RECEIPT_INVALID_VERSION);
    REJECT_FIELD(flags, 2U, NIYAH_RECEIPT_INVALID_ARGUMENT);
    REJECT_FIELD(producer_lobe, NIYAH_LOBE_NONE, NIYAH_RECEIPT_INVALID_LOBE);
    REJECT_FIELD(project_id, 0U, NIYAH_RECEIPT_INVALID_SCOPE);
    REJECT_FIELD(scope, 0U, NIYAH_RECEIPT_INVALID_SCOPE);
    REJECT_FIELD(state_before, (NiyahReceiptClaimState)0, NIYAH_RECEIPT_INVALID_STATE);
    REJECT_FIELD(state_after, (NiyahReceiptClaimState)6, NIYAH_RECEIPT_INVALID_STATE);
    REJECT_FIELD(next_action, (NiyahCoordinationAction)0, NIYAH_RECEIPT_INVALID_STATE);
#undef REJECT_FIELD
#define REJECT_DIGEST(field) do { \
    memset(&r.field, 0, sizeof(r.field)); \
    CHECK(niyah_receipt_v1_encode(&r, bytes) == NIYAH_RECEIPT_INVALID_IDENTITY); \
    CHECK(memcmp(bytes, before, sizeof(bytes)) == 0); \
    r = base_receipt(); \
} while (0)
    REJECT_DIGEST(task_id);
    REJECT_DIGEST(base_revision);
    REJECT_DIGEST(result_revision);
    REJECT_DIGEST(delta_digest);
    REJECT_DIGEST(evidence_root);
#undef REJECT_DIGEST
    r.project_id = UINT32_MAX;
    r.conflict_count = UINT32_MAX;
    r.state_after = NIYAH_RECEIPT_STATE_CONFLICT;
    CHECK(niyah_receipt_v1_validate(&r) == NIYAH_RECEIPT_OK);
    CHECK(niyah_receipt_v1_encode(NULL, bytes) == NIYAH_RECEIPT_INVALID_ARGUMENT);
    CHECK(niyah_receipt_v1_encode(&r, NULL) == NIYAH_RECEIPT_INVALID_ARGUMENT);
}

int main(void)
{
    test_canonical_known_answers();
    test_invalid_fields_and_encoding_preservation();
    test_deterministic_encoding_and_hash();
    test_delta_changes_identity();
    test_parent_changes_identity();
    test_genesis_parent_rule();
    test_conflict_is_explicit();
    test_bounded_representation();
    test_canonical_magic_and_version();

    test_coordination_binding_happy_path();
    test_coordination_binding_stale_base();
    test_coordination_binding_chain();
    test_coordination_binding_action();
    test_coordination_binding_unverified_promotion();
    test_coordination_binding_conflict();
    test_coordination_binding_self_verification();
    test_coordination_binding_scope();
    test_coordination_binding_writer();

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
