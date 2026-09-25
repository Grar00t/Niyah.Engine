#include "niyah/receipt_chain.h"

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
            (uint8_t)(seed + (uint8_t)i);
    }

    return d;
}


static NiyahReceiptChainV1 new_chain(void)
{
    NiyahReceiptChainV1 state;

    NiyahCoordinationDigest revision =
        digest(17U);

    memset(
        &state,
        0,
        sizeof(state));

    CHECK(
        niyah_receipt_chain_v1_init(
            &state,
            UINT32_C(0x4e495941),
            NIYAH_SCOPE_ENGINE_CORE |
                NIYAH_SCOPE_CHECKPOINT,
            &revision) ==
        NIYAH_RECEIPT_CHAIN_OK);

    return state;
}


static NiyahReceiptV1 receipt_for_state(
    const NiyahReceiptChainV1 *state,
    uint8_t result_seed)
{
    NiyahReceiptV1 r;

    memset(&r, 0, sizeof(r));

    r.version =
        NIYAH_RECEIPT_V1_VERSION;

    r.task_id =
        digest((uint8_t)(result_seed + 31U));

    r.producer_lobe =
        NIYAH_LOBE_COMPUTE;

    r.project_id =
        state->project_id;

    r.scope =
        NIYAH_SCOPE_ENGINE_CORE;

    r.base_revision =
        state->current_revision;

    r.result_revision =
        digest(result_seed);

    r.delta_digest =
        digest((uint8_t)(result_seed + 47U));

    r.evidence_root =
        digest((uint8_t)(result_seed + 79U));

    r.state_before =
        NIYAH_RECEIPT_STATE_UNVERIFIED;

    r.state_after =
        NIYAH_RECEIPT_STATE_VERIFIED;

    r.conflict_count = 0U;

    r.next_action =
        NIYAH_COORDINATION_PROMOTE;

    if (state->has_head != 0) {
        r.parent_receipt_sha256 =
            state->head_receipt_sha256;
    } else {
        memset(
            &r.parent_receipt_sha256,
            0,
            sizeof(r.parent_receipt_sha256));

        r.flags =
            NIYAH_RECEIPT_FLAG_GENESIS;
    }

    return r;
}


static NiyahReceiptChainStatus apply_promote(
    NiyahReceiptChainV1 *state,
    const NiyahReceiptV1 *receipt,
    NiyahCoordinationGate *gate,
    NiyahReceiptStatus *receipt_status)
{
    return niyah_receipt_chain_v1_apply(
        state,
        receipt,
        NIYAH_LOBE_VERIFIER,
        NIYAH_LOBE_COMPUTE,
        NIYAH_COORDINATION_PROMOTE,
        NIYAH_LOBE_VERIFIER,
        gate,
        receipt_status);
}


static void test_genesis_acceptance(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 r =
        receipt_for_state(&state, 61U);

    NiyahCoordinationDigest expected_hash;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    CHECK(
        niyah_receipt_v1_sha256(
            &r,
            &expected_hash) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        apply_promote(
            &state,
            &r,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_OK);

    CHECK(
        receipt_status ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_PASS);

    CHECK(state.has_head == 1);
    CHECK(state.sequence == UINT64_C(1));

    CHECK(
        niyah_coordination_digest_equal(
            &state.current_revision,
            &r.result_revision));

    CHECK(
        niyah_coordination_digest_equal(
            &state.head_receipt_sha256,
            &expected_hash));

    puts("CHAIN_GENESIS=PASS");
}


static void test_two_receipt_chain(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 first =
        receipt_for_state(&state, 61U);

    NiyahReceiptV1 second;

    NiyahCoordinationDigest first_hash;
    NiyahCoordinationDigest second_hash;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    CHECK(
        apply_promote(
            &state,
            &first,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_OK);

    CHECK(
        niyah_receipt_v1_sha256(
            &first,
            &first_hash) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        niyah_coordination_digest_equal(
            &state.head_receipt_sha256,
            &first_hash));

    second =
        receipt_for_state(&state, 93U);

    CHECK(
        niyah_coordination_digest_equal(
            &second.parent_receipt_sha256,
            &first_hash));

    CHECK(
        apply_promote(
            &state,
            &second,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_OK);

    CHECK(state.sequence == UINT64_C(2));

    CHECK(
        niyah_receipt_v1_sha256(
            &second,
            &second_hash) ==
        NIYAH_RECEIPT_OK);

    CHECK(
        niyah_coordination_digest_equal(
            &state.head_receipt_sha256,
            &second_hash));

    CHECK(
        niyah_coordination_digest_equal(
            &state.current_revision,
            &second.result_revision));

    puts("CHAIN_TWO_RECEIPTS=PASS");
}


static void test_replay_rejected_without_mutation(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 first =
        receipt_for_state(&state, 61U);

    NiyahReceiptChainV1 before_replay;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    CHECK(
        apply_promote(
            &state,
            &first,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_OK);

    before_replay = state;

    CHECK(
        apply_promote(
            &state,
            &first,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED);

    CHECK(
        receipt_status ==
        NIYAH_RECEIPT_CHAIN_MISMATCH);

    CHECK(
        memcmp(
            &state,
            &before_replay,
            sizeof(state)) == 0);

    puts("CHAIN_REPLAY=REJECTED");
}


static void test_stale_base_no_mutation(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 first =
        receipt_for_state(&state, 61U);

    NiyahReceiptV1 stale;

    NiyahReceiptChainV1 before;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    CHECK(
        apply_promote(
            &state,
            &first,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_OK);

    stale =
        receipt_for_state(&state, 93U);

    stale.base_revision.bytes[4] ^=
        UINT8_C(0x80);

    before = state;

    CHECK(
        apply_promote(
            &state,
            &stale,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED);

    CHECK(
        receipt_status ==
        NIYAH_RECEIPT_OK);

    CHECK(
        gate ==
        NIYAH_COORDINATION_STALE_BASE);

    CHECK(
        memcmp(
            &state,
            &before,
            sizeof(state)) == 0);

    puts("CHAIN_STALE_BASE=REJECTED");
}


static void test_scope_no_mutation(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 r =
        receipt_for_state(&state, 61U);

    NiyahReceiptChainV1 before =
        state;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    r.scope |=
        NIYAH_SCOPE_GRAPH;

    CHECK(
        apply_promote(
            &state,
            &r,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED);

    CHECK(
        gate ==
        NIYAH_COORDINATION_WRONG_SCOPE);

    CHECK(
        memcmp(
            &state,
            &before,
            sizeof(state)) == 0);

    puts("CHAIN_SCOPE=REJECTED");
}


static void test_conflict_no_mutation(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 r =
        receipt_for_state(&state, 61U);

    NiyahReceiptChainV1 before =
        state;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    r.state_after =
        NIYAH_RECEIPT_STATE_CONFLICT;

    r.conflict_count = 1U;

    CHECK(
        apply_promote(
            &state,
            &r,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED);

    CHECK(
        gate ==
        NIYAH_COORDINATION_CONTRADICTION);

    CHECK(
        memcmp(
            &state,
            &before,
            sizeof(state)) == 0);

    puts("CHAIN_CONTRADICTION=REJECTED");
}


static void test_self_verification_no_mutation(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 r =
        receipt_for_state(&state, 61U);

    NiyahReceiptChainV1 before =
        state;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    CHECK(
        niyah_receipt_chain_v1_apply(
            &state,
            &r,
            NIYAH_LOBE_COMPUTE,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_COMPUTE,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED);

    CHECK(
        gate ==
        NIYAH_COORDINATION_SELF_VERIFICATION);

    CHECK(
        memcmp(
            &state,
            &before,
            sizeof(state)) == 0);

    puts("CHAIN_SELF_VERIFICATION=REJECTED");
}


static void test_action_mismatch_no_mutation(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 r =
        receipt_for_state(&state, 61U);

    NiyahReceiptChainV1 before =
        state;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    CHECK(
        niyah_receipt_chain_v1_apply(
            &state,
            &r,
            NIYAH_LOBE_VERIFIER,
            NIYAH_LOBE_COMPUTE,
            NIYAH_COORDINATION_VERIFY,
            NIYAH_LOBE_VERIFIER,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED);

    CHECK(
        receipt_status ==
        NIYAH_RECEIPT_ACTION_MISMATCH);

    CHECK(
        memcmp(
            &state,
            &before,
            sizeof(state)) == 0);

    puts("CHAIN_ACTION_MISMATCH=REJECTED");
}


static void test_non_genesis_first_receipt_rejected(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 r =
        receipt_for_state(&state, 61U);

    NiyahReceiptChainV1 before =
        state;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    r.flags = 0U;

    CHECK(
        apply_promote(
            &state,
            &r,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_RECEIPT_REJECTED);

    CHECK(
        receipt_status ==
        NIYAH_RECEIPT_INVALID_PARENT);

    CHECK(
        memcmp(
            &state,
            &before,
            sizeof(state)) == 0);

    puts("CHAIN_GENESIS_REQUIRED=PASS");
}


static void test_sequence_overflow_no_mutation(void)
{
    NiyahReceiptChainV1 state =
        new_chain();

    NiyahReceiptV1 first =
        receipt_for_state(&state, 61U);

    NiyahReceiptV1 next;

    NiyahReceiptChainV1 before;

    NiyahCoordinationGate gate;
    NiyahReceiptStatus receipt_status;

    CHECK(
        apply_promote(
            &state,
            &first,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_OK);

    state.sequence = UINT64_MAX;

    next =
        receipt_for_state(&state, 93U);

    before = state;

    CHECK(
        apply_promote(
            &state,
            &next,
            &gate,
            &receipt_status) ==
        NIYAH_RECEIPT_CHAIN_SEQUENCE_OVERFLOW);

    CHECK(
        memcmp(
            &state,
            &before,
            sizeof(state)) == 0);

    puts("CHAIN_SEQUENCE_OVERFLOW=REJECTED");
}


int main(void)
{
    test_genesis_acceptance();
    test_two_receipt_chain();
    test_replay_rejected_without_mutation();
    test_stale_base_no_mutation();
    test_scope_no_mutation();
    test_conflict_no_mutation();
    test_self_verification_no_mutation();
    test_action_mismatch_no_mutation();
    test_non_genesis_first_receipt_rejected();
    test_sequence_overflow_no_mutation();

    if (failures != 0) {
        fprintf(
            stderr,
            "receipt_chain_failures=%d\n",
            failures);

        return 1;
    }

    puts("NIYAH_RECEIPT_CHAIN_V1=PASS");
    puts("REJECTED_RECEIPTS_MUTATE_STATE=NO");

    return 0;
}
