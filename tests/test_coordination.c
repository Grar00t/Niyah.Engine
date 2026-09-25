#include "niyah/coordination.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>


static int failures = 0;


#define CHECK_GATE(expr, expected) do { \
    NiyahCoordinationGate actual__ = (expr); \
    if (actual__ != (expected)) { \
        fprintf( \
            stderr, \
            "CHECK failed at %s:%d expected=%s actual=%s\n", \
            __FILE__, \
            __LINE__, \
            niyah_coordination_gate_name((expected)), \
            niyah_coordination_gate_name(actual__)); \
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


static NiyahCoordinationEnvelope base_envelope(void)
{
    NiyahCoordinationEnvelope e;

    memset(&e, 0, sizeof(e));

    e.base_revision = digest(17U);
    e.observed_revision = e.base_revision;

    e.project_id = UINT32_C(0x4e495941);
    e.expected_project_id = e.project_id;

    e.requested_scope =
        NIYAH_SCOPE_ENGINE_CORE |
        NIYAH_SCOPE_CHECKPOINT;

    e.allowed_scope =
        NIYAH_SCOPE_ENGINE_CORE |
        NIYAH_SCOPE_CHECKPOINT;

    e.producer_lobe =
        NIYAH_LOBE_COMPUTE;

    e.verifier_lobe =
        NIYAH_LOBE_VERIFIER;

    e.writer_lobe =
        NIYAH_LOBE_COMPUTE;

    e.proof_present = 1;
    e.contradiction_present = 0;

    return e;
}


static void test_happy_path(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_EXECUTE,
            NIYAH_LOBE_COMPUTE),
        NIYAH_COORDINATION_PASS);

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_VERIFY,
            NIYAH_LOBE_VERIFIER),
        NIYAH_COORDINATION_PASS);

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER),
        NIYAH_COORDINATION_PASS);
}


static void test_stale_base(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    e.observed_revision.bytes[7] ^= UINT8_C(0x80);

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_EXECUTE,
            NIYAH_LOBE_COMPUTE),
        NIYAH_COORDINATION_STALE_BASE);
}


static void test_wrong_scope(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    e.requested_scope |= NIYAH_SCOPE_GRAPH;

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_EXECUTE,
            NIYAH_LOBE_COMPUTE),
        NIYAH_COORDINATION_WRONG_SCOPE);

    e = base_envelope();

    e.expected_project_id =
        UINT32_C(0x47524150);

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_EXECUTE,
            NIYAH_LOBE_COMPUTE),
        NIYAH_COORDINATION_WRONG_SCOPE);
}


static void test_writer_violation(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    e.writer_lobe =
        NIYAH_LOBE_CHRONICLE;

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_EXECUTE,
            NIYAH_LOBE_COMPUTE),
        NIYAH_COORDINATION_WRITER_VIOLATION);
}


static void test_self_verification(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    e.verifier_lobe =
        NIYAH_LOBE_COMPUTE;

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_VERIFY,
            NIYAH_LOBE_COMPUTE),
        NIYAH_COORDINATION_SELF_VERIFICATION);
}


static void test_contradiction(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    e.contradiction_present = 1;

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_VERIFY,
            NIYAH_LOBE_VERIFIER),
        NIYAH_COORDINATION_CONTRADICTION);

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER),
        NIYAH_COORDINATION_CONTRADICTION);
}


static void test_promotion_requires_proof(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    e.proof_present = 0;

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_PROMOTE,
            NIYAH_LOBE_VERIFIER),
        NIYAH_COORDINATION_PROMOTION_WITHOUT_PROOF);
}


static void test_wrong_actor(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_VERIFY,
            NIYAH_LOBE_CHRONICLE),
        NIYAH_COORDINATION_UNAUTHORIZED_ACTOR);
}


static void test_chronicle_can_own_graph_artifact(void)
{
    NiyahCoordinationEnvelope e =
        base_envelope();

    e.requested_scope =
        NIYAH_SCOPE_GRAPH |
        NIYAH_SCOPE_EVIDENCE;

    e.allowed_scope =
        e.requested_scope;

    e.producer_lobe =
        NIYAH_LOBE_CHRONICLE;

    e.writer_lobe =
        NIYAH_LOBE_CHRONICLE;

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_EXECUTE,
            NIYAH_LOBE_CHRONICLE),
        NIYAH_COORDINATION_PASS);

    CHECK_GATE(
        niyah_coordination_gate(
            &e,
            NIYAH_COORDINATION_VERIFY,
            NIYAH_LOBE_VERIFIER),
        NIYAH_COORDINATION_PASS);
}


int main(void)
{
    test_happy_path();
    test_stale_base();
    test_wrong_scope();
    test_writer_violation();
    test_self_verification();
    test_contradiction();
    test_promotion_requires_proof();
    test_wrong_actor();
    test_chronicle_can_own_graph_artifact();

    if (failures != 0) {
        fprintf(
            stderr,
            "coordination_failures=%d\n",
            failures);

        return 1;
    }

    puts("NIYAH_TRI_LOBE_COORDINATION=PASS");
    puts("STALE_BASE=REJECTED");
    puts("WRONG_SCOPE=REJECTED");
    puts("WRITER_VIOLATION=REJECTED");
    puts("SELF_VERIFICATION=REJECTED");
    puts("CONTRADICTION=REJECTED");
    puts("PROMOTION_WITHOUT_PROOF=REJECTED");

    return 0;
}
