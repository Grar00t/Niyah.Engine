#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "niyah_control.h"

static NiyahActionRequest request(uint64_t nonce, uint32_t capability, const char* resource_id)
{
    NiyahActionRequest value;
    memset(&value, 0, sizeof(value));
    value.nonce = nonce;
    value.capability = capability;
    if (resource_id) {
        const size_t len = strlen(resource_id);
        assert(len < sizeof(value.resource_id));
        memcpy(value.resource_id, resource_id, len + 1u);
    }
    return value;
}

static void test_default_deny(void)
{
    NiyahControlPolicy policy;
    niyah_control_policy_init(&policy);

    const NiyahActionRequest req = request(1u, NIYAH_CAP_READ_LOCAL, "doc:alpha");
    const NiyahControlDecision decision = niyah_control_authorize(&policy, &req);

    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_NOT_GRANTED);
    assert(decision.matched_grant == SIZE_MAX);
}

static void test_exact_grant_only(void)
{
    NiyahControlPolicy policy;
    niyah_control_policy_init(&policy);
    assert(niyah_control_policy_add_grant(
        &policy, NIYAH_CAP_READ_LOCAL, "doc:alpha") == NIYAH_OK);

    NiyahActionRequest req = request(7u, NIYAH_CAP_READ_LOCAL, "doc:alpha");
    NiyahControlDecision decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_ALLOW);
    assert(decision.reason == NIYAH_CONTROL_REASON_ALLOWED);
    assert(decision.matched_grant == 0u);

    req = request(8u, NIYAH_CAP_WRITE_LOCAL, "doc:alpha");
    decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_NOT_GRANTED);

    req = request(9u, NIYAH_CAP_READ_LOCAL, "doc:alpha.child");
    decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_NOT_GRANTED);
}

static void test_invalid_requests_fail_closed(void)
{
    NiyahControlPolicy policy;
    niyah_control_policy_init(&policy);
    assert(niyah_control_policy_add_grant(
        &policy, NIYAH_CAP_PROCESS, "tool:compiler") == NIYAH_OK);

    NiyahActionRequest req = request(0u, NIYAH_CAP_PROCESS, "tool:compiler");
    NiyahControlDecision decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_INVALID_REQUEST);

    req = request(1u, NIYAH_CAP_PROCESS | NIYAH_CAP_NETWORK, "tool:compiler");
    decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_INVALID_CAPABILITY);

    req = request(1u, 1u << 31, "tool:compiler");
    decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_INVALID_CAPABILITY);

    req = request(1u, NIYAH_CAP_PROCESS, "/bin/sh");
    decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_INVALID_RESOURCE);

    req = request(1u, NIYAH_CAP_NETWORK, "https://example.com");
    decision = niyah_control_authorize(&policy, &req);
    assert(decision.verdict == NIYAH_CONTROL_DENY);
    assert(decision.reason == NIYAH_CONTROL_REASON_INVALID_RESOURCE);
}

static void test_duplicate_grant_is_idempotent(void)
{
    NiyahControlPolicy policy;
    niyah_control_policy_init(&policy);

    assert(niyah_control_policy_add_grant(
        &policy, NIYAH_CAP_READ_LOCAL, "doc:alpha") == NIYAH_OK);
    assert(niyah_control_policy_add_grant(
        &policy, NIYAH_CAP_READ_LOCAL, "doc:alpha") == NIYAH_OK);
    assert(policy.grant_count == 1u);
}

static void test_policy_capacity_is_bounded(void)
{
    NiyahControlPolicy policy;
    niyah_control_policy_init(&policy);

    char resource[NIYAH_CONTROL_RESOURCE_MAX];
    for (size_t i = 0u; i < NIYAH_CONTROL_MAX_GRANTS; ++i) {
        const int written = snprintf(resource, sizeof(resource), "doc:item%zu", i);
        assert(written > 0 && (size_t)written < sizeof(resource));
        assert(niyah_control_policy_add_grant(
            &policy, NIYAH_CAP_READ_LOCAL, resource) == NIYAH_OK);
    }

    assert(niyah_control_policy_add_grant(
        &policy, NIYAH_CAP_READ_LOCAL, "doc:overflow") == NIYAH_ERR_OVERFLOW);
}

int main(void)
{
    test_default_deny();
    test_exact_grant_only();
    test_invalid_requests_fail_closed();
    test_duplicate_grant_is_idempotent();
    test_policy_capacity_is_bounded();

    assert(strcmp(niyah_control_reason_to_string(NIYAH_CONTROL_REASON_ALLOWED), "allowed") == 0);
    assert(strcmp(niyah_control_reason_to_string(NIYAH_CONTROL_REASON_NOT_GRANTED), "not_granted") == 0);
    return 0;
}
