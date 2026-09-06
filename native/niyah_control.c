#include "niyah_control.h"

#include <ctype.h>
#include <string.h>

static int capability_is_single_known(uint32_t capability)
{
    const uint32_t known =
        NIYAH_CAP_READ_LOCAL |
        NIYAH_CAP_WRITE_LOCAL |
        NIYAH_CAP_NETWORK |
        NIYAH_CAP_PROCESS;

    if (capability == 0u || (capability & ~known) != 0u) {
        return 0;
    }

    return (capability & (capability - 1u)) == 0u;
}

static int resource_id_valid(const char* resource_id)
{
    if (!resource_id || !resource_id[0]) {
        return 0;
    }

    size_t length = 0u;
    for (const unsigned char* p = (const unsigned char*)resource_id; *p; ++p) {
        if (++length >= NIYAH_CONTROL_RESOURCE_MAX) {
            return 0;
        }

        if (!(isalnum(*p) || *p == '_' || *p == '-' || *p == '.' || *p == ':')) {
            return 0;
        }
    }

    return 1;
}

void niyah_control_policy_init(NiyahControlPolicy* policy)
{
    if (!policy) {
        return;
    }

    memset(policy, 0, sizeof(*policy));
}

NiyahStatus niyah_control_policy_add_grant(
    NiyahControlPolicy* policy,
    uint32_t capability,
    const char* resource_id)
{
    if (!policy ||
        !capability_is_single_known(capability) ||
        !resource_id_valid(resource_id)) {
        return NIYAH_ERR_INVALID_ARG;
    }

    for (size_t i = 0u; i < policy->grant_count; ++i) {
        const NiyahCapabilityGrant* grant = &policy->grants[i];
        if (grant->capability == capability &&
            strcmp(grant->resource_id, resource_id) == 0) {
            return NIYAH_OK;
        }
    }

    if (policy->grant_count >= NIYAH_CONTROL_MAX_GRANTS) {
        return NIYAH_ERR_OVERFLOW;
    }

    NiyahCapabilityGrant* grant = &policy->grants[policy->grant_count];
    grant->capability = capability;
    memcpy(grant->resource_id, resource_id, strlen(resource_id) + 1u);
    ++policy->grant_count;

    return NIYAH_OK;
}

NiyahControlDecision niyah_control_authorize(
    const NiyahControlPolicy* policy,
    const NiyahActionRequest* request)
{
    NiyahControlDecision decision;
    decision.verdict = NIYAH_CONTROL_DENY;
    decision.reason = NIYAH_CONTROL_REASON_INVALID_REQUEST;
    decision.matched_grant = SIZE_MAX;

    if (!policy || !request || request->nonce == 0u) {
        return decision;
    }

    if (!capability_is_single_known(request->capability)) {
        decision.reason = NIYAH_CONTROL_REASON_INVALID_CAPABILITY;
        return decision;
    }

    if (!resource_id_valid(request->resource_id)) {
        decision.reason = NIYAH_CONTROL_REASON_INVALID_RESOURCE;
        return decision;
    }

    for (size_t i = 0u; i < policy->grant_count; ++i) {
        const NiyahCapabilityGrant* grant = &policy->grants[i];
        if (grant->capability == request->capability &&
            strcmp(grant->resource_id, request->resource_id) == 0) {
            decision.verdict = NIYAH_CONTROL_ALLOW;
            decision.reason = NIYAH_CONTROL_REASON_ALLOWED;
            decision.matched_grant = i;
            return decision;
        }
    }

    decision.reason = NIYAH_CONTROL_REASON_NOT_GRANTED;
    return decision;
}

const char* niyah_control_reason_to_string(NiyahControlReason reason)
{
    switch (reason) {
        case NIYAH_CONTROL_REASON_ALLOWED:
            return "allowed";
        case NIYAH_CONTROL_REASON_INVALID_REQUEST:
            return "invalid_request";
        case NIYAH_CONTROL_REASON_INVALID_CAPABILITY:
            return "invalid_capability";
        case NIYAH_CONTROL_REASON_INVALID_RESOURCE:
            return "invalid_resource";
        case NIYAH_CONTROL_REASON_NOT_GRANTED:
            return "not_granted";
        default:
            return "unknown";
    }
}
