#ifndef NIYAH_CONTROL_H
#define NIYAH_CONTROL_H

#include "niyah.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NIYAH_CONTROL_MAX_GRANTS 32u
#define NIYAH_CONTROL_RESOURCE_MAX 96u

typedef enum {
    NIYAH_CAP_NONE        = 0u,
    NIYAH_CAP_READ_LOCAL  = 1u << 0,
    NIYAH_CAP_WRITE_LOCAL = 1u << 1,
    NIYAH_CAP_NETWORK     = 1u << 2,
    NIYAH_CAP_PROCESS     = 1u << 3
} NiyahCapability;

typedef struct {
    uint32_t capability;
    char resource_id[NIYAH_CONTROL_RESOURCE_MAX];
} NiyahCapabilityGrant;

typedef struct {
    NiyahCapabilityGrant grants[NIYAH_CONTROL_MAX_GRANTS];
    size_t grant_count;
} NiyahControlPolicy;

typedef struct {
    uint64_t nonce;
    uint32_t capability;
    char resource_id[NIYAH_CONTROL_RESOURCE_MAX];
    uint8_t payload_sha256[32];
} NiyahActionRequest;

typedef enum {
    NIYAH_CONTROL_DENY = 0,
    NIYAH_CONTROL_ALLOW = 1
} NiyahControlVerdict;

typedef enum {
    NIYAH_CONTROL_REASON_ALLOWED = 0,
    NIYAH_CONTROL_REASON_INVALID_REQUEST,
    NIYAH_CONTROL_REASON_INVALID_CAPABILITY,
    NIYAH_CONTROL_REASON_INVALID_RESOURCE,
    NIYAH_CONTROL_REASON_NOT_GRANTED
} NiyahControlReason;

typedef struct {
    NiyahControlVerdict verdict;
    NiyahControlReason reason;
    size_t matched_grant;
} NiyahControlDecision;

NIYAH_API void niyah_control_policy_init(NiyahControlPolicy* policy);
NIYAH_API NiyahStatus niyah_control_policy_add_grant(
    NiyahControlPolicy* policy,
    uint32_t capability,
    const char* resource_id);
NIYAH_API NiyahControlDecision niyah_control_authorize(
    const NiyahControlPolicy* policy,
    const NiyahActionRequest* request);
NIYAH_API const char* niyah_control_reason_to_string(NiyahControlReason reason);

#ifdef __cplusplus
}
#endif

#endif
