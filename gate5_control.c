#include <stdio.h>
#include <locale.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "native/niyah_control.h"

/* Gate 5: locale-sensitivity, nonce replay, boundary resource IDs */

static NiyahActionRequest make_req(uint64_t nonce, uint32_t cap, const char* rid) {
    NiyahActionRequest r;
    memset(&r, 0, sizeof(r));
    r.nonce = nonce;
    r.capability = cap;
    if (rid) {
        size_t n = strlen(rid);
        if (n < sizeof(r.resource_id)) memcpy(r.resource_id, rid, n+1);
    }
    return r;
}

int main(void) {
    /* --- Test 1: locale-dependent isalnum on bytes 0x80-0xFF --- */
    setlocale(LC_ALL, "en_US.UTF-8");
    printf("locale: %s\n", setlocale(LC_ALL, NULL));

    NiyahControlPolicy pol;
    niyah_control_policy_init(&pol);
    /* Grant a resource that starts with valid ASCII */
    niyah_control_policy_add_grant(&pol, NIYAH_CAP_READ_LOCAL, "doc:base");

    int accepted_high = 0;
    /* Try resource IDs containing individual high bytes */
    for (int b = 0x80; b <= 0xFF; b++) {
        char rid[8] = {'d','o','c',':','x', (char)b, '\0'};
        NiyahActionRequest req = make_req(1, NIYAH_CAP_READ_LOCAL, rid);
        /* Try to ADD such a grant */
        NiyahStatus s = niyah_control_policy_add_grant(&pol, NIYAH_CAP_READ_LOCAL, rid);
        if (s == NIYAH_OK) {
            accepted_high++;
            printf("  ACCEPTED_HIGH_BYTE 0x%02x: add_grant returned NIYAH_OK\n", b);
        }
    }
    printf("ACCEPTED_HIGH_BYTES=%d\n\n", accepted_high);

    /* --- Test 2: nonce replay --- */
    NiyahControlPolicy pol2;
    niyah_control_policy_init(&pol2);
    niyah_control_policy_add_grant(&pol2, NIYAH_CAP_READ_LOCAL, "doc:alpha");

    NiyahActionRequest req1 = make_req(42, NIYAH_CAP_READ_LOCAL, "doc:alpha");
    NiyahControlDecision d1 = niyah_control_authorize(&pol2, &req1);
    NiyahControlDecision d2 = niyah_control_authorize(&pol2, &req1); /* IDENTICAL */

    printf("NONCE_REPLAY_TEST:\n");
    printf("  DECISION_1=%s (reason=%s)\n",
        d1.verdict == NIYAH_CONTROL_ALLOW ? "ALLOW" : "DENY",
        niyah_control_reason_to_string(d1.reason));
    printf("  DECISION_2=%s (reason=%s)\n",
        d2.verdict == NIYAH_CONTROL_ALLOW ? "ALLOW" : "DENY",
        niyah_control_reason_to_string(d2.reason));
    if (d1.verdict == NIYAH_CONTROL_ALLOW && d2.verdict == NIYAH_CONTROL_ALLOW) {
        printf("  FINDING: nonce is decorative; identical (nonce=42,cap,resource) allowed twice\n");
    }
    printf("\n");

    /* --- Test 3: boundary resource IDs around NIYAH_CONTROL_RESOURCE_MAX --- */
    /* NIYAH_CONTROL_RESOURCE_MAX = 96 */
    /* resource_id_valid: ++length >= NIYAH_CONTROL_RESOURCE_MAX returns 0 */
    /* So max valid length is NIYAH_CONTROL_RESOURCE_MAX - 1 = 95 chars */
    printf("BOUNDARY_TEST (NIYAH_CONTROL_RESOURCE_MAX=%d):\n", NIYAH_CONTROL_RESOURCE_MAX);

    NiyahControlPolicy pol3;
    niyah_control_policy_init(&pol3);

    /* MAX-2 = 94 chars */
    char rid94[256];
    memset(rid94, 'a', 94);
    rid94[94] = '\0';
    NiyahStatus s94 = niyah_control_policy_add_grant(&pol3, NIYAH_CAP_READ_LOCAL, rid94);
    printf("  len=94 (MAX-2): %s\n", s94 == NIYAH_OK ? "ACCEPTED" : "REJECTED");

    /* MAX-1 = 95 chars */
    char rid95[256];
    memset(rid95, 'a', 95);
    rid95[95] = '\0';
    NiyahStatus s95 = niyah_control_policy_add_grant(&pol3, NIYAH_CAP_READ_LOCAL, rid95);
    printf("  len=95 (MAX-1): %s\n", s95 == NIYAH_OK ? "ACCEPTED" : "REJECTED");

    /* MAX = 96 chars */
    char rid96[256];
    memset(rid96, 'a', 96);
    rid96[96] = '\0';
    NiyahStatus s96 = niyah_control_policy_add_grant(&pol3, NIYAH_CAP_READ_LOCAL, rid96);
    printf("  len=96 (MAX):   %s\n", s96 == NIYAH_OK ? "ACCEPTED" : "REJECTED");

    return 0;
}
