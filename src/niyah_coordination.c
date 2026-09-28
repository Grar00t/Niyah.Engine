#include "niyah/coordination.h"

#include <stddef.h>


static int digest_is_zero(
    const NiyahCoordinationDigest *digest)
{
    size_t i;

    if (digest == NULL)
        return 1;

    for (i = 0U;
         i < NIYAH_COORDINATION_DIGEST_BYTES;
         ++i) {
        if (digest->bytes[i] != 0U)
            return 0;
    }

    return 1;
}


static int valid_lobe(
    NiyahCoordinationLobe lobe)
{
    return
        lobe == NIYAH_LOBE_COMPUTE ||
        lobe == NIYAH_LOBE_CHRONICLE ||
        lobe == NIYAH_LOBE_VERIFIER;
}


int niyah_coordination_digest_equal(
    const NiyahCoordinationDigest *a,
    const NiyahCoordinationDigest *b)
{
    uint8_t diff = 0U;
    size_t i;

    if (a == NULL || b == NULL)
        return 0;

    for (i = 0U;
         i < NIYAH_COORDINATION_DIGEST_BYTES;
         ++i) {
        diff |=
            (uint8_t)(
                a->bytes[i] ^
                b->bytes[i]);
    }

    return diff == 0U;
}


NiyahCoordinationGate niyah_coordination_gate(
    const NiyahCoordinationEnvelope *envelope,
    NiyahCoordinationAction action,
    NiyahCoordinationLobe actor)
{
    if (envelope == NULL)
        return NIYAH_COORDINATION_INVALID_ENVELOPE;

    if (digest_is_zero(&envelope->base_revision) ||
        digest_is_zero(&envelope->observed_revision) ||
        envelope->project_id == 0U ||
        envelope->expected_project_id == 0U ||
        envelope->requested_scope == 0U ||
        envelope->allowed_scope == 0U ||
        !valid_lobe(envelope->producer_lobe) ||
        !valid_lobe(envelope->verifier_lobe) ||
        !valid_lobe(envelope->writer_lobe) ||
        !valid_lobe(actor)) {
        return NIYAH_COORDINATION_INVALID_ENVELOPE;
    }

    /*
     * Never continue a task that was authorized against
     * a different source state.
     */
    if (!niyah_coordination_digest_equal(
            &envelope->base_revision,
            &envelope->observed_revision)) {
        return NIYAH_COORDINATION_STALE_BASE;
    }

    /*
     * Prevent Graph/Data/Core context from being silently
     * substituted for another project or capability scope.
     */
    if (envelope->project_id !=
            envelope->expected_project_id ||
        (envelope->requested_scope &
         ~envelope->allowed_scope) != 0U) {
        return NIYAH_COORDINATION_WRONG_SCOPE;
    }

    switch (action) {
        case NIYAH_COORDINATION_EXECUTE:
            /*
             * ONE ARTIFACT = ONE WRITER.
             */
            if (actor != envelope->producer_lobe ||
                actor != envelope->writer_lobe) {
                return NIYAH_COORDINATION_WRITER_VIOLATION;
            }

            return NIYAH_COORDINATION_PASS;

        case NIYAH_COORDINATION_VERIFY:
            if (actor != envelope->verifier_lobe)
                return NIYAH_COORDINATION_UNAUTHORIZED_ACTOR;

            if (envelope->verifier_lobe ==
                    envelope->producer_lobe) {
                return NIYAH_COORDINATION_SELF_VERIFICATION;
            }

            if (envelope->contradiction_present)
                return NIYAH_COORDINATION_CONTRADICTION;

            return NIYAH_COORDINATION_PASS;

        case NIYAH_COORDINATION_PROMOTE:
            if (actor != envelope->verifier_lobe)
                return NIYAH_COORDINATION_UNAUTHORIZED_ACTOR;

            if (envelope->verifier_lobe ==
                    envelope->producer_lobe) {
                return NIYAH_COORDINATION_SELF_VERIFICATION;
            }

            if (envelope->contradiction_present)
                return NIYAH_COORDINATION_CONTRADICTION;

            if (!envelope->proof_present)
                return
                    NIYAH_COORDINATION_PROMOTION_WITHOUT_PROOF;

            return NIYAH_COORDINATION_PASS;

        default:
            return NIYAH_COORDINATION_INVALID_ENVELOPE;
    }
}


const char *niyah_coordination_gate_name(
    NiyahCoordinationGate gate)
{
    switch (gate) {
        case NIYAH_COORDINATION_PASS:
            return "PASS";

        case NIYAH_COORDINATION_INVALID_ENVELOPE:
            return "INVALID_ENVELOPE";

        case NIYAH_COORDINATION_STALE_BASE:
            return "STALE_BASE";

        case NIYAH_COORDINATION_WRONG_SCOPE:
            return "WRONG_SCOPE";

        case NIYAH_COORDINATION_WRITER_VIOLATION:
            return "WRITER_VIOLATION";

        case NIYAH_COORDINATION_SELF_VERIFICATION:
            return "SELF_VERIFICATION";

        case NIYAH_COORDINATION_CONTRADICTION:
            return "CONTRADICTION";

        case NIYAH_COORDINATION_PROMOTION_WITHOUT_PROOF:
            return "PROMOTION_WITHOUT_PROOF";

        case NIYAH_COORDINATION_UNAUTHORIZED_ACTOR:
            return "UNAUTHORIZED_ACTOR";

        default:
            return "UNKNOWN";
    }
}
