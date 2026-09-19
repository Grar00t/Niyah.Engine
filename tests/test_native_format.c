#include "niyah/native_execute.h"
#include "niyah/native_format.h"

#include <stdint.h>
#include <string.h>

#define CHECK(x) \
    do { \
        if (!(x)) return __LINE__; \
    } while (0)

static int check_text(
    const char *input,
    const char *expected)
{
    NiyahNativeExecutionResult result;
    char output[256];
    size_t required = 0U;
    size_t written = 0U;
    size_t expected_size = strlen(expected);

    CHECK(
        niyah_native_execute_text(
            input,
            &result) ==
        NIYAH_OK);

    CHECK(
        niyah_native_execution_format(
            &result,
            NULL,
            0U,
            &required) ==
        NIYAH_OK);

    CHECK(required == expected_size);

    CHECK(
        niyah_native_execution_format(
            &result,
            output,
            sizeof(output),
            &written) ==
        NIYAH_OK);

    CHECK(written == expected_size);
    CHECK(strcmp(output, expected) == 0);

    return 0;
}

int main(void)
{
    CHECK(
        check_text(
            "Is 192.168.1.42 inside 192.168.1.0/24?",
            "route=NETWORK_IP_IN_CIDR\n"
            "address=192.168.1.42\n"
            "network=192.168.1.0/24\n"
            "match=true\n") == 0);

    CHECK(
        check_text(
            "Is 192.168.2.42 inside 192.168.1.0/24?",
            "route=NETWORK_IP_IN_CIDR\n"
            "address=192.168.2.42\n"
            "network=192.168.1.0/24\n"
            "match=false\n") == 0);

    CHECK(
        check_text(
            "هل 172.16.5.9 داخل الشبكة 172.16.0.0/16؟",
            "route=NETWORK_IP_IN_CIDR\n"
            "address=172.16.5.9\n"
            "network=172.16.0.0/16\n"
            "match=true\n") == 0);

    CHECK(
        check_text(
            "What is 2+2?",
            "route=NONE\n") == 0);

    /*
     * Exact capacity contract:
     * required length excludes NUL.
     */
    {
        NiyahNativeExecutionResult result;
        size_t required = 0U;
        size_t ignored = 0U;
        char exact_without_nul[11];

        CHECK(
            niyah_native_execute_text(
                "hello",
                &result) ==
            NIYAH_OK);

        CHECK(
            niyah_native_execution_format(
                &result,
                NULL,
                0U,
                &required) ==
            NIYAH_OK);

        CHECK(required == strlen("route=NONE\n"));

        CHECK(
            niyah_native_execution_format(
                &result,
                exact_without_nul,
                required,
                &ignored) ==
            NIYAH_ERR_BUFFER_TOO_SMALL);
    }

    /* Invalid API arguments. */
    {
        NiyahNativeExecutionResult result;
        size_t length = 0U;
        char output[32];

        memset(&result, 0, sizeof(result));

        CHECK(
            niyah_native_execution_format(
                NULL,
                output,
                sizeof(output),
                &length) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_native_execution_format(
                &result,
                output,
                sizeof(output),
                NULL) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        CHECK(
            niyah_native_execution_format(
                &result,
                NULL,
                1U,
                &length) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    /* Reject impossible typed network result. */
    {
        NiyahNativeExecutionResult result;
        size_t length = 0U;

        memset(&result, 0, sizeof(result));

        result.route_kind =
            NIYAH_ROUTE_NETWORK_IP_IN_CIDR;

        result.network_ir.op =
            NIYAH_NETWORK_IR_OP_INVALID;

        CHECK(
            niyah_native_execution_format(
                &result,
                NULL,
                0U,
                &length) ==
            NIYAH_ERR_INVALID_ARGUMENT);

        memset(&result, 0, sizeof(result));

        result.route_kind =
            NIYAH_ROUTE_NETWORK_IP_IN_CIDR;

        result.network_ir.op =
            NIYAH_NETWORK_IR_OP_IP_IN_CIDR;

        result.network_ir.prefix_length = 24U;
        result.network_match = 2;

        CHECK(
            niyah_native_execution_format(
                &result,
                NULL,
                0U,
                &length) ==
            NIYAH_ERR_INVALID_ARGUMENT);
    }

    return 0;
}
