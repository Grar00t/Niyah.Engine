#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "niyah_bridge.h"

int main(void)
{
    const char *conninfo = getenv("NIYAH_TEST_PG_CONNINFO");
    NiyahBridgeContext *ctx = NULL;
    char *json;

    assert(niyah_bridge_version() != NULL);
    assert(strcmp(niyah_bridge_version(), "0.2.0") == 0);
    assert(niyah_bridge_open(NULL, &ctx) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_bridge_open("", &ctx) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_bridge_open("dbname=niyah", NULL) == NIYAH_ERR_INVALID_ARG);
    assert(niyah_bridge_search_json(NULL, "x", 1) == NULL);
    niyah_bridge_close(NULL);
    niyah_bridge_free_string(NULL);

    if (!conninfo || conninfo[0] == '\0') {
        fprintf(stderr, "SKIP: set NIYAH_TEST_PG_CONNINFO\n");
        return 77;
    }

    assert(niyah_bridge_open(conninfo, &ctx) == NIYAH_OK);
    assert(ctx != NULL);

    json = niyah_bridge_search_json(ctx, "__niyah_no_such_token_9f1c__", 20);
    assert(json != NULL);
    assert(json[0] == '[');
    assert(json[strlen(json) - 1U] == ']');
    niyah_bridge_free_string(json);

    assert(niyah_bridge_search_json(ctx, "", 20) == NULL);
    assert(niyah_bridge_search_json(ctx, "x", 0) == NULL);
    assert(niyah_bridge_search_json(ctx, "x", 201) == NULL);

    niyah_bridge_close(ctx);
    return 0;
}
