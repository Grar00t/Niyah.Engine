#include "../tools/niyah_prompt_contract.h"
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "prompt CHECK line=%d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(void)
{
    /* Independent published SHA256 vectors: empty and ASCII abc. */
    const char *empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    const char *abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    const uint8_t checkpoint[32] = {0U};
    uint8_t wrong[32] = {1U};
    char text[220];
    int legacy = 99;
    CHECK(snprintf(text, sizeof(text), "NIYAH_PROMPT_V2\n%064d\n%s\n%s\n", 0, abc, empty) > 0);
    CHECK(niyah_prompt_contract_validate(text, checkpoint, "abc", "", &legacy) && !legacy);
    CHECK(!niyah_prompt_contract_validate(text, checkpoint, "abd", "", &legacy));
    CHECK(!niyah_prompt_contract_validate(text, checkpoint, "abc", "x", &legacy));
    CHECK(!niyah_prompt_contract_validate(text, wrong, "abc", "", &legacy));
    CHECK(niyah_prompt_contract_validate(text, checkpoint, NULL, "", &legacy));
    CHECK(niyah_prompt_contract_validate(text, checkpoint, "abc", NULL, &legacy));
    CHECK(snprintf(text, sizeof(text), "NIYAH_PROMPT_V2\r\n%064d\r\n%s\r\n%s\r\n", 0, abc, empty) > 0);
    CHECK(niyah_prompt_contract_validate(text, checkpoint, "abc", "", &legacy) && !legacy);
    CHECK(snprintf(text, sizeof(text), "NIYAH_PROMPT_V2\n%064d\n%s\n%s\n", 0, abc, empty) > 0);
    text[16U + 65U] = 'g';
    CHECK(!niyah_prompt_contract_validate(text, checkpoint, NULL, "", &legacy));
    CHECK(!niyah_prompt_contract_validate(NULL, checkpoint, "abc", "", &legacy));
    CHECK(!niyah_prompt_contract_validate("garbage", checkpoint, "abc", "", &legacy));
    CHECK(snprintf(text, sizeof(text), "%064d\r\n", 0) > 0);
    CHECK(niyah_prompt_contract_validate(text, checkpoint, "any legacy text", "", &legacy) && legacy);
    CHECK(!niyah_prompt_contract_validate(text, wrong, "", "", &legacy));
    puts("PROMPT_CONTRACT=PASS");
    return 0;
}
