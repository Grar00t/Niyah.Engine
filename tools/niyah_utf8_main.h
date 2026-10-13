/* Private CLI entry-point adapter. Define NIYAH_CLI_ENTRY before inclusion. */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <wchar.h>
#include <stdlib.h>
#include <stdio.h>
int wmain(int argc, wchar_t **wide_argv)
{
    char **argv;
    int i, rc = 1;
    if (argc < 0) return 1;
    argv = (char **)calloc((size_t)argc + 1U, sizeof(*argv));
    if (!argv) return 1;
    for (i = 0; i < argc; ++i) {
        int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
            wide_argv[i], -1, NULL, 0, NULL, NULL);
        if (count <= 0) goto done;
        argv[i] = (char *)malloc((size_t)count);
        if (!argv[i] || WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                wide_argv[i], -1, argv[i], count, NULL, NULL) != count) goto done;
    }
    rc = NIYAH_CLI_ENTRY(argc, argv);
done:
    if (i != argc) fprintf(stderr, "argv_utf8_conversion_failed index=%d\n", i);
    for (i = 0; i < argc; ++i) free(argv[i]);
    free(argv);
    return rc;
}
#else
int main(int argc, char **argv) { return NIYAH_CLI_ENTRY(argc, argv); }
#endif
