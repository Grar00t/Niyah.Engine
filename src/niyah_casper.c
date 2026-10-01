#include "niyah/casper.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define NIYAH_CASPER_MAGIC UINT32_C(0x4E595148)
#define NIYAH_CASPER_VERSION UINT32_C(0x0005)
#define NIYAH_CASPER_HEADER_BYTES 64U

static uint32_t niyah_casper_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) |
           ((uint32_t)p[2] << 16U) | ((uint32_t)p[3] << 24U);
}
static float niyah_casper_f32(const unsigned char *p)
{
    const uint32_t bits = niyah_casper_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int niyah_casper_host_is_little_endian(void)
{
    const uint16_t one = UINT16_C(1);
    return *((const unsigned char *)&one) == 1U;
}

static int niyah_casper_model_empty(const NiyahModel *model)
{
    return model != NULL && model->weights == NULL &&
           model->weight_count == 0U && model->mapping_base == NULL &&
           model->mapping_bytes == 0U;
}

#if defined(_WIN32)
static void *niyah_casper_map_file(const char *path, size_t *out_bytes)
{
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = NULL;
    LARGE_INTEGER size;
    void *view = NULL;
    file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return NULL;
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0 ||
        (uint64_t)size.QuadPart > (uint64_t)SIZE_MAX) {
        CloseHandle(file);
        return NULL;
    }
    mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0U, 0U, NULL);
    if (mapping != NULL) {
        view = MapViewOfFile(mapping, FILE_MAP_READ, 0U, 0U, 0U);
        CloseHandle(mapping);
    }
    CloseHandle(file);
    if (view != NULL) *out_bytes = (size_t)size.QuadPart;
    return view;
}
#else
static void *niyah_casper_map_file(const char *path, size_t *out_bytes)
{
    int fd;
    struct stat st;
    void *view;
    fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    if (fstat(fd, &st) != 0 || st.st_size <= 0 ||
        (uintmax_t)st.st_size > (uintmax_t)SIZE_MAX) {
        close(fd);
        return NULL;
    }
    view = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (view == MAP_FAILED) return NULL;
    *out_bytes = (size_t)st.st_size;
    return view;
}
#endif

static void niyah_casper_unmap(void *base, size_t bytes)
{
    if (base == NULL) return;
#if defined(_WIN32)
    (void)bytes;
    (void)UnmapViewOfFile(base);
#else
    if (bytes != 0U) (void)munmap(base, bytes);
#endif
}

NiyahStatus niyah_casper_mmap_load(const char *path, NiyahModel *out_model)
{
    unsigned char *base;
    size_t bytes = 0U;
    NiyahModelConfig config;
    NiyahModelLayout layout;
    uint32_t ffn_mult;
    uint64_t expected_bytes;
    NiyahStatus status;

    if (path == NULL || path[0] == '\0' || out_model == NULL ||
        !niyah_casper_model_empty(out_model) ||
        !niyah_casper_host_is_little_endian()) {
        return NIYAH_ERR_INVALID_ARGUMENT;
    }

    base = (unsigned char *)niyah_casper_map_file(path, &bytes);
    if (base == NULL) return NIYAH_ERR_IO;
    if (bytes < NIYAH_CASPER_HEADER_BYTES ||
        niyah_casper_u32(base + 0U) != NIYAH_CASPER_MAGIC ||
        niyah_casper_u32(base + 4U) != NIYAH_CASPER_VERSION) {
        niyah_casper_unmap(base, bytes);
        return NIYAH_ERR_UNSUPPORTED_VERSION;
    }

    memset(&config, 0, sizeof(config));
    config.embedding_dim = niyah_casper_u32(base + 8U);
    config.n_heads = niyah_casper_u32(base + 12U);
    config.n_kv_heads = niyah_casper_u32(base + 16U);
    config.n_layers = niyah_casper_u32(base + 20U);
    ffn_mult = niyah_casper_u32(base + 24U);
    config.vocab_size = niyah_casper_u32(base + 28U);
    config.context_length = niyah_casper_u32(base + 32U);
    config.rms_norm_eps = niyah_casper_f32(base + 40U);
    config.tie_word_embeddings = 0;
    config.n_segments = 0U;

    if (ffn_mult == 0U ||
        (uint64_t)config.embedding_dim * (uint64_t)ffn_mult > UINT32_MAX ||
        niyah_casper_f32(base + 36U) != 10000.0f ||
        niyah_casper_u32(base + 44U) != 0U) {
        niyah_casper_unmap(base, bytes);
        return NIYAH_ERR_INVALID_CONFIG;
    }
    config.ffn_hidden_dim = config.embedding_dim * ffn_mult;

    status = niyah_model_layout_compute(&config, &layout);
    if (status != NIYAH_OK) {
        niyah_casper_unmap(base, bytes);
        return status;
    }
    expected_bytes = (uint64_t)NIYAH_CASPER_HEADER_BYTES +
                     (uint64_t)layout.total_floats * UINT64_C(4);
    if (expected_bytes != (uint64_t)bytes) {
        niyah_casper_unmap(base, bytes);
        return NIYAH_ERR_CORRUPT_DATA;
    }

    out_model->config = config;
    out_model->layout = layout;
    out_model->weights = (float *)(void *)(base + NIYAH_CASPER_HEADER_BYTES);
    out_model->weight_count = layout.total_floats;
    out_model->storage_kind = NIYAH_MODEL_STORAGE_CASPER_MMAP;
    out_model->mapping_base = base;
    out_model->mapping_bytes = bytes;
    return NIYAH_OK;
}
