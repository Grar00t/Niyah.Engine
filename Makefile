# ╔══════════════════════════════════════════════════════════════════╗
# ║  NIYAH-KERNEL  Makefile                                          ║
# ╚══════════════════════════════════════════════════════════════════╝
#
#  make              → auto-detect platform, release build
#  make debug        → AddressSanitizer + UBSan
#  make test         → build + run SIMD unit tests
#  make clean        → remove build artefacts

CC       ?= gcc
CFLAGS    = -std=c11 -Wall -Wextra -Wcast-align -Wstrict-prototypes
LDFLAGS   = -lm

SRC_DIR   = niyah/src
INC_DIR   = niyah/include

SIMD_SRC  = $(SRC_DIR)/niyah_simd.c
GGUF_SRC  = $(SRC_DIR)/niyah_gguf.c
TEST_SRC  = $(SRC_DIR)/test_simd.c

SIMD_OBJ  = niyah_simd.o
GGUF_OBJ  = niyah_gguf.o
TEST_BIN  = niyah_test_simd

# ── Platform detection ────────────────────────────────────────────
UNAME := $(shell uname -m 2>/dev/null || echo unknown)

ifeq ($(UNAME),aarch64)
  OPT  = -O3 -march=armv8.2-a+simd
  INFO = ARM64 (aarch64)
else ifeq ($(UNAME),arm64)
  OPT  = -O3 -march=armv8.2-a
  INFO = ARM64 (Apple)
else
  OPT  = -O3 -march=native -mavx2 -mfma
  INFO = x86_64
endif

# ── Default target ────────────────────────────────────────────────
.PHONY: all debug test clean

all: $(SIMD_OBJ) $(GGUF_OBJ)
	@echo "[niyah] Built for $(INFO)"

$(SIMD_OBJ): $(SIMD_SRC)
	$(CC) $(CFLAGS) $(OPT) -I$(INC_DIR) -c $< -o $@

$(GGUF_OBJ): $(GGUF_SRC)
	$(CC) $(CFLAGS) $(OPT) -I$(INC_DIR) -c $< -o $@

# ── Debug build ───────────────────────────────────────────────────
debug:
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined \
	    -I$(INC_DIR) -c $(SIMD_SRC) -o niyah_simd_dbg.o $(LDFLAGS)
	@echo "[niyah] Debug build complete"

# ── Unit tests ────────────────────────────────────────────────────
test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(SIMD_OBJ)
	$(CC) $(CFLAGS) $(OPT) -I$(INC_DIR) $^ -o $@ $(LDFLAGS)

# ── Clean ─────────────────────────────────────────────────────────
clean:
	rm -f $(SIMD_OBJ) $(GGUF_OBJ) niyah_simd_dbg.o $(TEST_BIN)
