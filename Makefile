# ╔══════════════════════════════════════════════════════════════════╗
# ║  NIYAH-CORE  Makefile  ·  KHAWRIZM Sovereign Stack              ║
# ╚══════════════════════════════════════════════════════════════════╝
#
#  make                   → auto-detect platform
#  make arm64             → ARM64 + NEON (Snapdragon / Graviton / RPi)
#  make x86               → x86_64 + AVX2
#  make win               → Windows ARM64 (cross or native)
#  make debug             → with sanitizers
#  make bench             → build + run benchmark
#  make demo              → build + run architecture demo
#  make clean

CC      ?= gcc
CFLAGS   = -std=c99 -Wall -Wextra -Wno-unused-function
LDFLAGS  = -lm
TARGET   = niyah
SRCS     = niyah_core.c niyah_main.c

# ── detect platform ────────────────────────────────────────────────
UNAME   := $(shell uname -m)

ifeq ($(UNAME),aarch64)
  OPT  = -O3 -march=armv8.2-a+fp16+dotprod -mcpu=cortex-x1
  INFO = "ARM64 (aarch64)"
else ifeq ($(UNAME),arm64)
  OPT  = -O3 -march=armv8.2-a -mcpu=apple-m1
  INFO = "ARM64 (Apple)"
else
  OPT  = -O3 -march=native -mavx2 -mfma
  INFO = "x86_64"
endif

# ── default target ─────────────────────────────────────────────────
all:
	@echo "╔══════════════════════════════╗"
	@echo "║  NIYAH-CORE build            ║"
	@echo "║  Platform: $(INFO)  ║"
	@echo "╚══════════════════════════════╝"
	$(CC) $(CFLAGS) $(OPT) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "✓ Built: ./$(TARGET)"

# ── explicit platform targets ──────────────────────────────────────
arm64:
	$(CC) $(CFLAGS) -O3 -march=armv8.2-a+fp16 $(SRCS) -o $(TARGET) $(LDFLAGS)

x86:
	$(CC) $(CFLAGS) -O3 -march=x86-64-v3 -mavx2 -mfma $(SRCS) -o $(TARGET) $(LDFLAGS)

# Windows ARM64 (MSVC cl.exe)
win:
	cl /nologo /O2 /arch:ARM64 /std:c17 \
	   /D_CRT_SECURE_NO_WARNINGS \
	   $(SRCS) /Fe:$(TARGET).exe

# Debug + sanitizers
debug:
	$(CC) $(CFLAGS) -g -O1 -fsanitize=address,undefined \
	    $(SRCS) -o $(TARGET)_dbg $(LDFLAGS)

# ── convenience run targets ────────────────────────────────────────
demo: all
	./$(TARGET) --mode demo --size tiny

bench: all
	./$(TARGET) --mode bench --size tiny --steps 200

save: all
	./$(TARGET) --mode save --model tiny_test.niyah --size tiny
	@echo "Saved: tiny_test.niyah"

run: all
	./$(TARGET) --mode run --model tiny_test.niyah \
	            --prompt "NIYAH" --tokens 100 --temp 0.8

clean:
	rm -f $(TARGET) $(TARGET)_dbg $(TARGET).exe *.niyah *.o

# ── size report ────────────────────────────────────────────────────
size: all
	@wc -l $(SRCS) niyah_core.h
	@size $(TARGET)

.PHONY: all arm64 x86 win debug demo bench save run clean size
