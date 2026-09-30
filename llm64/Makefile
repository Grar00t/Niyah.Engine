NASM ?= nasm
LD   ?= ld
PYTHON ?= python3

TARGET := niyah-asm
ASM := $(wildcard src/*.asm)
OBJ := $(ASM:src/%.asm=build/%.o)

.PHONY: all clean check test-formats test-tokenizer-runtime

all: $(TARGET)

$(TARGET): $(OBJ)
	$(LD) -o $@ $(OBJ) -z noexecstack

build/%.o: src/%.asm
	@mkdir -p build
	$(NASM) -f elf64 -Wall -Werror -Wno-error=reloc-rel-dword -Iinclude/ $< -o $@

check: $(TARGET)
	@set +e; ./$(TARGET) >/dev/null 2>&1; rc=$$?; set -e; test $$rc -eq 64
	@! readelf -d ./$(TARGET) 2>/dev/null | grep -q '(NEEDED)'

test-formats: $(TARGET)
	$(PYTHON) tests/test_formats.py ./$(TARGET)

test-tokenizer-runtime: build/tokenizer_runtime.o
	$(LD) -shared -o build/libtokenizer_runtime.so build/tokenizer_runtime.o -z noexecstack
	$(PYTHON) tests/test_tokenizer_runtime.py build/libtokenizer_runtime.so

clean:
	rm -rf build $(TARGET)
