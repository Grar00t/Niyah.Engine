NASM ?= nasm
LD   ?= ld
PYTHON ?= python3

TARGET := niyah-asm
ASM := $(wildcard src/*.asm)
OBJ := $(ASM:src/%.asm=build/%.o)

# NASM 3.x added the reloc-rel-dword warning class. Older NASM releases
# reject the class name itself, so enable the exception only on major >= 3.
NASM_RELOC_WARNING_FLAG := $(shell major=$$($(NASM) -v 2>/dev/null | sed -n 's/^NASM version \([0-9][0-9]*\).*/\1/p'); if [ -n "$$major" ] && [ "$$major" -ge 3 ]; then printf '%s' '-Wno-error=reloc-rel-dword'; fi)

.PHONY: all clean check test-formats test-tokenizer-runtime test-math-runtime test-attention-runtime test-model-layout test-runtime-memory test-decode-runtime test-generate-cli

all: $(TARGET)

$(TARGET): $(OBJ)
	$(LD) -o $@ $(OBJ) -z noexecstack

build/%.o: src/%.asm
	@mkdir -p build
	$(NASM) -f elf64 -Wall -Werror $(NASM_RELOC_WARNING_FLAG) -Iinclude/ $< -o $@

check: $(TARGET)
	@set +e; ./$(TARGET) >/dev/null 2>&1; rc=$$?; set -e; test $$rc -eq 64
	@! readelf -d ./$(TARGET) 2>/dev/null | grep -q '(NEEDED)'

test-formats: $(TARGET)
	$(PYTHON) tests/test_formats.py ./$(TARGET)

test-tokenizer-runtime: build/tokenizer_runtime.o
	$(LD) -shared -o build/libtokenizer_runtime.so build/tokenizer_runtime.o -z noexecstack
	$(PYTHON) tests/test_tokenizer_runtime.py build/libtokenizer_runtime.so

test-math-runtime: build/math_scalar.o
	$(LD) -shared -o build/libmath_runtime.so build/math_scalar.o -z noexecstack
	$(PYTHON) tests/test_math_runtime.py build/libmath_runtime.so

test-attention-runtime: build/attention.o build/math_scalar.o
	$(LD) -shared -o build/libattention_runtime.so build/attention.o build/math_scalar.o -z noexecstack
	$(PYTHON) tests/test_attention_runtime.py build/libattention_runtime.so

test-model-layout: build/model_layout.o
	$(LD) -shared -o build/libmodel_layout.so build/model_layout.o -z noexecstack
	$(PYTHON) tests/test_model_layout.py build/libmodel_layout.so

test-runtime-memory: build/runtime_memory.o build/model_layout.o build/syscalls.o
	$(LD) -shared -o build/libruntime_memory.so build/runtime_memory.o build/model_layout.o build/syscalls.o -z noexecstack
	$(PYTHON) tests/test_runtime_memory.py build/libruntime_memory.so

test-decode-runtime: build/decode.o build/attention.o build/math_scalar.o build/model_layout.o build/runtime_memory.o build/syscalls.o
	$(LD) -shared -o build/libdecode_runtime.so build/decode.o build/attention.o build/math_scalar.o build/model_layout.o build/runtime_memory.o build/syscalls.o -z noexecstack
	$(PYTHON) tests/test_decode_runtime.py build/libdecode_runtime.so

test-generate-cli: $(TARGET)
	$(PYTHON) tests/test_generate_cli.py ./$(TARGET)

clean:
	rm -rf build $(TARGET)
