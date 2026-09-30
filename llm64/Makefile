NASM ?= nasm
LD   ?= ld
PYTHON ?= python3

TARGET := niyah-asm
ASM := $(wildcard src/*.asm)
OBJ := $(ASM:src/%.asm=build/%.o)

.PHONY: all clean check test-formats test-tokenizer-runtime test-math-runtime test-attention-runtime test-model-layout

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

test-math-runtime: build/math_scalar.o
	$(LD) -shared -o build/libmath_runtime.so build/math_scalar.o -z noexecstack
	$(PYTHON) tests/test_math_runtime.py build/libmath_runtime.so

test-attention-runtime: build/attention.o build/math_scalar.o
	$(LD) -shared -o build/libattention_runtime.so build/attention.o build/math_scalar.o -z noexecstack
	$(PYTHON) tests/test_attention_runtime.py build/libattention_runtime.so

test-model-layout: build/model_layout.o
	$(LD) -shared -o build/libmodel_layout.so build/model_layout.o -z noexecstack
	$(PYTHON) tests/test_model_layout.py build/libmodel_layout.so

clean:
	rm -rf build $(TARGET)
