NASM ?= nasm
LD   ?= ld

TARGET := niyah-asm
ASM := $(wildcard src/*.asm)
OBJ := $(ASM:src/%.asm=build/%.o)

.PHONY: all clean check

all: $(TARGET)

$(TARGET): $(OBJ)
	$(LD) -o $@ $(OBJ) -z noexecstack

build/%.o: src/%.asm
	@mkdir -p build
	$(NASM) -f elf64 -Wall -Werror -Iinclude/ $< -o $@

check: $(TARGET)
	@set +e; ./$(TARGET) >/dev/null 2>&1; rc=$$?; set -e; test $$rc -eq 64
	@! readelf -d ./$(TARGET) 2>/dev/null | grep -q '(NEEDED)'

clean:
	rm -rf build $(TARGET)
