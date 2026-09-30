BITS 64
DEFAULT REL

global crc32_ieee

section .text

; rdi = bytes, rsi = length -> eax = IEEE CRC32
crc32_ieee:
    mov     eax, 0xffffffff
.byte_loop:
    test    rsi, rsi
    jz      .done
    movzx   edx, byte [rdi]
    xor     eax, edx
    mov     ecx, 8
.bit_loop:
    mov     edx, eax
    and     edx, 1
    shr     eax, 1
    test    edx, edx
    jz      .no_poly
    xor     eax, 0xedb88320
.no_poly:
    dec     ecx
    jnz     .bit_loop
    inc     rdi
    dec     rsi
    jmp     .byte_loop
.done:
    xor     eax, 0xffffffff
    ret
