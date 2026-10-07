BITS 64
DEFAULT REL

%include "model.inc"
%include "tokenizer_map.inc"

extern sys_open_ro
extern sys_close
extern sys_fstat_size
extern sys_mmap_ro
extern sys_munmap
extern crc32_ieee

global tokenizer_validate_v1
global tokenizer_map_v1
global tokenizer_unmap_v1

section .text

; rdi = tok.bin path
; eax = 0 success, 1 format/corruption, 2 I/O
tokenizer_validate_v1:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    call    sys_open_ro
    test    rax, rax
    js      .io_no_fd
    mov     r12, rax

    mov     rdi, r12
    call    sys_fstat_size
    test    rax, rax
    js      .io_close
    mov     r13, rax
    cmp     r13, TOK_HEADER_SIZE + FILE_FOOTER_SIZE
    jb      .format_close

    mov     rdi, r12
    mov     rsi, r13
    call    sys_mmap_ro
    test    rax, rax
    js      .io_close
    mov     r14, rax

    mov     rdi, r12
    call    sys_close

    mov     rax, [r14]
    mov     rcx, 0x4b4f54484159494e      ; "NIYAHTOK" little-endian
    cmp     rax, rcx
    jne     .format_unmap
    cmp     dword [r14 + TOK_VERSION_OFF], TOK_VERSION_V1
    jne     .format_unmap
    cmp     dword [r14 + TOK_FLAGS_OFF], 0
    jne     .format_unmap
    cmp     dword [r14 + TOK_BASE_VOCAB_OFF], TOK_BASE_VOCAB
    jne     .format_unmap
    cmp     dword [r14 + TOK_RESERVED_OFF], 0
    jne     .format_unmap

    mov     eax, [r14 + TOK_VOCAB_OFF]
    cmp     eax, TOK_BASE_VOCAB
    jb      .format_unmap
    mov     edx, eax
    sub     edx, TOK_BASE_VOCAB
    cmp     edx, [r14 + TOK_MERGE_COUNT_OFF]
    jne     .format_unmap

    ; Exact file size = 32-byte header + 12 bytes per merge + 8-byte footer.
    mov     eax, [r14 + TOK_MERGE_COUNT_OFF]
    mov     ecx, TOK_MERGE_SIZE
    mul     rcx
    test    rdx, rdx
    jnz     .format_unmap
    add     rax, TOK_HEADER_SIZE + FILE_FOOTER_SIZE
    jc      .format_unmap
    cmp     rax, r13
    jne     .format_unmap

    lea     r15, [r14 + r13 - FILE_FOOTER_SIZE]
    cmp     dword [r15], CHECKSUM_CRC32
    jne     .format_unmap
    mov     rdi, r14
    mov     rsi, r13
    sub     rsi, FILE_FOOTER_SIZE
    call    crc32_ieee
    cmp     eax, [r15 + 4]
    jne     .format_unmap

    mov     r11d, [r14 + TOK_MERGE_COUNT_OFF]
    xor     r10d, r10d
    lea     r8, [r14 + TOK_HEADER_SIZE]

.merge_loop:
    cmp     r10d, r11d
    jae     .merge_done

    mov     eax, r10d
    add     eax, TOK_BASE_VOCAB

    cmp     dword [r8 + 8], eax
    jne     .format_unmap

    mov     edx, [r8]
    cmp     edx, eax
    jae     .format_unmap
    cmp     edx, TOK_BOS
    je      .format_unmap
    cmp     edx, TOK_EOS
    je      .format_unmap

    mov     edx, [r8 + 4]
    cmp     edx, eax
    jae     .format_unmap
    cmp     edx, TOK_BOS
    je      .format_unmap
    cmp     edx, TOK_EOS
    je      .format_unmap

    add     r8, TOK_MERGE_SIZE
    inc     r10d
    jmp     .merge_loop

.merge_done:
    cmp     r8, r15
    jne     .format_unmap

    mov     rdi, r14
    mov     rsi, r13
    call    sys_munmap
    xor     eax, eax
    jmp     .return

.format_close:
    mov     rdi, r12
    call    sys_close
    mov     eax, 1
    jmp     .return

.format_unmap:
    mov     rdi, r14
    mov     rsi, r13
    call    sys_munmap
    mov     eax, 1
    jmp     .return

.io_close:
    mov     rdi, r12
    call    sys_close
.io_no_fd:
    mov     eax, 2

.return:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret


; =====================================================================
; int tokenizer_map_v1(const char *path, TokenizerMap *out)
;
; Keeps the exact validated tok.bin mapping alive for inference.
;
; eax:
;   0 success
;   1 format / CRC / merge-order failure
;   2 I/O failure
; =====================================================================

tokenizer_map_v1:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    test    rdi, rdi
    jz      .tm_format
    test    rsi, rsi
    jz      .tm_format

    mov     r12, rdi
    mov     r13, rsi

    xor     eax, eax
    mov     [r13 + TM_MAP_BASE], rax
    mov     [r13 + TM_MAP_BYTES], rax

    mov     rdi, r12
    call    sys_open_ro
    test    rax, rax
    js      .tm_io

    mov     r14, rax

    mov     rdi, r14
    call    sys_fstat_size
    test    rax, rax
    js      .tm_io_close

    mov     r15, rax

    cmp     r15, TOK_HEADER_SIZE + FILE_FOOTER_SIZE
    jb      .tm_format_close

    mov     rdi, r14
    mov     rsi, r15
    call    sys_mmap_ro
    test    rax, rax
    js      .tm_io_close

    mov     rbx, rax

    mov     rdi, r14
    call    sys_close

    ; Header identity.
    mov     rax, [rbx]
    mov     rcx, 0x4b4f54484159494e
    cmp     rax, rcx
    jne     .tm_bad_unmap

    cmp     dword [rbx + TOK_VERSION_OFF], TOK_VERSION_V1
    jne     .tm_bad_unmap

    cmp     dword [rbx + TOK_FLAGS_OFF], 0
    jne     .tm_bad_unmap

    cmp     dword [rbx + TOK_BASE_VOCAB_OFF], TOK_BASE_VOCAB
    jne     .tm_bad_unmap

    cmp     dword [rbx + TOK_RESERVED_OFF], 0
    jne     .tm_bad_unmap

    ; vocab = base + merge_count
    mov     eax, [rbx + TOK_VOCAB_OFF]
    cmp     eax, TOK_BASE_VOCAB
    jb      .tm_bad_unmap

    mov     edx, eax
    sub     edx, TOK_BASE_VOCAB
    cmp     edx, [rbx + TOK_MERGE_COUNT_OFF]
    jne     .tm_bad_unmap

    ; Exact size.
    mov     eax, [rbx + TOK_MERGE_COUNT_OFF]
    mov     ecx, TOK_MERGE_SIZE
    mul     rcx

    test    rdx, rdx
    jnz     .tm_bad_unmap

    add     rax, TOK_HEADER_SIZE + FILE_FOOTER_SIZE
    jc      .tm_bad_unmap

    cmp     rax, r15
    jne     .tm_bad_unmap

    ; CRC footer.
    lea     r12, [rbx + r15 - FILE_FOOTER_SIZE]

    cmp     dword [r12], CHECKSUM_CRC32
    jne     .tm_bad_unmap

    mov     rdi, rbx
    mov     rsi, r15
    sub     rsi, FILE_FOOTER_SIZE
    call    crc32_ieee

    cmp     eax, [r12 + 4]
    jne     .tm_bad_unmap

    ; Ordered merge triples.
    mov     r11d, [rbx + TOK_MERGE_COUNT_OFF]
    xor     r10d, r10d
    lea     r8, [rbx + TOK_HEADER_SIZE]

.tm_merge_loop:
    cmp     r10d, r11d
    jae     .tm_merge_done

    mov     eax, r10d
    add     eax, TOK_BASE_VOCAB

    ; output token must be exactly BASE + merge_index
    cmp     dword [r8 + 8], eax
    jne     .tm_bad_unmap

    ; left must refer to an already-existing non-BOS/EOS token.
    mov     edx, [r8]
    cmp     edx, eax
    jae     .tm_bad_unmap
    cmp     edx, TOK_BOS
    je      .tm_bad_unmap
    cmp     edx, TOK_EOS
    je      .tm_bad_unmap

    ; right same rule.
    mov     edx, [r8 + 4]
    cmp     edx, eax
    jae     .tm_bad_unmap
    cmp     edx, TOK_BOS
    je      .tm_bad_unmap
    cmp     edx, TOK_EOS
    je      .tm_bad_unmap

    add     r8, TOK_MERGE_SIZE
    inc     r10d
    jmp     .tm_merge_loop

.tm_merge_done:
    cmp     r8, r12
    jne     .tm_bad_unmap

    mov     [r13 + TM_MAP_BASE], rbx
    mov     [r13 + TM_MAP_BYTES], r15

    xor     eax, eax
    jmp     .tm_return

.tm_bad_unmap:
    mov     rdi, rbx
    mov     rsi, r15
    call    sys_munmap

.tm_format:
    mov     eax, 1
    jmp     .tm_return

.tm_format_close:
    mov     rdi, r14
    call    sys_close
    mov     eax, 1
    jmp     .tm_return

.tm_io_close:
    mov     rdi, r14
    call    sys_close

.tm_io:
    mov     eax, 2

.tm_return:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret


; =====================================================================
; int tokenizer_unmap_v1(TokenizerMap *map)
; =====================================================================

tokenizer_unmap_v1:
    push    r12

    test    rdi, rdi
    jz      .tu_bad

    mov     r12, rdi

    mov     rax, [r12 + TM_MAP_BASE]
    test    rax, rax
    jz      .tu_clear

    mov     rsi, [r12 + TM_MAP_BYTES]
    test    rsi, rsi
    jz      .tu_bad

    mov     rdi, rax
    call    sys_munmap
    test    rax, rax
    js      .tu_io

.tu_clear:
    xor     eax, eax
    mov     [r12 + TM_MAP_BASE], rax
    mov     [r12 + TM_MAP_BYTES], rax

    pop     r12
    ret

.tu_bad:
    mov     eax, 1
    pop     r12
    ret

.tu_io:
    mov     eax, 2
    pop     r12
    ret
