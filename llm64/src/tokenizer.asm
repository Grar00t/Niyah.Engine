BITS 64
DEFAULT REL

%include "model.inc"

extern sys_open_ro
extern sys_close
extern sys_fstat_size
extern sys_mmap_ro
extern sys_munmap
extern crc32_ieee

global tokenizer_validate_v1

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
