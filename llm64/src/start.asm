BITS 64
DEFAULT REL

%include "errors.inc"
%include "model_map.inc"
%include "tokenizer_map.inc"

extern checkpoint_map_model_v2
extern checkpoint_unmap_model
extern tokenizer_map_v1
extern tokenizer_unmap_v1
extern sys_write_all
extern sys_exit

global _start

section .rodata
usage_msg: db "usage: niyah-asm <model.ckpt> <tok.bin>", 10
usage_len equ $ - usage_msg

ckpt_msg: db "niyah-asm: checkpoint V2 rejected", 10
ckpt_len equ $ - ckpt_msg

tok_msg: db "niyah-asm: tokenizer v1 rejected", 10
tok_len equ $ - tok_msg

io_msg: db "niyah-asm: file I/O failed", 10
io_len equ $ - io_msg

ok_msg: db "NIYAH_ASM64_LOADER=OK", 10
ok_len equ $ - ok_msg

section .text

_start:
    mov     rbx, rsp

    cmp     qword [rbx], 3
    jne     .usage

    ; Stack layout:
    ;   rsp+0  ModelMap  (40 bytes)
    ;   rsp+48 TokenizerMap (16 bytes)
    sub     rsp, 64

    mov     r12, rsp
    lea     r14, [rsp + 48]

    xor     eax, eax

    mov     [r12 + MM_MAP_BASE], rax
    mov     [r12 + MM_MAP_BYTES], rax
    mov     [r12 + MM_HEADER], rax
    mov     [r12 + MM_WEIGHTS], rax
    mov     [r12 + MM_WEIGHT_BYTES], rax

    mov     [r14 + TM_MAP_BASE], rax
    mov     [r14 + TM_MAP_BYTES], rax

    ; -------------------------------------------------------------
    ; Persistent checkpoint mapping.
    ; -------------------------------------------------------------

    mov     rdi, [rbx + 16]
    mov     rsi, r12
    call    checkpoint_map_model_v2

    test    eax, eax
    jz      .checkpoint_ok

    mov     r13d, eax

    mov     rdi, r12
    call    checkpoint_unmap_model

    cmp     r13d, 2
    je      .io_fail

    mov     edi, 2
    lea     rsi, [rel ckpt_msg]
    mov     edx, ckpt_len
    call    sys_write_all

    mov     edi, EXIT_FORMAT
    jmp     sys_exit


.checkpoint_ok:
    cmp     qword [r12 + MM_MAP_BASE], 0
    je      .internal_fail

    cmp     qword [r12 + MM_WEIGHTS], 0
    je      .internal_fail

    cmp     qword [r12 + MM_WEIGHT_BYTES], 0
    je      .internal_fail

    ; -------------------------------------------------------------
    ; Persistent tokenizer mapping.
    ; -------------------------------------------------------------

    mov     rdi, [rbx + 24]
    mov     rsi, r14
    call    tokenizer_map_v1

    test    eax, eax
    jz      .tokenizer_ok

    mov     r13d, eax

    mov     rdi, r14
    call    tokenizer_unmap_v1

    mov     rdi, r12
    call    checkpoint_unmap_model

    cmp     r13d, 2
    je      .io_fail

    mov     edi, 2
    lea     rsi, [rel tok_msg]
    mov     edx, tok_len
    call    sys_write_all

    mov     edi, EXIT_FORMAT
    jmp     sys_exit


.tokenizer_ok:
    cmp     qword [r14 + TM_MAP_BASE], 0
    je      .internal_fail

    cmp     qword [r14 + TM_MAP_BYTES], 0
    je      .internal_fail

    ; No generation yet: prove lifecycle then cleanly unmap.
    mov     rdi, r14
    call    tokenizer_unmap_v1
    test    eax, eax
    jnz     .internal_fail

    mov     rdi, r12
    call    checkpoint_unmap_model
    test    eax, eax
    jnz     .internal_fail

    jmp     .success


.internal_fail:
    mov     rdi, r14
    call    tokenizer_unmap_v1

    mov     rdi, r12
    call    checkpoint_unmap_model

    mov     edi, EXIT_INTERNAL
    jmp     sys_exit


.io_fail:
    mov     edi, 2
    lea     rsi, [rel io_msg]
    mov     edx, io_len
    call    sys_write_all

    mov     edi, EXIT_IO
    jmp     sys_exit


.usage:
    mov     edi, 2
    lea     rsi, [rel usage_msg]
    mov     edx, usage_len
    call    sys_write_all

    mov     edi, EXIT_USAGE
    jmp     sys_exit


.success:
    mov     edi, 1
    lea     rsi, [rel ok_msg]
    mov     edx, ok_len
    call    sys_write_all

    mov     edi, EXIT_OK
    jmp     sys_exit
