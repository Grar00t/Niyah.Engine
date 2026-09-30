BITS 64
DEFAULT REL

%include "errors.inc"
%include "model_map.inc"
%include "tokenizer_map.inc"
%include "runtime.inc"

extern checkpoint_map_model_v2
extern checkpoint_unmap_model
extern tokenizer_map_v1
extern tokenizer_unmap_v1
extern runtime_memory_create
extern runtime_memory_destroy
extern generate_greedy_stdout
extern sys_write_all
extern sys_exit

global _start

section .rodata
usage_msg: db "usage: niyah-asm <model.ckpt> <tok.bin> [prompt]", 10
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

    mov     rax, [rbx]
    cmp     rax, 3
    je      .argc_ok
    cmp     rax, 4
    jne     .usage

.argc_ok:
    ; Stack layout:
    ;   rsp+0   ModelMap      (40 bytes)
    ;   rsp+48  TokenizerMap  (16 bytes)
    ;   rsp+64  RuntimeMemory (56 bytes)
    sub     rsp, 128

    mov     r12, rsp
    lea     r14, [rsp + 48]
    lea     r15, [rsp + 64]

    xor     eax, eax
    mov     [r12 + MM_MAP_BASE], rax
    mov     [r12 + MM_MAP_BYTES], rax
    mov     [r12 + MM_HEADER], rax
    mov     [r12 + MM_WEIGHTS], rax
    mov     [r12 + MM_WEIGHT_BYTES], rax
    mov     [r14 + TM_MAP_BASE], rax
    mov     [r14 + TM_MAP_BYTES], rax

    ; Persistent checkpoint mapping.
    mov     rdi, [rbx + 16]
    mov     rsi, r12
    call    checkpoint_map_model_v2
    test    eax, eax
    jz      .checkpoint_ok

    mov     r13d, eax
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
    je      .internal_cleanup_model
    cmp     qword [r12 + MM_WEIGHTS], 0
    je      .internal_cleanup_model
    cmp     qword [r12 + MM_WEIGHT_BYTES], 0
    je      .internal_cleanup_model

    ; Persistent tokenizer mapping.
    mov     rdi, [rbx + 24]
    mov     rsi, r14
    call    tokenizer_map_v1
    test    eax, eax
    jz      .tokenizer_ok

    mov     r13d, eax
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
    je      .internal_cleanup_maps
    cmp     qword [r14 + TM_MAP_BYTES], 0
    je      .internal_cleanup_maps

    ; argc=3 retains the existing loader-only contract.
    cmp     qword [rbx], 3
    jne     .generate

    mov     rdi, r14
    call    tokenizer_unmap_v1
    test    eax, eax
    jnz     .internal_cleanup_model

    mov     rdi, r12
    call    checkpoint_unmap_model
    test    eax, eax
    jnz     .internal_fail

    mov     edi, 1
    lea     rsi, [rel ok_msg]
    mov     edx, ok_len
    call    sys_write_all
    mov     edi, EXIT_OK
    jmp     sys_exit

.generate:
    mov     rdi, [r12 + MM_HEADER]
    mov     rsi, r15
    call    runtime_memory_create
    test    eax, eax
    jnz     .internal_cleanup_maps

    mov     rdi, [r12 + MM_HEADER]
    mov     rsi, [r12 + MM_WEIGHTS]
    mov     rdx, [r14 + TM_MAP_BASE]
    mov     rcx, [rbx + 32]
    mov     r8, r15
    call    generate_greedy_stdout
    mov     r13d, eax

    mov     rdi, r15
    call    runtime_memory_destroy
    test    eax, eax
    jnz     .internal_cleanup_maps

    mov     rdi, r14
    call    tokenizer_unmap_v1
    test    eax, eax
    jnz     .internal_cleanup_model

    mov     rdi, r12
    call    checkpoint_unmap_model
    test    eax, eax
    jnz     .internal_fail

    test    r13d, r13d
    jz      .success_silent
    cmp     r13d, 2
    je      .io_fail
    mov     edi, EXIT_INTERNAL
    jmp     sys_exit

.internal_cleanup_maps:
    mov     rdi, r14
    call    tokenizer_unmap_v1

.internal_cleanup_model:
    mov     rdi, r12
    call    checkpoint_unmap_model

.internal_fail:
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

.success_silent:
    mov     edi, EXIT_OK
    jmp     sys_exit
