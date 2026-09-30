BITS 64
DEFAULT REL

%include "errors.inc"

extern checkpoint_validate_v2
extern tokenizer_validate_v1
extern sys_write_all
extern sys_exit

global _start

section .rodata
usage_msg:      db "usage: niyah-asm <model.ckpt> <tok.bin>", 10
usage_len       equ $ - usage_msg
ckpt_msg:       db "niyah-asm: checkpoint V2 rejected", 10
ckpt_len        equ $ - ckpt_msg
tok_msg:        db "niyah-asm: tokenizer v1 rejected", 10
tok_len         equ $ - tok_msg
io_msg:         db "niyah-asm: file I/O failed", 10
io_len          equ $ - io_msg
ok_msg:         db "NIYAH_ASM64_LOADER=OK", 10
ok_len          equ $ - ok_msg

section .text
_start:
    mov     rbx, rsp
    cmp     qword [rbx], 3
    jne     .usage

    mov     rdi, [rbx + 16]
    call    checkpoint_validate_v2
    test    eax, eax
    jz      .checkpoint_ok
    cmp     eax, 2
    je      .io_fail
    mov     edi, 2
    lea     rsi, [rel ckpt_msg]
    mov     edx, ckpt_len
    call    sys_write_all
    mov     edi, EXIT_FORMAT
    jmp     sys_exit

.checkpoint_ok:
    mov     rdi, [rbx + 24]
    call    tokenizer_validate_v1
    test    eax, eax
    jz      .success
    cmp     eax, 2
    je      .io_fail
    mov     edi, 2
    lea     rsi, [rel tok_msg]
    mov     edx, tok_len
    call    sys_write_all
    mov     edi, EXIT_FORMAT
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
