BITS 64
DEFAULT REL

%include "model.inc"
%include "runtime.inc"

extern decode_token_f32
extern argmax_f32
extern tokenizer_encode_mapped
extern tokenizer_decode_one_mapped
extern sys_mmap_rw_anon
extern sys_munmap
extern sys_write_all

global generate_one_stdout

section .text

; int generate_one_stdout(const void *header,
;                         const float *weights,
;                         const void *tok_map,
;                         const char *prompt_cstr,
;                         RuntimeMemory *runtime)
;
; Emits exactly one greedy autoregressive token to stdout.
; Returns eax=0 success, eax=1 invalid/kernel failure, eax=2 mmap/write failure.
generate_one_stdout:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 96

    test    rdi, rdi
    jz      .bad
    test    rsi, rsi
    jz      .bad
    test    rdx, rdx
    jz      .bad
    test    rcx, rcx
    jz      .bad
    test    r8, r8
    jz      .bad

    mov     r12, rdi                ; checkpoint header
    mov     r13, rsi                ; model weights
    mov     r14, rdx                ; mapped tokenizer
    mov     rbx, rcx                ; prompt C string
    mov     r15, r8                 ; RuntimeMemory

    xor     eax, eax
    mov     [rsp + 0], rax          ; cache position
    mov     [rsp + 16], rax         ; token buffer mapping
    mov     [rsp + 24], rax         ; token buffer bytes
    mov     [rsp + 40], rax         ; decode stack mapping
    mov     [rsp + 48], rax         ; decode stack bytes
    mov     [rsp + 56], rax         ; output mapping
    mov     [rsp + 64], rax         ; output bytes

    cmp     dword [r12 + CKPT_CONTEXT_OFF], 1
    jb      .bad

    ; Measure raw UTF-8 prompt bytes. argv guarantees NUL termination.
    xor     rax, rax
.len_loop:
    cmp     byte [rbx + rax], 0
    je      .len_done
    inc     rax
    jnz     .len_loop
    jmp     .bad
.len_done:
    mov     [rsp + 8], rax          ; prompt byte count

    ; BOS occupies position 0 and produces initial logits.
    mov     rdi, r12
    mov     rsi, r13
    mov     rdx, r15
    lea     rcx, [rsp + 0]
    mov     r8d, TOK_BOS
    call    decode_token_f32 wrt ..plt
    test    eax, eax
    jnz     .bad_cleanup

    mov     rax, [rsp + 8]
    test    rax, rax
    jz      .prompt_done

    ; BPE output count cannot exceed raw input byte count.
    mov     rdx, rax
    shr     rdx, 62
    jnz     .bad_cleanup
    shl     rax, 2
    mov     [rsp + 24], rax

    mov     rdi, rax
    call    sys_mmap_rw_anon wrt ..plt
    test    rax, rax
    js      .io_cleanup
    mov     [rsp + 16], rax

    mov     rdi, r14
    mov     rsi, rbx
    mov     rdx, [rsp + 8]
    mov     rcx, [rsp + 16]
    mov     r8, [rsp + 8]
    call    tokenizer_encode_mapped wrt ..plt
    test    rax, rax
    js      .bad_cleanup
    mov     [rsp + 32], rax         ; prompt token count

    ; BOS + prompt must fit the model context.
    mov     rcx, [rsp + 0]
    add     rcx, rax
    jc      .bad_cleanup
    mov     edx, [r12 + CKPT_CONTEXT_OFF]
    cmp     rcx, rdx
    ja      .bad_cleanup

    xor     r10d, r10d
.prompt_loop:
    cmp     r10, [rsp + 32]
    jae     .prompt_done

    mov     r11, [rsp + 16]
    mov     r8d, [r11 + r10*4]
    mov     [rsp + 88], r10

    mov     rdi, r12
    mov     rsi, r13
    mov     rdx, r15
    lea     rcx, [rsp + 0]
    call    decode_token_f32 wrt ..plt
    test    eax, eax
    jnz     .bad_cleanup

    mov     r10, [rsp + 88]
    inc     r10
    jmp     .prompt_loop

.prompt_done:
    mov     rdi, [r15 + RT_LOGITS]
    mov     esi, [r12 + CKPT_VOCAB_OFF]
    call    argmax_f32 wrt ..plt
    cmp     eax, 0xffffffff
    je      .bad_cleanup
    mov     [rsp + 72], eax         ; generated token

    cmp     eax, TOK_EOS
    je      .ok_cleanup

    ; Explicit decode stack and byte output each receive 4*vocab bytes.
    mov     eax, [r12 + CKPT_VOCAB_OFF]
    mov     rdx, rax
    shr     rdx, 62
    jnz     .bad_cleanup
    shl     rax, 2
    test    rax, rax
    jz      .bad_cleanup
    mov     [rsp + 48], rax
    mov     [rsp + 64], rax

    mov     rdi, rax
    call    sys_mmap_rw_anon wrt ..plt
    test    rax, rax
    js      .io_cleanup
    mov     [rsp + 40], rax

    mov     rdi, [rsp + 64]
    call    sys_mmap_rw_anon wrt ..plt
    test    rax, rax
    js      .io_cleanup
    mov     [rsp + 56], rax

    mov     rdi, r14
    mov     esi, [rsp + 72]
    mov     rdx, [rsp + 56]
    mov     rcx, [rsp + 64]
    mov     r8, [rsp + 40]
    mov     r9d, [r12 + CKPT_VOCAB_OFF]
    call    tokenizer_decode_one_mapped wrt ..plt
    test    rax, rax
    js      .bad_cleanup
    mov     [rsp + 80], rax

    test    rax, rax
    jz      .ok_cleanup

    mov     edi, 1
    mov     rsi, [rsp + 56]
    mov     rdx, rax
    call    sys_write_all wrt ..plt
    test    rax, rax
    js      .io_cleanup

.ok_cleanup:
    xor     ebx, ebx
    jmp     .cleanup

.bad_cleanup:
    mov     ebx, 1
    jmp     .cleanup

.io_cleanup:
    mov     ebx, 2

.cleanup:
    mov     rax, [rsp + 56]
    test    rax, rax
    jz      .no_output
    mov     rdi, rax
    mov     rsi, [rsp + 64]
    call    sys_munmap wrt ..plt
.no_output:

    mov     rax, [rsp + 40]
    test    rax, rax
    jz      .no_stack
    mov     rdi, rax
    mov     rsi, [rsp + 48]
    call    sys_munmap wrt ..plt
.no_stack:

    mov     rax, [rsp + 16]
    test    rax, rax
    jz      .done_cleanup
    mov     rdi, rax
    mov     rsi, [rsp + 24]
    call    sys_munmap wrt ..plt
.done_cleanup:
    mov     eax, ebx
    jmp     .ret

.bad:
    mov     eax, 1

.ret:
    add     rsp, 96
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
