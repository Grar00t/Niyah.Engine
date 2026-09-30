BITS 64
DEFAULT REL

%include "model.inc"

global tokenizer_encode_mapped
global tokenizer_decode_one_mapped

section .text

; ssize_t tokenizer_encode_mapped(
;   const void *tok,
;   const uint8_t *input,
;   size_t input_size,
;   uint32_t *output,
;   size_t output_capacity)
;
; Returns token count, or -1 on invalid arguments.
; output_capacity is measured in uint32_t tokens.
; Since BPE merges only reduce count, input_size entries are sufficient.
tokenizer_encode_mapped:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    test    rdi, rdi
    jz      .enc_bad

    test    rdx, rdx
    jz      .enc_zero

    test    rsi, rsi
    jz      .enc_bad
    test    rcx, rcx
    jz      .enc_bad
    cmp     r8, rdx
    jb      .enc_bad

    mov     r12, rcx                ; output token array
    mov     r13, rdx                ; current token count

    xor     r9d, r9d
.enc_copy_bytes:
    movzx   eax, byte [rsi + r9]
    mov     [r12 + r9*4], eax
    inc     r9
    cmp     r9, r13
    jb      .enc_copy_bytes

    mov     r14d, [rdi + TOK_MERGE_COUNT_OFF]
    lea     r15, [rdi + TOK_HEADER_SIZE]
    xor     ebx, ebx                ; merge index

.enc_merge_loop:
    cmp     ebx, r14d
    jae     .enc_ok

    mov     r10d, [r15 + 0]         ; left
    mov     r11d, [r15 + 4]         ; right
    mov     edx,  [r15 + 8]         ; output

    xor     eax, eax                ; read index
    xor     ecx, ecx                ; write index

.enc_scan:
    cmp     rax, r13
    jae     .enc_merge_done

    mov     r9d, [r12 + rax*4]
    cmp     r9d, r10d
    jne     .enc_copy_one

    lea     r8, [rax + 1]
    cmp     r8, r13
    jae     .enc_copy_one
    cmp     dword [r12 + r8*4], r11d
    jne     .enc_copy_one

    mov     [r12 + rcx*4], edx
    add     rax, 2
    inc     rcx
    jmp     .enc_scan

.enc_copy_one:
    mov     [r12 + rcx*4], r9d
    inc     rax
    inc     rcx
    jmp     .enc_scan

.enc_merge_done:
    mov     r13, rcx
    add     r15, TOK_MERGE_SIZE
    inc     ebx
    jmp     .enc_merge_loop

.enc_ok:
    mov     rax, r13
    jmp     .enc_return

.enc_zero:
    xor     eax, eax
    jmp     .enc_return

.enc_bad:
    mov     rax, -1

.enc_return:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret


; ssize_t tokenizer_decode_one_mapped(
;   const void *tok,
;   uint32_t token,
;   uint8_t *output,
;   size_t output_capacity,
;   uint32_t *stack,
;   size_t stack_capacity)
;
; Expands one token recursively through the stored merge triples.
; BOS/EOS emit zero bytes.
; Returns emitted byte count, or -1 on invalid/capacity error.
tokenizer_decode_one_mapped:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    test    rdi, rdi
    jz      .dec_bad

    mov     eax, esi

    cmp     eax, TOK_BOS
    je      .dec_zero
    cmp     eax, TOK_EOS
    je      .dec_zero

    cmp     eax, [rdi + TOK_VOCAB_OFF]
    jae     .dec_bad

    cmp     eax, 256
    jb      .dec_raw_byte

    test    rdx, rdx
    jz      .dec_bad
    test    r8, r8
    jz      .dec_bad
    test    r9, r9
    jz      .dec_bad

    mov     rbx, rdi                ; tokenizer base
    mov     r12, rdx                ; output bytes
    mov     r13, rcx                ; output capacity
    mov     r14, r8                 ; explicit token stack
    mov     r15, r9                 ; stack capacity

    xor     r10d, r10d              ; stack size
    xor     r11d, r11d              ; output size

    mov     [r14], eax
    inc     r10

.dec_loop:
    test    r10, r10
    jz      .dec_ok

    dec     r10
    mov     eax, [r14 + r10*4]

    cmp     eax, 256
    jb      .dec_emit

    cmp     eax, TOK_BOS
    je      .dec_loop
    cmp     eax, TOK_EOS
    je      .dec_loop

    sub     eax, TOK_BASE_VOCAB
    cmp     eax, [rbx + TOK_MERGE_COUNT_OFF]
    jae     .dec_bad

    mov     ecx, eax
    imul    rcx, TOK_MERGE_SIZE
    lea     rcx, [rbx + rcx + TOK_HEADER_SIZE]

    ; LIFO: push right first, then left, so left is emitted first.
    lea     rax, [r10 + 2]
    cmp     rax, r15
    ja      .dec_bad

    mov     edx, [rcx + 4]
    mov     [r14 + r10*4], edx
    mov     edx, [rcx + 0]
    mov     [r14 + r10*4 + 4], edx
    add     r10, 2
    jmp     .dec_loop

.dec_emit:
    cmp     r11, r13
    jae     .dec_bad
    mov     [r12 + r11], al
    inc     r11
    jmp     .dec_loop

.dec_raw_byte:
    test    rdx, rdx
    jz      .dec_bad
    test    rcx, rcx
    jz      .dec_bad
    mov     [rdx], sil
    mov     eax, 1
    jmp     .dec_return

.dec_ok:
    mov     rax, r11
    jmp     .dec_return

.dec_zero:
    xor     eax, eax
    jmp     .dec_return

.dec_bad:
    mov     rax, -1

.dec_return:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
