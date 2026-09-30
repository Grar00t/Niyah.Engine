BITS 64
DEFAULT REL

extern softmax_f32

global kv_store_f32
global attention_gqa_f32

section .rodata align=4
one_f32: dd 0x3f800000

section .text


; ===========================================================================
; int kv_store_f32(
;     float *keys,
;     float *values,
;     size_t context_length,
;     size_t kv_dim,
;     size_t position,
;     const float *k,
;     const float *v)
;
; Stores one K/V vector in a single layer cache:
;
;     cache[position][0..kv_dim)
;
; Returns:
;   eax=0 success
;   eax=1 invalid args / overflow
; ===========================================================================

kv_store_f32:
    mov     r10, [rsp + 8]          ; v source

    push    rbx

    test    rdi, rdi
    jz      .kv_bad
    test    rsi, rsi
    jz      .kv_bad
    test    rcx, rcx
    jz      .kv_bad
    test    r9, r9
    jz      .kv_bad
    test    r10, r10
    jz      .kv_bad

    cmp     r8, rdx                 ; position < context_length
    jae     .kv_bad

    mov     rbx, rsi                ; preserve values base
    mov     r11, rcx                ; kv_dim

    mov     rax, r8
    mul     r11                     ; rdx:rax = position * kv_dim
    test    rdx, rdx
    jnz     .kv_bad

    ; Require offset <= (2^62 - 1) before scaling by sizeof(float).
    ; CMP r64 has no imm64 form, so inspect the upper two bits directly.
    mov     rdx, rax
    shr     rdx, 62
    jnz     .kv_bad

    mov     r8, rax                 ; float offset

    ; keys[position]
    lea     rdi, [rdi + r8*4]
    mov     rsi, r9
    mov     rcx, r11
    rep movsd

    ; values[position]
    lea     rdi, [rbx + r8*4]
    mov     rsi, r10
    mov     rcx, r11
    rep movsd

    xor     eax, eax
    pop     rbx
    ret

.kv_bad:
    mov     eax, 1
    pop     rbx
    ret


; ===========================================================================
; int attention_gqa_f32(
;     float *out,             rdi
;     const float *q,         rsi
;     const float *keys,      rdx
;     const float *values,    rcx
;     size_t n_heads,         r8
;     size_t n_kv_heads,      r9
;     size_t head_dim,        [rsp+8]
;     size_t position,        [rsp+16]
;     float *scores)          [rsp+24]
;
; Cache layout for this function is one layer:
;
;   keys  [context][kv_dim]
;   values[context][kv_dim]
;
; kv_dim     = n_kv_heads * head_dim
; group_size = n_heads / n_kv_heads
;
; Sources used: 0..position inclusive.
;
; Returns:
;   eax=0 success
;   eax=1 invalid geometry/data
; ===========================================================================

attention_gqa_f32:
    ; Capture stack arguments before changing rsp.
    mov     r10, [rsp + 8]          ; head_dim
    mov     r11, [rsp + 16]         ; position
    mov     rax, [rsp + 24]         ; scores

    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    sub     rsp, 112

    ; locals
    mov     [rsp + 0],  rdi         ; out
    mov     [rsp + 8],  rsi         ; q
    mov     [rsp + 16], rdx         ; keys
    mov     [rsp + 24], rcx         ; values
    mov     [rsp + 32], r8          ; n_heads
    mov     [rsp + 40], r9          ; n_kv_heads
    mov     [rsp + 48], r10         ; head_dim
    mov     [rsp + 56], r11         ; position
    mov     [rsp + 64], rax         ; scores

    test    rdi, rdi
    jz      .attn_bad
    test    rsi, rsi
    jz      .attn_bad
    test    rdx, rdx
    jz      .attn_bad
    test    rcx, rcx
    jz      .attn_bad
    test    rax, rax
    jz      .attn_bad

    test    r8, r8
    jz      .attn_bad
    test    r9, r9
    jz      .attn_bad
    test    r10, r10
    jz      .attn_bad

    ; group_size = n_heads / n_kv_heads, exact division required.
    mov     rax, r8
    xor     edx, edx
    div     r9

    test    rdx, rdx
    jnz     .attn_bad
    test    rax, rax
    jz      .attn_bad

    mov     [rsp + 72], rax         ; group_size

    ; kv_dim = n_kv_heads * head_dim
    mov     rax, r9
    mul     r10
    test    rdx, rdx
    jnz     .attn_bad

    mov     [rsp + 80], rax         ; kv_dim

    ; scale = 1 / sqrt(head_dim)
    cvtsi2ss xmm0, r10
    sqrtss  xmm0, xmm0

    movss   xmm1, [rel one_f32]
    divss   xmm1, xmm0
    movss   [rsp + 88], xmm1

    xor     ebx, ebx                ; head = 0

.attn_head_loop:
    cmp     rbx, [rsp + 32]
    jae     .attn_ok

    ; kv_head = head / group_size
    mov     rax, rbx
    xor     edx, edx
    div     qword [rsp + 72]

    ; kv-head base float offset
    imul    rax, [rsp + 48]
    mov     [rsp + 104], rax

    ; q_head
    mov     rax, rbx
    imul    rax, [rsp + 48]
    shl     rax, 2
    add     rax, [rsp + 8]
    mov     [rsp + 96], rax

    ; ---------------------------------------------------------------
    ; Compute scaled Q·K scores for sources 0..position.
    ; ---------------------------------------------------------------

    xor     r12d, r12d              ; source=0

.attn_score_source:
    cmp     r12, [rsp + 56]
    ja      .attn_scores_ready

    ; key float index =
    ; source * kv_dim + kv_head * head_dim
    mov     rax, r12
    imul    rax, [rsp + 80]
    add     rax, [rsp + 104]

    mov     r10, [rsp + 16]
    lea     r10, [r10 + rax*4]      ; key head ptr

    mov     r11, [rsp + 96]         ; q head ptr

    xorps   xmm0, xmm0
    xor     r13d, r13d              ; d=0

.attn_dot_loop:
    cmp     r13, [rsp + 48]
    jae     .attn_dot_done

    movss   xmm1, [r11 + r13*4]
    mulss   xmm1, [r10 + r13*4]
    addss   xmm0, xmm1

    inc     r13
    jmp     .attn_dot_loop

.attn_dot_done:
    mulss   xmm0, [rsp + 88]

    mov     rax, [rsp + 64]
    movss   [rax + r12*4], xmm0

    inc     r12
    jmp     .attn_score_source


.attn_scores_ready:
    mov     rdi, [rsp + 64]
    mov     rsi, [rsp + 56]
    inc     rsi

    call    softmax_f32 wrt ..plt
    test    eax, eax
    jnz     .attn_bad

    ; ---------------------------------------------------------------
    ; Weighted sum over values.
    ; ---------------------------------------------------------------

    xor     r13d, r13d              ; d=0

.attn_dim_loop:
    cmp     r13, [rsp + 48]
    jae     .attn_next_head

    xorps   xmm0, xmm0              ; accumulated output[d]
    xor     r14d, r14d              ; source=0

.attn_value_source:
    cmp     r14, [rsp + 56]
    ja      .attn_value_done

    ; value index =
    ; source*kv_dim + kv_head*head_dim + d
    mov     rax, r14
    imul    rax, [rsp + 80]
    add     rax, [rsp + 104]
    add     rax, r13

    mov     r10, [rsp + 24]
    movss   xmm1, [r10 + rax*4]

    mov     r11, [rsp + 64]
    movss   xmm2, [r11 + r14*4]

    mulss   xmm1, xmm2
    addss   xmm0, xmm1

    inc     r14
    jmp     .attn_value_source

.attn_value_done:
    ; out[head*head_dim + d]
    mov     rax, rbx
    imul    rax, [rsp + 48]
    add     rax, r13

    mov     r10, [rsp + 0]
    movss   [r10 + rax*4], xmm0

    inc     r13
    jmp     .attn_dim_loop


.attn_next_head:
    inc     rbx
    jmp     .attn_head_loop


.attn_ok:
    xor     eax, eax
    jmp     .attn_return

.attn_bad:
    mov     eax, 1

.attn_return:
    add     rsp, 112

    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
