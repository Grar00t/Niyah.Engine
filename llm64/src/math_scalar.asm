BITS 64
DEFAULT REL

global matvec_f32_scalar
global rmsnorm_f32
global argmax_f32

section .rodata align=4
one_f32: dd 0x3f800000

section .text

; void matvec_f32_scalar(float *out, const float *matrix,
;                        const float *x, size_t rows, size_t cols)
; rdi=out, rsi=matrix, rdx=x, rcx=rows, r8=cols
matvec_f32_scalar:
    test    rdi, rdi
    jz      .mv_ret
    test    rsi, rsi
    jz      .mv_ret
    test    rdx, rdx
    jz      .mv_ret
    test    rcx, rcx
    jz      .mv_ret
    test    r8, r8
    jz      .mv_ret

    xor     r9d, r9d
    mov     r11, rsi
.mv_row:
    pxor    xmm0, xmm0
    xor     eax, eax
.mv_col:
    movss   xmm1, [r11 + rax*4]
    mulss   xmm1, [rdx + rax*4]
    addss   xmm0, xmm1
    inc     rax
    cmp     rax, r8
    jb      .mv_col
    movss   [rdi + r9*4], xmm0
    lea     r11, [r11 + r8*4]
    inc     r9
    cmp     r9, rcx
    jb      .mv_row
.mv_ret:
    ret

; int rmsnorm_f32(float *out, const float *x, const float *weight,
;                 size_t n, float eps)
; rdi=out, rsi=x, rdx=weight, rcx=n, xmm0=eps
; returns eax=0 on success, eax=1 on invalid arguments
rmsnorm_f32:
    test    rdi, rdi
    jz      .rn_bad
    test    rsi, rsi
    jz      .rn_bad
    test    rdx, rdx
    jz      .rn_bad
    test    rcx, rcx
    jz      .rn_bad

    pxor    xmm3, xmm3
    ucomiss xmm0, xmm3
    jp      .rn_bad
    jbe     .rn_bad

    pxor    xmm1, xmm1              ; FP64 sum_sq
    xor     r8d, r8d
.rn_sum:
    movss   xmm2, [rsi + r8*4]
    cvtss2sd xmm2, xmm2
    mulsd   xmm2, xmm2
    addsd   xmm1, xmm2
    inc     r8
    cmp     r8, rcx
    jb      .rn_sum

    cvtsi2sd xmm2, rcx
    divsd   xmm1, xmm2
    cvtsd2ss xmm1, xmm1             ; reference path narrows before eps
    addss   xmm1, xmm0
    sqrtss  xmm1, xmm1
    movss   xmm2, [rel one_f32]
    divss   xmm2, xmm1              ; inv_rms

    xor     r8d, r8d
.rn_write:
    movss   xmm1, [rsi + r8*4]
    mulss   xmm1, xmm2
    mulss   xmm1, [rdx + r8*4]
    movss   [rdi + r8*4], xmm1
    inc     r8
    cmp     r8, rcx
    jb      .rn_write

    xor     eax, eax
    ret
.rn_bad:
    mov     eax, 1
    ret

; uint32_t argmax_f32(const float *values, size_t count)
; rdi=values, rsi=count. Returns UINT32_MAX when count==0 or values==NULL.
argmax_f32:
    test    rdi, rdi
    jz      .am_bad
    test    rsi, rsi
    jz      .am_bad
    movss   xmm0, [rdi]
    xor     eax, eax                  ; best index
    mov     rcx, 1
.am_loop:
    cmp     rcx, rsi
    jae     .am_done
    movss   xmm1, [rdi + rcx*4]
    ucomiss xmm1, xmm0
    jbe     .am_next                  ; stable: first maximum wins
    movss   xmm0, xmm1
    mov     eax, ecx
.am_next:
    inc     rcx
    jmp     .am_loop
.am_done:
    ret
.am_bad:
    mov     eax, 0xffffffff
    ret
