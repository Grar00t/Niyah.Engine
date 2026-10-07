BITS 64
DEFAULT REL

global matvec_f32_scalar
global rmsnorm_f32
global argmax_f32
global exp_f32
global softmax_f32
global rope_f32
global swiglu_f32

section .rodata align=4
one_f32:           dd 0x3f800000
exp_overflow_f32:  dd 88.72283935546875
exp_zero_f32:      dd -103.97208404541016
pos_inf_f32:       dd 0x7f800000
neg_2ln10000_f32:  dd -18.420680743952367

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
; rdi=values, rsi=count. Returns UINT32_MAX for invalid input or any
; non-finite logit. This matches the native sampler's fail-closed contract.
argmax_f32:
    test    rdi, rdi
    jz      .am_bad
    test    rsi, rsi
    jz      .am_bad

    mov     edx, [rdi]
    mov     r8d, edx
    and     r8d, 0x7f800000
    cmp     r8d, 0x7f800000
    je      .am_bad
    movd    xmm0, edx

    xor     eax, eax                  ; best index
    mov     rcx, 1
.am_loop:
    cmp     rcx, rsi
    jae     .am_done

    mov     edx, [rdi + rcx*4]
    mov     r8d, edx
    and     r8d, 0x7f800000
    cmp     r8d, 0x7f800000
    je      .am_bad
    movd    xmm1, edx

    ucomiss xmm1, xmm0
    jbe     .am_next                  ; stable: first maximum wins
    movaps  xmm0, xmm1
    mov     eax, ecx
.am_next:
    inc     rcx
    jmp     .am_loop
.am_done:
    ret
.am_bad:
    mov     eax, 0xffffffff
    ret


; ---------------------------------------------------------------------------
; float exp_f32(float x)
;
; Pure x86 implementation. No libc/libm.
; Input/output: xmm0.
; Preserve expf-style FP32 overflow/underflow instead of clamping finite
; inputs, because the clamp changes softmax and SwiGLU inference results.
; ---------------------------------------------------------------------------
exp_f32:
    sub     rsp, 16

    ucomiss xmm0, xmm0
    jp      .exp_return             ; preserve NaN

    movss   xmm1, [rel exp_overflow_f32]
    ucomiss xmm0, xmm1
    ja      .exp_inf

    movss   xmm1, [rel exp_zero_f32]
    ucomiss xmm0, xmm1
    jb      .exp_zero

    movss   [rsp], xmm0

    ; exp(x) = 2^(x * log2(e))
    fld     dword [rsp]
    fldl2e
    fmulp   st1, st0               ; y = x*log2(e)

    ; y = n + f, with f in approximately [-0.5,+0.5].
    fld     st0
    frndint                         ; n, y
    fxch    st1                     ; y, n
    fsub    st0, st1               ; f, n

    f2xm1                           ; 2^f - 1
    fld1
    faddp   st1, st0               ; 2^f, n
    fscale                          ; 2^f * 2^n

    fstp    dword [rsp + 4]
    fstp    st0                     ; discard n

    movss   xmm0, [rsp + 4]
    jmp     .exp_return

.exp_inf:
    movss   xmm0, [rel pos_inf_f32]
    jmp     .exp_return

.exp_zero:
    xorps   xmm0, xmm0

.exp_return:
    add     rsp, 16
    ret


; ---------------------------------------------------------------------------
; int softmax_f32(float *values, size_t count)
;
; Stable in-place softmax.
; rdi = values
; rsi = count
; eax = 0 success, 1 invalid input
; ---------------------------------------------------------------------------
softmax_f32:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    test    rdi, rdi
    jz      .sm_bad
    test    rsi, rsi
    jz      .sm_bad

    mov     r12, rdi
    mov     r13, rsi

    ; max
    movss   xmm3, [r12]
    ucomiss xmm3, xmm3
    jp      .sm_bad

    mov     rbx, 1
.sm_max_loop:
    cmp     rbx, r13
    jae     .sm_max_done

    movss   xmm1, [r12 + rbx*4]
    ucomiss xmm1, xmm1
    jp      .sm_bad

    ucomiss xmm1, xmm3
    jbe     .sm_max_next
    movaps  xmm3, xmm1

.sm_max_next:
    inc     rbx
    jmp     .sm_max_loop

.sm_max_done:
    xorps   xmm4, xmm4              ; sum
    xor     ebx, ebx

.sm_exp_loop:
    cmp     rbx, r13
    jae     .sm_exp_done

    movss   xmm0, [r12 + rbx*4]
    subss   xmm0, xmm3
    call    exp_f32

    movss   [r12 + rbx*4], xmm0
    addss   xmm4, xmm0

    inc     rbx
    jmp     .sm_exp_loop

.sm_exp_done:
    xorps   xmm0, xmm0
    ucomiss xmm4, xmm0
    jp      .sm_bad
    jbe     .sm_bad

    xor     ebx, ebx
.sm_norm_loop:
    cmp     rbx, r13
    jae     .sm_ok

    movss   xmm0, [r12 + rbx*4]
    divss   xmm0, xmm4
    movss   [r12 + rbx*4], xmm0

    inc     rbx
    jmp     .sm_norm_loop

.sm_ok:
    xor     eax, eax
    jmp     .sm_return

.sm_bad:
    mov     eax, 1

.sm_return:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret


; ---------------------------------------------------------------------------
; int rope_f32(float *vector,
;              size_t n_heads,
;              size_t head_dim,
;              size_t position)
;
; Same geometry as Niyah C decode:
; inv_freq(i) = 10000^(-i/head_dim), i = 0,2,4...
;
; Uses:
;   - exp_f32 once to derive the pair-frequency multiplier
;   - x87 fsincos for sin/cos
;
; eax = 0 success, 1 invalid arguments
; ---------------------------------------------------------------------------
rope_f32:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 16

    test    rdi, rdi
    jz      .rope_bad
    test    rsi, rsi
    jz      .rope_bad

    cmp     rdx, 2
    jb      .rope_bad
    test    rdx, 1
    jnz     .rope_bad

    mov     r12, rdi                ; vector
    mov     r13, rsi                ; n_heads
    mov     r14, rdx                ; head_dim
    mov     r15, rcx                ; position

    ; pair frequency multiplier:
    ; exp(-2*ln(10000)/head_dim)
    movss   xmm0, [rel neg_2ln10000_f32]
    cvtsi2ss xmm1, r14
    divss   xmm0, xmm1
    call    exp_f32
    movaps  xmm7, xmm0              ; frequency multiplier

    cvtsi2ss xmm5, r15              ; float(position)

    xor     ebx, ebx                ; head index

.rope_head_loop:
    cmp     rbx, r13
    jae     .rope_ok

    mov     rax, rbx
    imul    rax, r14                ; base float index for head

    movss   xmm6, [rel one_f32]     ; inv_freq = 1
    xor     r10d, r10d              ; i = 0

.rope_pair_loop:
    cmp     r10, r14
    jae     .rope_next_head

    ; angle = position * inv_freq
    movaps  xmm0, xmm5
    mulss   xmm0, xmm6
    movss   [rsp], xmm0

    ; ST0=cos(angle), ST1=sin(angle)
    fld     dword [rsp]
    fsincos
    fstp    dword [rsp + 4]         ; cos
    fstp    dword [rsp + 8]         ; sin

    lea     r11, [rax + r10]

    movss   xmm2, [r12 + r11*4]       ; x0
    movss   xmm3, [r12 + r11*4 + 4]   ; x1

    ; y0 = x0*c - x1*s
    movss   xmm0, [rsp + 4]
    mulss   xmm0, xmm2

    movss   xmm1, [rsp + 8]
    mulss   xmm1, xmm3
    subss   xmm0, xmm1

    ; y1 = x0*s + x1*c
    movss   xmm4, [rsp + 8]
    mulss   xmm4, xmm2

    movss   xmm1, [rsp + 4]
    mulss   xmm1, xmm3
    addss   xmm4, xmm1

    movss   [r12 + r11*4], xmm0
    movss   [r12 + r11*4 + 4], xmm4

    mulss   xmm6, xmm7
    add     r10, 2
    jmp     .rope_pair_loop

.rope_next_head:
    inc     rbx
    jmp     .rope_head_loop

.rope_ok:
    xor     eax, eax
    jmp     .rope_return

.rope_bad:
    mov     eax, 1

.rope_return:
    add     rsp, 16
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret


; int swiglu_f32(float *gate, const float *up, size_t n)
; gate[i] = SiLU(gate[i]) * up[i]
; SiLU(x) = x / (1 + exp(-x))
swiglu_f32:
    push    rbx
    push    r12
    push    r13
    push    r14
    sub     rsp, 24
    test    rdi, rdi
    jz      .sg_bad
    test    rsi, rsi
    jz      .sg_bad
    test    rdx, rdx
    jz      .sg_bad
    mov     r12, rdi
    mov     r13, rsi
    mov     r14, rdx
    xor     ebx, ebx
.sg_loop:
    cmp     rbx, r14
    jae     .sg_ok
    movss   xmm2, [r12 + rbx*4]
    movss   [rsp], xmm2
    xorps   xmm0, xmm0
    subss   xmm0, xmm2
    call    exp_f32
    addss   xmm0, [rel one_f32]
    movss   xmm2, [rsp]
    divss   xmm2, xmm0
    mulss   xmm2, [r13 + rbx*4]
    movss   [r12 + rbx*4], xmm2
    inc     rbx
    jmp     .sg_loop
.sg_ok:
    xor     eax, eax
    jmp     .sg_return
.sg_bad:
    mov     eax, 1
.sg_return:
    add     rsp, 24
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
