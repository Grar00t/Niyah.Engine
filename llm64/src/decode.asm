BITS 64
DEFAULT REL

%include "model.inc"
%include "layout.inc"
%include "runtime.inc"

extern model_layout_from_header
extern model_layer_layout_from_header
extern matvec_f32_scalar
extern rmsnorm_f32
extern rope_f32
extern swiglu_f32
extern kv_store_f32
extern attention_gqa_f32
extern argmax_f32

global decode_token_f32

section .text

; =====================================================================
; int decode_token_f32(
;     const void *header,       rdi
;     const float *weights,     rsi
;     RuntimeMemory *runtime,   rdx
;     uint64_t *position,       rcx
;     uint32_t token)           r8d
;
; Returns eax=0 success, eax=1 invalid/kernel failure.
;
; Uses the exact Niyah order:
; embedding -> attn RMS -> QKV -> RoPE -> KV -> GQA -> Wo
; -> FFN RMS -> gate/up -> SwiGLU -> Wdown
; -> final RMS -> lm_head logits
; =====================================================================

decode_token_f32:
    push rbx
    push r12
    push r13
    push r14
    push r15

    sub rsp, 320
    mov r15, rsp

    test rdi, rdi
    jz .bad
    test rsi, rsi
    jz .bad
    test rdx, rdx
    jz .bad
    test rcx, rcx
    jz .bad

    mov r12, rdi                    ; header
    mov r13, rsi                    ; weights
    mov rbx, rdx                    ; RuntimeMemory
    mov [r15 + 312], rcx            ; position pointer
    mov [r15 + 308], r8d            ; token

    ; Canonical model layout.
    mov rdi, r12
    lea rsi, [r15 + 0]
    call model_layout_from_header wrt ..plt
    test eax, eax
    jnz .bad

    mov eax, [r12 + CKPT_DIM_OFF]
    mov [r15 + 240], rax            ; dim

    mov eax, [r12 + CKPT_FFN_OFF]
    mov [r15 + 248], rax            ; ffn

    mov eax, [r12 + CKPT_VOCAB_OFF]
    mov [r15 + 256], rax            ; vocab

    mov rax, [r15 + L_KV_DIM]
    mov [r15 + 264], rax            ; kv_dim

    mov rax, [r15 + L_HEAD_DIM]
    mov [r15 + 272], rax            ; head_dim

    ; token < vocab
    mov eax, [r15 + 308]
    cmp rax, [r15 + 256]
    jae .bad

    ; position < context
    mov rcx, [r15 + 312]
    mov rax, [rcx]
    mov [r15 + 280], rax            ; position

    mov edx, [r12 + CKPT_CONTEXT_OFF]
    cmp rax, rdx
    jae .bad

    ; Runtime buffers must exist.
    cmp qword [rbx + RT_KEYS], 0
    je .bad
    cmp qword [rbx + RT_VALUES], 0
    je .bad
    cmp qword [rbx + RT_WORKSPACE], 0
    je .bad
    cmp qword [rbx + RT_LOGITS], 0
    je .bad

    ; -------------------------------------------------------------
    ; Workspace slicing:
    ; hidden dim
    ; norm   dim
    ; q      dim
    ; k      kv_dim
    ; v      kv_dim
    ; attn   dim
    ; proj   dim
    ; gate   ffn
    ; up     ffn
    ; scores context
    ; -------------------------------------------------------------

    mov rax, [rbx + RT_WORKSPACE]
    mov [r15 + 160], rax            ; hidden

    mov rcx, [r15 + 240]
    lea rax, [rax + rcx*4]
    mov [r15 + 168], rax            ; norm

    lea rax, [rax + rcx*4]
    mov [r15 + 176], rax            ; q

    lea rax, [rax + rcx*4]
    mov [r15 + 184], rax            ; k

    mov rcx, [r15 + 264]
    lea rax, [rax + rcx*4]
    mov [r15 + 192], rax            ; v

    lea rax, [rax + rcx*4]
    mov [r15 + 200], rax            ; attn

    mov rcx, [r15 + 240]
    lea rax, [rax + rcx*4]
    mov [r15 + 208], rax            ; proj

    lea rax, [rax + rcx*4]
    mov [r15 + 216], rax            ; gate

    mov rcx, [r15 + 248]
    lea rax, [rax + rcx*4]
    mov [r15 + 224], rax            ; up

    lea rax, [rax + rcx*4]
    mov [r15 + 232], rax            ; scores

    ; -------------------------------------------------------------
    ; hidden = token_embedding[token]
    ; -------------------------------------------------------------

    mov eax, [r15 + 308]
    imul rax, [r15 + 240]
    add rax, [r15 + L_TOKEN_EMBED]

    lea rsi, [r13 + rax*4]
    mov rdi, [r15 + 160]
    mov rcx, [r15 + 240]
    rep movsd

    xor r14d, r14d                  ; layer index

.layer_loop:
    mov eax, [r12 + CKPT_LAYERS_OFF]
    cmp r14d, eax
    jae .layers_done

    ; layer layout at local +80
    mov rdi, r12
    lea rsi, [r15 + 0]
    mov edx, r14d
    lea rcx, [r15 + 80]
    call model_layer_layout_from_header wrt ..plt
    test eax, eax
    jnz .bad

    ; norm = RMSNorm(hidden, attn_norm)
    mov rax, [r15 + 80 + LL_ATTN_NORM]
    lea rdx, [r13 + rax*4]

    mov rdi, [r15 + 168]
    mov rsi, [r15 + 160]
    mov rcx, [r15 + 240]
    movss xmm0, [r12 + CKPT_RMS_EPS_OFF]
    call rmsnorm_f32 wrt ..plt
    test eax, eax
    jnz .bad

    ; q = Wq * norm
    mov rax, [r15 + 80 + LL_WQ]
    lea rsi, [r13 + rax*4]

    mov rdi, [r15 + 176]
    mov rdx, [r15 + 168]
    mov rcx, [r15 + 240]
    mov r8,  [r15 + 240]
    call matvec_f32_scalar wrt ..plt

    ; k = Wk * norm
    mov rax, [r15 + 80 + LL_WK]
    lea rsi, [r13 + rax*4]

    mov rdi, [r15 + 184]
    mov rdx, [r15 + 168]
    mov rcx, [r15 + 264]
    mov r8,  [r15 + 240]
    call matvec_f32_scalar wrt ..plt

    ; v = Wv * norm
    mov rax, [r15 + 80 + LL_WV]
    lea rsi, [r13 + rax*4]

    mov rdi, [r15 + 192]
    mov rdx, [r15 + 168]
    mov rcx, [r15 + 264]
    mov r8,  [r15 + 240]
    call matvec_f32_scalar wrt ..plt

    ; RoPE(q)
    mov rdi, [r15 + 176]
    mov esi, [r12 + CKPT_HEADS_OFF]
    mov rdx, [r15 + 272]
    mov rcx, [r15 + 280]
    call rope_f32 wrt ..plt
    test eax, eax
    jnz .bad

    ; RoPE(k)
    mov rdi, [r15 + 184]
    mov esi, [r12 + CKPT_KV_HEADS_OFF]
    mov rdx, [r15 + 272]
    mov rcx, [r15 + 280]
    call rope_f32 wrt ..plt
    test eax, eax
    jnz .bad

    ; Layer-cache byte offset:
    ; layer * context * kv_dim * sizeof(float)
    mov rax, r14
    mov ecx, [r12 + CKPT_CONTEXT_OFF]
    imul rax, rcx
    imul rax, [r15 + 264]
    shl rax, 2

    mov r10, [rbx + RT_KEYS]
    add r10, rax

    mov r11, [rbx + RT_VALUES]
    add r11, rax

    ; KV store: 7th arg is v on stack.
    sub rsp, 16
    mov rax, [r15 + 192]
    mov [rsp], rax

    mov rdi, r10
    mov rsi, r11
    mov edx, [r12 + CKPT_CONTEXT_OFF]
    mov rcx, [r15 + 264]
    mov r8,  [r15 + 280]
    mov r9,  [r15 + 184]
    call kv_store_f32 wrt ..plt

    add rsp, 16

    test eax, eax
    jnz .bad

    ; Recompute layer cache pointers: volatile regs were clobbered.
    mov rax, r14
    mov ecx, [r12 + CKPT_CONTEXT_OFF]
    imul rax, rcx
    imul rax, [r15 + 264]
    shl rax, 2

    mov r10, [rbx + RT_KEYS]
    add r10, rax

    mov r11, [rbx + RT_VALUES]
    add r11, rax

    ; attention_gqa:
    ; stack args = head_dim, position, scores + alignment pad.
    sub rsp, 32

    mov rax, [r15 + 272]
    mov [rsp + 0], rax

    mov rax, [r15 + 280]
    mov [rsp + 8], rax

    mov rax, [r15 + 232]
    mov [rsp + 16], rax

    mov rdi, [r15 + 200]
    mov rsi, [r15 + 176]
    mov rdx, r10
    mov rcx, r11
    mov r8d, [r12 + CKPT_HEADS_OFF]
    mov r9d, [r12 + CKPT_KV_HEADS_OFF]

    call attention_gqa_f32 wrt ..plt

    add rsp, 32

    test eax, eax
    jnz .bad

    ; proj = Wo * attn
    mov rax, [r15 + 80 + LL_WO]
    lea rsi, [r13 + rax*4]

    mov rdi, [r15 + 208]
    mov rdx, [r15 + 200]
    mov rcx, [r15 + 240]
    mov r8,  [r15 + 240]
    call matvec_f32_scalar wrt ..plt

    ; hidden += proj
    xor ecx, ecx
.attn_residual:
    cmp rcx, [r15 + 240]
    jae .attn_residual_done

    mov rdi, [r15 + 160]
    mov rsi, [r15 + 208]

    movss xmm0, [rdi + rcx*4]
    addss xmm0, [rsi + rcx*4]
    movss [rdi + rcx*4], xmm0

    inc rcx
    jmp .attn_residual

.attn_residual_done:

    ; norm = RMSNorm(hidden, ffn_norm)
    mov rax, [r15 + 80 + LL_FFN_NORM]
    lea rdx, [r13 + rax*4]

    mov rdi, [r15 + 168]
    mov rsi, [r15 + 160]
    mov rcx, [r15 + 240]
    movss xmm0, [r12 + CKPT_RMS_EPS_OFF]
    call rmsnorm_f32 wrt ..plt
    test eax, eax
    jnz .bad

    ; gate = Wgate * norm
    mov rax, [r15 + 80 + LL_W_GATE]
    lea rsi, [r13 + rax*4]

    mov rdi, [r15 + 216]
    mov rdx, [r15 + 168]
    mov rcx, [r15 + 248]
    mov r8,  [r15 + 240]
    call matvec_f32_scalar wrt ..plt

    ; up = Wup * norm
    mov rax, [r15 + 80 + LL_W_UP]
    lea rsi, [r13 + rax*4]

    mov rdi, [r15 + 224]
    mov rdx, [r15 + 168]
    mov rcx, [r15 + 248]
    mov r8,  [r15 + 240]
    call matvec_f32_scalar wrt ..plt

    ; gate = SiLU(gate) * up
    mov rdi, [r15 + 216]
    mov rsi, [r15 + 224]
    mov rdx, [r15 + 248]
    call swiglu_f32 wrt ..plt
    test eax, eax
    jnz .bad

    ; proj = Wdown * gate
    mov rax, [r15 + 80 + LL_W_DOWN]
    lea rsi, [r13 + rax*4]

    mov rdi, [r15 + 208]
    mov rdx, [r15 + 216]
    mov rcx, [r15 + 240]
    mov r8,  [r15 + 248]
    call matvec_f32_scalar wrt ..plt

    ; hidden += proj
    xor ecx, ecx
.ffn_residual:
    cmp rcx, [r15 + 240]
    jae .ffn_residual_done

    mov rdi, [r15 + 160]
    mov rsi, [r15 + 208]

    movss xmm0, [rdi + rcx*4]
    addss xmm0, [rsi + rcx*4]
    movss [rdi + rcx*4], xmm0

    inc rcx
    jmp .ffn_residual

.ffn_residual_done:
    inc r14d
    jmp .layer_loop


.layers_done:

    ; final RMSNorm
    mov rax, [r15 + L_FINAL_NORM]
    lea rdx, [r13 + rax*4]

    mov rdi, [r15 + 168]
    mov rsi, [r15 + 160]
    mov rcx, [r15 + 240]
    movss xmm0, [r12 + CKPT_RMS_EPS_OFF]
    call rmsnorm_f32 wrt ..plt
    test eax, eax
    jnz .bad

    ; logits = LM_head * norm.
    ; For tied embeddings L_LM_HEAD == token embedding offset (0).
    mov rax, [r15 + L_LM_HEAD]
    lea rsi, [r13 + rax*4]

    mov rdi, [rbx + RT_LOGITS]
    mov rdx, [r15 + 168]
    mov rcx, [r15 + 256]
    mov r8,  [r15 + 240]
    call matvec_f32_scalar wrt ..plt

    ; position++
    mov rcx, [r15 + 312]
    mov rax, [r15 + 280]
    inc rax
    mov [rcx], rax

    xor eax, eax
    jmp .ret


.bad:
    mov eax, 1

.ret:
    add rsp, 320
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
