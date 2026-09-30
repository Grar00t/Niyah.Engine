BITS 64
DEFAULT REL
%include "model.inc"
%include "layout.inc"
global model_layout_from_header
global model_layer_layout_from_header
section .text

model_layout_from_header:
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 64
    test rdi, rdi
    jz .bad
    test rsi, rsi
    jz .bad
    mov rbx, rdi
    mov r12, rsi
    mov r8d,  [rbx + CKPT_VOCAB_OFF]
    mov r9d,  [rbx + CKPT_CONTEXT_OFF]
    mov r10d, [rbx + CKPT_DIM_OFF]
    mov r11d, [rbx + CKPT_LAYERS_OFF]
    mov r13d, [rbx + CKPT_HEADS_OFF]
    mov r14d, [rbx + CKPT_KV_HEADS_OFF]
    mov r15d, [rbx + CKPT_FFN_OFF]
    mov eax,  [rbx + CKPT_SEGMENTS_OFF]
    mov [rsp + 0], rax
    mov eax,  [rbx + CKPT_TIE_OFF]
    mov [rsp + 8], rax
    cmp r8, 2
    jb .bad
    test r9, r9
    jz .bad
    test r10, r10
    jz .bad
    test r11, r11
    jz .bad
    test r13, r13
    jz .bad
    test r14, r14
    jz .bad
    test r15, r15
    jz .bad
    cmp r14, r13
    ja .bad
    mov eax, [rbx + CKPT_RMS_EPS_OFF]
    mov edx, eax
    and edx, 0x7fffffff
    jz .bad
    test eax, 0x80000000
    jnz .bad
    and eax, 0x7f800000
    cmp eax, 0x7f800000
    je .bad
    mov rax, r10
    xor edx, edx
    div r13
    test rdx, rdx
    jnz .bad
    cmp rax, 2
    jb .bad
    test rax, 1
    jnz .bad
    mov [rsp + 16], rax
    mov rax, r13
    xor edx, edx
    div r14
    test rdx, rdx
    jnz .bad
    mov rax, [rsp + 16]
    mul r14
    test rdx, rdx
    jnz .bad
    mov [rsp + 24], rax
    mov rax, r8
    mul r10
    test rdx, rdx
    jnz .bad
    mov [rsp + 32], rax
    mov rax, [rsp + 0]
    mul r10
    test rdx, rdx
    jnz .bad
    mov [rsp + 40], rax
    mov rax, r10
    mul r10
    test rdx, rdx
    jnz .bad
    mov [rsp + 48], rax
    mov rax, [rsp + 24]
    mul r10
    test rdx, rdx
    jnz .bad
    mov [rsp + 56], rax
    mov rax, r15
    mul r10
    test rdx, rdx
    jnz .bad
    mov rcx, rax
    mov qword [r12 + L_TOKEN_EMBED], 0
    mov rax, [rsp + 32]
    mov [r12 + L_SEGMENT_EMBED], rax
    add rax, [rsp + 40]
    jc .bad
    mov [r12 + L_LAYERS], rax
    xor rsi, rsi
    add rsi, r10
    jc .bad
    add rsi, [rsp + 48]
    jc .bad
    add rsi, [rsp + 56]
    jc .bad
    add rsi, [rsp + 56]
    jc .bad
    add rsi, [rsp + 48]
    jc .bad
    add rsi, r10
    jc .bad
    add rsi, rcx
    jc .bad
    add rsi, rcx
    jc .bad
    add rsi, rcx
    jc .bad
    mov [r12 + L_LAYER_STRIDE], rsi
    mov rax, rsi
    mul r11
    test rdx, rdx
    jnz .bad
    add rax, [r12 + L_LAYERS]
    jc .bad
    mov [r12 + L_FINAL_NORM], rax
    add rax, r10
    jc .bad
    cmp qword [rsp + 8], 0
    jne .tied
    mov [r12 + L_LM_HEAD], rax
    add rax, [rsp + 32]
    jc .bad
    jmp .finish
.tied:
    mov qword [r12 + L_LM_HEAD], 0
.finish:
    mov [r12 + L_TOTAL_FLOATS], rax
    mov rdx, [rsp + 16]
    mov [r12 + L_HEAD_DIM], rdx
    mov rdx, [rsp + 24]
    mov [r12 + L_KV_DIM], rdx
    xor eax, eax
    jmp .return
.bad:
    mov eax, 1
.return:
    add rsp, 64
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

model_layer_layout_from_header:
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 48
    test rdi, rdi
    jz .lbad
    test rsi, rsi
    jz .lbad
    test rcx, rcx
    jz .lbad
    mov rbx, rdi
    mov r12, rsi
    mov r13d, edx
    mov r14, rcx
    cmp r13d, [rbx + CKPT_LAYERS_OFF]
    jae .lbad
    mov r8d, [rbx + CKPT_DIM_OFF]
    mov r9d, [rbx + CKPT_FFN_OFF]
    mov rax, r8
    mul r8
    test rdx, rdx
    jnz .lbad
    mov [rsp + 0], rax
    mov rax, [r12 + L_KV_DIM]
    mul r8
    test rdx, rdx
    jnz .lbad
    mov [rsp + 8], rax
    mov rax, r9
    mul r8
    test rdx, rdx
    jnz .lbad
    mov [rsp + 16], rax
    mov rax, r13
    mul qword [r12 + L_LAYER_STRIDE]
    test rdx, rdx
    jnz .lbad
    add rax, [r12 + L_LAYERS]
    jc .lbad
    mov r15, rax
    mov [r14 + LL_ATTN_NORM], r15
    add r15, r8
    jc .lbad
    mov [r14 + LL_WQ], r15
    add r15, [rsp + 0]
    jc .lbad
    mov [r14 + LL_WK], r15
    add r15, [rsp + 8]
    jc .lbad
    mov [r14 + LL_WV], r15
    add r15, [rsp + 8]
    jc .lbad
    mov [r14 + LL_WO], r15
    add r15, [rsp + 0]
    jc .lbad
    mov [r14 + LL_FFN_NORM], r15
    add r15, r8
    jc .lbad
    mov [r14 + LL_W_GATE], r15
    add r15, [rsp + 16]
    jc .lbad
    mov [r14 + LL_W_UP], r15
    add r15, [rsp + 16]
    jc .lbad
    mov [r14 + LL_W_DOWN], r15
    add r15, [rsp + 16]
    jc .lbad
    mov rax, r13
    inc rax
    mul qword [r12 + L_LAYER_STRIDE]
    test rdx, rdx
    jnz .lbad
    add rax, [r12 + L_LAYERS]
    jc .lbad
    cmp r15, rax
    jne .lbad
    xor eax, eax
    jmp .lreturn
.lbad:
    mov eax, 1
.lreturn:
    add rsp, 48
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
