BITS 64
DEFAULT REL

%include "model.inc"
%include "layout.inc"
%include "runtime.inc"

extern model_layout_from_header
extern sys_mmap_rw_anon
extern sys_munmap

global runtime_memory_create
global runtime_memory_destroy

section .text

; int runtime_memory_create(const header*, RuntimeMemory*)
; eax=0 success, 1 invalid/overflow, 2 mmap failure
runtime_memory_create:
    push rbx
    push r12
    push r13
    sub rsp, L_SIZE

    test rdi, rdi
    jz .bad
    test rsi, rsi
    jz .bad

    mov r12, rdi
    mov r13, rsi

    xor eax, eax
    mov [r13 + RT_KEYS], rax
    mov [r13 + RT_VALUES], rax
    mov [r13 + RT_KV_BYTES], rax
    mov [r13 + RT_WORKSPACE], rax
    mov [r13 + RT_WORKSPACE_BYTES], rax
    mov [r13 + RT_LOGITS], rax
    mov [r13 + RT_LOGITS_BYTES], rax

    mov rdi, r12
    mov rsi, rsp
    call model_layout_from_header wrt ..plt
    test eax, eax
    jnz .bad

    ; KV floats/tensor =
    ; layers * context * kv_dim
    mov eax, [r12 + CKPT_LAYERS_OFF]
    mov ebx, [r12 + CKPT_CONTEXT_OFF]
    mul rbx
    test rdx, rdx
    jnz .bad

    mul qword [rsp + L_KV_DIM]
    test rdx, rdx
    jnz .bad

    mov rdx, rax
    shr rdx, 62
    jnz .bad
    shl rax, 2
    mov [r13 + RT_KV_BYTES], rax

    ; keys
    mov rdi, rax
    call sys_mmap_rw_anon wrt ..plt
    test rax, rax
    js .map_fail
    mov [r13 + RT_KEYS], rax

    ; values
    mov rdi, [r13 + RT_KV_BYTES]
    call sys_mmap_rw_anon wrt ..plt
    test rax, rax
    js .map_fail
    mov [r13 + RT_VALUES], rax

    ; workspace floats =
    ; 5*dim + 2*kv_dim + 2*ffn + context
    mov eax, [r12 + CKPT_DIM_OFF]
    imul rax, 5

    mov rcx, [rsp + L_KV_DIM]
    shl rcx, 1
    add rax, rcx
    jc .bad_cleanup

    mov ecx, [r12 + CKPT_FFN_OFF]
    shl rcx, 1
    add rax, rcx
    jc .bad_cleanup

    mov ecx, [r12 + CKPT_CONTEXT_OFF]
    add rax, rcx
    jc .bad_cleanup

    mov rdx, rax
    shr rdx, 62
    jnz .bad_cleanup

    shl rax, 2
    mov [r13 + RT_WORKSPACE_BYTES], rax

    mov rdi, rax
    call sys_mmap_rw_anon wrt ..plt
    test rax, rax
    js .map_fail
    mov [r13 + RT_WORKSPACE], rax

    ; logits = vocab * sizeof(float)
    mov eax, [r12 + CKPT_VOCAB_OFF]
    mov rdx, rax
    shr rdx, 62
    jnz .bad_cleanup

    shl rax, 2
    mov [r13 + RT_LOGITS_BYTES], rax

    mov rdi, rax
    call sys_mmap_rw_anon wrt ..plt
    test rax, rax
    js .map_fail
    mov [r13 + RT_LOGITS], rax

    xor eax, eax
    jmp .ret

.map_fail:
    mov rbx, 2
    jmp .cleanup

.bad_cleanup:
    mov rbx, 1

.cleanup:
    mov rdi, r13
    call runtime_memory_destroy
    mov rax, rbx
    jmp .ret

.bad:
    mov eax, 1

.ret:
    add rsp, L_SIZE
    pop r13
    pop r12
    pop rbx
    ret


runtime_memory_destroy:
    push rbx
    push r12

    test rdi, rdi
    jz .destroy_bad

    mov r12, rdi

    mov rax, [r12 + RT_KEYS]
    test rax, rax
    jz .no_keys
    mov rdi, rax
    mov rsi, [r12 + RT_KV_BYTES]
    call sys_munmap wrt ..plt
.no_keys:

    mov rax, [r12 + RT_VALUES]
    test rax, rax
    jz .no_values
    mov rdi, rax
    mov rsi, [r12 + RT_KV_BYTES]
    call sys_munmap wrt ..plt
.no_values:

    mov rax, [r12 + RT_WORKSPACE]
    test rax, rax
    jz .no_workspace
    mov rdi, rax
    mov rsi, [r12 + RT_WORKSPACE_BYTES]
    call sys_munmap wrt ..plt
.no_workspace:

    mov rax, [r12 + RT_LOGITS]
    test rax, rax
    jz .clear
    mov rdi, rax
    mov rsi, [r12 + RT_LOGITS_BYTES]
    call sys_munmap wrt ..plt

.clear:
    xor eax, eax
    mov [r12 + RT_KEYS], rax
    mov [r12 + RT_VALUES], rax
    mov [r12 + RT_KV_BYTES], rax
    mov [r12 + RT_WORKSPACE], rax
    mov [r12 + RT_WORKSPACE_BYTES], rax
    mov [r12 + RT_LOGITS], rax
    mov [r12 + RT_LOGITS_BYTES], rax

    pop r12
    pop rbx
    ret

.destroy_bad:
    mov eax, 1
    pop r12
    pop rbx
    ret
