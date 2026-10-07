BITS 64
DEFAULT REL

%include "model.inc"

extern sha256_bytes
extern sys_mmap_rw_anon
extern sys_munmap

global tokenizer_identity_sha256_mapped

section .rodata
tokenizer_identity_domain:
    db "NIYAH-TOKENIZER-V1"
tokenizer_identity_domain_len equ $ - tokenizer_identity_domain

section .text

; int tokenizer_identity_sha256_mapped(const void *tok, uint8_t out[32])
; Hashes the canonical semantic identity used by the native C runtime:
; domain || LE32(base_vocab) || LE32(vocab) || LE32(merge_count) || merge triples.
; eax=0 success, eax=1 invalid/overflow/hash failure, eax=2 mmap/munmap failure.
tokenizer_identity_sha256_mapped:
    push rbx
    push r12
    push r13
    push r14
    push r15

    test rdi, rdi
    jz .bad
    test rsi, rsi
    jz .bad

    mov r12, rdi
    mov r13, rsi

    cmp dword [r12 + TOK_BASE_VOCAB_OFF], TOK_BASE_VOCAB
    jne .bad

    mov eax, [r12 + TOK_VOCAB_OFF]
    cmp eax, TOK_BASE_VOCAB
    jb .bad
    mov edx, eax
    sub edx, TOK_BASE_VOCAB
    cmp edx, [r12 + TOK_MERGE_COUNT_OFF]
    jne .bad

    mov eax, [r12 + TOK_MERGE_COUNT_OFF]
    mov ecx, TOK_MERGE_SIZE
    mul rcx
    test rdx, rdx
    jnz .bad
    add rax, tokenizer_identity_domain_len + 12
    jc .bad
    mov r15, rax

    mov rdi, r15
    call sys_mmap_rw_anon wrt ..plt
    test rax, rax
    js .io
    mov r14, rax

    cld
    mov rdi, r14
    lea rsi, [rel tokenizer_identity_domain]
    mov rcx, tokenizer_identity_domain_len
    rep movsb

    lea rsi, [r12 + TOK_BASE_VOCAB_OFF]
    mov rcx, 12
    rep movsb

    lea rsi, [r12 + TOK_HEADER_SIZE]
    mov eax, [r12 + TOK_MERGE_COUNT_OFF]
    imul rax, TOK_MERGE_SIZE
    mov rcx, rax
    rep movsb

    mov rdi, r14
    mov rsi, r15
    mov rdx, r13
    call sha256_bytes wrt ..plt
    mov ebx, eax

    mov rdi, r14
    mov rsi, r15
    call sys_munmap wrt ..plt
    test rax, rax
    js .unmap_failed

    mov eax, ebx
    jmp .return

.unmap_failed:
    test ebx, ebx
    jnz .hash_failed
    mov eax, 2
    jmp .return

.hash_failed:
    mov eax, ebx
    jmp .return

.io:
    mov eax, 2
    jmp .return

.bad:
    mov eax, 1

.return:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
