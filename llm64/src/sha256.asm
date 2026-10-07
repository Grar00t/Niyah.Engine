BITS 64
DEFAULT REL

global sha256_bytes

section .rodata
align 4
sha256_k:
    dd 0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5
    dd 0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5
    dd 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3
    dd 0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174
    dd 0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc
    dd 0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da
    dd 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7
    dd 0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967
    dd 0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13
    dd 0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85
    dd 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3
    dd 0xd192e819,0xd6990624,0xf40e3585,0x106aa070
    dd 0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5
    dd 0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3
    dd 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208
    dd 0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2

section .text

; int sha256_bytes(const uint8_t *data, size_t size, uint8_t out[32])
; eax=0 success, eax=1 invalid/bit-length overflow.
sha256_bytes:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub rsp, 456

    test rdi, rdi
    jz .bad
    test rdx, rdx
    jz .bad

    mov r12, rdi
    mov r13, rsi
    mov r14, rdx

    mov rax, r13
    shr rax, 61
    jnz .bad

    mov dword [rsp + 0],  0x6a09e667
    mov dword [rsp + 4],  0xbb67ae85
    mov dword [rsp + 8],  0x3c6ef372
    mov dword [rsp + 12], 0xa54ff53a
    mov dword [rsp + 16], 0x510e527f
    mov dword [rsp + 20], 0x9b05688c
    mov dword [rsp + 24], 0x1f83d9ab
    mov dword [rsp + 28], 0x5be0cd19

    xor r15d, r15d

.full_blocks:
    mov rax, r13
    sub rax, r15
    cmp rax, 64
    jb .final_blocks

    mov rdi, rsp
    lea rsi, [r12 + r15]
    lea rdx, [rsp + 32]
    call .transform
    add r15, 64
    jmp .full_blocks

.final_blocks:
    mov rbx, r13
    sub rbx, r15

    lea rdi, [rsp + 320]
    xor eax, eax
    mov ecx, 128
    rep stosb

    lea rdi, [rsp + 320]
    lea rsi, [r12 + r15]
    mov rcx, rbx
    rep movsb

    mov byte [rsp + 320 + rbx], 0x80

    mov rax, r13
    shl rax, 3
    bswap rax

    cmp rbx, 55
    ja .two_final
    mov [rsp + 320 + 56], rax

    mov rdi, rsp
    lea rsi, [rsp + 320]
    lea rdx, [rsp + 32]
    call .transform
    jmp .emit

.two_final:
    mov [rsp + 320 + 120], rax

    mov rdi, rsp
    lea rsi, [rsp + 320]
    lea rdx, [rsp + 32]
    call .transform

    mov rdi, rsp
    lea rsi, [rsp + 384]
    lea rdx, [rsp + 32]
    call .transform

.emit:
    xor ecx, ecx
.emit_loop:
    mov eax, [rsp + rcx*4]
    bswap eax
    mov [r14 + rcx*4], eax
    inc ecx
    cmp ecx, 8
    jb .emit_loop

    xor eax, eax
    jmp .return

.bad:
    mov eax, 1

.return:
    add rsp, 456
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; rdi=state[8] dwords, rsi=64-byte block, rdx=w[64] dwords.
.transform:
    push rbp
    push rbx

    mov rbp, rdi
    mov rbx, rdx

    xor ecx, ecx
.load16:
    mov eax, [rsi + rcx*4]
    bswap eax
    mov [rbx + rcx*4], eax
    inc ecx
    cmp ecx, 16
    jb .load16

.extend:
    cmp ecx, 64
    jae .init_work

    mov eax, [rbx + rcx*4 - 60]
    mov r8d, eax
    ror r8d, 7
    mov r9d, eax
    ror r9d, 18
    xor r8d, r9d
    shr eax, 3
    xor r8d, eax

    mov eax, [rbx + rcx*4 - 8]
    mov r9d, eax
    ror r9d, 17
    mov r10d, eax
    ror r10d, 19
    xor r9d, r10d
    shr eax, 10
    xor r9d, eax

    mov eax, [rbx + rcx*4 - 64]
    add eax, r8d
    add eax, [rbx + rcx*4 - 28]
    add eax, r9d
    mov [rbx + rcx*4], eax

    inc ecx
    jmp .extend

.init_work:
    mov eax, [rbp + 0]
    mov [rbx + 256], eax
    mov eax, [rbp + 4]
    mov [rbx + 260], eax
    mov eax, [rbp + 8]
    mov [rbx + 264], eax
    mov eax, [rbp + 12]
    mov [rbx + 268], eax
    mov eax, [rbp + 16]
    mov [rbx + 272], eax
    mov eax, [rbp + 20]
    mov [rbx + 276], eax
    mov eax, [rbp + 24]
    mov [rbx + 280], eax
    mov eax, [rbp + 28]
    mov [rbx + 284], eax

    lea rdi, [rel sha256_k]
    xor esi, esi

.round:
    mov eax, [rbx + 272]
    mov ecx, eax
    ror ecx, 6
    mov edx, eax
    ror edx, 11
    xor ecx, edx
    mov edx, eax
    ror edx, 25
    xor ecx, edx

    mov r8d, [rbx + 276]
    mov r9d, [rbx + 280]
    mov r10d, eax
    and r10d, r8d
    not eax
    and eax, r9d
    xor r10d, eax

    add ecx, r10d
    add ecx, [rbx + 284]
    add ecx, [rdi + rsi*4]
    add ecx, [rbx + rsi*4]

    mov eax, [rbx + 256]
    mov edx, eax
    ror edx, 2
    mov r8d, eax
    ror r8d, 13
    xor edx, r8d
    mov r8d, eax
    ror r8d, 22
    xor edx, r8d

    mov r8d, [rbx + 260]
    mov r9d, [rbx + 264]
    mov r10d, eax
    and r10d, r8d
    mov r11d, eax
    and r11d, r9d
    xor r10d, r11d
    mov r11d, r8d
    and r11d, r9d
    xor r10d, r11d
    add edx, r10d

    mov r10d, [rbx + 280]
    mov [rbx + 284], r10d
    mov r10d, [rbx + 276]
    mov [rbx + 280], r10d
    mov r10d, [rbx + 272]
    mov [rbx + 276], r10d
    mov r10d, [rbx + 268]
    add r10d, ecx
    mov [rbx + 272], r10d
    mov r10d, [rbx + 264]
    mov [rbx + 268], r10d
    mov r10d, [rbx + 260]
    mov [rbx + 264], r10d
    mov r10d, [rbx + 256]
    mov [rbx + 260], r10d
    add ecx, edx
    mov [rbx + 256], ecx

    inc esi
    cmp esi, 64
    jb .round

    xor ecx, ecx
.add_state:
    mov eax, [rbx + 256 + rcx*4]
    add [rbp + rcx*4], eax
    inc ecx
    cmp ecx, 8
    jb .add_state

    pop rbx
    pop rbp
    ret
