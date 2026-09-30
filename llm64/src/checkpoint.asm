BITS 64
DEFAULT REL

%include "model.inc"

extern sys_open_ro
extern sys_close
extern sys_fstat_size
extern sys_mmap_ro
extern sys_munmap
extern crc32_ieee

global checkpoint_validate_v2

section .text

; rdi = checkpoint path
; eax = 0 success, 1 format/corruption, 2 I/O
checkpoint_validate_v2:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    call    sys_open_ro
    test    rax, rax
    js      .io_no_fd
    mov     r12, rax

    mov     rdi, r12
    call    sys_fstat_size
    test    rax, rax
    js      .io_close
    mov     r13, rax
    cmp     r13, CKPT_HEADER_SIZE + FILE_FOOTER_SIZE
    jb      .format_close

    mov     rdi, r12
    mov     rsi, r13
    call    sys_mmap_ro
    test    rax, rax
    js      .io_close
    mov     r14, rax

    mov     rdi, r12
    call    sys_close

    lea     r15, [r14 + r13 - FILE_FOOTER_SIZE]

    mov     rax, [r14 + CKPT_MAGIC_OFF]
    mov     rcx, 0x504b43484159494e      ; "NIYAHCKP" little-endian
    cmp     rax, rcx
    jne     .format_unmap
    cmp     dword [r14 + CKPT_VERSION_OFF], CKPT_VERSION_V2
    jne     .format_unmap
    cmp     dword [r14 + CKPT_FLAGS_OFF], 0
    jne     .format_unmap
    cmp     dword [r14 + CKPT_RESERVED_OFF], 0
    jne     .format_unmap
    mov     eax, [r14 + CKPT_SECTION_COUNT_OFF]
    cmp     eax, 5
    jb      .format_unmap
    cmp     eax, CKPT_MAX_SECTIONS
    ja      .format_unmap

    cmp     dword [r15], CHECKSUM_CRC32
    jne     .format_unmap
    mov     rdi, r14
    mov     rsi, r13
    sub     rsi, FILE_FOOTER_SIZE
    call    crc32_ieee
    cmp     eax, [r15 + 4]
    jne     .format_unmap

    mov     rdi, r14
    call    .calc_weight_count
    test    rax, rax
    jz      .format_unmap
    cmp     rax, [r14 + CKPT_WEIGHT_COUNT_OFF]
    jne     .format_unmap

    mov     r11d, [r14 + CKPT_SECTION_COUNT_OFF]
    xor     r10d, r10d
    lea     r8, [r14 + CKPT_HEADER_SIZE]
    xor     ebx, ebx                    ; seen known-section bitmask

.section_loop:
    cmp     r10d, r11d
    jae     .sections_done

    mov     rax, r15
    sub     rax, r8
    cmp     rax, CKPT_SECTION_HEADER
    jb      .format_unmap

    mov     ecx, [r8 + 4]               ; section flags
    test    ecx, ~CKPT_SECTION_REQUIRED
    jnz     .format_unmap
    mov     edx, [r8]                   ; section id
    mov     r9, [r8 + 8]                ; payload bytes
    add     r8, CKPT_SECTION_HEADER

    mov     rax, r15
    sub     rax, r8
    cmp     r9, rax
    ja      .format_unmap

    cmp     edx, CKPT_SECTION_MODEL
    je      .section_model
    cmp     edx, CKPT_SECTION_ADAMW_M
    je      .section_adamw_m
    cmp     edx, CKPT_SECTION_ADAMW_V
    je      .section_adamw_v
    cmp     edx, CKPT_SECTION_ADAMW_META
    je      .section_adamw_meta
    cmp     edx, CKPT_SECTION_TOKENIZER_IDENTITY
    je      .section_tokenizer_identity
    cmp     edx, CKPT_SECTION_LR_SCHEDULE
    je      .section_lr_schedule

    ; Forward-compatible unknown sections are allowed only when optional.
    test    ecx, CKPT_SECTION_REQUIRED
    jnz     .format_unmap
    jmp     .section_advance

.section_model:
    test    ecx, CKPT_SECTION_REQUIRED
    jz      .format_unmap
    test    ebx, 0x01
    jnz     .format_unmap
    or      ebx, 0x01
    call    .require_tensor_bytes
    test    eax, eax
    jnz     .format_unmap

    ; Reject NaN and infinity in model weights.
    mov     rcx, r9
    shr     rcx, 2
    mov     rax, r8
.weight_scan:
    test    rcx, rcx
    jz      .section_advance
    mov     edx, [rax]
    and     edx, 0x7f800000
    cmp     edx, 0x7f800000
    je      .format_unmap
    add     rax, 4
    dec     rcx
    jmp     .weight_scan

.section_adamw_m:
    test    ecx, CKPT_SECTION_REQUIRED
    jz      .format_unmap
    test    ebx, 0x02
    jnz     .format_unmap
    or      ebx, 0x02
    call    .require_tensor_bytes
    test    eax, eax
    jnz     .format_unmap
    jmp     .section_advance

.section_adamw_v:
    test    ecx, CKPT_SECTION_REQUIRED
    jz      .format_unmap
    test    ebx, 0x04
    jnz     .format_unmap
    or      ebx, 0x04
    call    .require_tensor_bytes
    test    eax, eax
    jnz     .format_unmap
    jmp     .section_advance

.section_adamw_meta:
    test    ecx, CKPT_SECTION_REQUIRED
    jz      .format_unmap
    test    ebx, 0x08
    jnz     .format_unmap
    cmp     r9, CKPT_ADAMW_META_SIZE
    jne     .format_unmap
    or      ebx, 0x08
    jmp     .section_advance

.section_tokenizer_identity:
    test    ecx, CKPT_SECTION_REQUIRED
    jz      .format_unmap
    test    ebx, 0x10
    jnz     .format_unmap
    cmp     r9, CKPT_TOKENIZER_IDENTITY_SIZE
    jne     .format_unmap
    or      ebx, 0x10
    jmp     .section_advance

.section_lr_schedule:
    test    ecx, CKPT_SECTION_REQUIRED
    jz      .format_unmap
    test    ebx, 0x20
    jnz     .format_unmap
    cmp     r9, CKPT_LR_SCHEDULE_SIZE
    jne     .format_unmap
    cmp     qword [r8], 0
    je      .format_unmap
    or      ebx, 0x20

.section_advance:
    add     r8, r9
    inc     r10d
    jmp     .section_loop

.sections_done:
    cmp     r8, r15
    jne     .format_unmap
    cmp     ebx, CKPT_SEEN_V2_REQUIRED
    je      .sections_valid
    cmp     ebx, CKPT_SEEN_V2_WITH_SCHEDULE
    jne     .format_unmap
.sections_valid:

    mov     rdi, r14
    mov     rsi, r13
    call    sys_munmap
    xor     eax, eax
    jmp     .return

; r8 points to current payload, r9 is payload size.
; eax=0 when payload size equals canonical model tensor bytes, 1 otherwise.
.require_tensor_bytes:
    mov     rax, [r14 + CKPT_WEIGHT_COUNT_OFF]
    mov     rdx, rax
    shr     rdx, 62
    jnz     .tensor_bad
    shl     rax, 2
    cmp     r9, rax
    jne     .tensor_bad
    xor     eax, eax
    ret
.tensor_bad:
    mov     eax, 1
    ret

.format_close:
    mov     rdi, r12
    call    sys_close
    mov     eax, 1
    jmp     .return

.format_unmap:
    mov     rdi, r14
    mov     rsi, r13
    call    sys_munmap
    mov     eax, 1
    jmp     .return

.io_close:
    mov     rdi, r12
    call    sys_close
.io_no_fd:
    mov     eax, 2

.return:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; rdi = mapped 72-byte current checkpoint header
; rax = canonical FP32 weight count, or 0 on invalid geometry/overflow
.calc_weight_count:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 24

    mov     rbx, rdi
    mov     r8d,  [rbx + CKPT_VOCAB_OFF]
    mov     eax,  [rbx + CKPT_CONTEXT_OFF]
    mov     r9d,  [rbx + CKPT_DIM_OFF]
    mov     r10d, [rbx + CKPT_LAYERS_OFF]
    mov     r11d, [rbx + CKPT_HEADS_OFF]
    mov     r12d, [rbx + CKPT_KV_HEADS_OFF]
    mov     r13d, [rbx + CKPT_FFN_OFF]
    mov     r14d, [rbx + CKPT_SEGMENTS_OFF]
    mov     r15d, [rbx + CKPT_TIE_OFF]

    cmp     r8, 2
    jb      .count_bad
    test    eax, eax
    jz      .count_bad
    test    r9, r9
    jz      .count_bad
    test    r10, r10
    jz      .count_bad
    test    r11, r11
    jz      .count_bad
    test    r12, r12
    jz      .count_bad
    test    r13, r13
    jz      .count_bad
    cmp     r12, r11
    ja      .count_bad
    cmp     r15, 1
    ja      .count_bad

    mov     eax, [rbx + CKPT_RMS_EPS_OFF]
    test    eax, 0x80000000
    jnz     .count_bad
    mov     edx, eax
    and     edx, 0x7fffffff
    jz      .count_bad
    mov     edx, eax
    and     edx, 0x7f800000
    cmp     edx, 0x7f800000
    je      .count_bad

    mov     rax, r9
    xor     edx, edx
    div     r11
    test    rdx, rdx
    jnz     .count_bad
    cmp     rax, 2
    jb      .count_bad
    test    rax, 1
    jnz     .count_bad
    mov     rsi, rax                    ; head_dim

    mov     rax, r11
    xor     edx, edx
    div     r12
    test    rdx, rdx
    jnz     .count_bad

    ; token_embedding count
    mov     rax, r8
    mul     r9
    test    rdx, rdx
    jnz     .count_bad
    mov     [rsp], rax
    mov     rbx, rax

    ; optional segment_embedding count
    mov     rax, r14
    mul     r9
    test    rdx, rdx
    jnz     .count_bad
    add     rbx, rax
    jc      .count_bad

    ; q_count = dim * dim
    mov     rax, r9
    mul     r9
    test    rdx, rdx
    jnz     .count_bad
    mov     [rsp + 8], rax

    ; kv_count = head_dim * n_kv_heads * dim
    mov     rax, rsi
    mul     r12
    test    rdx, rdx
    jnz     .count_bad
    mul     r9
    test    rdx, rdx
    jnz     .count_bad
    mov     [rsp + 16], rax

    ; each FFN matrix has ffn * dim elements
    mov     rax, r13
    mul     r9
    test    rdx, rdx
    jnz     .count_bad
    mov     rsi, rax

    ; layer_stride = 2*dim + 2*q + 2*kv + 3*ffn_matrix
    xor     rcx, rcx
    add     rcx, r9
    jc      .count_bad
    add     rcx, r9
    jc      .count_bad
    mov     rax, [rsp + 8]
    add     rcx, rax
    jc      .count_bad
    add     rcx, rax
    jc      .count_bad
    mov     rax, [rsp + 16]
    add     rcx, rax
    jc      .count_bad
    add     rcx, rax
    jc      .count_bad
    add     rcx, rsi
    jc      .count_bad
    add     rcx, rsi
    jc      .count_bad
    add     rcx, rsi
    jc      .count_bad

    mov     rax, rcx
    mul     r10
    test    rdx, rdx
    jnz     .count_bad
    add     rbx, rax
    jc      .count_bad

    add     rbx, r9                    ; final_norm
    jc      .count_bad

    test    r15, r15
    jnz     .count_ok
    mov     rax, [rsp]                 ; separate lm_head
    add     rbx, rax
    jc      .count_bad

.count_ok:
    mov     rax, rbx
    jmp     .count_return

.count_bad:
    xor     eax, eax

.count_return:
    add     rsp, 24
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
