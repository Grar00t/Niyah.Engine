BITS 64
DEFAULT REL

%include "model.inc"
%include "model_map.inc"
%include "layout.inc"

extern sys_open_ro
extern sys_close
extern sys_fstat_size
extern sys_mmap_ro
extern sys_munmap
extern crc32_ieee
extern model_layout_from_header

global checkpoint_validate_v2
global checkpoint_map_model_v2
global checkpoint_unmap_model

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


; ===========================================================================
; int checkpoint_map_model_v2(const char *path, ModelMap *out)
;
; Keeps the validated V2 mapping alive for inference and exposes Section 1.
;
; eax:
;   0 = success
;   1 = format/corruption
;   2 = I/O
; ===========================================================================

checkpoint_map_model_v2:
    push    rbp
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 88

    test    rdi, rdi
    jz      .map_format
    test    rsi, rsi
    jz      .map_format

    mov     r12, rdi                ; path
    mov     r13, rsi                ; ModelMap*

    ; Fail closed: clear the output handle first.
    xor     eax, eax
    mov     [r13 + MM_MAP_BASE], rax
    mov     [r13 + MM_MAP_BYTES], rax
    mov     [r13 + MM_HEADER], rax
    mov     [r13 + MM_WEIGHTS], rax
    mov     [r13 + MM_WEIGHT_BYTES], rax
    mov     [r13 + MM_TOKENIZER_IDENTITY], rax

    ; Open once. Every validation below is performed on the exact mapping
    ; retained for inference, eliminating the validate/reopen TOCTOU window.
    mov     rdi, r12
    call    sys_open_ro
    test    rax, rax
    js      .map_io
    mov     r14, rax

    mov     rdi, r14
    call    sys_fstat_size
    test    rax, rax
    js      .map_io_close
    mov     r15, rax
    cmp     r15, CKPT_HEADER_SIZE + FILE_FOOTER_SIZE
    jb      .map_format_close

    mov     rdi, r14
    mov     rsi, r15
    call    sys_mmap_ro
    test    rax, rax
    js      .map_io_close
    mov     rbx, rax

    mov     rdi, r14
    call    sys_close

    ; Header contract.
    mov     rax, [rbx + CKPT_MAGIC_OFF]
    mov     rcx, 0x504b43484159494e
    cmp     rax, rcx
    jne     .map_bad_unmap
    cmp     dword [rbx + CKPT_VERSION_OFF], CKPT_VERSION_V2
    jne     .map_bad_unmap
    cmp     dword [rbx + CKPT_FLAGS_OFF], 0
    jne     .map_bad_unmap
    cmp     dword [rbx + CKPT_RESERVED_OFF], 0
    jne     .map_bad_unmap

    mov     eax, [rbx + CKPT_SECTION_COUNT_OFF]
    cmp     eax, 5
    jb      .map_bad_unmap
    cmp     eax, CKPT_MAX_SECTIONS
    ja      .map_bad_unmap

    ; Footer and CRC on the retained mapping.
    lea     rbp, [rbx + r15 - FILE_FOOTER_SIZE]
    cmp     dword [rbp], CHECKSUM_CRC32
    jne     .map_bad_unmap
    mov     rdi, rbx
    mov     rsi, r15
    sub     rsi, FILE_FOOTER_SIZE
    call    crc32_ieee
    cmp     eax, [rbp + 4]
    jne     .map_bad_unmap

    ; Canonical geometry/layout must reproduce header weight_count.
    mov     rdi, rbx
    mov     rsi, rsp
    call    model_layout_from_header
    test    eax, eax
    jnz     .map_bad_unmap
    mov     rax, [rsp + L_TOTAL_FLOATS]
    cmp     rax, [rbx + CKPT_WEIGHT_COUNT_OFF]
    jne     .map_bad_unmap

    ; Layout scratch is no longer needed; reuse tail slots as validation locals.
    xor     eax, eax
    mov     [rsp + 56], rax         ; tokenizer identity payload
    mov     [rsp + 64], rax         ; seen-section bitmask
    mov     [rsp + 72], rax         ; model payload
    mov     [rsp + 80], rax         ; model payload bytes

    mov     r10d, [rbx + CKPT_SECTION_COUNT_OFF]
    xor     r12d, r12d
    lea     r8, [rbx + CKPT_HEADER_SIZE]

.map_scan:
    cmp     r12d, r10d
    jae     .map_scan_done

    mov     rax, rbp
    sub     rax, r8
    cmp     rax, CKPT_SECTION_HEADER
    jb      .map_bad_unmap

    mov     edx, [r8]
    mov     esi, [r8 + 4]
    mov     r9, [r8 + 8]
    add     r8, CKPT_SECTION_HEADER

    test    esi, ~CKPT_SECTION_REQUIRED
    jnz     .map_bad_unmap

    mov     rax, rbp
    sub     rax, r8
    cmp     r9, rax
    ja      .map_bad_unmap

    cmp     edx, CKPT_SECTION_MODEL
    je      .map_model
    cmp     edx, CKPT_SECTION_ADAMW_M
    je      .map_adamw_m
    cmp     edx, CKPT_SECTION_ADAMW_V
    je      .map_adamw_v
    cmp     edx, CKPT_SECTION_ADAMW_META
    je      .map_adamw_meta
    cmp     edx, CKPT_SECTION_TOKENIZER_IDENTITY
    je      .map_tokenizer_identity
    cmp     edx, CKPT_SECTION_LR_SCHEDULE
    je      .map_lr_schedule

    test    esi, CKPT_SECTION_REQUIRED
    jnz     .map_bad_unmap
    jmp     .map_advance

.map_model:
    test    esi, CKPT_SECTION_REQUIRED
    jz      .map_bad_unmap
    mov     eax, [rsp + 64]
    test    eax, 0x01
    jnz     .map_bad_unmap
    or      eax, 0x01
    mov     [rsp + 64], eax

    mov     rax, [rbx + CKPT_WEIGHT_COUNT_OFF]
    mov     rdx, rax
    shr     rdx, 62
    jnz     .map_bad_unmap
    shl     rax, 2
    cmp     r9, rax
    jne     .map_bad_unmap

    mov     [rsp + 72], r8
    mov     [rsp + 80], r9

    ; Match the canonical validator: model weights must all be finite.
    mov     rcx, r9
    shr     rcx, 2
    mov     rax, r8
.map_weight_scan:
    test    rcx, rcx
    jz      .map_advance
    mov     edx, [rax]
    and     edx, 0x7f800000
    cmp     edx, 0x7f800000
    je      .map_bad_unmap
    add     rax, 4
    dec     rcx
    jmp     .map_weight_scan

.map_adamw_m:
    test    esi, CKPT_SECTION_REQUIRED
    jz      .map_bad_unmap
    mov     eax, [rsp + 64]
    test    eax, 0x02
    jnz     .map_bad_unmap
    or      eax, 0x02
    mov     [rsp + 64], eax
    jmp     .map_require_tensor

.map_adamw_v:
    test    esi, CKPT_SECTION_REQUIRED
    jz      .map_bad_unmap
    mov     eax, [rsp + 64]
    test    eax, 0x04
    jnz     .map_bad_unmap
    or      eax, 0x04
    mov     [rsp + 64], eax

.map_require_tensor:
    mov     rax, [rbx + CKPT_WEIGHT_COUNT_OFF]
    mov     rdx, rax
    shr     rdx, 62
    jnz     .map_bad_unmap
    shl     rax, 2
    cmp     r9, rax
    jne     .map_bad_unmap
    jmp     .map_advance

.map_adamw_meta:
    test    esi, CKPT_SECTION_REQUIRED
    jz      .map_bad_unmap
    mov     eax, [rsp + 64]
    test    eax, 0x08
    jnz     .map_bad_unmap
    or      eax, 0x08
    mov     [rsp + 64], eax
    cmp     r9, CKPT_ADAMW_META_SIZE
    jne     .map_bad_unmap
    jmp     .map_advance

.map_tokenizer_identity:
    test    esi, CKPT_SECTION_REQUIRED
    jz      .map_bad_unmap
    mov     eax, [rsp + 64]
    test    eax, 0x10
    jnz     .map_bad_unmap
    or      eax, 0x10
    mov     [rsp + 64], eax
    cmp     r9, CKPT_TOKENIZER_IDENTITY_SIZE
    jne     .map_bad_unmap
    mov     [rsp + 56], r8
    jmp     .map_advance

.map_lr_schedule:
    test    esi, CKPT_SECTION_REQUIRED
    jz      .map_bad_unmap
    mov     eax, [rsp + 64]
    test    eax, 0x20
    jnz     .map_bad_unmap
    or      eax, 0x20
    mov     [rsp + 64], eax
    cmp     r9, CKPT_LR_SCHEDULE_SIZE
    jne     .map_bad_unmap
    cmp     qword [r8], 0
    je      .map_bad_unmap

.map_advance:
    add     r8, r9
    jc      .map_bad_unmap
    inc     r12d
    jmp     .map_scan

.map_scan_done:
    cmp     r8, rbp
    jne     .map_bad_unmap

    mov     eax, [rsp + 64]
    cmp     eax, CKPT_SEEN_V2_REQUIRED
    je      .map_sections_valid
    cmp     eax, CKPT_SEEN_V2_WITH_SCHEDULE
    jne     .map_bad_unmap

.map_sections_valid:
    cmp     qword [rsp + 72], 0
    je      .map_bad_unmap
    cmp     qword [rsp + 56], 0
    je      .map_bad_unmap

    mov     [r13 + MM_MAP_BASE], rbx
    mov     [r13 + MM_MAP_BYTES], r15
    mov     [r13 + MM_HEADER], rbx

    mov     rax, [rsp + 72]
    mov     [r13 + MM_WEIGHTS], rax
    mov     rax, [rsp + 80]
    mov     [r13 + MM_WEIGHT_BYTES], rax
    mov     rax, [rsp + 56]
    mov     [r13 + MM_TOKENIZER_IDENTITY], rax

    xor     eax, eax
    jmp     .map_return

.map_bad_unmap:
    mov     rdi, rbx
    mov     rsi, r15
    call    sys_munmap
    mov     eax, 1
    jmp     .map_return

.map_format_close:
    mov     rdi, r14
    call    sys_close
.map_format:
    mov     eax, 1
    jmp     .map_return

.map_io_close:
    mov     rdi, r14
    call    sys_close
.map_io:
    mov     eax, 2

.map_return:
    add     rsp, 88
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    pop     rbp
    ret

; ===========================================================================
; int checkpoint_unmap_model(ModelMap *map)
; Safe no-op for an already-empty map.
; ===========================================================================

checkpoint_unmap_model:
    push    r12

    test    rdi, rdi
    jz      .unmap_bad

    mov     r12, rdi
    mov     rax, [r12 + MM_MAP_BASE]
    test    rax, rax
    jz      .unmap_clear

    mov     rsi, [r12 + MM_MAP_BYTES]
    test    rsi, rsi
    jz      .unmap_bad

    mov     rdi, rax
    call    sys_munmap
    test    rax, rax
    js      .unmap_io

.unmap_clear:
    xor     eax, eax
    mov     [r12 + MM_MAP_BASE], rax
    mov     [r12 + MM_MAP_BYTES], rax
    mov     [r12 + MM_HEADER], rax
    mov     [r12 + MM_WEIGHTS], rax
    mov     [r12 + MM_WEIGHT_BYTES], rax
    mov     [r12 + MM_TOKENIZER_IDENTITY], rax

    pop     r12
    ret

.unmap_bad:
    mov     eax, 1
    pop     r12
    ret

.unmap_io:
    mov     eax, 2
    pop     r12
    ret
