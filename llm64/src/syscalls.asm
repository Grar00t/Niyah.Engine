BITS 64
DEFAULT REL

%include "syscalls.inc"

global sys_open_ro
global sys_close
global sys_fstat_size
global sys_mmap_ro
global sys_mmap_rw_anon
global sys_munmap
global sys_write_all
global sys_exit

section .text

; rdi = path -> rax = fd or -errno
sys_open_ro:
    mov     rsi, rdi
    mov     edi, AT_FDCWD
    xor     edx, edx
    xor     r10d, r10d
    mov     eax, SYS_openat
    syscall
    ret

; rdi = fd -> rax = 0 or -errno
sys_close:
    mov     eax, SYS_close
    syscall
    ret

; rdi = fd -> rax = size or -errno
sys_fstat_size:
    sub     rsp, STAT_BUF_SIZE
    mov     rsi, rsp
    mov     eax, SYS_fstat
    syscall
    test    rax, rax
    js      .done
    mov     rax, [rsp + STAT_SIZE_OFF]
.done:
    add     rsp, STAT_BUF_SIZE
    ret

; rdi = fd, rsi = size -> rax = mapping or -errno
sys_mmap_ro:
    mov     r8, rdi
    xor     edi, edi
    mov     edx, PROT_READ
    mov     r10d, MAP_PRIVATE
    xor     r9d, r9d
    mov     eax, SYS_mmap
    syscall
    ret

; rdi = size -> rax = RW anonymous mapping or -errno
sys_mmap_rw_anon:
    mov     rsi, rdi
    xor     edi, edi
    mov     edx, PROT_READ | PROT_WRITE
    mov     r10d, MAP_PRIVATE | MAP_ANONYMOUS
    mov     r8, -1
    xor     r9d, r9d
    mov     eax, SYS_mmap
    syscall
    ret

; rdi = address, rsi = size -> rax = 0 or -errno
sys_munmap:
    mov     eax, SYS_munmap
    syscall
    ret

; rdi = fd, rsi = buffer, rdx = bytes -> rax = 0 or -errno
sys_write_all:
.loop:
    test    rdx, rdx
    jz      .ok
    mov     eax, SYS_write
    syscall
    cmp     rax, -EINTR
    je      .loop
    test    rax, rax
    jle     .ret
    add     rsi, rax
    sub     rdx, rax
    jmp     .loop
.ok:
    xor     eax, eax
.ret:
    ret

; rdi = exit status, does not return
sys_exit:
    mov     eax, SYS_exit
    syscall
    ud2
