%ifndef NIYAH_LLM64_SYSCALLS_INC
%define NIYAH_LLM64_SYSCALLS_INC 1

%define SYS_read       0
%define SYS_write      1
%define SYS_close      3
%define SYS_fstat      5
%define SYS_mmap       9
%define SYS_munmap     11
%define SYS_exit       60
%define SYS_openat     257

%define AT_FDCWD       -100
%define O_RDONLY       0
%define PROT_READ      1
%define PROT_WRITE     2
%define MAP_PRIVATE    2
%define MAP_ANONYMOUS  0x20

%define EINTR          4
%define STAT_SIZE_OFF  48
%define STAT_BUF_SIZE  160

%endif
