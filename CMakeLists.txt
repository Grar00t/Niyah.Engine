cmake_minimum_required(VERSION 3.16)
project(niyah-kernel VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ── Compiler flags ────────────────────────────────────────────────
add_compile_options(-Wall -Wextra -Wcast-align -Wstrict-prototypes)

# ── Platform / SIMD detection ─────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(-march=armv8.2-a+simd)
    message(STATUS "[niyah] Target: ARM64 + NEON")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    add_compile_options(-march=native -mavx2 -mfma)
    message(STATUS "[niyah] Target: x86_64 + AVX2")
else()
    message(STATUS "[niyah] Target: scalar fallback")
endif()

# ── Release / Debug flags ─────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG   "-O1 -g -fsanitize=address,undefined")

# ── Source layout ─────────────────────────────────────────────────
set(SRC_DIR  ${CMAKE_SOURCE_DIR}/niyah/src)
set(INC_DIR  ${CMAKE_SOURCE_DIR}/niyah/include)

include_directories(${INC_DIR})

# ── SIMD primitives library ───────────────────────────────────────
add_library(niyah_simd STATIC
    ${SRC_DIR}/niyah_simd.c
)
target_include_directories(niyah_simd PUBLIC ${INC_DIR})
target_link_libraries(niyah_simd m)

# ── GGUF loader library ───────────────────────────────────────────
add_library(niyah_gguf STATIC
    ${SRC_DIR}/niyah_gguf.c
)
target_include_directories(niyah_gguf PUBLIC ${INC_DIR})

# ── SIMD unit test ────────────────────────────────────────────────
add_executable(niyah_test_simd
    ${SRC_DIR}/test_simd.c
)
target_link_libraries(niyah_test_simd niyah_simd m)

# ── Install ───────────────────────────────────────────────────────
install(TARGETS niyah_simd niyah_gguf
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${INC_DIR}/
    DESTINATION include/niyah
    FILES_MATCHING PATTERN "*.h"
)
