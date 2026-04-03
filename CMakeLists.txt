cmake_minimum_required(VERSION 3.10)
project(Khawrizm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ARM64 specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_FLAGS "-march=armv8-a+simd")
    message(STATUS "Building for ARM64 with NEON")
else()
    set(ARCH_FLAGS "-march=native")
    message(STATUS "Building for x86_64")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS} -Wall -Wextra -O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/Downloads)
include_directories(${CMAKE_SOURCE_DIR}/Downloads/include)

# NIYAH inference engine
add_executable(niyah 
    ${CMAKE_SOURCE_DIR}/Downloads/niyah_infer.c
    ${CMAKE_SOURCE_DIR}/Downloads/niyah_gguf.c
    ${CMAKE_SOURCE_DIR}/Downloads/niyah_simd.c
    ${CMAKE_SOURCE_DIR}/Downloads/niyah_tokenizer.c
)
target_link_libraries(niyah m pthread)

# SIMD test
add_executable(niyah_test
    ${CMAKE_SOURCE_DIR}/Downloads/test_simd.c
    ${CMAKE_SOURCE_DIR}/Downloads/niyah_simd.c
)
target_link_libraries(niyah_test m pthread)

# Phalanx Gate (userspace demo)
add_executable(phalanx_gate
    ${CMAKE_SOURCE_DIR}/Downloads/phalanx_gate.c
)
target_link_libraries(phalanx_gate m)
