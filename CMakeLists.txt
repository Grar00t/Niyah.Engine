cmake_minimum_required(VERSION 3.20)
project(NiyahEngine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(NIYAH_BUILD_TESTS "Build Niyah.Engine tests" ON)

add_library(niyah STATIC
    src/niyah_model.c
    src/niyah_math.c
)

target_include_directories(niyah
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(niyah PRIVATE /W4 /WX)
else()
    target_compile_options(niyah PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(NOT MSVC)
    target_link_libraries(niyah PUBLIC m)
endif()

if(NIYAH_BUILD_TESTS)
    enable_testing()

    add_executable(niyah_model_test tests/test_model.c)
    target_link_libraries(niyah_model_test PRIVATE niyah)

    if(MSVC)
        target_compile_options(niyah_model_test PRIVATE /W4 /WX)
    else()
        target_compile_options(niyah_model_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()

    add_test(NAME niyah_model_test COMMAND niyah_model_test)
endif()
