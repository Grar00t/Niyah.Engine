cmake_minimum_required(VERSION 3.20)
project(NiyahEngine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(NIYAH_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

add_library(niyah STATIC
    src/niyah_model.c
    src/niyah_io.c
)
target_include_directories(niyah PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
    target_compile_definitions(niyah PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(niyah PRIVATE /W4)
    if(NIYAH_WARNINGS_AS_ERRORS)
        target_compile_options(niyah PRIVATE /WX)
    endif()
else()
    target_compile_options(niyah PRIVATE -Wall -Wextra -Wpedantic)
    if(NIYAH_WARNINGS_AS_ERRORS)
        target_compile_options(niyah PRIVATE -Werror)
    endif()
endif()

add_executable(niyah-cli tools/niyah.c)
target_link_libraries(niyah-cli PRIVATE niyah m)
set_target_properties(niyah-cli PROPERTIES OUTPUT_NAME niyah)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_model tests/test_model.c)
    target_link_libraries(test_model PRIVATE niyah m)
    add_test(NAME model_unit COMMAND test_model)

    add_executable(test_sha256 tests/test_sha256.c src/niyah_sha256.c)
    target_link_libraries(test_sha256 PRIVATE niyah m)
    add_test(NAME sha256_unit COMMAND test_sha256)

    add_test(NAME cli_smoke
        COMMAND ${CMAKE_COMMAND}
            -DNIYAH_EXE=$<TARGET_FILE:niyah-cli>
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_smoke.cmake)
endif()
