cmake_minimum_required(VERSION 3.16)
project(niyah_storage LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(PostgreSQL REQUIRED)

add_library(
    niyah_store
    STATIC
    postgres_store.c
    store.h
)

target_include_directories(
    niyah_store
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    niyah_store
    PUBLIC
    PostgreSQL::PostgreSQL
)

if(MSVC)
    target_compile_options(
        niyah_store
        PRIVATE
        /W4
        /utf-8
    )
else()
    target_compile_options(
        niyah_store
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

enable_testing()

add_executable(
    postgres_store_test
    postgres_store_test.c
)

target_link_libraries(
    postgres_store_test
    PRIVATE
    niyah_store
    PostgreSQL::PostgreSQL
)

if(MSVC)
    target_compile_options(
        postgres_store_test
        PRIVATE
        /W4
        /utf-8
    )
else()
    target_compile_options(
        postgres_store_test
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_test(
    NAME postgres_store_test
    COMMAND postgres_store_test
)

set_tests_properties(
    postgres_store_test
    PROPERTIES
    SKIP_RETURN_CODE 77
    RUN_SERIAL TRUE
)
