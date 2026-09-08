cmake_minimum_required(VERSION 3.16)
project(niyah_storage LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Transitional SQLite compatibility backend. It remains until PostgreSQL
# parity is proven and the SQLite-specific schema path can be retired safely.
find_package(SQLite3 REQUIRED)

add_library(niyah_local_store STATIC local_store.c local_store.h store.h)
target_include_directories(niyah_local_store PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(niyah_local_store PUBLIC SQLite::SQLite3)

if(MSVC)
    target_compile_options(niyah_local_store PRIVATE /W4 /utf-8)
else()
    target_compile_options(niyah_local_store PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_executable(local_store_test local_store_test.c)
target_link_libraries(local_store_test PRIVATE niyah_local_store)
if(MSVC)
    target_compile_options(local_store_test PRIVATE /W4 /utf-8)
else()
    target_compile_options(local_store_test PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME local_store_test COMMAND local_store_test)

# Canonical runtime backend. Keep discovery non-fatal during the transition so
# environments without libpq can still build the legacy target; once parity is
# complete PostgreSQL becomes required and the SQLite target is removed.
find_package(PostgreSQL QUIET)

if(PostgreSQL_FOUND)
    add_library(niyah_postgres_store STATIC postgres_store.c store.h)
    target_include_directories(niyah_postgres_store PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(niyah_postgres_store PUBLIC PostgreSQL::PostgreSQL)

    if(MSVC)
        target_compile_options(niyah_postgres_store PRIVATE /W4 /utf-8)
    else()
        target_compile_options(niyah_postgres_store PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(postgres_store_test postgres_store_test.c)
    target_link_libraries(postgres_store_test PRIVATE niyah_postgres_store PostgreSQL::PostgreSQL)

    if(MSVC)
        target_compile_options(postgres_store_test PRIVATE /W4 /utf-8)
    else()
        target_compile_options(postgres_store_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME postgres_store_test COMMAND postgres_store_test)
    set_tests_properties(postgres_store_test PROPERTIES
        SKIP_RETURN_CODE 77
        RUN_SERIAL TRUE)
endif()
