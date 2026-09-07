cmake_minimum_required(VERSION 3.16)
project(niyah_storage LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(SQLite3 REQUIRED)

add_library(niyah_local_store STATIC local_store.c local_store.h)
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
