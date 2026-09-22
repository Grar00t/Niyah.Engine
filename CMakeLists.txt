cmake_minimum_required(VERSION 3.20)
project(NiyahEngine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(NIYAH_BUILD_TESTS "Build tests" ON)

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_library(niyah STATIC
  src/common.c
  src/sha256.c
  src/tokenizer.c
  src/dataset.c
  src/model.c
  src/checkpoint.c
  src/eval.c
  src/train.c
)

target_include_directories(niyah PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT MSVC)
  target_link_libraries(niyah PUBLIC m)
endif()

add_executable(niyah-cli tools/niyah.c)
target_link_libraries(niyah-cli PRIVATE niyah)
set_target_properties(niyah-cli PROPERTIES OUTPUT_NAME niyah)

add_executable(niyah-train tools/niyah_train.c)
target_link_libraries(niyah-train PRIVATE niyah)

if(NIYAH_BUILD_TESTS)
  enable_testing()

  foreach(t sha256 tokenizer dataset checkpoint eval train_resume binding)
    add_executable(test_${t} tests/test_${t}.c)
    target_link_libraries(test_${t} PRIVATE niyah)
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
endif()
