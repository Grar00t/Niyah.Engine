cmake_minimum_required(VERSION 3.20)
project(NiyahEngine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(NIYAH_BUILD_TESTS "Build Niyah.Engine tests" ON)

add_library(niyah STATIC
    src/niyah_model.c
    src/niyah_math.c
    src/niyah_tokenizer.c
    src/niyah_transformer.c
    src/niyah_decode.c
    src/niyah_sampler.c
    src/niyah_generate.c
    src/niyah_train.c
    src/niyah_backward.c
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

    add_executable(niyah_tokenizer_test tests/test_tokenizer.c)
    target_link_libraries(niyah_tokenizer_test PRIVATE niyah)

    add_executable(niyah_transformer_test tests/test_transformer.c)
    target_link_libraries(niyah_transformer_test PRIVATE niyah)

    add_executable(niyah_decode_test tests/test_decode.c)
    target_link_libraries(niyah_decode_test PRIVATE niyah)

    add_executable(niyah_generation_test tests/test_generation.c)
    target_link_libraries(niyah_generation_test PRIVATE niyah)

    add_executable(niyah_train_test tests/test_train.c)
    target_link_libraries(niyah_train_test PRIVATE niyah)

    add_executable(niyah_backward_test tests/test_backward.c)
    target_link_libraries(niyah_backward_test PRIVATE niyah)

    add_executable(niyah_text_pipeline_test tests/test_text_pipeline.c)
    target_link_libraries(niyah_text_pipeline_test PRIVATE niyah)

    if(MSVC)
        target_compile_options(niyah_model_test PRIVATE /W4 /WX)
        target_compile_options(niyah_tokenizer_test PRIVATE /W4 /WX)
        target_compile_options(niyah_transformer_test PRIVATE /W4 /WX)
        target_compile_options(niyah_decode_test PRIVATE /W4 /WX)
        target_compile_options(niyah_generation_test PRIVATE /W4 /WX)
        target_compile_options(niyah_train_test PRIVATE /W4 /WX)
        target_compile_options(niyah_backward_test PRIVATE /W4 /WX)
        target_compile_options(niyah_text_pipeline_test PRIVATE /W4 /WX)
    else()
        target_compile_options(niyah_model_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_tokenizer_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_transformer_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_decode_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_generation_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_train_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_backward_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_text_pipeline_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()

    add_test(NAME niyah_model_test COMMAND niyah_model_test)
    add_test(NAME niyah_tokenizer_test COMMAND niyah_tokenizer_test)
    add_test(NAME niyah_transformer_test COMMAND niyah_transformer_test)
    add_test(NAME niyah_decode_test COMMAND niyah_decode_test)
    add_test(NAME niyah_generation_test COMMAND niyah_generation_test)
    add_test(NAME niyah_train_test COMMAND niyah_train_test)
    add_test(NAME niyah_backward_test COMMAND niyah_backward_test)
    add_test(NAME niyah_text_pipeline_test COMMAND niyah_text_pipeline_test)
endif()
