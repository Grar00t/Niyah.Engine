cmake_minimum_required(VERSION 3.20)
project(NiyahEngine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(NIYAH_BUILD_TESTS "Build Niyah.Engine tests" ON)
option(NIYAH_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(NIYAH_ENABLE_SANITIZERS)
    if(MSVC)
        message(FATAL_ERROR "NIYAH_ENABLE_SANITIZERS is not supported for MSVC in this CI configuration")
    endif()

    include(CheckCCompilerFlag)
    include(CheckCSourceCompiles)

    set(_NIYAH_SAVED_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
    set(_NIYAH_SAVED_REQUIRED_LINK_OPTIONS "${CMAKE_REQUIRED_LINK_OPTIONS}")
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fsanitize=address,undefined")
    list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address,undefined")
    check_c_source_compiles("int main(void) { return 0; }"
                            NIYAH_HAS_ADDRESS_UNDEFINED_SANITIZERS)
    set(CMAKE_REQUIRED_FLAGS "${_NIYAH_SAVED_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_LINK_OPTIONS "${_NIYAH_SAVED_REQUIRED_LINK_OPTIONS}")
    unset(_NIYAH_SAVED_REQUIRED_FLAGS)
    unset(_NIYAH_SAVED_REQUIRED_LINK_OPTIONS)

    check_c_compiler_flag("-fno-omit-frame-pointer" NIYAH_HAS_FRAME_POINTER_FLAG)

    if(NOT NIYAH_HAS_ADDRESS_UNDEFINED_SANITIZERS)
        message(FATAL_ERROR "AddressSanitizer and UndefinedBehaviorSanitizer could not compile and link with the active C compiler")
    endif()
    if(NOT NIYAH_HAS_FRAME_POINTER_FLAG)
        message(FATAL_ERROR "The active C compiler does not support -fno-omit-frame-pointer")
    endif()

    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

add_library(niyah STATIC
    src/niyah_model.c
    src/niyah_math.c
    src/niyah_tokenizer.c
    src/niyah_dataset.c
    src/niyah_dataset_shard.c
    src/niyah_transformer.c
    src/niyah_decode.c
    src/niyah_sampler.c
    src/niyah_generate.c
    src/niyah_train.c
    src/niyah_eval.c
    src/niyah_backward.c
    src/niyah_training_loop.c
    src/niyah_optimizer.c
    src/niyah_checkpoint.c
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

    add_executable(niyah_dataset_test tests/test_dataset.c)
    target_link_libraries(niyah_dataset_test PRIVATE niyah)

    add_executable(niyah_dataset_shard_test tests/test_dataset_shard.c)
    target_link_libraries(niyah_dataset_shard_test PRIVATE niyah)

    add_executable(niyah_transformer_test tests/test_transformer.c)
    target_link_libraries(niyah_transformer_test PRIVATE niyah)

    add_executable(niyah_decode_test tests/test_decode.c)
    target_link_libraries(niyah_decode_test PRIVATE niyah)

    add_executable(niyah_generation_test tests/test_generation.c)
    target_link_libraries(niyah_generation_test PRIVATE niyah)

    add_executable(niyah_train_test tests/test_train.c)
    target_link_libraries(niyah_train_test PRIVATE niyah)

    add_executable(niyah_eval_test tests/test_eval.c)
    target_link_libraries(niyah_eval_test PRIVATE niyah)

    add_executable(niyah_training_loop_test tests/test_training_loop.c)
    target_link_libraries(niyah_training_loop_test PRIVATE niyah)

    add_executable(niyah_backward_test tests/test_backward.c)
    target_link_libraries(niyah_backward_test PRIVATE niyah)

    add_executable(niyah_text_pipeline_test tests/test_text_pipeline.c)
    target_link_libraries(niyah_text_pipeline_test PRIVATE niyah)

    add_executable(niyah_optimizer_test tests/test_optimizer.c)
    target_link_libraries(niyah_optimizer_test PRIVATE niyah)

    add_executable(niyah_checkpoint_test tests/test_checkpoint.c)
    target_link_libraries(niyah_checkpoint_test PRIVATE niyah)

    if(MSVC)
        target_compile_options(niyah_model_test PRIVATE /W4 /WX)
        target_compile_options(niyah_tokenizer_test PRIVATE /W4 /WX)
        target_compile_options(niyah_dataset_test PRIVATE /W4 /WX)
        target_compile_options(niyah_dataset_shard_test PRIVATE /W4 /WX)
        target_compile_options(niyah_transformer_test PRIVATE /W4 /WX)
        target_compile_options(niyah_decode_test PRIVATE /W4 /WX)
        target_compile_options(niyah_generation_test PRIVATE /W4 /WX)
        target_compile_options(niyah_train_test PRIVATE /W4 /WX)
        target_compile_options(niyah_eval_test PRIVATE /W4 /WX)
        target_compile_options(niyah_training_loop_test PRIVATE /W4 /WX)
        target_compile_options(niyah_backward_test PRIVATE /W4 /WX)
        target_compile_options(niyah_text_pipeline_test PRIVATE /W4 /WX)
        target_compile_options(niyah_optimizer_test PRIVATE /W4 /WX)
        target_compile_options(niyah_checkpoint_test PRIVATE /W4 /WX)
    else()
        target_compile_options(niyah_model_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_tokenizer_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_dataset_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_dataset_shard_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_transformer_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_decode_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_generation_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_train_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_eval_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_training_loop_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_backward_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_text_pipeline_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_optimizer_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_checkpoint_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()

    add_test(NAME niyah_model_test COMMAND niyah_model_test)
    add_test(NAME niyah_tokenizer_test COMMAND niyah_tokenizer_test)
    add_test(NAME niyah_dataset_test COMMAND niyah_dataset_test)
    add_test(NAME niyah_dataset_shard_test COMMAND niyah_dataset_shard_test)
    add_test(NAME niyah_transformer_test COMMAND niyah_transformer_test)
    add_test(NAME niyah_decode_test COMMAND niyah_decode_test)
    add_test(NAME niyah_generation_test COMMAND niyah_generation_test)
    add_test(NAME niyah_train_test COMMAND niyah_train_test)
    add_test(NAME niyah_eval_test COMMAND niyah_eval_test)
    add_test(NAME niyah_training_loop_test COMMAND niyah_training_loop_test)
    add_test(NAME niyah_backward_test COMMAND niyah_backward_test)
    add_test(NAME niyah_text_pipeline_test COMMAND niyah_text_pipeline_test)
    add_test(NAME niyah_optimizer_test COMMAND niyah_optimizer_test)
    add_test(NAME niyah_checkpoint_test COMMAND niyah_checkpoint_test)
endif()
