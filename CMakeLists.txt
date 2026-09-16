cmake_minimum_required(VERSION 3.20)
project(NiyahEngine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(NIYAH_BUILD_TESTS "Build Niyah.Engine tests" ON)
option(NIYAH_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(NIYAH_ENABLE_CUDA "Build the optional CUDA backend" OFF)

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

if(NIYAH_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "NIYAH_ENABLE_CUDA=ON but no CUDA compiler was found")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(niyah_cuda_backend STATIC
        src/niyah_cuda_matvec.cu
    )
    target_include_directories(niyah_cuda_backend
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(niyah_cuda_backend PRIVATE CUDA::cudart)
    set_target_properties(niyah_cuda_backend PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
endif()

add_library(niyah STATIC
    src/niyah_model.c
    src/niyah_math.c
    src/niyah_sha256.c
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

add_executable(niyah_train tools/niyah_train.c)
set_target_properties(niyah_train PROPERTIES OUTPUT_NAME "niyah-train")
target_link_libraries(niyah_train PRIVATE niyah)

if(MSVC)
    target_compile_options(niyah_train PRIVATE /W4 /WX)
else()
    target_compile_options(niyah_train PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(NIYAH_BUILD_TESTS)
    enable_testing()

    if(NIYAH_ENABLE_CUDA)
        add_executable(niyah_cuda_matvec_test tests/test_cuda_matvec.c)
        target_include_directories(niyah_cuda_matvec_test
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(niyah_cuda_matvec_test
            PRIVATE
                niyah
                niyah_cuda_backend
        )

        if(MSVC)
            target_compile_options(niyah_cuda_matvec_test PRIVATE /W4 /WX)
        else()
            target_compile_options(niyah_cuda_matvec_test
                PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()

        add_test(NAME niyah_cuda_matvec_test COMMAND niyah_cuda_matvec_test)

        add_executable(niyah_cuda_model_test tests/test_cuda_model.c)
        target_include_directories(niyah_cuda_model_test
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(niyah_cuda_model_test
            PRIVATE
                niyah
                niyah_cuda_backend
        )

        if(MSVC)
            target_compile_options(niyah_cuda_model_test PRIVATE /W4 /WX)
        else()
            target_compile_options(niyah_cuda_model_test
                PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()

        add_test(NAME niyah_cuda_model_test COMMAND niyah_cuda_model_test)

        add_executable(niyah_cuda_decode_state_test
            tests/test_cuda_decode_state.c
        )
        target_include_directories(niyah_cuda_decode_state_test
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(niyah_cuda_decode_state_test
            PRIVATE
                niyah
                niyah_cuda_backend
        )

        if(MSVC)
            target_compile_options(niyah_cuda_decode_state_test
                PRIVATE /W4 /WX)
        else()
            target_compile_options(niyah_cuda_decode_state_test
                PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()

        add_test(
            NAME niyah_cuda_decode_state_test
            COMMAND niyah_cuda_decode_state_test
        )

        add_executable(niyah_cuda_device_matvec_test
            tests/test_cuda_device_matvec.c
        )
        target_include_directories(niyah_cuda_device_matvec_test
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(niyah_cuda_device_matvec_test
            PRIVATE
                niyah
                niyah_cuda_backend
                CUDA::cudart
        )

        if(MSVC)
            target_compile_options(niyah_cuda_device_matvec_test
                PRIVATE /W4 /WX)
        else()
            target_compile_options(niyah_cuda_device_matvec_test
                PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()

        add_test(
            NAME niyah_cuda_device_matvec_test
            COMMAND niyah_cuda_device_matvec_test
        )
    endif()

    add_executable(niyah_model_test tests/test_model.c)
    target_link_libraries(niyah_model_test PRIVATE niyah)

    add_executable(niyah_tokenizer_test tests/test_tokenizer.c)
    target_link_libraries(niyah_tokenizer_test PRIVATE niyah)

    add_executable(niyah_dataset_test tests/test_dataset.c)
    target_link_libraries(niyah_dataset_test PRIVATE niyah)

    add_executable(niyah_dataset_shard_test tests/test_dataset_shard.c)
    target_link_libraries(niyah_dataset_shard_test PRIVATE niyah)

    add_executable(niyah_dataset_collection_test tests/test_dataset_collection.c)
    target_link_libraries(niyah_dataset_collection_test PRIVATE niyah)

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

    add_executable(niyah_training_fixture tests/make_training_fixture.c)
    target_link_libraries(niyah_training_fixture PRIVATE niyah)

    if(MSVC)
        target_compile_options(niyah_model_test PRIVATE /W4 /WX)
        target_compile_options(niyah_tokenizer_test PRIVATE /W4 /WX)
        target_compile_options(niyah_dataset_test PRIVATE /W4 /WX)
        target_compile_options(niyah_dataset_shard_test PRIVATE /W4 /WX)
        target_compile_options(niyah_dataset_collection_test PRIVATE /W4 /WX)
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
        target_compile_options(niyah_training_fixture PRIVATE /W4 /WX)
    else()
        target_compile_options(niyah_model_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_tokenizer_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_dataset_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_dataset_shard_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
        target_compile_options(niyah_dataset_collection_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
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
        target_compile_options(niyah_training_fixture PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()

    add_test(NAME niyah_model_test COMMAND niyah_model_test)
    add_test(NAME niyah_tokenizer_test COMMAND niyah_tokenizer_test)
    add_test(NAME niyah_dataset_test COMMAND niyah_dataset_test)
    add_test(NAME niyah_dataset_shard_test COMMAND niyah_dataset_shard_test)
    add_test(NAME niyah_dataset_collection_test COMMAND niyah_dataset_collection_test)
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

    set(_NIYAH_TRAIN_TOK "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture.tok")
    set(_NIYAH_TRAIN_SHARD "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture.srd")
    set(_NIYAH_TRAIN_SHARD_ALT "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture_alt.srd")
    set(_NIYAH_TRAIN_CKPT1 "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture_1.ckpt")
    set(_NIYAH_TRAIN_CUR1 "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture_1.cursor")
    set(_NIYAH_TRAIN_CKPT2 "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture_2.ckpt")
    set(_NIYAH_TRAIN_CUR2 "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture_2.cursor")
    set(_NIYAH_TRAIN_CKPT3 "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture_3.ckpt")
    set(_NIYAH_TRAIN_CUR3 "${CMAKE_CURRENT_BINARY_DIR}/niyah_train_fixture_3.cursor")

    add_test(NAME niyah_train_help_test COMMAND niyah_train --help)

    add_test(NAME niyah_training_fixture_test
        COMMAND niyah_training_fixture
            "${_NIYAH_TRAIN_TOK}" "${_NIYAH_TRAIN_SHARD}"
            "${_NIYAH_TRAIN_SHARD_ALT}"
            "${_NIYAH_TRAIN_CKPT1}" "${_NIYAH_TRAIN_CUR1}"
            "${_NIYAH_TRAIN_CKPT2}" "${_NIYAH_TRAIN_CUR2}")
    set_tests_properties(niyah_training_fixture_test PROPERTIES
        FIXTURES_SETUP niyah_train_files)

    add_test(NAME niyah_train_new_test
        COMMAND niyah_train new
            --tokenizer "${_NIYAH_TRAIN_TOK}"
            --shard "${_NIYAH_TRAIN_SHARD}"
            --checkpoint-out "${_NIYAH_TRAIN_CKPT1}"
            --cursor-out "${_NIYAH_TRAIN_CUR1}"
            --updates 2
            --batch-size 1
            --accumulation-steps 1
            --model-seed 42
            --data-seed 7
            --context-length 4
            --embedding-dim 8
            --layers 1
            --heads 2
            --kv-heads 1
            --ffn-hidden-dim 16
            --rms-norm-eps 0.00001
            --tie-word-embeddings 1
            --learning-rate 0.001
            --beta1 0.9
            --beta2 0.999
            --epsilon 0.00000001
            --weight-decay 0.0
            --max-grad-norm 1.0)
    set_tests_properties(niyah_train_new_test PROPERTIES
        FIXTURES_REQUIRED niyah_train_files)

    add_test(NAME niyah_train_resume_test
        COMMAND niyah_train resume
            --tokenizer "${_NIYAH_TRAIN_TOK}"
            --shard "${_NIYAH_TRAIN_SHARD}"
            --checkpoint-in "${_NIYAH_TRAIN_CKPT1}"
            --cursor-in "${_NIYAH_TRAIN_CUR1}"
            --checkpoint-out "${_NIYAH_TRAIN_CKPT2}"
            --cursor-out "${_NIYAH_TRAIN_CUR2}"
            --updates 2
            --batch-size 1
            --accumulation-steps 1)
    set_tests_properties(niyah_train_resume_test PROPERTIES
        FIXTURES_REQUIRED niyah_train_files
        DEPENDS niyah_train_new_test)

    add_test(NAME niyah_train_resume_dataset_mismatch_test
        COMMAND ${CMAKE_COMMAND}
            "-DNIYAH_TRAIN_EXE=$<TARGET_FILE:niyah_train>"
            "-DNIYAH_TRAIN_TOK=${_NIYAH_TRAIN_TOK}"
            "-DNIYAH_TRAIN_SHARD=${_NIYAH_TRAIN_SHARD}"
            "-DNIYAH_TRAIN_SHARD_ALT=${_NIYAH_TRAIN_SHARD_ALT}"
            "-DNIYAH_TRAIN_CKPT=${_NIYAH_TRAIN_CKPT3}"
            "-DNIYAH_TRAIN_CUR=${_NIYAH_TRAIN_CUR3}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_train_resume_mismatch.cmake")
    set_tests_properties(niyah_train_resume_dataset_mismatch_test PROPERTIES
        FIXTURES_REQUIRED niyah_train_files)

    add_test(NAME niyah_train_multishard_test
        COMMAND ${CMAKE_COMMAND}
            "-DNIYAH_TRAIN_EXE=$<TARGET_FILE:niyah_train>"
            "-DNIYAH_TRAIN_TOK=${_NIYAH_TRAIN_TOK}"
            "-DNIYAH_TRAIN_SHARD_A=${_NIYAH_TRAIN_SHARD}"
            "-DNIYAH_TRAIN_SHARD_B=${_NIYAH_TRAIN_SHARD_ALT}"
            "-DNIYAH_TRAIN_MULTI_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/niyah_train_multishard"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_train_multishard.cmake")
    set_tests_properties(niyah_train_multishard_test PROPERTIES
        FIXTURES_REQUIRED niyah_train_files)

    add_test(NAME niyah_train_pair_mismatch_test
        COMMAND ${CMAKE_COMMAND}
            "-DNIYAH_TRAIN_EXE=$<TARGET_FILE:niyah_train>"
            "-DNIYAH_TRAIN_TOK=${_NIYAH_TRAIN_TOK}"
            "-DNIYAH_TRAIN_SHARD=${_NIYAH_TRAIN_SHARD}"
            "-DNIYAH_TRAIN_PAIR_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/niyah_train_pair"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_train_pair_mismatch.cmake")
    set_tests_properties(niyah_train_pair_mismatch_test PROPERTIES
        FIXTURES_REQUIRED niyah_train_files)
endif()
