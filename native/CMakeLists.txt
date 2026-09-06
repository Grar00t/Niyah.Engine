cmake_minimum_required(VERSION 3.16)
project(niyah_native LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(NIYAH_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(NIYAH_SANITIZE)
    if(MSVC)
        add_compile_options(/fsanitize=address /Zi)
    else()
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

set(NIYAH_SOURCES
    niyah_core.c
    niyah_sha256.c
    niyah_proof.c
    niyah_identity.c
    niyah_control.c
    niyah_matmul.c
    niyah_softmax.c
    niyah_rmsnorm.c
    niyah_swiglu.c
    niyah_rope.c
    niyah_attention.c
    niyah_transformer_layer.c
    niyah_sampler.c
    niyah_tokenizer.c
    niyah_runtime.c
    niyah_telemetry.c
    niyah_model.c
    niyah_llm.c
    niyah_graph.c
    niyah_embedding.c
    niyah_source.c
    niyah_storage.c
    niyah_search.c
    niyah_crawler.c
    niyah_document.c
    niyah_bridge.c
    evidence_envelope.c
    evidence_graph.c
    evidence_reasoner.c
    constraint_solver.c
)

set(NIYAH_HEADERS
    niyah.h
    niyah_sha256.h
    niyah_proof.h
    niyah_identity.h
    niyah_control.h
    niyah_attention.h
    niyah_bridge.h
    niyah_document.h
    niyah_llm.h
    niyah_runtime.h
    niyah_telemetry.h
    niyah_tokenizer.h
    niyah_transformer_layer.h
    evidence_envelope.h
    evidence_graph.h
    evidence_reasoner.h
    constraint_solver.h
)

function(niyah_configure_target target)
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${target} PRIVATE NIYAH_BRIDGE_EXPORTS)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /utf-8)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
    if(WIN32)
        target_link_libraries(${target} PUBLIC wininet psapi)
    endif()
    if(UNIX)
        target_link_libraries(${target} PUBLIC m)
    endif()
endfunction()

add_library(niyah STATIC ${NIYAH_SOURCES} ${NIYAH_HEADERS})
niyah_configure_target(niyah)

add_library(niyah_shared SHARED ${NIYAH_SOURCES} ${NIYAH_HEADERS})
set_target_properties(niyah_shared PROPERTIES OUTPUT_NAME niyah C_VISIBILITY_PRESET hidden)
target_compile_definitions(niyah_shared PRIVATE NIYAH_BUILD_SHARED)
niyah_configure_target(niyah_shared)

add_subdirectory(niyah_mini)

add_executable(niyah_token_tax niyah_token_tax.c)
target_link_libraries(niyah_token_tax PRIVATE niyah)
target_compile_definitions(niyah_token_tax PRIVATE NIYAH_BRIDGE_EXPORTS)
if(MSVC)
    target_compile_options(niyah_token_tax PRIVATE /W4 /utf-8)
    target_compile_definitions(niyah_token_tax PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(niyah_token_tax PRIVATE -Wall -Wextra)
endif()

add_executable(niyah_cli niyah_cli.c)
target_link_libraries(niyah_cli PRIVATE niyah)
target_compile_definitions(niyah_cli PRIVATE NIYAH_BRIDGE_EXPORTS)
set_target_properties(niyah_cli PROPERTIES OUTPUT_NAME "niyah-core-cli")

add_executable(niyah_dispatch niyah_dispatch.c)
target_link_libraries(niyah_dispatch PRIVATE niyah)
target_compile_definitions(niyah_dispatch PRIVATE NIYAH_BRIDGE_EXPORTS)
set_target_properties(niyah_dispatch PROPERTIES OUTPUT_NAME "niyah")

foreach(cli_target niyah_cli niyah_dispatch)
    if(MSVC)
        target_compile_options(${cli_target} PRIVATE /W4 /utf-8)
        target_compile_definitions(${cli_target} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${cli_target} PRIVATE -Wall -Wextra)
    endif()
endforeach()

enable_testing()

set(NIYAH_TESTS
    test_niyah_core.c
    test_niyah_document.c
    niyah_control_test.c
    niyah_matmul_test.c
    niyah_softmax_test.c
    niyah_rmsnorm_test.c
    niyah_swiglu_test.c
    niyah_rope_test.c
    niyah_attention_test.c
    niyah_transformer_layer_test.c
    niyah_sampler_test.c
    niyah_tokenizer_test.c
    niyah_tokenizer_roundtrip_test.c
    niyah_config_test.c
    niyah_identity_test.c
    niyah_runtime_test.c
    niyah_telemetry_test.c
    niyah_graph_test.c
    niyah_embedding_test.c
    niyah_search_test.c
    niyah_storage_test.c
    niyah_crawler_test.c
    niyah_bridge_test.c
    niyah_llm_test.c
    niyah_llm_generation_test.c
    niyah_evidence_test.c
    niyah_csp_test.c
    niyah_proof_test.c
)

foreach(test_source ${NIYAH_TESTS})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE niyah)
    target_compile_definitions(${test_name} PRIVATE NIYAH_BRIDGE_EXPORTS)
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /W4 /utf-8)
        target_compile_definitions(${test_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${test_name} PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_program(NIYAH_BASH_EXECUTABLE bash)
    find_program(NIYAH_SHA256SUM_EXECUTABLE sha256sum)
    if(NOT NIYAH_BASH_EXECUTABLE OR NOT NIYAH_SHA256SUM_EXECUTABLE)
        message(FATAL_ERROR "niyah_run_proof test requires bash and sha256sum")
    endif()

    add_executable(niyah_test_llama_cli niyah_test_llama_cli.c)
    target_compile_options(niyah_test_llama_cli PRIVATE -Wall -Wextra)
    add_test(
        NAME niyah_run_proof
        COMMAND ${NIYAH_BASH_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/niyah_run_proof_test.sh
                $<TARGET_FILE:niyah_dispatch>
                $<TARGET_FILE:niyah_test_llama_cli>)
endif()
