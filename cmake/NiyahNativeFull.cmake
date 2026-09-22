# Recovered native Transformer source surface.
#
# The source list is not guessed: it matches the archived successful Linux
# static-library link receipt at build-wsl-utf8/CMakeFiles/niyah.dir/link.txt.

set(NIYAH_NATIVE_FULL_SOURCES
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

foreach(_source IN LISTS NIYAH_NATIVE_FULL_SOURCES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
        message(FATAL_ERROR "Recovered native source is missing: ${_source}")
    endif()
endforeach()

add_library(niyah-native-full STATIC ${NIYAH_NATIVE_FULL_SOURCES})
target_include_directories(niyah-native-full PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(niyah-native-full PUBLIC c_std_11)

if(MSVC)
    target_compile_definitions(niyah-native-full PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(niyah-native-full PRIVATE /W4)
    if(NIYAH_WARNINGS_AS_ERRORS)
        target_compile_options(niyah-native-full PRIVATE /WX)
    endif()
else()
    target_compile_options(niyah-native-full PRIVATE -Wall -Wextra -Wpedantic)
    if(NIYAH_WARNINGS_AS_ERRORS)
        target_compile_options(niyah-native-full PRIVATE -Werror)
    endif()
    target_link_libraries(niyah-native-full PUBLIC m)
endif()

add_executable(niyah-native-train tools/niyah_train.c)
target_link_libraries(niyah-native-train PRIVATE niyah-native-full)

add_executable(niyah-native-probe tools/niyah_probe.c)
target_link_libraries(niyah-native-probe PRIVATE niyah-native-full)
