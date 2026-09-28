if(NOT DEFINED NIYAH_FIXTURE OR
   NOT DEFINED NIYAH_TRAIN OR
   NOT DEFINED NIYAH_CLI OR
   NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "missing runtime test arguments")
endif()

if(NOT DEFINED BACKEND)
    set(BACKEND "cpu")
endif()

if(NOT BACKEND STREQUAL "cpu" AND NOT BACKEND STREQUAL "cuda")
    message(FATAL_ERROR "invalid backend: ${BACKEND}")
endif()

set(PREFIX "${WORK_DIR}/niyah_cli_${BACKEND}_fixture")
set(TOK "${PREFIX}.tok")
set(SHARD "${PREFIX}.srd")
set(SHARD_ALT "${PREFIX}_alt.srd")
set(CKPT "${PREFIX}.ckpt")
set(PROMPT_PREFIX "${CKPT}.prompt-prefix")
set(PROMPT_SUFFIX "${CKPT}.prompt-suffix")
set(PROMPT_SHA "${CKPT}.prompt-sha256")
set(CURSOR "${PREFIX}.cursor")
set(OUTPUT "${PREFIX}.out")

file(REMOVE
    "${TOK}"
    "${SHARD}"
    "${SHARD_ALT}"
    "${CKPT}"
    "${CURSOR}"
    "${OUTPUT}")
file(REMOVE_RECURSE
    "${PROMPT_PREFIX}"
    "${PROMPT_SUFFIX}"
    "${PROMPT_SHA}")

execute_process(
    COMMAND "${NIYAH_FIXTURE}"
        "${TOK}"
        "${SHARD}"
        "${SHARD_ALT}"
    RESULT_VARIABLE fixture_result
)
if(NOT fixture_result EQUAL 0)
    message(FATAL_ERROR "runtime fixture creation failed: ${fixture_result}")
endif()

execute_process(
    COMMAND "${NIYAH_TRAIN}" new
        --tokenizer "${TOK}"
        --shard "${SHARD}"
        --checkpoint-out "${CKPT}"
        --cursor-out "${CURSOR}"
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
        --max-grad-norm 1.0
    RESULT_VARIABLE train_result
    OUTPUT_QUIET
)
if(NOT train_result EQUAL 0)
    message(FATAL_ERROR "runtime checkpoint training failed: ${train_result}")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" run
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --prompt "a"
        --max-new-tokens 1
        --temperature 0
        --seed 1
        --backend "${BACKEND}"
    RESULT_VARIABLE run_result
    OUTPUT_FILE "${OUTPUT}"
)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "niyah run failed: ${run_result}")
endif()

if(NOT EXISTS "${OUTPUT}")
    message(FATAL_ERROR "runtime output missing")
endif()

file(SIZE "${OUTPUT}" output_size)
if(output_size LESS 1)
    message(FATAL_ERROR "runtime output empty")
endif()

file(MAKE_DIRECTORY "${PROMPT_PREFIX}")
execute_process(
    COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
    RESULT_VARIABLE unreadable_result ERROR_VARIABLE unreadable_error OUTPUT_QUIET)
if(unreadable_result EQUAL 0 OR NOT unreadable_error MATCHES "error_stage=prompt_contract")
    message(FATAL_ERROR "unreadable prompt sidecar did not fail closed: ${unreadable_error}")
endif()
file(REMOVE_RECURSE "${PROMPT_PREFIX}")

file(WRITE "${PROMPT_PREFIX}" "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
file(WRITE "${PROMPT_SUFFIX}" "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

execute_process(
    COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
    RESULT_VARIABLE unbound_result ERROR_VARIABLE unbound_error OUTPUT_QUIET)
if(unbound_result EQUAL 0 OR NOT unbound_error MATCHES "error_stage=prompt_contract")
    message(FATAL_ERROR "unbound prompt contract was accepted: ${unbound_error}")
endif()

file(SHA256 "${CKPT}" checkpoint_sha256)
file(WRITE "${PROMPT_SHA}" "${checkpoint_sha256}\n")
execute_process(
    COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
    RESULT_VARIABLE sidecar_result ERROR_VARIABLE sidecar_error OUTPUT_QUIET)
if(sidecar_result EQUAL 0 OR NOT sidecar_error MATCHES "error_stage=context_capacity")
    message(FATAL_ERROR "bound prompt sidecars were not applied: ${sidecar_error}")
endif()

file(WRITE "${PROMPT_SHA}" "0000000000000000000000000000000000000000000000000000000000000000\n")
execute_process(
    COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
    RESULT_VARIABLE stale_result ERROR_VARIABLE stale_error OUTPUT_QUIET)
if(stale_result EQUAL 0 OR NOT stale_error MATCHES "error_stage=prompt_contract")
    message(FATAL_ERROR "stale prompt contract identity was accepted: ${stale_error}")
endif()

file(WRITE "${PROMPT_SHA}" "${checkpoint_sha256}")
file(WRITE "${PROMPT_PREFIX}" "")
file(WRITE "${PROMPT_SUFFIX}" "")
execute_process(
    COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
    RESULT_VARIABLE empty_result OUTPUT_QUIET)
if(NOT empty_result EQUAL 0)
    message(FATAL_ERROR "empty prompt sidecars were rejected")
endif()

file(WRITE "${PROMPT_PREFIX}" "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
file(WRITE "${PROMPT_SUFFIX}" "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
execute_process(
    COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --prompt-prefix "" --prompt-suffix "" --max-new-tokens 1
        --temperature 0 --seed 1 --backend "${BACKEND}"
    RESULT_VARIABLE sidecar_override_result OUTPUT_QUIET)
if(NOT sidecar_override_result EQUAL 0)
    message(FATAL_ERROR "explicit prompt flags did not override sidecars")
endif()
file(REMOVE "${PROMPT_PREFIX}" "${PROMPT_SUFFIX}" "${PROMPT_SHA}")

execute_process(
    COMMAND "${NIYAH_CLI}" run
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --prompt "a"
        --prompt-prefix "b"
        --max-new-tokens 1
        --temperature 0
        --seed 1
        --backend "${BACKEND}"
    RESULT_VARIABLE prefix_result
    OUTPUT_QUIET
)
if(NOT prefix_result EQUAL 0)
    message(FATAL_ERROR "runtime prompt prefix failed: ${prefix_result}")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" run
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --prompt "a"
        --prompt-suffix "b"
        --max-new-tokens 1
        --temperature 0
        --seed 1
        --backend "${BACKEND}"
    RESULT_VARIABLE suffix_result
    OUTPUT_QUIET
)
if(NOT suffix_result EQUAL 0)
    message(FATAL_ERROR "runtime prompt suffix failed: ${suffix_result}")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" run
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --prompt "xyz"
        --max-new-tokens 1
        --temperature 0
        --seed 1
        --backend "${BACKEND}"
    RESULT_VARIABLE bos_capacity_result
    ERROR_VARIABLE bos_capacity_error
    OUTPUT_QUIET
)
if(bos_capacity_result EQUAL 0)
    message(FATAL_ERROR "runtime BOS capacity contract was not enforced")
endif()

string(FIND
    "${bos_capacity_error}"
    "error_stage=context_capacity"
    bos_capacity_error_index)
if(bos_capacity_error_index EQUAL -1)
    message(FATAL_ERROR
        "runtime BOS capacity rejection used unexpected error: ${bos_capacity_error}")
endif()

message("P8B_RUNTIME_BOS_CONTRACT_${BACKEND}=PASS")
message("P8B_NATIVE_RUNTIME_BACKEND_${BACKEND}=PASS")
