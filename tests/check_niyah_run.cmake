string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef _niyah_fixture_id)
if(DEFINED WORK_DIR)
    set(WORK_DIR "${WORK_DIR}/fixture-${_niyah_fixture_id}")
    file(MAKE_DIRECTORY "${WORK_DIR}")
endif()
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

# V2 metadata binds template bytes as well as checkpoint identity. Keep these
# fixtures isolated from user checkpoints; CLI overrides remain authoritative.
file(WRITE "${PROMPT_PREFIX}" "b")
file(WRITE "${PROMPT_SUFFIX}" "")
file(SHA256 "${PROMPT_PREFIX}" prefix_sha)
file(SHA256 "${PROMPT_SUFFIX}" suffix_sha)
set(v2_metadata "NIYAH_PROMPT_V2\n${checkpoint_sha256}\n${prefix_sha}\n${suffix_sha}\n")
file(WRITE "${PROMPT_SHA}" "${v2_metadata}")
execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
    --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
    RESULT_VARIABLE v2_result ERROR_VARIABLE v2_error OUTPUT_QUIET)
if(NOT v2_result EQUAL 0 OR v2_error MATCHES "legacy_checkpoint_only")
    message(FATAL_ERROR "V2 template was rejected: ${v2_error}")
endif()
foreach(component IN ITEMS "${PROMPT_PREFIX}" "${PROMPT_SUFFIX}")
    file(WRITE "${component}" "c")
    execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
        RESULT_VARIABLE tamper_result ERROR_VARIABLE tamper_error OUTPUT_QUIET)
    if(tamper_result EQUAL 0 OR NOT tamper_error MATCHES "error_stage=prompt_contract")
        message(FATAL_ERROR "modified template was accepted: ${component}: ${tamper_error}")
    endif()
    if(component STREQUAL PROMPT_PREFIX)
        set(override_flag "--prompt-prefix")
    else()
        set(override_flag "--prompt-suffix")
    endif()
    execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" "${override_flag}" "" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
        RESULT_VARIABLE v2_override_result ERROR_VARIABLE v2_override_error OUTPUT_QUIET)
    if(NOT v2_override_result EQUAL 0)
        message(FATAL_ERROR "V2 explicit override failed: ${v2_override_error}")
    endif()
    file(WRITE "${PROMPT_PREFIX}" "b")
    file(WRITE "${PROMPT_SUFFIX}" "")
endforeach()
foreach(bad_metadata IN ITEMS "NIYAH_PROMPT_V2\ngarbage" "NIYAH_PROMPT_V3" "")
    file(WRITE "${PROMPT_SHA}" "${bad_metadata}")
    execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --temperature 0 --seed 1 --backend "${BACKEND}"
        RESULT_VARIABLE malformed_result ERROR_VARIABLE malformed_error OUTPUT_QUIET)
    if(malformed_result EQUAL 0 OR NOT malformed_error MATCHES "error_stage=prompt_contract")
        message(FATAL_ERROR "malformed V2 metadata accepted: ${malformed_error}")
    endif()
endforeach()
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

# New generation options must not enter eval; all invalid values fail in parsing.
foreach(bad_penalty IN ITEMS "NaN" "Inf" "-Inf" "-1" "0" "0.99" "abc" "1.1junk" "1e999")
    execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --backend "${BACKEND}" --repetition-penalty "${bad_penalty}"
        RESULT_VARIABLE bad_result OUTPUT_QUIET ERROR_QUIET)
    if(NOT bad_result EQUAL 2)
        message(FATAL_ERROR "invalid repetition penalty not rejected by parser: ${bad_penalty}: ${bad_result}")
    endif()
endforeach()
foreach(bad_window IN ITEMS "0" "-1" " -1" "+1" "1.5" "NaN" "abc" "184467440737095516160")
    execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --backend "${BACKEND}" --repetition-window "${bad_window}"
        RESULT_VARIABLE bad_result OUTPUT_QUIET ERROR_QUIET)
    if(NOT bad_result EQUAL 2)
        message(FATAL_ERROR "invalid repetition window not rejected by parser: ${bad_window}: ${bad_result}")
    endif()
endforeach()
foreach(flag IN ITEMS --repetition-penalty --repetition-window)
    execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --prompt "a" --max-new-tokens 1 --backend "${BACKEND}" "${flag}"
        RESULT_VARIABLE missing_result OUTPUT_QUIET ERROR_QUIET)
    if(NOT missing_result EQUAL 2)
        message(FATAL_ERROR "missing repetition value accepted: ${flag}")
    endif()
    execute_process(COMMAND "${NIYAH_CLI}" eval --tokenizer "${TOK}" --checkpoint "${CKPT}"
        --heldout "${SHARD}" "${flag}" 2
        RESULT_VARIABLE eval_result OUTPUT_QUIET ERROR_QUIET)
    if(NOT eval_result EQUAL 2)
        message(FATAL_ERROR "generation-only option accepted by eval: ${flag}")
    endif()
endforeach()
execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
    --prompt "a" --max-new-tokens 1 --temperature 0.7 --seed 42 --backend "${BACKEND}"
    RESULT_VARIABLE default_result OUTPUT_VARIABLE default_output ERROR_VARIABLE default_error)
execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
    --prompt "a" --max-new-tokens 1 --temperature 0.7 --seed 42 --backend "${BACKEND}"
    --repetition-penalty 1 --repetition-window 64
    RESULT_VARIABLE disabled_result OUTPUT_VARIABLE disabled_output ERROR_VARIABLE disabled_error)
if(NOT default_result EQUAL 0 OR NOT disabled_result EQUAL 0 OR
   NOT default_output STREQUAL disabled_output OR NOT default_error STREQUAL disabled_error)
    message(FATAL_ERROR "disabled repetition changed seeded CLI behavior: ${BACKEND}")
endif()
execute_process(COMMAND "${NIYAH_CLI}" run --tokenizer "${TOK}" --checkpoint "${CKPT}"
    --prompt "a" --max-new-tokens 1 --backend "${BACKEND}" --repetition-penalty 1.1 --repetition-window 64
    RESULT_VARIABLE enabled_result OUTPUT_QUIET)
if(NOT enabled_result EQUAL 0)
    message(FATAL_ERROR "enabled repetition CLI failed: ${BACKEND}: ${enabled_result}")
endif()
message("REPETITION_CLI_${BACKEND}=PASS")
