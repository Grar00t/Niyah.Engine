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
set(CURSOR "${PREFIX}.cursor")
set(OUTPUT "${PREFIX}.out")

file(REMOVE
    "${TOK}"
    "${SHARD}"
    "${SHARD_ALT}"
    "${CKPT}"
    "${CURSOR}"
    "${OUTPUT}")

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


# ----------------------------------------------------------
# P9O guarded model-proposal integration
#
# The tiny runtime fixture is NOT expected to emit a valid
# typed proposal. The proven fixture output for this input is
# model garbage from the proposal parser's perspective.
#
# Required behavior:
#   generation succeeds
#   strict proposal parse rejects
#   guarded pipeline returns canonical fail-closed output
#   no symbolic execution occurs
# ----------------------------------------------------------

set(GUARDED_OUTPUT
    "${PREFIX}_guarded.out")

file(REMOVE "${GUARDED_OUTPUT}")

execute_process(
    COMMAND "${NIYAH_CLI}" run
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --prompt "a"
        --max-new-tokens 1
        --temperature 0
        --seed 1
        --backend "${BACKEND}"
        --guarded-proposal-network
    RESULT_VARIABLE guarded_result
    OUTPUT_FILE "${GUARDED_OUTPUT}"
    ERROR_VARIABLE guarded_error
)

if(NOT guarded_result EQUAL 0)
    message(FATAL_ERROR
        "guarded run failed: ${guarded_result}; "
        "stderr=${guarded_error}")
endif()

if(NOT EXISTS "${GUARDED_OUTPUT}")
    message(FATAL_ERROR
        "guarded output missing")
endif()

file(READ
    "${GUARDED_OUTPUT}"
    guarded_output_text)

string(CONCAT EXPECTED_GUARDED_OUTPUT
    "state=PARSE_REJECTED\n"
    "proposal_status=NIYAH_ERR_INVALID_ARGUMENT\n")

if(NOT guarded_output_text STREQUAL
       EXPECTED_GUARDED_OUTPUT)
    message(FATAL_ERROR
        "unexpected guarded output: "
        "[${guarded_output_text}]")
endif()

# Guarded network mode and legacy arithmetic execution are
# separate contracts and cannot be enabled together.
execute_process(
    COMMAND "${NIYAH_CLI}" run
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --prompt "a"
        --max-new-tokens 1
        --backend "${BACKEND}"
        --execute-ir
        --guarded-proposal-network
    RESULT_VARIABLE guarded_execute_ir_result
    OUTPUT_QUIET
    ERROR_QUIET
)

if(guarded_execute_ir_result EQUAL 0)
    message(FATAL_ERROR
        "guarded mode accepted --execute-ir")
endif()

# Receipt/evidence V1 remain legacy arithmetic-only.
execute_process(
    COMMAND "${NIYAH_CLI}" run
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --prompt "a"
        --max-new-tokens 1
        --backend "${BACKEND}"
        --guarded-proposal-network
        --receipt
    RESULT_VARIABLE guarded_receipt_result
    OUTPUT_QUIET
    ERROR_QUIET
)

if(guarded_receipt_result EQUAL 0)
    message(FATAL_ERROR
        "guarded mode accepted --receipt")
endif()

message("P9O_GUARDED_RUN_${BACKEND}=PASS")

message("P8B_RUNTIME_BOS_CONTRACT_${BACKEND}=PASS")
message("P8B_NATIVE_RUNTIME_BACKEND_${BACKEND}=PASS")
