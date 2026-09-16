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

message("P8B_NATIVE_RUNTIME_BACKEND_${BACKEND}=PASS")
