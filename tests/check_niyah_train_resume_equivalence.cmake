if(NOT DEFINED NIYAH_TRAIN)
    message(FATAL_ERROR "NIYAH_TRAIN is required")
endif()

if(NOT DEFINED TOKENIZER)
    message(FATAL_ERROR "TOKENIZER is required")
endif()

if(NOT DEFINED SHARD)
    message(FATAL_ERROR "SHARD is required")
endif()

if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "WORK_DIR is required")
endif()

if(NOT EXISTS "${NIYAH_TRAIN}")
    message(FATAL_ERROR "niyah-train executable does not exist")
endif()

if(NOT EXISTS "${TOKENIZER}")
    message(FATAL_ERROR "training tokenizer does not exist")
endif()

if(NOT EXISTS "${SHARD}")
    message(FATAL_ERROR "training shard does not exist")
endif()

set(ONE_CKPT
    "${WORK_DIR}/niyah_resume_eq_oneshot.ckpt")
set(ONE_CURSOR
    "${WORK_DIR}/niyah_resume_eq_oneshot.cursor")

set(SPLIT_A_CKPT
    "${WORK_DIR}/niyah_resume_eq_split_a.ckpt")
set(SPLIT_A_CURSOR
    "${WORK_DIR}/niyah_resume_eq_split_a.cursor")

set(SPLIT_B_CKPT
    "${WORK_DIR}/niyah_resume_eq_split_b.ckpt")
set(SPLIT_B_CURSOR
    "${WORK_DIR}/niyah_resume_eq_split_b.cursor")

file(REMOVE
    "${ONE_CKPT}"
    "${ONE_CURSOR}"
    "${SPLIT_A_CKPT}"
    "${SPLIT_A_CURSOR}"
    "${SPLIT_B_CKPT}"
    "${SPLIT_B_CURSOR}"
)

set(NEW_COMMON
    --tokenizer "${TOKENIZER}"
    --shard "${SHARD}"
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
)

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" new
        --checkpoint-out "${ONE_CKPT}"
        --cursor-out "${ONE_CURSOR}"
        --updates 4
        ${NEW_COMMON}
    RESULT_VARIABLE ONE_RESULT
    OUTPUT_VARIABLE ONE_STDOUT
    ERROR_VARIABLE ONE_STDERR
)

if(NOT ONE_RESULT EQUAL 0)
    message(FATAL_ERROR
        "one-shot training failed\n"
        "stdout:\n${ONE_STDOUT}\n"
        "stderr:\n${ONE_STDERR}")
endif()

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" new
        --checkpoint-out "${SPLIT_A_CKPT}"
        --cursor-out "${SPLIT_A_CURSOR}"
        --updates 2
        ${NEW_COMMON}
    RESULT_VARIABLE SPLIT_A_RESULT
    OUTPUT_VARIABLE SPLIT_A_STDOUT
    ERROR_VARIABLE SPLIT_A_STDERR
)

if(NOT SPLIT_A_RESULT EQUAL 0)
    message(FATAL_ERROR
        "split stage A failed\n"
        "stdout:\n${SPLIT_A_STDOUT}\n"
        "stderr:\n${SPLIT_A_STDERR}")
endif()

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" resume
        --tokenizer "${TOKENIZER}"
        --shard "${SHARD}"
        --checkpoint-in "${SPLIT_A_CKPT}"
        --cursor-in "${SPLIT_A_CURSOR}"
        --checkpoint-out "${SPLIT_B_CKPT}"
        --cursor-out "${SPLIT_B_CURSOR}"
        --updates 2
        --batch-size 1
        --accumulation-steps 1
    RESULT_VARIABLE SPLIT_B_RESULT
    OUTPUT_VARIABLE SPLIT_B_STDOUT
    ERROR_VARIABLE SPLIT_B_STDERR
)

if(NOT SPLIT_B_RESULT EQUAL 0)
    message(FATAL_ERROR
        "split stage B failed\n"
        "stdout:\n${SPLIT_B_STDOUT}\n"
        "stderr:\n${SPLIT_B_STDERR}")
endif()

file(SHA256 "${ONE_CKPT}" ONE_CKPT_SHA)
file(SHA256 "${SPLIT_B_CKPT}" SPLIT_CKPT_SHA)

file(SHA256 "${ONE_CURSOR}" ONE_CURSOR_SHA)
file(SHA256 "${SPLIT_B_CURSOR}" SPLIT_CURSOR_SHA)

message(STATUS
    "ONE_CHECKPOINT_SHA256=${ONE_CKPT_SHA}")
message(STATUS
    "SPLIT_CHECKPOINT_SHA256=${SPLIT_CKPT_SHA}")

message(STATUS
    "ONE_CURSOR_SHA256=${ONE_CURSOR_SHA}")
message(STATUS
    "SPLIT_CURSOR_SHA256=${SPLIT_CURSOR_SHA}")

if(NOT ONE_CKPT_SHA STREQUAL SPLIT_CKPT_SHA)
    message(FATAL_ERROR
        "resume checkpoint is not byte-equivalent")
endif()

if(NOT ONE_CURSOR_SHA STREQUAL SPLIT_CURSOR_SHA)
    message(FATAL_ERROR
        "resume cursor is not byte-equivalent")
endif()

message(STATUS
    "NIYAH_TRAIN_RESUME_EQUIVALENCE=PASS")