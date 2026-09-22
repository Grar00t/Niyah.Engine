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

set(ONE_CKPT "${WORK_DIR}/warmup_one.ckpt")
set(ONE_CURSOR "${WORK_DIR}/warmup_one.cursor")
set(SPLIT_A_CKPT "${WORK_DIR}/warmup_split_a.ckpt")
set(SPLIT_A_CURSOR "${WORK_DIR}/warmup_split_a.cursor")
set(SPLIT_B_CKPT "${WORK_DIR}/warmup_split_b.ckpt")
set(SPLIT_B_CURSOR "${WORK_DIR}/warmup_split_b.cursor")
set(REJECT_CKPT "${WORK_DIR}/warmup_reject.ckpt")
set(REJECT_CURSOR "${WORK_DIR}/warmup_reject.cursor")

file(REMOVE
    "${ONE_CKPT}" "${ONE_CURSOR}"
    "${SPLIT_A_CKPT}" "${SPLIT_A_CURSOR}"
    "${SPLIT_B_CKPT}" "${SPLIT_B_CURSOR}"
    "${REJECT_CKPT}" "${REJECT_CURSOR}"
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
    --warmup-steps 4
)

# One-shot 4 updates. Also prove reported effective LR.
execute_process(
    COMMAND
        "${NIYAH_TRAIN}" new
        --checkpoint-out "${ONE_CKPT}"
        --cursor-out "${ONE_CURSOR}"
        --updates 4
        --progress-details
        ${NEW_COMMON}
    RESULT_VARIABLE ONE_RC
    OUTPUT_VARIABLE ONE_OUT
    ERROR_VARIABLE ONE_ERR
)

if(NOT ONE_RC EQUAL 0)
    message(FATAL_ERROR
        "warmup one-shot failed\nstdout:\n${ONE_OUT}\nstderr:\n${ONE_ERR}")
endif()

foreach(PATTERN
    "update=1/4 loss=[^ \r\n]+ optimizer_step=1 learning_rate=0[.]00025[0-9]*"
    "update=2/4 loss=[^ \r\n]+ optimizer_step=2 learning_rate=0[.]0005[0-9]*"
    "update=3/4 loss=[^ \r\n]+ optimizer_step=3 learning_rate=0[.]00075[0-9]*"
    "update=4/4 loss=[^ \r\n]+ optimizer_step=4 learning_rate=0[.]001[0-9]*"
)
    if(NOT ONE_ERR MATCHES "${PATTERN}")
        message(FATAL_ERROR
            "missing expected warmup progress pattern: ${PATTERN}\n"
            "stderr:\n${ONE_ERR}")
    endif()
endforeach()

# Split: first 2 updates.
execute_process(
    COMMAND
        "${NIYAH_TRAIN}" new
        --checkpoint-out "${SPLIT_A_CKPT}"
        --cursor-out "${SPLIT_A_CURSOR}"
        --updates 2
        ${NEW_COMMON}
    RESULT_VARIABLE SPLIT_A_RC
    OUTPUT_VARIABLE SPLIT_A_OUT
    ERROR_VARIABLE SPLIT_A_ERR
)

if(NOT SPLIT_A_RC EQUAL 0)
    message(FATAL_ERROR
        "warmup split A failed\n"
        "stdout:\n${SPLIT_A_OUT}\n"
        "stderr:\n${SPLIT_A_ERR}")
endif()

# Resume without supplying schedule again. It must come from checkpoint.
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
        --progress-details
    RESULT_VARIABLE SPLIT_B_RC
    OUTPUT_VARIABLE SPLIT_B_OUT
    ERROR_VARIABLE SPLIT_B_ERR
)

if(NOT SPLIT_B_RC EQUAL 0)
    message(FATAL_ERROR
        "warmup resume failed\n"
        "stdout:\n${SPLIT_B_OUT}\n"
        "stderr:\n${SPLIT_B_ERR}")
endif()

if(NOT SPLIT_B_ERR MATCHES
   "update=1/2 loss=[^ \r\n]+ optimizer_step=3 learning_rate=0[.]00075[0-9]*")
    message(FATAL_ERROR
        "resume did not restore warmup step 3\nstderr:\n${SPLIT_B_ERR}")
endif()

if(NOT SPLIT_B_ERR MATCHES
   "update=2/2 loss=[^ \r\n]+ optimizer_step=4 learning_rate=0[.]001[0-9]*")
    message(FATAL_ERROR
        "resume did not restore warmup step 4\nstderr:\n${SPLIT_B_ERR}")
endif()

# Final state must be byte-identical.
file(SHA256 "${ONE_CKPT}" ONE_CKPT_SHA)
file(SHA256 "${SPLIT_B_CKPT}" SPLIT_CKPT_SHA)
file(SHA256 "${ONE_CURSOR}" ONE_CURSOR_SHA)
file(SHA256 "${SPLIT_B_CURSOR}" SPLIT_CURSOR_SHA)

message(STATUS "ONE_CHECKPOINT_SHA256=${ONE_CKPT_SHA}")
message(STATUS "SPLIT_CHECKPOINT_SHA256=${SPLIT_CKPT_SHA}")
message(STATUS "ONE_CURSOR_SHA256=${ONE_CURSOR_SHA}")
message(STATUS "SPLIT_CURSOR_SHA256=${SPLIT_CURSOR_SHA}")

if(NOT ONE_CKPT_SHA STREQUAL SPLIT_CKPT_SHA)
    message(FATAL_ERROR
        "warmup checkpoint resume is not byte-equivalent")
endif()

if(NOT ONE_CURSOR_SHA STREQUAL SPLIT_CURSOR_SHA)
    message(FATAL_ERROR
        "warmup cursor resume is not byte-equivalent")
endif()

# Resume must not permit schedule override from CLI.
execute_process(
    COMMAND
        "${NIYAH_TRAIN}" resume
        --tokenizer "${TOKENIZER}"
        --shard "${SHARD}"
        --checkpoint-in "${SPLIT_A_CKPT}"
        --cursor-in "${SPLIT_A_CURSOR}"
        --checkpoint-out "${REJECT_CKPT}"
        --cursor-out "${REJECT_CURSOR}"
        --updates 2
        --batch-size 1
        --accumulation-steps 1
        --warmup-steps 4
    RESULT_VARIABLE REJECT_RC
    OUTPUT_VARIABLE REJECT_OUT
    ERROR_VARIABLE REJECT_ERR
)

if(REJECT_RC EQUAL 0)
    message(FATAL_ERROR
        "resume unexpectedly accepted --warmup-steps")
endif()

if(EXISTS "${REJECT_CKPT}" OR EXISTS "${REJECT_CURSOR}")
    message(FATAL_ERROR
        "rejected resume created output artifacts")
endif()

message(STATUS "NIYAH_TRAIN_WARMUP_CONTRACT=PASS")