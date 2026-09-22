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

set(DETAILED_CKPT "${WORK_DIR}/niyah_progress_details.ckpt")
set(DETAILED_CURSOR "${WORK_DIR}/niyah_progress_details.cursor")
set(BASIC_CKPT "${WORK_DIR}/niyah_progress_basic.ckpt")
set(BASIC_CURSOR "${WORK_DIR}/niyah_progress_basic.cursor")

file(REMOVE
    "${DETAILED_CKPT}"
    "${DETAILED_CURSOR}"
    "${BASIC_CKPT}"
    "${BASIC_CURSOR}"
)

set(COMMON
    --tokenizer "${TOKENIZER}"
    --shard "${SHARD}"
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
)

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" new
        --checkpoint-out "${DETAILED_CKPT}"
        --cursor-out "${DETAILED_CURSOR}"
        ${COMMON}
        --progress-details
    RESULT_VARIABLE DETAILED_RESULT
    OUTPUT_VARIABLE DETAILED_STDOUT
    ERROR_VARIABLE DETAILED_STDERR
)

if(NOT DETAILED_RESULT EQUAL 0)
    message(FATAL_ERROR
        "detailed training failed\n"
        "stdout:\n${DETAILED_STDOUT}\n"
        "stderr:\n${DETAILED_STDERR}")
endif()

if(NOT DETAILED_STDERR MATCHES
   "update=1/2 loss=[^ \r\n]+ optimizer_step=1 learning_rate=[^ \r\n]+ cursor_epoch=[0-9]+ cursor_position=[0-9]+")
    message(FATAL_ERROR
        "first detailed progress record missing\n${DETAILED_STDERR}")
endif()

if(NOT DETAILED_STDERR MATCHES
   "update=2/2 loss=[^ \r\n]+ optimizer_step=2 learning_rate=[^ \r\n]+ cursor_epoch=[0-9]+ cursor_position=[0-9]+")
    message(FATAL_ERROR
        "second detailed progress record missing\n${DETAILED_STDERR}")
endif()

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" new
        --checkpoint-out "${BASIC_CKPT}"
        --cursor-out "${BASIC_CURSOR}"
        ${COMMON}
    RESULT_VARIABLE BASIC_RESULT
    OUTPUT_VARIABLE BASIC_STDOUT
    ERROR_VARIABLE BASIC_STDERR
)

if(NOT BASIC_RESULT EQUAL 0)
    message(FATAL_ERROR
        "basic training failed\n"
        "stdout:\n${BASIC_STDOUT}\n"
        "stderr:\n${BASIC_STDERR}")
endif()

if(BASIC_STDERR MATCHES "optimizer_step=")
    message(FATAL_ERROR
        "default progress output changed unexpectedly\n${BASIC_STDERR}")
endif()

if(BASIC_STDERR MATCHES "learning_rate=")
    message(FATAL_ERROR
        "default progress leaked detailed fields\n${BASIC_STDERR}")
endif()

if(NOT BASIC_STDERR MATCHES "update=1/2 loss=[^ \r\n]+")
    message(FATAL_ERROR
        "legacy progress record missing\n${BASIC_STDERR}")
endif()

message(STATUS "NIYAH_TRAIN_PROGRESS_DETAILS=PASS")