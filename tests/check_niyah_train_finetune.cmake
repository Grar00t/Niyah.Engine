foreach(VAR_NAME
    IN ITEMS
        NIYAH_TRAIN
        TOKENIZER
        BASE_SHARD
        FINETUNE_SHARD
        BASE_CHECKPOINT
        WORK_DIR
)
    if(NOT DEFINED ${VAR_NAME})
        message(FATAL_ERROR
            "${VAR_NAME} is required")
    endif()
endforeach()

foreach(INPUT_PATH
    IN ITEMS
        "${NIYAH_TRAIN}"
        "${TOKENIZER}"
        "${BASE_SHARD}"
        "${FINETUNE_SHARD}"
        "${BASE_CHECKPOINT}"
)
    if(NOT EXISTS "${INPUT_PATH}")
        message(FATAL_ERROR
            "required input missing: ${INPUT_PATH}")
    endif()
endforeach()

set(FT_CKPT
    "${WORK_DIR}/niyah_finetune_reconstructed.ckpt")
set(FT_CURSOR
    "${WORK_DIR}/niyah_finetune_reconstructed.cursor")

set(RESUME_CKPT
    "${WORK_DIR}/niyah_finetune_resume.ckpt")
set(RESUME_CURSOR
    "${WORK_DIR}/niyah_finetune_resume.cursor")

set(MISMATCH_CKPT
    "${WORK_DIR}/niyah_finetune_mismatch.ckpt")
set(MISMATCH_CURSOR
    "${WORK_DIR}/niyah_finetune_mismatch.cursor")

file(REMOVE
    "${FT_CKPT}"
    "${FT_CURSOR}"
    "${RESUME_CKPT}"
    "${RESUME_CURSOR}"
    "${MISMATCH_CKPT}"
    "${MISMATCH_CURSOR}"
)

file(SHA256
    "${BASE_CHECKPOINT}"
    BASE_SHA_BEFORE
)

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" finetune
        --tokenizer "${TOKENIZER}"
        --shard "${FINETUNE_SHARD}"
        --checkpoint-in "${BASE_CHECKPOINT}"
        --checkpoint-out "${FT_CKPT}"
        --cursor-out "${FT_CURSOR}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
        --data-seed 19
        --learning-rate 0.001
        --beta1 0.9
        --beta2 0.999
        --epsilon 0.00000001
        --weight-decay 0.0
        --max-grad-norm 1.0
    RESULT_VARIABLE FT_RESULT
    OUTPUT_VARIABLE FT_STDOUT
    ERROR_VARIABLE FT_STDERR
)

if(NOT FT_RESULT EQUAL 0)
    message(FATAL_ERROR
        "finetune failed\n"
        "stdout:\n${FT_STDOUT}\n"
        "stderr:\n${FT_STDERR}")
endif()

if(NOT EXISTS "${FT_CKPT}")
    message(FATAL_ERROR
        "finetune checkpoint missing")
endif()

if(NOT EXISTS "${FT_CURSOR}")
    message(FATAL_ERROR
        "finetune cursor missing")
endif()

string(FIND
    "${FT_STDOUT}"
    "mode=finetune"
    MODE_POS
)

if(MODE_POS LESS 0)
    message(FATAL_ERROR
        "finetune mode marker missing")
endif()

string(FIND
    "${FT_STDOUT}"
    "optimizer_step=1"
    STEP_POS
)

if(STEP_POS LESS 0)
    message(FATAL_ERROR
        "optimizer was not observably reset")
endif()

file(SHA256
    "${BASE_CHECKPOINT}"
    BASE_SHA_AFTER
)

if(NOT BASE_SHA_BEFORE STREQUAL BASE_SHA_AFTER)
    message(FATAL_ERROR
        "input checkpoint was mutated")
endif()

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" resume
        --tokenizer "${TOKENIZER}"
        --shard "${FINETUNE_SHARD}"
        --checkpoint-in "${FT_CKPT}"
        --cursor-in "${FT_CURSOR}"
        --checkpoint-out "${RESUME_CKPT}"
        --cursor-out "${RESUME_CURSOR}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
    RESULT_VARIABLE RESUME_RESULT
    OUTPUT_VARIABLE RESUME_STDOUT
    ERROR_VARIABLE RESUME_STDERR
)

if(NOT RESUME_RESULT EQUAL 0)
    message(FATAL_ERROR
        "resume after finetune failed\n"
        "stdout:\n${RESUME_STDOUT}\n"
        "stderr:\n${RESUME_STDERR}")
endif()

if(NOT EXISTS "${RESUME_CKPT}")
    message(FATAL_ERROR
        "resume checkpoint missing")
endif()

if(NOT EXISTS "${RESUME_CURSOR}")
    message(FATAL_ERROR
        "resume cursor missing")
endif()

execute_process(
    COMMAND
        "${NIYAH_TRAIN}" resume
        --tokenizer "${TOKENIZER}"
        --shard "${BASE_SHARD}"
        --checkpoint-in "${FT_CKPT}"
        --cursor-in "${FT_CURSOR}"
        --checkpoint-out "${MISMATCH_CKPT}"
        --cursor-out "${MISMATCH_CURSOR}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
    RESULT_VARIABLE MISMATCH_RESULT
    OUTPUT_VARIABLE MISMATCH_STDOUT
    ERROR_VARIABLE MISMATCH_STDERR
)

if(MISMATCH_RESULT EQUAL 0)
    message(FATAL_ERROR
        "finetune cursor accepted wrong dataset")
endif()

message(STATUS
    "FINETUNE_MODE=PASS")

message(STATUS
    "FINETUNE_OPTIMIZER_RESET=PASS")

message(STATUS
    "FINETUNE_INPUT_CHECKPOINT_IMMUTABLE=PASS")

message(STATUS
    "FINETUNE_RESUME=PASS")

message(STATUS
    "FINETUNE_DATASET_MISMATCH=REJECTED")

message(STATUS
    "NIYAH_TRAIN_FINETUNE_RECONSTRUCTED=PASS")
