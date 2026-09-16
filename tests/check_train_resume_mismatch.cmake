if(NOT DEFINED NIYAH_TRAIN_EXE OR
   NOT DEFINED NIYAH_TRAIN_TOK OR
   NOT DEFINED NIYAH_TRAIN_SHARD OR
   NOT DEFINED NIYAH_TRAIN_SHARD_ALT OR
   NOT DEFINED NIYAH_TRAIN_CKPT OR
   NOT DEFINED NIYAH_TRAIN_CUR)
    message(FATAL_ERROR "missing required mismatch-test argument")
endif()

file(REMOVE "${NIYAH_TRAIN_CKPT}" "${NIYAH_TRAIN_CUR}")

execute_process(
    COMMAND "${NIYAH_TRAIN_EXE}" new
        --tokenizer "${NIYAH_TRAIN_TOK}"
        --shard "${NIYAH_TRAIN_SHARD}"
        --checkpoint-out "${NIYAH_TRAIN_CKPT}"
        --cursor-out "${NIYAH_TRAIN_CUR}"
        --updates 1
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
    RESULT_VARIABLE new_result
    OUTPUT_VARIABLE new_stdout
    ERROR_VARIABLE new_stderr)

if(NOT new_result EQUAL 0)
    message(FATAL_ERROR
        "mismatch-test setup failed: result=${new_result}\n"
        "stdout=${new_stdout}\nstderr=${new_stderr}")
endif()

set(mismatch_ckpt "${NIYAH_TRAIN_CKPT}.mismatch")
set(mismatch_cur "${NIYAH_TRAIN_CUR}.mismatch")
file(REMOVE "${mismatch_ckpt}" "${mismatch_cur}")

execute_process(
    COMMAND "${NIYAH_TRAIN_EXE}" resume
        --tokenizer "${NIYAH_TRAIN_TOK}"
        --shard "${NIYAH_TRAIN_SHARD_ALT}"
        --checkpoint-in "${NIYAH_TRAIN_CKPT}"
        --cursor-in "${NIYAH_TRAIN_CUR}"
        --checkpoint-out "${mismatch_ckpt}"
        --cursor-out "${mismatch_cur}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
    RESULT_VARIABLE resume_result
    OUTPUT_VARIABLE resume_stdout
    ERROR_VARIABLE resume_stderr)

file(REMOVE
    "${NIYAH_TRAIN_CKPT}" "${NIYAH_TRAIN_CUR}"
    "${mismatch_ckpt}" "${mismatch_cur}")

if(resume_result EQUAL 0)
    message(FATAL_ERROR
        "dataset mismatch unexpectedly resumed successfully\n"
        "stdout=${resume_stdout}\nstderr=${resume_stderr}")
endif()

set(resume_output "${resume_stdout}\n${resume_stderr}")
if(NOT resume_output MATCHES "stage=resume_compatibility")
    message(FATAL_ERROR
        "dataset mismatch failed for the wrong reason: result=${resume_result}\n"
        "stdout=${resume_stdout}\nstderr=${resume_stderr}")
endif()

message(STATUS "P6KB_RESUME_DATASET_MISMATCH_REASON=PASS")
