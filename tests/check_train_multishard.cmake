if(NOT DEFINED NIYAH_TRAIN_EXE OR
   NOT DEFINED NIYAH_TRAIN_TOK OR
   NOT DEFINED NIYAH_TRAIN_SHARD_A OR
   NOT DEFINED NIYAH_TRAIN_SHARD_B OR
   NOT DEFINED NIYAH_TRAIN_MULTI_PREFIX)
    message(FATAL_ERROR "missing required multi-shard test argument")
endif()

set(ckpt1 "${NIYAH_TRAIN_MULTI_PREFIX}_1.ckpt")
set(cur1 "${NIYAH_TRAIN_MULTI_PREFIX}_1.cursor")
set(ckpt2 "${NIYAH_TRAIN_MULTI_PREFIX}_2.ckpt")
set(cur2 "${NIYAH_TRAIN_MULTI_PREFIX}_2.cursor")
set(bad_ckpt "${NIYAH_TRAIN_MULTI_PREFIX}_bad.ckpt")
set(bad_cur "${NIYAH_TRAIN_MULTI_PREFIX}_bad.cursor")
file(REMOVE "${ckpt1}" "${cur1}" "${ckpt2}" "${cur2}" "${bad_ckpt}" "${bad_cur}")

execute_process(
    COMMAND "${NIYAH_TRAIN_EXE}" new
        --tokenizer "${NIYAH_TRAIN_TOK}"
        --shard "${NIYAH_TRAIN_SHARD_A}"
        --shard "${NIYAH_TRAIN_SHARD_B}"
        --checkpoint-out "${ckpt1}"
        --cursor-out "${cur1}"
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
    RESULT_VARIABLE new_result
    OUTPUT_VARIABLE new_stdout
    ERROR_VARIABLE new_stderr)

if(NOT new_result EQUAL 0)
    message(FATAL_ERROR
        "multi-shard new failed: result=${new_result}\n"
        "stdout=${new_stdout}\nstderr=${new_stderr}")
endif()
if(NOT new_stdout MATCHES "shards=2")
    message(FATAL_ERROR "multi-shard new did not report shards=2\n${new_stdout}")
endif()

execute_process(
    COMMAND "${NIYAH_TRAIN_EXE}" resume
        --tokenizer "${NIYAH_TRAIN_TOK}"
        --shard "${NIYAH_TRAIN_SHARD_A}"
        --shard "${NIYAH_TRAIN_SHARD_B}"
        --checkpoint-in "${ckpt1}"
        --cursor-in "${cur1}"
        --checkpoint-out "${ckpt2}"
        --cursor-out "${cur2}"
        --updates 2
        --batch-size 1
        --accumulation-steps 1
    RESULT_VARIABLE resume_result
    OUTPUT_VARIABLE resume_stdout
    ERROR_VARIABLE resume_stderr)

if(NOT resume_result EQUAL 0)
    message(FATAL_ERROR
        "multi-shard resume failed: result=${resume_result}\n"
        "stdout=${resume_stdout}\nstderr=${resume_stderr}")
endif()
if(NOT resume_stdout MATCHES "shards=2")
    message(FATAL_ERROR "multi-shard resume did not report shards=2\n${resume_stdout}")
endif()

execute_process(
    COMMAND "${NIYAH_TRAIN_EXE}" resume
        --tokenizer "${NIYAH_TRAIN_TOK}"
        --shard "${NIYAH_TRAIN_SHARD_B}"
        --shard "${NIYAH_TRAIN_SHARD_A}"
        --checkpoint-in "${ckpt1}"
        --cursor-in "${cur1}"
        --checkpoint-out "${bad_ckpt}"
        --cursor-out "${bad_cur}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
    RESULT_VARIABLE bad_result
    OUTPUT_VARIABLE bad_stdout
    ERROR_VARIABLE bad_stderr)

file(REMOVE "${ckpt1}" "${cur1}" "${ckpt2}" "${cur2}" "${bad_ckpt}" "${bad_cur}")

if(bad_result EQUAL 0)
    message(FATAL_ERROR "reordered multi-shard resume unexpectedly succeeded")
endif()
set(bad_output "${bad_stdout}\n${bad_stderr}")
if(NOT bad_output MATCHES "stage=resume_compatibility")
    message(FATAL_ERROR
        "reordered multi-shard resume failed for wrong reason: result=${bad_result}\n"
        "stdout=${bad_stdout}\nstderr=${bad_stderr}")
endif()

message(STATUS "P6KC_MULTI_SHARD_NEW_RESUME=PASS")
message(STATUS "P6KC_MULTI_SHARD_ORDER_BINDING=PASS")
