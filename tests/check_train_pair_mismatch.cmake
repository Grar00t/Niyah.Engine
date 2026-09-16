if(NOT DEFINED NIYAH_TRAIN_EXE OR
   NOT DEFINED NIYAH_TRAIN_TOK OR
   NOT DEFINED NIYAH_TRAIN_SHARD OR
   NOT DEFINED NIYAH_TRAIN_PAIR_PREFIX)
    message(FATAL_ERROR "missing required pair-binding test argument")
endif()

set(a_ckpt "${NIYAH_TRAIN_PAIR_PREFIX}_a.ckpt")
set(a_cur  "${NIYAH_TRAIN_PAIR_PREFIX}_a.cursor")
set(b_ckpt "${NIYAH_TRAIN_PAIR_PREFIX}_b.ckpt")
set(b_cur  "${NIYAH_TRAIN_PAIR_PREFIX}_b.cursor")
set(out_ckpt "${NIYAH_TRAIN_PAIR_PREFIX}_out.ckpt")
set(out_cur  "${NIYAH_TRAIN_PAIR_PREFIX}_out.cursor")
file(REMOVE "${a_ckpt}" "${a_cur}" "${b_ckpt}" "${b_cur}"
            "${out_ckpt}" "${out_cur}")

function(make_pair seed ckpt cur)
    execute_process(
        COMMAND "${NIYAH_TRAIN_EXE}" new
            --tokenizer "${NIYAH_TRAIN_TOK}"
            --shard "${NIYAH_TRAIN_SHARD}"
            --checkpoint-out "${ckpt}"
            --cursor-out "${cur}"
            --updates 2
            --batch-size 1
            --accumulation-steps 1
            --model-seed "${seed}"
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
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR
            "pair setup failed: result=${result}\nstdout=${stdout}\nstderr=${stderr}")
    endif()
endfunction()

make_pair(42 "${a_ckpt}" "${a_cur}")
make_pair(43 "${b_ckpt}" "${b_cur}")

execute_process(
    COMMAND "${NIYAH_TRAIN_EXE}" resume
        --tokenizer "${NIYAH_TRAIN_TOK}"
        --shard "${NIYAH_TRAIN_SHARD}"
        --checkpoint-in "${a_ckpt}"
        --cursor-in "${b_cur}"
        --checkpoint-out "${out_ckpt}"
        --cursor-out "${out_cur}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
    RESULT_VARIABLE mixed_result
    OUTPUT_VARIABLE mixed_stdout
    ERROR_VARIABLE mixed_stderr)

file(REMOVE "${a_ckpt}" "${a_cur}" "${b_ckpt}" "${b_cur}"
            "${out_ckpt}" "${out_cur}")

if(mixed_result EQUAL 0)
    message(FATAL_ERROR "mixed checkpoint/cursor pair unexpectedly resumed")
endif()
set(mixed_output "${mixed_stdout}\n${mixed_stderr}")
if(NOT mixed_output MATCHES "stage=resume_compatibility")
    message(FATAL_ERROR
        "mixed pair failed for wrong reason: result=${mixed_result}\n"
        "stdout=${mixed_stdout}\nstderr=${mixed_stderr}")
endif()

message(STATUS "P6KD_CHECKPOINT_CURSOR_PAIR_BINDING=PASS")
