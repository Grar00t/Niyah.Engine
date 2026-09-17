if(NOT DEFINED NIYAH_CLI)
    message(FATAL_ERROR "NIYAH_CLI is required")
endif()
if(NOT DEFINED NIYAH_TRAIN)
    message(FATAL_ERROR "NIYAH_TRAIN is required")
endif()
if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "WORK_DIR is required")
endif()

set(CORPUS "${WORK_DIR}/p8c_corpus.txt")
set(TOK_A "${WORK_DIR}/p8c_a.tok")
set(SHARD_A "${WORK_DIR}/p8c_a.srd")
set(TOK_B "${WORK_DIR}/p8c_b.tok")
set(SHARD_B "${WORK_DIR}/p8c_b.srd")
set(CKPT "${WORK_DIR}/p8c_train.ckpt")
set(CURSOR "${WORK_DIR}/p8c_train.cursor")
set(RECORD_CORPUS "${WORK_DIR}/p8d_records.txt")
set(RECORD_TOK "${WORK_DIR}/p8d_records.tok")
set(RECORD_SHARD "${WORK_DIR}/p8d_records.srd")
set(RECORD_CKPT "${WORK_DIR}/p8d_records.ckpt")
set(RECORD_CURSOR "${WORK_DIR}/p8d_records.cursor")
set(SUP_CORPUS "${WORK_DIR}/p8f_supervised.txt")
set(SUP_TOK "${WORK_DIR}/p8f_supervised.tok")
set(SUP_SHARD "${WORK_DIR}/p8f_supervised.srd")
set(SUP_CKPT "${WORK_DIR}/p8f_supervised.ckpt")
set(SUP_CURSOR "${WORK_DIR}/p8f_supervised.cursor")

file(REMOVE
    "${CORPUS}"
    "${TOK_A}"
    "${SHARD_A}"
    "${TOK_B}"
    "${SHARD_B}"
    "${CKPT}"
    "${CURSOR}"
    "${RECORD_CORPUS}"
    "${RECORD_TOK}"
    "${RECORD_SHARD}"
    "${RECORD_CKPT}"
    "${RECORD_CURSOR}"
    "${SUP_CORPUS}"
    "${SUP_TOK}"
    "${SUP_SHARD}"
    "${SUP_CKPT}"
    "${SUP_CURSOR}")

file(WRITE "${CORPUS}"
    "hello world hello world\n"
    "native corpus training path\n"
    "hello assistant hello user\n"
    "deterministic tokenizer shard\n")

execute_process(
    COMMAND "${NIYAH_CLI}" prepare
        --corpus "${CORPUS}"
        --tokenizer-out "${TOK_A}"
        --shard-out "${SHARD_A}"
        --target-vocab 270
        --min-pair-frequency 2
        --sequence-length 4
    RESULT_VARIABLE prepare_a_result
    OUTPUT_VARIABLE prepare_a_output
    ERROR_VARIABLE prepare_a_error)

if(NOT prepare_a_result EQUAL 0)
    message(FATAL_ERROR
        "prepare A failed: ${prepare_a_result}\n"
        "stdout=${prepare_a_output}\n"
        "stderr=${prepare_a_error}")
endif()

if(NOT prepare_a_output MATCHES "P8C_PREPARE=PASS")
    message(FATAL_ERROR "prepare A marker missing: ${prepare_a_output}")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" prepare
        --corpus "${CORPUS}"
        --tokenizer-out "${TOK_B}"
        --shard-out "${SHARD_B}"
        --target-vocab 270
        --min-pair-frequency 2
        --sequence-length 4
    RESULT_VARIABLE prepare_b_result
    OUTPUT_VARIABLE prepare_b_output
    ERROR_VARIABLE prepare_b_error)

if(NOT prepare_b_result EQUAL 0)
    message(FATAL_ERROR
        "prepare B failed: ${prepare_b_result}\n"
        "stdout=${prepare_b_output}\n"
        "stderr=${prepare_b_error}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${TOK_A}" "${TOK_B}"
    RESULT_VARIABLE tokenizer_compare_result)
if(NOT tokenizer_compare_result EQUAL 0)
    message(FATAL_ERROR "tokenizer output is not deterministic")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${SHARD_A}" "${SHARD_B}"
    RESULT_VARIABLE shard_compare_result)
if(NOT shard_compare_result EQUAL 0)
    message(FATAL_ERROR "shard output is not deterministic")
endif()

execute_process(
    COMMAND "${NIYAH_TRAIN}" new
        --tokenizer "${TOK_A}"
        --shard "${SHARD_A}"
        --checkpoint-out "${CKPT}"
        --cursor-out "${CURSOR}"
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
        --weight-decay 0
        --max-grad-norm 1
    RESULT_VARIABLE train_result
    OUTPUT_VARIABLE train_output
    ERROR_VARIABLE train_error)

if(NOT train_result EQUAL 0)
    message(FATAL_ERROR
        "prepared artifacts were rejected by niyah-train: ${train_result}\n"
        "stdout=${train_output}\n"
        "stderr=${train_error}")
endif()

if(NOT EXISTS "${CKPT}" OR NOT EXISTS "${CURSOR}")
    message(FATAL_ERROR "training outputs missing")
endif()

file(WRITE "${RECORD_CORPUS}"
    "User: A\n"
    "Assistant: 1\n"
    "\n"
    "User: B\n"
    "Assistant: 2\n")

execute_process(
    COMMAND "${NIYAH_CLI}" prepare
        --corpus "${RECORD_CORPUS}"
        --tokenizer-out "${RECORD_TOK}"
        --shard-out "${RECORD_SHARD}"
        --target-vocab 258
        --min-pair-frequency 1
        --sequence-length 64
        --record-mode blank-line
    RESULT_VARIABLE record_prepare_result
    OUTPUT_VARIABLE record_prepare_output
    ERROR_VARIABLE record_prepare_error)

if(NOT record_prepare_result EQUAL 0)
    message(FATAL_ERROR
        "boundary-aware prepare failed: ${record_prepare_result}\n"
        "stdout=${record_prepare_output}\n"
        "stderr=${record_prepare_error}")
endif()

execute_process(
    COMMAND "${NIYAH_TRAIN}" new
        --tokenizer "${RECORD_TOK}"
        --shard "${RECORD_SHARD}"
        --checkpoint-out "${RECORD_CKPT}"
        --cursor-out "${RECORD_CURSOR}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
        --model-seed 42
        --data-seed 7
        --context-length 64
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
        --weight-decay 0
        --max-grad-norm 1
    RESULT_VARIABLE record_train_result
    OUTPUT_VARIABLE record_train_output
    ERROR_VARIABLE record_train_error)

if(NOT record_train_result EQUAL 0)
    message(FATAL_ERROR
        "boundary-aware shard rejected by niyah-train: ${record_train_result}\n"
        "stdout=${record_train_output}\n"
        "stderr=${record_train_error}")
endif()

message("P8D_BOUNDARY_AWARE_PREPARE=PASS")

file(WRITE "${SUP_CORPUS}"
    "User: A\n"
    "Assistant: 1\n"
    "\n"
    "User: B\n"
    "Bot: 2\n")

execute_process(
    COMMAND "${NIYAH_CLI}" prepare
        --corpus "${SUP_CORPUS}"
        --tokenizer-out "${SUP_TOK}"
        --shard-out "${SUP_SHARD}"
        --target-vocab 258
        --min-pair-frequency 1
        --sequence-length 64
        --record-mode blank-line
        --response-delimiter "Assistant: "
        --response-delimiter "Bot: "
    RESULT_VARIABLE sup_prepare_result
    OUTPUT_VARIABLE sup_prepare_output
    ERROR_VARIABLE sup_prepare_error)

if(NOT sup_prepare_result EQUAL 0)
    message(FATAL_ERROR
        "supervised prepare failed: ${sup_prepare_result}\n"
        "stdout=${sup_prepare_output}\n"
        "stderr=${sup_prepare_error}")
endif()

if(NOT sup_prepare_output MATCHES
    "P8F_SUPERVISED_PREPARE=PASS delimiters=2")
    message(FATAL_ERROR
        "supervised prepare marker missing: ${sup_prepare_output}")
endif()

execute_process(
    COMMAND "${NIYAH_TRAIN}" new
        --tokenizer "${SUP_TOK}"
        --shard "${SUP_SHARD}"
        --checkpoint-out "${SUP_CKPT}"
        --cursor-out "${SUP_CURSOR}"
        --updates 1
        --batch-size 1
        --accumulation-steps 1
        --model-seed 42
        --data-seed 7
        --context-length 64
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
        --weight-decay 0
        --max-grad-norm 1
    RESULT_VARIABLE sup_train_result
    OUTPUT_VARIABLE sup_train_output
    ERROR_VARIABLE sup_train_error)

if(NOT sup_train_result EQUAL 0)
    message(FATAL_ERROR
        "supervised shard rejected by niyah-train: ${sup_train_result}\n"
        "stdout=${sup_train_output}\n"
        "stderr=${sup_train_error}")
endif()

if(NOT EXISTS "${SUP_CKPT}" OR
   NOT EXISTS "${SUP_CURSOR}")
    message(FATAL_ERROR
        "supervised training outputs missing")
endif()

message("P8F_SUPERVISED_PREPARE=PASS")
message("P8C_NATIVE_CORPUS_PREPARE=PASS")
