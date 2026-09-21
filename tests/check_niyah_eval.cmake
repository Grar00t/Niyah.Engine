if(NOT DEFINED NIYAH_FIXTURE OR
   NOT DEFINED NIYAH_TRAIN OR
   NOT DEFINED NIYAH_CLI OR
   NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "missing eval test arguments")
endif()

set(PREFIX "${WORK_DIR}/niyah_eval_fixture")
set(TOK "${PREFIX}.tok")
set(SHARD "${PREFIX}.srd")
set(SHARD_ALT "${PREFIX}_alt.srd")
set(CKPT "${PREFIX}.ckpt")
set(CURSOR "${PREFIX}.cursor")
set(HELDOUT "${PREFIX}_heldout.txt")
set(MISSING_TOK "${PREFIX}_missing.tok")
set(SUP_CORPUS "${PREFIX}_supervised.txt")
set(SUP_TOK "${PREFIX}_supervised.tok")
set(SUP_SHARD "${PREFIX}_supervised.srd")
set(SUP_CKPT "${PREFIX}_supervised.ckpt")
set(SUP_CURSOR "${PREFIX}_supervised.cursor")

file(REMOVE
    "${TOK}"
    "${SHARD}"
    "${SHARD_ALT}"
    "${CKPT}"
    "${CURSOR}"
    "${HELDOUT}"
    "${MISSING_TOK}"
    "${SUP_CORPUS}"
    "${SUP_TOK}"
    "${SUP_SHARD}"
    "${SUP_CKPT}"
    "${SUP_CURSOR}")

execute_process(
    COMMAND "${NIYAH_FIXTURE}"
        "${TOK}"
        "${SHARD}"
        "${SHARD_ALT}"
    RESULT_VARIABLE fixture_result
)
if(NOT fixture_result EQUAL 0)
    message(FATAL_ERROR "eval fixture creation failed: ${fixture_result}")
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
    ERROR_QUIET
)
if(NOT train_result EQUAL 0)
    message(FATAL_ERROR "eval checkpoint training failed: ${train_result}")
endif()

file(WRITE "${HELDOUT}" "held-out Ω eval text\n")
file(SHA256 "${CKPT}" checkpoint_sha_before)

execute_process(
    COMMAND "${NIYAH_CLI}" eval
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --heldout "${HELDOUT}"
    RESULT_VARIABLE eval_result
    OUTPUT_VARIABLE eval_output
    ERROR_VARIABLE eval_error
)
if(NOT eval_result EQUAL 0)
    message(FATAL_ERROR "niyah eval text failed: ${eval_result}: ${eval_error}")
endif()

string(FIND "${eval_output}" "EVAL_EXIT=0" eval_exit_index)
if(eval_exit_index EQUAL -1)
    message(FATAL_ERROR "eval output missing EVAL_EXIT=0: ${eval_output}")
endif()
string(FIND "${eval_output}" "mode=eval" eval_mode_index)
if(eval_mode_index EQUAL -1)
    message(FATAL_ERROR "eval output missing mode=eval: ${eval_output}")
endif()
string(FIND "${eval_output}" "format=text" eval_format_index)
if(eval_format_index EQUAL -1)
    message(FATAL_ERROR "eval output missing format=text: ${eval_output}")
endif()
string(FIND "${eval_output}" "objective=all_tokens" eval_objective_index)
if(eval_objective_index EQUAL -1)
    message(FATAL_ERROR "eval output missing objective=all_tokens: ${eval_output}")
endif()

string(TOLOWER "${eval_output}" eval_output_lower)
if(eval_output_lower MATCHES "(^|[^a-z])(nan|inf|infinity)([^a-z]|$)")
    message(FATAL_ERROR "eval output contains non-finite metric: ${eval_output}")
endif()
string(REGEX MATCH "(^|\n)mean_loss=([0-9]+\\.[0-9]+)" mean_match "${eval_output}")
if(NOT mean_match)
    message(FATAL_ERROR "eval output missing finite mean_loss: ${eval_output}")
endif()
string(REGEX MATCH "(^|\n)baseline_mean_loss=([0-9]+\\.[0-9]+)" baseline_match "${eval_output}")
if(NOT baseline_match)
    message(FATAL_ERROR "eval output missing finite baseline_mean_loss: ${eval_output}")
endif()

file(SHA256 "${CKPT}" checkpoint_sha_after_first)
if(NOT checkpoint_sha_before STREQUAL checkpoint_sha_after_first)
    message(FATAL_ERROR "checkpoint bytes changed after first eval")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" eval
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --heldout "${HELDOUT}"
    RESULT_VARIABLE eval_second_result
    OUTPUT_VARIABLE eval_second_output
    ERROR_VARIABLE eval_second_error
)
if(NOT eval_second_result EQUAL 0)
    message(FATAL_ERROR "second niyah eval failed: ${eval_second_result}: ${eval_second_error}")
endif()
file(SHA256 "${CKPT}" checkpoint_sha_after_second)
if(NOT checkpoint_sha_before STREQUAL checkpoint_sha_after_second)
    message(FATAL_ERROR "checkpoint bytes changed after second eval")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" eval
        --tokenizer "${MISSING_TOK}"
        --checkpoint "${CKPT}"
        --heldout "${HELDOUT}"
    RESULT_VARIABLE missing_tokenizer_result
    OUTPUT_QUIET
    ERROR_QUIET
)
if(missing_tokenizer_result EQUAL 0)
    message(FATAL_ERROR "missing tokenizer unexpectedly succeeded")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" eval
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --heldout "${HELDOUT}"
        --format shard
    RESULT_VARIABLE raw_as_shard_result
    OUTPUT_QUIET
    ERROR_QUIET
)
if(raw_as_shard_result EQUAL 0)
    message(FATAL_ERROR "raw text accepted as shard")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" eval
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --heldout "${SHARD}"
        --format shard
    RESULT_VARIABLE shard_eval_result
    OUTPUT_VARIABLE shard_eval_output
    ERROR_VARIABLE shard_eval_error
)
if(NOT shard_eval_result EQUAL 0)
    message(FATAL_ERROR "niyah eval shard failed: ${shard_eval_result}: ${shard_eval_error}")
endif()
string(FIND "${shard_eval_output}" "format=shard" shard_format_index)
if(shard_format_index EQUAL -1)
    message(FATAL_ERROR "shard eval output missing format=shard: ${shard_eval_output}")
endif()
string(FIND "${shard_eval_output}" "objective=all_tokens" shard_objective_index)
if(shard_objective_index EQUAL -1)
    message(FATAL_ERROR "shard eval output missing objective=all_tokens: ${shard_eval_output}")
endif()
string(FIND "${shard_eval_output}" "EVAL_EXIT=0" shard_exit_index)
if(shard_exit_index EQUAL -1)
    message(FATAL_ERROR "shard eval output missing EVAL_EXIT=0: ${shard_eval_output}")
endif()
string(FIND "${shard_eval_output}" "bits_per_byte=" shard_bpb_index)
if(NOT shard_bpb_index EQUAL -1)
    message(FATAL_ERROR "shard eval emitted bits_per_byte: ${shard_eval_output}")
endif()

execute_process(
    COMMAND "${NIYAH_CLI}" eval
        --tokenizer "${TOK}"
        --checkpoint "${CKPT}"
        --heldout "${SHARD}"
        --format shard
        --sequence-length 4
    RESULT_VARIABLE shard_sequence_result
    OUTPUT_QUIET
    ERROR_QUIET
)
if(shard_sequence_result EQUAL 0)
    message(FATAL_ERROR "shard eval accepted explicit --sequence-length")
endif()

file(WRITE "${SUP_CORPUS}" "User: q\nAssistant: a\n")
execute_process(
    COMMAND "${NIYAH_CLI}" prepare
        --corpus "${SUP_CORPUS}"
        --tokenizer-out "${SUP_TOK}"
        --shard-out "${SUP_SHARD}"
        --target-vocab 258
        --min-pair-frequency 2
        --sequence-length 32
        --record-mode blank-line
        --response-delimiter "Assistant:"
    RESULT_VARIABLE supervised_prepare_result
    OUTPUT_QUIET
    ERROR_VARIABLE supervised_prepare_error
)
if(NOT supervised_prepare_result EQUAL 0)
    message(FATAL_ERROR "supervised eval fixture prepare failed: ${supervised_prepare_result}: ${supervised_prepare_error}")
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
        --context-length 32
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
    RESULT_VARIABLE supervised_train_result
    OUTPUT_QUIET
    ERROR_VARIABLE supervised_train_error
)
if(NOT supervised_train_result EQUAL 0)
    message(FATAL_ERROR "supervised eval checkpoint training failed: ${supervised_train_result}: ${supervised_train_error}")
endif()

file(SHA256 "${SUP_CKPT}" supervised_checkpoint_sha_before)
execute_process(
    COMMAND "${NIYAH_CLI}" eval
        --tokenizer "${SUP_TOK}"
        --checkpoint "${SUP_CKPT}"
        --heldout "${SUP_SHARD}"
        --format shard
    RESULT_VARIABLE supervised_eval_result
    OUTPUT_VARIABLE supervised_eval_output
    ERROR_VARIABLE supervised_eval_error
)
if(NOT supervised_eval_result EQUAL 0)
    message(FATAL_ERROR "loss-masked supervised shard failed: ${supervised_eval_result}: ${supervised_eval_error}")
endif()
string(FIND "${supervised_eval_output}" "format=shard" supervised_format_index)
if(supervised_format_index EQUAL -1)
    message(FATAL_ERROR "supervised eval output missing format=shard: ${supervised_eval_output}")
endif()
string(FIND "${supervised_eval_output}" "objective=loss_masked" supervised_objective_index)
if(supervised_objective_index EQUAL -1)
    message(FATAL_ERROR "supervised eval output missing objective=loss_masked: ${supervised_eval_output}")
endif()
string(FIND "${supervised_eval_output}" "EVAL_EXIT=0" supervised_exit_index)
if(supervised_exit_index EQUAL -1)
    message(FATAL_ERROR "supervised eval output missing EVAL_EXIT=0: ${supervised_eval_output}")
endif()
string(FIND "${supervised_eval_output}" "bits_per_byte=" supervised_bpb_index)
if(NOT supervised_bpb_index EQUAL -1)
    message(FATAL_ERROR "supervised shard eval emitted bits_per_byte: ${supervised_eval_output}")
endif()
string(TOLOWER "${supervised_eval_output}" supervised_eval_output_lower)
if(supervised_eval_output_lower MATCHES "(^|[^a-z])(nan|inf|infinity)([^a-z]|$)")
    message(FATAL_ERROR "supervised eval output contains non-finite metric: ${supervised_eval_output}")
endif()
string(REGEX MATCH "(^|\n)mean_loss=([0-9]+\\.[0-9]+)" supervised_mean_match "${supervised_eval_output}")
if(NOT supervised_mean_match)
    message(FATAL_ERROR "supervised eval output missing finite mean_loss: ${supervised_eval_output}")
endif()
string(REGEX MATCH "(^|\n)baseline_mean_loss=([0-9]+\\.[0-9]+)" supervised_baseline_match "${supervised_eval_output}")
if(NOT supervised_baseline_match)
    message(FATAL_ERROR "supervised eval output missing finite baseline_mean_loss: ${supervised_eval_output}")
endif()
file(SHA256 "${SUP_CKPT}" supervised_checkpoint_sha_after)
if(NOT supervised_checkpoint_sha_before STREQUAL supervised_checkpoint_sha_after)
    message(FATAL_ERROR "supervised checkpoint bytes changed after eval")
endif()

message("EVAL_CLI=PASS")
