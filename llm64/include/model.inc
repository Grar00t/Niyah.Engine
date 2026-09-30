%ifndef NIYAH_LLM64_MODEL_INC
%define NIYAH_LLM64_MODEL_INC 1

; Current Niyah checkpoint V2 header (72 bytes)
%define CKPT_HEADER_SIZE        72
%define CKPT_MAGIC_OFF          0
%define CKPT_VERSION_OFF        8
%define CKPT_FLAGS_OFF          12
%define CKPT_SECTION_COUNT_OFF  16
%define CKPT_RESERVED_OFF       20
%define CKPT_VOCAB_OFF          24
%define CKPT_CONTEXT_OFF        28
%define CKPT_DIM_OFF            32
%define CKPT_LAYERS_OFF         36
%define CKPT_HEADS_OFF          40
%define CKPT_KV_HEADS_OFF       44
%define CKPT_FFN_OFF            48
%define CKPT_RMS_EPS_OFF        52
%define CKPT_TIE_OFF            56
%define CKPT_SEGMENTS_OFF       60
%define CKPT_WEIGHT_COUNT_OFF   64

%define CKPT_VERSION_V2                  2
%define CKPT_MAX_SECTIONS                64
%define CKPT_SECTION_HEADER              16
%define CKPT_SECTION_REQUIRED            1
%define CKPT_SECTION_MODEL               1
%define CKPT_SECTION_ADAMW_M             2
%define CKPT_SECTION_ADAMW_V             3
%define CKPT_SECTION_ADAMW_META          4
%define CKPT_SECTION_TOKENIZER_IDENTITY  5
%define CKPT_SECTION_LR_SCHEDULE         6
%define CKPT_ADAMW_META_SIZE             32
%define CKPT_TOKENIZER_IDENTITY_SIZE     32
%define CKPT_LR_SCHEDULE_SIZE            8
%define CKPT_SEEN_V2_REQUIRED            0x1f
%define CKPT_SEEN_V2_WITH_SCHEDULE       0x3f
%define CHECKSUM_CRC32                   1
%define FILE_FOOTER_SIZE                 8

; Tokenizer v1 header (32 bytes)
%define TOK_HEADER_SIZE         32
%define TOK_VERSION_OFF         8
%define TOK_FLAGS_OFF           12
%define TOK_BASE_VOCAB_OFF      16
%define TOK_VOCAB_OFF           20
%define TOK_MERGE_COUNT_OFF     24
%define TOK_RESERVED_OFF        28
%define TOK_VERSION_V1          1
%define TOK_BASE_VOCAB          258
%define TOK_BOS                 256
%define TOK_EOS                 257
%define TOK_MERGE_SIZE          12

%endif
