%ifndef NIYAH_LLM64_LAYOUT_INC
%define NIYAH_LLM64_LAYOUT_INC 1
%define L_TOKEN_EMBED 0
%define L_SEGMENT_EMBED 8
%define L_LAYERS 16
%define L_LAYER_STRIDE 24
%define L_FINAL_NORM 32
%define L_LM_HEAD 40
%define L_TOTAL_FLOATS 48
%define L_HEAD_DIM 56
%define L_KV_DIM 64
%define L_SIZE 72
%define LL_ATTN_NORM 0
%define LL_WQ 8
%define LL_WK 16
%define LL_WV 24
%define LL_WO 32
%define LL_FFN_NORM 40
%define LL_W_GATE 48
%define LL_W_UP 56
%define LL_W_DOWN 64
%define LL_SIZE 72
%endif
