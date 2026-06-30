#!/usr/bin/env bash
# Clean canonical head-to-head on an UNCONTENDED GPU (pinned to physical GPU 1 via
# CUDA_VISIBLE_DEVICES=1, which both engines see as logical device 0). Same model, same card.
set -u
source $CUDAOXIDE_ROOT/env.sh
export CUDA_VISIBLE_DEVICES=1
ROOT=$CUDAOXIDE_ROOT/cuda-oxide

echo "############ GPU pinned to physical device 1 (clean) ############"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader

echo
echo "############ OURS — pure-Rust cuda-oxide MXFP4, prompt 'Hello' (BOS+15043), 5 runs ############"
cd "$ROOT" || exit 1
for i in 1 2 3 4 5; do
  printf "run %d: " "$i"
  PROMPT_IDS="1 15043" cargo oxide run llama_run 2>/dev/null | grep -E "gen:"
done

echo
echo "############ llama.cpp — TinyLlama-1.1B Q4_K_M, all layers GPU ############"
BIN=$CUDAOXIDE_ROOT/llama.cpp/build/bin/llama-bench
GGUF=$CUDAOXIDE_ROOT/tinyllama-q4.gguf
"$BIN" -m "$GGUF" -ngl 99 2>&1 | grep -E "\| (model|llama)" | tail -4
echo "ALL_DONE"
