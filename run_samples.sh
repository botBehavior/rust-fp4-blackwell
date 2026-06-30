#!/usr/bin/env bash
# Re-verify the capstone artifact: generate coherent text from TinyLlama-1.1B (MXFP4, pure Rust
# via cuda-oxide) for several candidate headline prompts, and report decode tok/s for each.
set -u
source $CUDAOXIDE_ROOT/env.sh
cd $CUDAOXIDE_ROOT/tok && cargo build --release -q 2>&1 | tail -3
TOK=$CUDAOXIDE_ROOT/target/release/tok
ROOT=$CUDAOXIDE_ROOT/cuda-oxide

PROMPTS=(
  "Rust is a systems programming language that"
  "The main advantage of running a neural network on a GPU instead of a CPU is"
  "Quantization makes large language models smaller by"
)

cd "$ROOT" || exit 1
for P in "${PROMPTS[@]}"; do
  IDS="$("$TOK" encode "$P")"
  echo "############################################################"
  echo "PROMPT: $P"
  echo "IDS: $IDS"
  OUT="$(PROMPT_IDS="$IDS" cargo oxide run llama_run 2>/dev/null)"
  echo "$OUT" | grep -E "gen:|output ids:"
  OIDS="$(echo "$OUT" | sed -n 's/^output ids: //p')"
  echo "----- FULL TEXT -----"
  "$TOK" decode $IDS $OIDS
  echo
done
echo "ALL_DONE"
