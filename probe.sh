#!/usr/bin/env bash
# sm_120 capability probe — corrected classifier (per a corrected-classifier review).
# RUN-OK requires a REAL kernel-execution marker; sm_100/Hopper reject+skip are caught
# BEFORE any generic success/rc=0 fallthrough (the old grep matched PTX-gen boilerplate).
source $CUDAOXIDE_ROOT/env.sh
cd $CUDAOXIDE_ROOT/cuda-oxide
mkdir -p $CUDAOXIDE_ROOT/probe-logs
SUMMARY=$CUDAOXIDE_ROOT/probe-logs/SUMMARY.txt
: > "$SUMMARY"

# vecadd = baseline; POS expected to execute on sm_120; NEG = datacenter/Hopper features.
POS="vecadd atomics sharedmem warp_reduce cp_async_small bf16x2_arith dotprod inline_ptx tiled_gemm gemm"
NEG="tma_copy cluster tcgen05 tcgen05_matmul wgmma tma_multicast gemm_sol"

# prints "VERDICT<TAB>decisive-line"
classify() {
  local rc="$1" log="$2" rl
  # 1. HARD REJECT by driver/ptxas (e.g. sm_100a PTX cannot load on sm_120)
  rl=$(grep -iE "cannot be compiled to future architecture|CUDA_ERROR_INVALID_PTX|rejected the PTX|cuModuleLoad failed|PTX JIT compilation failed" "$log" | head -1)
  if [ -n "$rl" ]; then printf "REJECT(driver)\t%s" "$rl"; return; fi
  # 2. GRACEFUL SKIP (example self-detects it needs sm_100/sm_100a/Hopper)
  rl=$(grep -iE "requires sm_100|run on sm_100|WGMMA is Hopper|don.t exist on this architecture|requires sm_100a|skipping:|has no tcgen05|Example \(sm_100a\)" "$log" | head -1)
  if [ -n "$rl" ]; then printf "SKIP(sm_100/sm_90-only)\t%s" "$rl"; return; fi
  # 3. timeout
  if [ "$rc" = "124" ]; then printf "TIMEOUT\t-"; return; fi
  # 4. REAL kernel-execution success (NOT "PTX generated successfully" boilerplate)
  rl=$(grep -iE "All [0-9]+ [a-z ]*(correct|match|copied|passed|completed)|tests? passed|PASSED|🎉|results are correct|results match|computed correctly|SUCCESS:|SUCCESS!|PASS:|sums correct|neighbor (read|sum)" "$log" | head -1)
  if [ -n "$rl" ]; then printf "RUN-OK\t%s" "$rl"; return; fi
  # 5. build/codegen failure
  rl=$(grep -iE "error\[|panicked|not yet implemented|unimplemented|ptxas fatal|could not compile|CannotYetSelect|LLVM ERROR|report_fatal_error" "$log" | head -1)
  if [ -n "$rl" ]; then printf "BUILD-FAIL\t%s" "$rl"; return; fi
  [ "$rc" = "0" ] && printf "RAN(no-marker)\t-" || printf "FAIL(rc=%s)\t-" "$rc"
}

run_one() {
  local ex="$1" tag="$2"
  local log=$CUDAOXIDE_ROOT/probe-logs/$ex.log
  # auto-detect arch (resolves to sm_120a via nvidia-smi) = representative user behavior.
  # Datacenter examples self-target sm_100a; we record their natural reject/skip.
  timeout 240 cargo oxide run "$ex" > "$log" 2>&1
  local rc=$?
  local res verdict line
  res=$(classify "$rc" "$log")
  verdict="${res%%$'\t'*}"; line="${res#*$'\t'}"
  printf "%-16s %-9s %-22s rc=%s\n  decisive: %s\n" "$ex" "$tag" "$verdict" "$rc" "$(echo "$line" | sed 's/^[[:space:]]*//')" | tee -a "$SUMMARY"
}

echo "===== POSITIVE (expect execute on sm_120) =====" | tee -a "$SUMMARY"
for ex in $POS; do run_one "$ex" positive; done
echo "===== NEGATIVE (datacenter/Hopper features) =====" | tee -a "$SUMMARY"
for ex in $NEG; do run_one "$ex" negative; done
echo; echo "==== SUMMARY ===="; cat "$SUMMARY"
