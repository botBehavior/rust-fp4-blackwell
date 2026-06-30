: "${CUDAOXIDE_ROOT:?Set CUDAOXIDE_ROOT to your working checkout — see README}"
export RUSTUP_HOME=$CUDAOXIDE_ROOT/.rustup
export CARGO_HOME=$CUDAOXIDE_ROOT/.cargo
export CARGO_TARGET_DIR=$CUDAOXIDE_ROOT/target
export PATH="$CARGO_HOME/bin:/usr/local/cuda-13.1/bin:$PATH"
export CUDA_OXIDE_BACKEND=$CUDAOXIDE_ROOT/target/debug/deps/librustc_codegen_cuda.so
