#!/usr/bin/env bash
set -e
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd $CUDAOXIDE_ROOT
echo "START $(date)"
# download Q4 GGUF (same model, 4-bit) in parallel
if [ ! -f tinyllama-q4.gguf ]; then
  echo "downloading TinyLlama Q4_K_M GGUF..."
  curl -sL -o tinyllama-q4.gguf "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" &
fi
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp 2>&1 | tail -2
fi
cd llama.cpp
echo "configuring (CUDA sm_120)..."
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -4
echo "building llama-bench..."
cmake --build build --config Release -j --target llama-bench 2>&1 | tail -6
wait
echo "GGUF size:"; ls -la $CUDAOXIDE_ROOT/tinyllama-q4.gguf 2>/dev/null
echo "EXIT=$? DONE $(date)"
