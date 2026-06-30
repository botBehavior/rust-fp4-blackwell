# Real FP4 tensor-core code in pure Rust on a gaming GPU — with NVIDIA's own compiler

**[🌐 Project page](https://botbehavior.github.io/rust-fp4-blackwell/) · [📄 Paper (Zenodo)](https://doi.org/10.5281/zenodo.21056347) · [ORCID](https://orcid.org/0009-0005-8660-6453)**

This repository accompanies the paper *Real FP4 Tensor-Core Code in Pure Rust on a Gaming GPU — with
NVIDIA's Own Compiler*. It is a **viability** result, not a supremacy claim: an entire Llama-class
decoder written in pure Rust, compiled to PTX by NVIDIA's experimental first-party Rust→PTX backend
([`cuda-oxide`](https://github.com/NVlabs/cuda-oxide)), running FP4-quantized weights on a consumer
RTX 5070 Ti (`sm_120`) and generating coherent text — within ~3.7× of `llama.cpp`'s decode throughput
on a first, unoptimized cut. `llama.cpp` is faster today; we say where and why.

## What's here

```
main.tex, figs/            the paper + its figures (vector PDF)
examples/                  our pure-Rust GPU kernels (drop into a cuda-oxide checkout to build):
  fp4_mma/  fp4_gemv/         FP4 m16n8k64 block-scaled MMA; FP4 decode GEMV
  llama_rmsnorm/ llama_rope/ llama_attn_decode/ llama_decode/ llama_run/   the decode engine
trackb/                    the validated FP4 m16n8k64 operand layout: doc + CPU-reference .cu + PTX templates
tok/                       standalone Hugging Face tokenizer helper (kept out of the cuda-oxide build)
env.sh probe.sh run_samples.sh reverify.sh build_llamacpp.sh install-llvm.sh   repro scripts
probe-logs/SUMMARY.txt     the sm_120 capability-probe results (executes vs. correctly rejects)
fig_gen.py paperstyle.mplstyle   regenerate the paper figures
```

## Dependencies (not redistributed)

- **[`cuda-oxide`](https://github.com/NVlabs/cuda-oxide)** (NVIDIA Labs) — the Rust→PTX compiler. This
  repo contains only *our* kernels and scripts; obtain cuda-oxide separately.
- **CUDA toolkit** — our PTX path uses CUDA 13.1; the `llama.cpp` baseline build uses CUDA 12.9 (full,
  with cuBLAS). Keep them distinct (see `build_llamacpp.sh`).
- **TinyLlama-1.1B** ([arXiv:2401.02385](https://arxiv.org/abs/2401.02385)) — expected at
  `./models/tinyllama/` (`model.safetensors`, `tokenizer.json`); not redistributed.
- **[`llama.cpp`](https://github.com/ggml-org/llama.cpp)** — the head-to-head baseline.

## Build & run

The kernels build with the cuda-oxide backend. Point `CUDAOXIDE_ROOT` at a working checkout laid out
like ours (a cuda-oxide clone with these `examples/` copied in, plus the toolchain), then:

```bash
export CUDAOXIDE_ROOT=/path/to/your/checkout
source env.sh
bash probe.sh         # the sm_120 capability probe -> probe-logs/SUMMARY.txt
bash run_samples.sh   # build the tokenizer + run the decode samples
bash reverify.sh      # re-measure the ours-vs-llama.cpp decode head-to-head
```

Paths in the scripts are expressed relative to `$CUDAOXIDE_ROOT`; adapt to your layout. The hardcoded
model paths in `examples/llama_run` and `tok` (`./models/tinyllama/...`) assume you run from the
checkout root.

## Reproduce the figures

```bash
pip install -r requirements.txt
python fig_gen.py     # figs/fig_throughput.pdf, figs/fig_bandwidth.pdf
```

## Attribution & honesty

We redistribute no third-party compiler, model weights, or `llama.cpp` build. The capability table and
every speed figure carry their measurement and configuration; the `tcgen05` datacenter pipeline is
reported as correctly rejecting on consumer `sm_120` (driver `INVALID_PTX`), separately from the
weaker observation that the toolchain's `sm_120a` lowering hits an LLVM selection gap — cuda-oxide
*does* implement `tcgen05`. We make no supremacy claim.

## License

Our code (kernels, scripts, figure scripts) is under the MIT License (`LICENSE`). `cuda-oxide` and all
other dependencies are governed by their own licenses.
