# =============================================================================
# llama.cpp (CUDA backend) built specifically for NVIDIA Tesla V100 (Volta, SM70)
#
# Why this instead of vLLM:
#   - Modern PyTorch/vLLM wheels are compiled only for sm_75/80/86/90/100/120.
#     Volta (sm_70) was dropped, so the V100 has no kernels -> the cascade of
#     "NCCL internal error / WorkerProc failed to start / Engine core init failed".
#   - llama.cpp has no PyTorch dependency, builds its own CUDA kernels, and we
#     pin CMAKE_CUDA_ARCHITECTURES=70 so the binary contains real sm_70 kernels.
#   - It uses its own CUDA peer-to-peer (over NVLink), NOT NCCL.
#
# CUDA 12.1 is chosen as a balance: it supports sm_70 and needs host driver
# >= 525.60.13. If your host driver is older, switch both base images to
# 11.8.0 (driver >= 450/520). Do NOT use CUDA 13 (drops Volta tooling).
# =============================================================================

# ---------- build stage ----------
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS build

ARG LLAMA_CPP_REF=master
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential \
        libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ggml-org/llama.cpp.git
WORKDIR /opt/llama.cpp
RUN git checkout "${LLAMA_CPP_REF}"

# - GGML_CUDA=ON            : enable CUDA backend
# - CMAKE_CUDA_ARCHITECTURES=70 : compile ONLY for Volta/V100 -> guarantees sm_70
# - BUILD_SHARED_LIBS=OFF   : statically link ggml/llama into the binaries so the
#                             runtime image only needs the CUDA runtime libs
# - GGML_NATIVE=OFF         : don't tune CPU code to this build host's CPU
# - LLAMA_CURL=ON           : enable `-hf` model downloads from Hugging Face
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=ON \
    && cmake --build build --config Release -j --target llama-server llama-cli

# ---------- runtime stage ----------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

# curl: model downloads (-hf) + container healthcheck
# libgomp1: OpenMP runtime used by ggml CPU paths
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /opt/llama.cpp/build/bin/llama-cli   /usr/local/bin/llama-cli

ENV LLAMA_CACHE=/root/.cache/llama.cpp
EXPOSE 8000

ENTRYPOINT ["llama-server"]
