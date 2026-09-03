# syntax=docker/dockerfile:1.12

# One Dockerfile builds all three accelerator images. ACCEL picks the builder
# base and the runtime base; everything between them — the whisper.cpp build,
# the Python environment, the user, the ENV block, the healthcheck — is written
# once and shared, so the three images cannot drift apart.
#
#   cpu     Debian build with runtime CPU dispatch (default, portable)
#   cuda    NVIDIA CUDA build on the CUDA runtime base
#   vulkan  Vulkan build on Ubuntu with the Mesa ICDs
#
# Compose selects it per service; by hand it is
# `docker build --build-arg ACCEL=cuda .`
ARG ACCEL=cpu

ARG WHISPER_CPP_VERSION=v1.9.1
ARG UV_VERSION=0.8.0
ARG PYTHON_VERSION=3.12
# The CPU builder and its runtime must share a Debian release. The ARM variant
# list below reaches armv9.2+sme, which needs a compiler that knows the flag
# (GCC 14, in trixie; bookworm's GCC 12 rejects it), and a binary linked against
# trixie's glibc will not start on a bookworm runtime.
ARG DEBIAN_VERSION=trixie
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1

# ---------------------------------------------------------------------------
# Builder bases — one per accelerator, each naming its own cmake flags.
# ---------------------------------------------------------------------------
# The apt cache mounts below need Docker's auto-clean hook gone, otherwise apt
# deletes every .deb the moment it is unpacked and the cache stays empty.

FROM debian:${DEBIAN_VERSION}-slim AS builder-base-cpu
ENV WHISPER_ACCEL_CMAKE="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,id=apt-cache-debian,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-debian,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      build-essential ca-certificates ccache cmake curl libopenblas-dev pkg-config

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder-base-cuda
ENV WHISPER_ACCEL_CMAKE="-DGGML_CUDA=ON"
# GCC 13, Ubuntu 24.04's default, rejects `-march=armv9.2-a+...+sme` with
# "invalid feature modifier 'sme'", which fails the armv9.2 CPU variants on an
# arm64 builder. GCC 14 knows the flag, and noble's runtime libstdc++6 is
# already 14.2, so nothing downstream has to change. (CUDA 12.6+ supports GCC 14
# as its host compiler.)
ENV CC=gcc-14 CXX=g++-14
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,id=apt-cache-cuda,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-cuda,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      build-essential ca-certificates ccache cmake curl gcc-14 g++-14

FROM ubuntu:${UBUNTU_VERSION} AS builder-base-vulkan
ENV WHISPER_ACCEL_CMAKE="-DGGML_VULKAN=ON"
# GCC 13, Ubuntu 24.04's default, rejects `-march=armv9.2-a+...+sme` with
# "invalid feature modifier 'sme'", which fails the armv9.2 CPU variants on an
# arm64 builder. GCC 14 knows the flag, and noble's runtime libstdc++6 is
# already 14.2, so nothing downstream has to change. (CUDA 12.6+ supports GCC 14
# as its host compiler.)
ENV CC=gcc-14 CXX=g++-14
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,id=apt-cache-ubuntu,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-ubuntu,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      build-essential ca-certificates ccache cmake curl gcc-14 g++-14 \
      glslc libvulkan-dev spirv-headers

# ---------------------------------------------------------------------------
# whisper.cpp — one build, shared by all three accelerators.
# ---------------------------------------------------------------------------

FROM builder-base-${ACCEL} AS whisper-builder
ARG ACCEL
ARG WHISPER_CPP_VERSION
# Appended last to the cmake line, so a -D here overrides one of the defaults
# below it. Two worth knowing:
#   -DCMAKE_CUDA_ARCHITECTURES=89-real   build for one known GPU instead of the
#     portable virtual+real spread ggml picks; much faster nvcc, smaller binary
#   -DGGML_BLAS=OFF                      drop OpenBLAS from the CPU image
ARG WHISPER_CMAKE_EXTRA=""
# Empty means "one job per core". nvcc instantiates a lot of templates and each
# job can want most of a gigabyte, so the CUDA build is the one that runs a
# constrained builder out of memory — `cmake --build --parallel` with no cap
# turns that into an Error 137 partway through, not a clean diagnostic. An 8 GB
# builder wants BUILD_JOBS=3 or so for ACCEL=cuda.
ARG BUILD_JOBS=""
WORKDIR /src
RUN curl --fail --location --show-error \
      "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/${WHISPER_CPP_VERSION}.tar.gz" \
      | tar --extract --gzip --strip-components=1

# GGML_CPU_ALL_VARIANTS compiles one CPU backend per micro-architecture
# (x86: sse42 … haswell … zen4, alderlake, sapphirerapids; arm64: armv8.0 …
# armv8.2+dotprod … armv8.6+i8mm … armv9.2+sme) and ggml dlopens the best one
# the host reports at startup. That is what makes GGML_NATIVE=OFF cheap: the
# image stays portable across registries but still runs AVX2/AVX-512 code on
# x86 and dotprod/i8mm code on arm64, instead of the baseline scalar kernels a
# plain non-native build would ship. It requires GGML_BACKEND_DL, which in turn
# requires shared libraries.
#
# GGML_BACKEND_DIR is compiled in as the first search path (the fallbacks are
# the executable's own directory and the cwd, neither of which holds the
# backends once whisper-cli is installed to /usr/local/bin).
#
# The default target is built rather than `--target whisper-cli`: the variant
# backends are standalone dlopen'd modules that nothing links against, so a
# whisper-cli-only build would produce a binary with no CPU backend to load.
RUN --mount=type=cache,id=ccache-${ACCEL},target=/root/.cache/ccache \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
      -DBUILD_SHARED_LIBS=ON \
      -DGGML_NATIVE=OFF \
      -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DGGML_BACKEND_DIR=/usr/local/lib/ggml \
      -DWHISPER_BUILD_TESTS=OFF \
      -DWHISPER_BUILD_EXAMPLES=ON \
      ${WHISPER_ACCEL_CMAKE} \
      ${WHISPER_CMAKE_EXTRA} \
    && cmake --build build --config Release --parallel ${BUILD_JOBS}

# Sort the build output the way the runtime expects it: the linked libraries
# onto the loader path, the dlopen'd backends into GGML_BACKEND_DIR. Only
# whisper-cli is taken from the examples; the rest of the default target is
# build fallout.
RUN set -eu; \
    mkdir -p /out/bin /out/lib/ggml; \
    cp build/bin/whisper-cli /out/bin/; \
    for lib in build/bin/*.so*; do \
      case "${lib##*/}" in \
        libggml-cpu*|libggml-blas*|libggml-cuda*|libggml-vulkan*) \
          cp -P "${lib}" /out/lib/ggml/ ;; \
        *) \
          cp -P "${lib}" /out/lib/ ;; \
      esac; \
    done

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------------------------------------------------------------------
# Runtime bases — one per accelerator.
# ---------------------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS runtime-base-cpu
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,id=apt-cache-debian,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-debian,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates ffmpeg libgomp1 libopenblas0-pthread libportaudio2

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime-base-cuda
ARG PYTHON_VERSION
# faster-whisper reaches CUDA through CTranslate2, which wants cuDNN as well as
# cuBLAS; the whisper.cpp CUDA backend needs only the latter. compute,utility is
# what both need from the injected driver — stated here rather than inherited so
# a base image default cannot quietly change it.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,id=apt-cache-cuda,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-cuda,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates ffmpeg libgomp1 libportaudio2 \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python

FROM ubuntu:${UBUNTU_VERSION} AS runtime-base-vulkan
ARG PYTHON_VERSION
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
      > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,id=apt-cache-ubuntu,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-ubuntu,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install --yes --no-install-recommends \
      ca-certificates ffmpeg libgomp1 libportaudio2 libvulkan1 mesa-vulkan-drivers \
      python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python

# ---------------------------------------------------------------------------
# Runtime — shared by all three accelerators.
# ---------------------------------------------------------------------------

FROM runtime-base-${ACCEL} AS runtime
ARG ACCEL
COPY --from=uv /uv /uvx /bin/
COPY --from=whisper-builder /out/bin/ /usr/local/bin/
COPY --from=whisper-builder /out/lib/ /usr/local/lib/
RUN ldconfig

WORKDIR /app
# The uv cache lives on a build cache mount rather than inside the layer: the
# engines extra pulls ctranslate2, onnxruntime, transformers and scipy, and a
# cached copy of all that used to ship in the image for nothing. Copy link mode
# is what makes a cache mount usable — the default hardlinks, which cannot cross
# the mount boundary. Bytecode is compiled at build time so the first request
# after a restart does not pay for it.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
# The manifests alone first, so the dependency layer survives an app/ edit.
# README.md is here because hatchling reads it for the wheel metadata.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,id=uv-${ACCEL},target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra engines --no-install-project
COPY app ./app
RUN --mount=type=cache,id=uv-${ACCEL},target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra engines --no-editable \
    && groupadd --gid 10001 vocaphone \
    && useradd --uid 10001 --gid vocaphone --no-create-home \
      --home-dir /app --shell /usr/sbin/nologin vocaphone \
    && mkdir --parents /data \
    && chown --recursive vocaphone:vocaphone /data /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    VOCAGATEWAY_BIND_HOST=0.0.0.0 \
    VOCAGATEWAY_PORT=8765 \
    VOCAGATEWAY_DATA_DIR=/data \
    VOCAGATEWAY_MODELS_DIR=/data/models \
    VOCAGATEWAY_CONFIG_FILE=/data/config/config.json \
    VOCAGATEWAY_TOKEN_FILE=/run/secrets/vocagateway_token \
    VOCAGATEWAY_WHISPER_BINARY=/usr/local/bin/whisper-cli \
    VOCAGATEWAY_ENGINE=auto

# OPENBLAS_NUM_THREADS is deliberately not set here. It looks like it should be:
# OpenBLAS-pthread sizes its pool from the host CPU count and cannot see the
# cgroup quota, so a quota-limited container reads nproc=10 while it is allowed
# two. Measured, it makes no difference — ggml's BLAS backend issues a matmul
# from one thread at a time rather than from each of its workers, so the pools
# never nest. ggml-tiny.en and ggml-base.en on jfk.wav, --cpus 4 and --cpus 2 on
# a 10-CPU host, were all within run-to-run noise pinned and unpinned. Pinning it
# would only cap OpenBLAS's own parallelism for no measured gain.

# Source commit of the image, surfaced in /v1/admin/status and the WebUI.
# Pass at build time: --build-arg VOCAGATEWAY_GIT_COMMIT="$(git rev-parse HEAD)"
ARG VOCAGATEWAY_GIT_COMMIT=""
ARG VOCAGATEWAY_GIT_COMMIT_SUBJECT=""
ARG VOCAGATEWAY_GIT_COMMIT_DATE=""
ENV VOCAGATEWAY_GIT_COMMIT="${VOCAGATEWAY_GIT_COMMIT}" \
    VOCAGATEWAY_GIT_COMMIT_SUBJECT="${VOCAGATEWAY_GIT_COMMIT_SUBJECT}" \
    VOCAGATEWAY_GIT_COMMIT_DATE="${VOCAGATEWAY_GIT_COMMIT_DATE}"

USER vocaphone
EXPOSE 8765
VOLUME ["/data"]
# start-interval polls every 2s until the first success, so a fresh container is
# reported healthy as soon as it is, instead of at the next 30s tick.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --start-interval=2s --retries=3 \
  CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('VOCAGATEWAY_PORT', '8765'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health/live', timeout=3)"]
ENTRYPOINT ["vocagateway"]
