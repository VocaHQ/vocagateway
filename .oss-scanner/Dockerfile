# Build image for Anthropic's OSS Scanner (https://github.com/anthropics/oss-scanner).
#
# Not the production image (that is ../Dockerfile). This one keeps the source
# tree at /src, installs the dev group so the test suite runs, and builds
# whisper.cpp with debug info so the gateway can transcribe end to end. The scan
# runs with no network, so the tiny.en weights and a sample WAV are fetched here.
FROM python:3.12-slim-trixie

ARG WHISPER_CPP_VERSION=v1.9.1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      build-essential ca-certificates cmake curl ffmpeg git libgomp1 pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

# whisper.cpp: whisper-cli and whisper-server, the runtimes the gateway drives.
RUN mkdir /opt/whisper.cpp && cd /opt/whisper.cpp \
    && curl --fail --location --show-error \
      "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/${WHISPER_CPP_VERSION}.tar.gz" \
      | tar --extract --gzip --strip-components=1 \
    && cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_C_FLAGS="-g -fno-omit-frame-pointer" \
      -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" \
      -DGGML_NATIVE=OFF \
      -DWHISPER_BUILD_TESTS=OFF \
      -DWHISPER_BUILD_EXAMPLES=ON \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --install build \
    && cp build/bin/whisper-cli build/bin/whisper-server /usr/local/bin/ \
    && ldconfig \
    && whisper-cli --help >/dev/null

# Offline fixtures: the smallest English model and whisper.cpp's sample clip.
RUN mkdir -p /opt/models /opt/samples \
    && curl --fail --location --show-error --output /opt/models/ggml-tiny.en.bin \
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin \
    && cp /opt/whisper.cpp/samples/jfk.wav /opt/samples/jfk.wav

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src
# Installed into the system interpreter rather than a venv, so `python`,
# `pytest` and `vocagateway` resolve in any shell, login shells included.
# .oss-scanner/Dockerfile.dockerignore keeps tests/ in the build context.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/usr/local
RUN uv sync --frozen --inexact --all-groups --extra engines
ENV UV_NO_SYNC=1 \
    UV_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    VOCAGATEWAY_BIND_HOST=127.0.0.1 \
    VOCAGATEWAY_PORT=8765 \
    VOCAGATEWAY_DATA_DIR=/tmp/vocagateway-data \
    VOCAGATEWAY_MODELS_DIR=/tmp/vocagateway-data/models \
    VOCAGATEWAY_WHISPER_BINARY=/usr/local/bin/whisper-cli \
    VOCAGATEWAY_WHISPER_MODEL=/opt/models/ggml-tiny.en.bin \
    VOCAGATEWAY_ENGINE=whisper.cpp \
    VOCAGATEWAY_TOKEN=oss-scanner-local-test-token-not-a-secret-0001

# Cheap checks so the image is known to work; a failing test does not fail the build.
RUN python -c "import app.main" \
    && (python -m pytest -n auto -q -p no:cacheprovider || echo "WARNING: pytest reported failures")
