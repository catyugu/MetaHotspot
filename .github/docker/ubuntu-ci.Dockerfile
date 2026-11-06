FROM ubuntu:24.04

LABEL org.opencontainers.image.source="https://github.com/catyugu/MetaHotspot"

ENV DEBIAN_FRONTEND=noninteractive

# Install Ubuntu SDK build dependencies once rather than in every CI run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      clang-18 \
      curl \
      git \
      libclang-rt-18-dev \
      libbz2-1.0 \
      libffi8 \
      liblzma5 \
      libsqlite3-0 \
      libssl3t64 \
      libomp-18-dev \
      pkg-config \
      python3 \
      tar \
      unzip \
      xz-utils \
      zstd \
    && rm -rf /var/lib/apt/lists/*
