# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.14-slim-bookworm

COPY --from=node:24.16.0-bookworm-slim /usr/local /usr/local

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl git pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal \
    && rm -f /tmp/rustup-init.sh
ENV PATH=/root/.cargo/bin:$PATH

RUN npm install --global openclaw@2026.9.4 --allow-scripts=openclaw

COPY vendor/nemo-fabric /opt/nemo-fabric
RUN pip install --no-cache-dir \
        -e /opt/nemo-fabric/adapter-contract/python \
        -e /opt/nemo-fabric/sdk/python/nemo-fabric-runtime \
        -e /opt/nemo-fabric/adapters/python/common \
        -e /opt/nemo-fabric/adapters/python/claude \
        -e '/opt/nemo-fabric/adapters/python/codex[harness]' \
        -e /opt/nemo-fabric/adapters/python/openclaw \
        -e "/opt/nemo-fabric/sdk/python/nemo-fabric[claude,openclaw]"

# Build the Pi adapter and its pinned SDK harness from the vendored source.
RUN npm ci --prefix /opt/nemo-fabric/adapter-contract/typescript --ignore-scripts \
    && npm ci --prefix /opt/nemo-fabric/adapters/typescript \
        --workspace nemo-fabric-adapters-pi --include-workspace-root --ignore-scripts \
    && npm run build --prefix /opt/nemo-fabric/adapter-contract/typescript \
    && npm run build --prefix /opt/nemo-fabric/adapters/typescript \
        --workspace nemo-fabric-adapters-common \
    && npm run build --prefix /opt/nemo-fabric/adapters/typescript \
        --workspace nemo-fabric-adapters-pi

COPY calculator.py /app/calculator.py
COPY fabric /opt/fabric-calculator

WORKDIR /app
