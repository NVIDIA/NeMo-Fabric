# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:0.7.13 AS uv

FROM rust:1.98-bookworm AS fabric-builder
COPY --from=uv /uv /uvx /usr/local/bin/
COPY vendor/nemo-fabric-source.tar.gz /tmp/nemo-fabric-source.tar.gz
RUN mkdir /src /packages \
    && tar -xzf /tmp/nemo-fabric-source.tar.gz -C /src \
    && uv build --wheel --out-dir /packages /src/sdk/python/nemo-fabric \
    && cd /src/sdk/python/nemo-fabric-runtime \
    && uvx --from 'maturin[zig]>=1.9.3,<2.0' maturin build \
        --release \
        --locked \
        --compatibility manylinux_2_17 \
        --zig \
        --out /packages

FROM node:24.16.0-bookworm-slim AS kiro-builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://cli.kiro.dev/install -o /tmp/install-kiro.sh \
    && bash /tmp/install-kiro.sh \
    && rm -f /tmp/install-kiro.sh
COPY vendor/nemo-fabric-source.tar.gz /tmp/nemo-fabric-source.tar.gz
RUN mkdir /src \
    && tar -xzf /tmp/nemo-fabric-source.tar.gz -C /src \
    && npm ci --prefix /src/adapter-contract/typescript --ignore-scripts \
    && npm ci --prefix /src/adapters/typescript \
        --workspace nemo-fabric-adapters-kiro \
        --include-workspace-root \
        --ignore-scripts \
    && npm run build --prefix /src/adapter-contract/typescript \
    && npm run build --prefix /src/adapters/typescript \
        --workspace nemo-fabric-adapters-common \
    && npm run build --prefix /src/adapters/typescript \
        --workspace nemo-fabric-adapters-kiro

FROM swebench/sweb.eval.x86_64.django_1776_django-13741:latest

WORKDIR /testbed

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=kiro-builder /usr/local/ /usr/local/
COPY --from=kiro-builder /root/.local/bin/ /root/.local/bin/
COPY --from=fabric-builder /packages /opt/nemo-fabric/wheelhouse
COPY --from=kiro-builder /src/adapter-contract/typescript /opt/nemo-fabric-js/adapter-contract/typescript
COPY --from=kiro-builder /src/adapters/typescript/package.json /src/adapters/typescript/package-lock.json /opt/nemo-fabric-js/adapters/typescript/
COPY --from=kiro-builder /src/adapters/typescript/node_modules /opt/nemo-fabric-js/adapters/typescript/node_modules
COPY --from=kiro-builder /src/adapters/typescript/common /opt/nemo-fabric-js/adapters/typescript/common
COPY --from=kiro-builder /src/adapters/typescript/kiro /opt/nemo-fabric-js/adapters/typescript/kiro

RUN uv venv --python /opt/miniconda3/bin/python /opt/nemo-fabric-venv \
    && uv pip install --python /opt/nemo-fabric-venv/bin/python \
        /opt/nemo-fabric/wheelhouse/*.whl \
    && mkdir -p /opt/nemo-fabric-config/adapters/kiro \
    && cp -R \
        /opt/nemo-fabric-js/adapters/typescript/kiro/. \
        /opt/nemo-fabric-config/adapters/kiro/ \
    && ln -s \
        /opt/nemo-fabric-js/adapters/typescript/node_modules \
        /opt/nemo-fabric-config/node_modules

ENV PATH="/opt/nemo-fabric-venv/bin:/root/.local/bin:/usr/local/bin:${PATH}"
