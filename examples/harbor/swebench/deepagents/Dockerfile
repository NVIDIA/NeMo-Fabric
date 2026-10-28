# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:0.7.13 AS uv
FROM rust:1.94-bookworm AS fabric-builder
COPY --from=uv /uv /uvx /usr/local/bin/
COPY vendor/nemo-fabric-source.tar.gz /tmp/nemo-fabric-source.tar.gz
RUN mkdir /src /packages \
    && tar -xzf /tmp/nemo-fabric-source.tar.gz -C /src \
    && uv build --wheel --out-dir /packages /src/sdk/python/nemo-fabric-evidence \
    && cd /src/sdk/python/nemo-fabric-runtime \
    && uvx --from 'maturin[zig]>=1.9.3,<2.0' maturin build \
        --release --locked --compatibility manylinux_2_17 --zig --out /packages

FROM swebench/sweb.eval.x86_64.django_1776_django-13741:latest
WORKDIR /testbed
COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=fabric-builder /packages /opt/nemo-fabric/wheelhouse
COPY vendor/harness-constraints.txt /opt/nemo-fabric/harness-constraints.txt
COPY --from=fabric-builder /src/LICENSE /opt/nemo-fabric-source/LICENSE
COPY --from=fabric-builder /src/sdk/python/nemo-fabric-evidence /opt/nemo-fabric-source/sdk/python/nemo-fabric-evidence
COPY --from=fabric-builder /src/adapter-contract/python /opt/nemo-fabric-source/adapter-contract/python
COPY --from=fabric-builder /src/adapters/python/common /opt/nemo-fabric-source/adapters/python/common
COPY --from=fabric-builder /src/adapters/python/deepagents /opt/nemo-fabric-source/adapters/python/deepagents
RUN uv venv --python /opt/miniconda3/bin/python /opt/nemo-fabric-venv \
    && uv pip install --python /opt/nemo-fabric-venv/bin/python \
        --constraint /opt/nemo-fabric/harness-constraints.txt \
        /opt/nemo-fabric/wheelhouse/*.whl \
        /opt/nemo-fabric-source/adapter-contract/python \
        /opt/nemo-fabric-source/adapters/python/common \
        '/opt/nemo-fabric-source/adapters/python/deepagents[harness]'
ENV PATH="/opt/nemo-fabric-venv/bin:/usr/local/bin:${PATH}"
ENV ADAPTER_PYTHON="/opt/nemo-fabric-venv/bin/python"
