# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-bookworm

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
        -e /opt/nemo-fabric/adapters/python/hermes \
        -e /opt/nemo-fabric/adapters/python/openclaw \
        -e "/opt/nemo-fabric/sdk/python/nemo-fabric[claude,hermes-agent,openclaw,relay]"

# Pin the upstream Hermes Relay 0.9 snapshot until a compatible release exists.
RUN python -m venv /opt/hermes-venv \
    && git clone https://github.com/NousResearch/hermes-agent.git /opt/hermes-agent \
    && git -C /opt/hermes-agent checkout --detach 47edd6455bbb66ac94e00be0ca3e888680f02882 \
    && /opt/hermes-venv/bin/pip install --no-cache-dir \
        -e /opt/nemo-fabric/adapter-contract/python \
        -e /opt/nemo-fabric/adapters/python/common \
        -e '/opt/nemo-fabric/adapters/python/hermes[relay]' \
    && /opt/hermes-venv/bin/pip install --no-cache-dir \
        --config-settings editable_mode=compat -e /opt/hermes-agent

COPY calculator.py /app/calculator.py
COPY fabric /opt/fabric-calculator

WORKDIR /app
