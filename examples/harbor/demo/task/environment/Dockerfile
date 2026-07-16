# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl git pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal \
    && rm -f /tmp/rustup-init.sh
ENV PATH=/root/.cargo/bin:$PATH

COPY vendor/nemo-fabric /opt/nemo-fabric
RUN pip install --no-cache-dir \
        -e /opt/nemo-fabric/python \
        -e /opt/nemo-fabric/adapters/common \
        -e /opt/nemo-fabric/adapters/claude \
        -e /opt/nemo-fabric/adapters/hermes \
        -e "/opt/nemo-fabric[harbor,hermes,relay,runtime]"

COPY calculator.py /app/calculator.py
COPY fabric /opt/fabric-demo

WORKDIR /app
