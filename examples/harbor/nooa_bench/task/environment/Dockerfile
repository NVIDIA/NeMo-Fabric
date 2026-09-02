# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY vendor/wheelhouse /opt/nemo-fabric-wheelhouse
COPY vendor/nooa-adapter /opt/nemo-fabric-nooa
COPY vendor/labs-OO-Agents /opt/labs-OO-Agents

RUN pip install --no-cache-dir \
        /opt/nemo-fabric-wheelhouse/*.whl \
        "nemo-relay>=0.7.2,<0.8" \
        -e /opt/labs-OO-Agents \
        -e /opt/labs-OO-Agents/packages/nooa-cli \
        -e /opt/labs-OO-Agents/packages/nooa-bench

RUN python -c \
    "import importlib.metadata as m; from nooa_bench.bench_agent import BenchAgent; print(BenchAgent.__name__, m.version('nemo-relay'))"

ENV PYTHONPATH=/opt/nemo-fabric-nooa/src

COPY calculator.py /app/calculator.py

RUN useradd --create-home --uid 10001 fabric \
    && chown -R fabric:fabric /app

USER fabric
WORKDIR /app
