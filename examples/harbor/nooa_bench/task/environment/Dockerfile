# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY vendor/wheelhouse /opt/nemo-fabric-wheelhouse

RUN pip install --no-cache-dir \
        /opt/nemo-fabric-wheelhouse/*.whl \
    && pip install --no-cache-dir "nemo-fabric-adapters-nooa[full]"

RUN python -c \
    "import importlib.metadata as m; from nooa_bench.bench_agent import BenchAgent; print(BenchAgent.__name__, m.version('nemo-relay'))"

COPY calculator.py /app/calculator.py

RUN useradd --create-home --uid 10001 fabric \
    && chown -R fabric:fabric /app

USER fabric
WORKDIR /app
