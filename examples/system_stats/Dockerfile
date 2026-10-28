# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Development build from this checkout; no unpublished registry version is used.
FROM rust:1.94-trixie AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/build && /opt/build/bin/pip install 'maturin>=1.9.3,<2'
WORKDIR /src
COPY Cargo.toml Cargo.lock LICENSE ./
COPY crates/ crates/
COPY environment-providers/ environment-providers/
COPY sdk/python/nemo-fabric-evidence/ sdk/python/nemo-fabric-evidence/
RUN /opt/build/bin/pip wheel --no-deps sdk/python/nemo-fabric-evidence --wheel-dir /wheels
COPY sdk/python/nemo-fabric-runtime/ sdk/python/nemo-fabric-runtime/
WORKDIR /src/sdk/python/nemo-fabric-runtime
RUN /opt/build/bin/maturin build --locked --compatibility linux --out /wheels

# Match the glibc generation used by the Rust build image.
FROM python:3.12-slim-trixie
COPY --from=build /wheels/ /wheels/
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
RUN useradd --uid 10001 --user-group --create-home stats
COPY --chown=stats:stats examples/system_stats/ /example/
USER stats:stats
WORKDIR /example
ENTRYPOINT ["python", "/example/run.py"]
