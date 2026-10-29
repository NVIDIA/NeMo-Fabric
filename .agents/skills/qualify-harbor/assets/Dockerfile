# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG TASK_IMAGE=python:3.12-slim-bookworm
FROM ${TASK_IMAGE}
COPY --from=node:24.16.0-bookworm-slim /usr/local /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git pkg-config libssl-dev unzip ripgrep \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_BUILD_JOBS=4
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.94.0 && rm /tmp/rustup.sh
ENV PATH=/root/.cargo/bin:/opt/harness/node_modules/.bin:$PATH
COPY source /src
ARG FABRIC_VERSION
ARG INSTALL_MODE
RUN mkdir /wheels && if [ "$INSTALL_MODE" = source ]; then \
      pip wheel --no-deps --wheel-dir /wheels /src/adapter-contract/python \
        /src/sdk/python/nemo-fabric-runtime /src/sdk/python/nemo-fabric \
        /src/sdk/python/nemo-fabric-evidence \
        /src/sdk/python/nemo-fabric-adapter-catalog \
        /src/adapters/python/*/; \
    else pip download --only-binary=:all: --no-deps --dest /wheels \
        "nemo-fabric==$FABRIC_VERSION" "nemo-fabric-runtime==$FABRIC_VERSION" \
        "nemo-fabric-evidence==$FABRIC_VERSION" \
        "nemo-fabric-adapter-catalog==$FABRIC_VERSION" \
        "nemo-fabric-adapter-contract==$FABRIC_VERSION" "nemo-fabric-adapters-common==$FABRIC_VERSION"; fi
RUN pip install /wheels/nemo_fabric-*.whl /wheels/nemo_fabric_runtime-*.whl \
    /wheels/nemo_fabric_evidence-*.whl \
    /wheels/nemo_fabric_adapter_catalog-*.whl /wheels/nemo_fabric_adapter_contract-*.whl \
    /wheels/nemo_fabric_adapters_common-*.whl
COPY identity.py /qualification/identity.py
WORKDIR /app
