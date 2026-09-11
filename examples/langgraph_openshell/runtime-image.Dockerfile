# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim-trixie

ARG LANGGRAPH_VERSION=1.2.11

RUN apt-get update \
    && apt-get install --yes --no-install-recommends iproute2 nftables \
    && rm -rf /var/lib/apt/lists/*

COPY adapter-contract /opt/build/adapter-contract
COPY adapters-common /opt/build/adapters-common

RUN python -m pip install --no-cache-dir \
      "langgraph==${LANGGRAPH_VERSION}" \
      /opt/build/adapter-contract \
      /opt/build/adapters-common \
    && rm -rf /opt/build \
    && groupadd --gid 1500 sandbox \
    && useradd --uid 1500 --gid 1500 --create-home sandbox \
    && mkdir -p /opt/nemo-fabric/examples /sandbox/artifacts /sandbox/.fabric/control \
    && chown -R sandbox:sandbox /sandbox

COPY --chmod=0755 fabric-runtime-server fabric-runtime-ctl /usr/local/bin/
COPY examples/__init__.py /opt/nemo-fabric/examples/__init__.py
COPY examples/langgraph_openshell /opt/nemo-fabric/examples/langgraph_openshell

ENV PYTHONPATH=/opt/nemo-fabric
WORKDIR /sandbox
USER sandbox:sandbox

CMD ["fabric-runtime-server", "serve"]
