# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM swebench/sweb.eval.x86_64.django_1776_django-13741:latest

WORKDIR /testbed

RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh \
    && mkdir -p /logs

ENV PATH="/root/.local/bin:${PATH}"

COPY vendor /opt/nemo-fabric-nooa

RUN uv venv --python 3.12 /opt/nemo-fabric-venv \
    && uv pip install --python /opt/nemo-fabric-venv/bin/python \
        /opt/nemo-fabric-nooa/wheelhouse/*.whl \
        "nemo-relay>=0.7.2,<0.8" \
        /opt/nemo-fabric-nooa/labs-OO-Agents \
        /opt/nemo-fabric-nooa/labs-OO-Agents/packages/nooa-cli \
        /opt/nemo-fabric-nooa/labs-OO-Agents/packages/nooa-bench

ENV PYTHONPATH=/opt/nemo-fabric-nooa/nooa-adapter/src
