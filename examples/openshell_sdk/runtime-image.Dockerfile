# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Add the Fabric-ready echo adapter to the reusable OpenShell task image.
FROM fabric-openshell-sdk:source
USER root
RUN install -d -o sandbox -g sandbox /opt/fabric/agents/echo
COPY --chown=sandbox:sandbox examples/openshell_sdk/adapter/echo_adapter.py /opt/fabric/agents/echo/echo_adapter.py
COPY --chown=sandbox:sandbox examples/openshell_sdk/adapter/echo.fabric-adapter.json /opt/fabric/agents/echo/echo.fabric-adapter.json
USER sandbox:sandbox
WORKDIR /sandbox/workspace
