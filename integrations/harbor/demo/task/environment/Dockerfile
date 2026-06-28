FROM python:3.12-slim-bookworm

ARG FABRIC_REPO=https://github.com/AjayThorve/NeMo-Fabric.git
ARG FABRIC_REF=ajay/harbor-multi-harness-demo

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl git nodejs npm pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH

RUN git clone "$FABRIC_REPO" /opt/nemo-fabric \
    && git -C /opt/nemo-fabric checkout "$FABRIC_REF" \
    && pip install --no-cache-dir -e "/opt/nemo-fabric[hermes,relay]" \
    && npm install --global @openai/codex@0.134.0

COPY calculator.py /app/calculator.py
COPY fabric /opt/fabric-demo

WORKDIR /app
