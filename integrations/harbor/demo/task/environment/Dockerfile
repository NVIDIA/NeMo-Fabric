FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl git nodejs npm pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal \
    && rm -f /tmp/rustup-init.sh
ENV PATH=/root/.cargo/bin:$PATH

COPY vendor/nemo-fabric /opt/nemo-fabric
RUN pip install --no-cache-dir -e "/opt/nemo-fabric[codex,harbor,hermes,relay]" \
    && npm install --global @openai/codex@0.142.4

COPY calculator.py /app/calculator.py
COPY fabric /opt/fabric-demo

WORKDIR /app
