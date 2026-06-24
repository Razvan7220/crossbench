FROM public.ecr.aws/docker/library/python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pytest numpy

WORKDIR /workspace

RUN useradd -m -s /bin/bash runner
USER runner
