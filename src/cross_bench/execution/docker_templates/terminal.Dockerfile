FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget python3 python3-pip \
    nodejs npm golang rustc cargo \
    net-tools iproute2 openssh-client \
    vim nano jq tree htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
