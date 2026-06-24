FROM cross-bench-base:latest

USER root

# BigCodeBench requires ~139 libraries. Install the most common ones.
COPY requirements_bigcodebench.txt /tmp/requirements_bigcodebench.txt
RUN pip install --no-cache-dir -r /tmp/requirements_bigcodebench.txt || true

USER runner
WORKDIR /workspace
