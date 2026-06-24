FROM cross-bench-base:latest

USER root
RUN pip install --no-cache-dir \
    pandas scipy sympy matplotlib networkx scikit-learn \
    requests beautifulsoup4 regex tqdm

USER runner
WORKDIR /workspace
