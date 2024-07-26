FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get install wget -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pip-tools

RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64  -O bazelisk && \
    chmod +x bazelisk && \
    mv bazelisk /usr/local/bin/bazel    
    

