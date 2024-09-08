FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
                                         
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y sudo && \
    apt-get install -y screen && \
    apt-get install gdb -y && \
    apt-get install -y net-tools && \
    apt-get install -y adduser libfontconfig1 musl  && \
    apt-get install -y --no-install-recommends git && \
    apt-get install -y build-essential && \
    apt-get install wget -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



#  Install Bazelisk
RUN wget -c --tries=10 https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    -O /usr/local/bin/bazelisk && \
    chmod +x /usr/local/bin/bazelisk
RUN ln -s /usr/local/bin/bazelisk /usr/local/bin/bazel


# install grafana
RUN wget -c --tries=10   https://dl.grafana.com/enterprise/release/grafana-enterprise_11.1.3_amd64.deb &&\
    dpkg -i grafana-enterprise_11.1.3_amd64.deb &&\
    rm grafana-enterprise_11.1.3_amd64.deb

# Install Prometheus
ENV PROMETHEUS_VERSION=2.54.0
RUN mkdir -p /etc/prometheus /var/lib/prometheus
RUN wget -c --tries=10   https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz \
    && tar -xvzf prometheus-${PROMETHEUS_VERSION}.linux-amd64.tar.gz \
    && mv prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus /usr/local/bin/ \
    && mv prometheus-${PROMETHEUS_VERSION}.linux-amd64/promtool /usr/local/bin/ \
    && mv prometheus-${PROMETHEUS_VERSION}.linux-amd64/consoles /etc/prometheus/ \
    && mv prometheus-${PROMETHEUS_VERSION}.linux-amd64/console_libraries /etc/prometheus/ \
    && mv prometheus-${PROMETHEUS_VERSION}.linux-amd64/prometheus.yml /etc/prometheus/ \
    && rm -rf prometheus-${PROMETHEUS_VERSION}.linux-amd64* 


# Install pip-tools
RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple  pip-tools    


# Install dependencies if requirements.in has changed
COPY requirements.in .

# Function to check if requirements.in has changed and run pip-compile if needed
RUN set -e; \
    if [ -f requirements.in.sha256 ] && sha256sum -c requirements.in.sha256; then \
        echo "requirements.in has not changed, skipping pip-compile"; \
    else \
        echo "requirements.in has changed, running pip-compile"; \
        pip-compile --index-url https://pypi.tuna.tsinghua.edu.cn/simple  -v  requirements.in; \
        sha256sum requirements.in > requirements.in.sha256; \
    fi

RUN pip install  --index-url https://pypi.tuna.tsinghua.edu.cn/simple   -r requirements.txt


    
