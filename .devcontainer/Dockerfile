FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime


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

RUN wget https://dl.grafana.com/enterprise/release/grafana-enterprise_11.1.3_amd64.deb &&\
    dpkg -i grafana-enterprise_11.1.3_amd64.deb &&\
    rm grafana-enterprise_11.1.3_amd64.deb

RUN pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple  pip-tools    


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

# Install dependencies
RUN pip install  --index-url https://pypi.tuna.tsinghua.edu.cn/simple   -r requirements.txt

    
