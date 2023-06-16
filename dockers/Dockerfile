FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
LABEL maintainer="decu@microsoft"
LABEL repository="unitorch-microsoft"

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1
RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   locales \
                   ca-certificates \
                   zip \
                   unzip \
                   ninja-build \
                   cmake \
                   libopenmpi-dev \
                   libgl1-mesa-glx \
                   gfortran libopenblas-dev liblapack-dev \
                   python3.8-dev python3-pip && \
    rm -rf /var/lib/apt/lists

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python -m pip install --no-cache-dir --upgrade pip setuptools && \
    python -m pip install --no-cache-dir mkl ninja

RUN python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PATH "$PATH:~/.local/bin"

WORKDIR /workspace

CMD ["/bin/bash"]
