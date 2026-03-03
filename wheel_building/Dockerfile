# Copyright (c) 2025, AA-I Technologies Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
ARG CUDA_VERSION=13.0.2
ARG PYTHON_VERSION=3.13.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux8
ARG PYTHON_VERSION

RUN dnf install -y \
    bash \
    binutils \
    bzip2-devel \
    ca-certificates \
    curl \
    gcc \
    gcc-c++ \
    git \
    gzip \
    libffi-devel \
    make \
    ncurses-devel \
    openssl-devel \
    pkgconfig \
    readline-devel \
    sqlite-devel \
    tar \
    which \
    xz \
    xz-devel \
    zlib-devel \
    && dnf clean all

RUN dnf install -y \
    gcc-toolset-12 \
    gcc-toolset-12-gcc \
    gcc-toolset-12-gcc-c++ \
    && dnf clean all

ARG CCACHE_VERSION=4.10.2
RUN curl -fsSL "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz" \
    -o /tmp/ccache.tar.xz \
    && tar -xf /tmp/ccache.tar.xz -C /tmp \
    && cp "/tmp/ccache-${CCACHE_VERSION}-linux-x86_64/ccache" /usr/local/bin/ccache \
    && rm -rf /tmp/ccache*

RUN curl -fsSL "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" -o /tmp/python.tgz \
    && tar -xzf /tmp/python.tgz -C /tmp \
    && cd "/tmp/Python-${PYTHON_VERSION}" \
    && ./configure --prefix=/usr/local --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make altinstall \
    && ln -sfn /usr/local/bin/python3.13 /usr/local/bin/python3 \
    && ln -sfn /usr/local/bin/pip3.13 /usr/local/bin/pip3 \
    && rm -rf /tmp/python.tgz "/tmp/Python-${PYTHON_VERSION}"

RUN python3 -m pip install --no-cache-dir \
    "cmake>=3.30.4" \
    "cython>=3.1.2,<3.2.0" \
    "ninja" \
    "packaging>=24.2" \
    "patchelf>=0.17.2.4" \
    "rapids-build-backend>=0.4.0,<0.5.0" \
    "scikit-build-core[pyproject]>=0.11.0" \
    "auditwheel"

WORKDIR /src
