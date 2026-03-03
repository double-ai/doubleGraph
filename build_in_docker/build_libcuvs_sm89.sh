#!/usr/bin/env bash
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
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Why this script exists:
# Some prebuilt libcuvs conda artifacts are not compiled for sm_89 (L4),
# which can cause runtime failures when benchmarks execute ANN/cuVS paths.
# We rebuild libcuvs from source with --gpu-arch=89-real and install it into
# the active cugraph environment so public L4 docker builds are deterministic.
set -euxo pipefail

CUVS_VERSION="v26.02.00"
CUVS_REPO="https://github.com/rapidsai/cuvs.git"
CUVS_SRC_DIR="/tmp/cuvs"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cugraph}"
NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
PARALLEL_LEVEL="${PARALLEL_LEVEL:-$(nproc)}"
ENABLE_CCACHE="${ENABLE_CCACHE:-1}"
CCACHE_DIR="${CCACHE_DIR:-/ccache}"
CCACHE_BASEDIR="${CCACHE_BASEDIR:-${CUVS_SRC_DIR}}"
CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"

echo "============================================="
echo "Building libcuvs ${CUVS_VERSION} for sm_89 (L4)"
echo "  PARALLEL_LEVEL=${PARALLEL_LEVEL}"
echo "  ENABLE_CCACHE=${ENABLE_CCACHE}"
echo "============================================="

# Clone
git clone --depth 1 --branch "${CUVS_VERSION}" "${CUVS_REPO}" "${CUVS_SRC_DIR}"
cd "${CUVS_SRC_DIR}"

# Configure ccache
cmake_extra_args=""
case "${ENABLE_CCACHE,,}" in
  1|true|yes|on)
    export CCACHE_DIR
    export CCACHE_BASEDIR
    export CCACHE_MAXSIZE
    export CCACHE_COMPILERCHECK=${CCACHE_COMPILERCHECK:-content}
    ccache -M "${CCACHE_MAXSIZE}" >/dev/null 2>&1 || true
    cmake_extra_args="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
    ;;
esac

PARALLEL_LEVEL="${PARALLEL_LEVEL}" \
  /opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" -v --no-capture-output ./build.sh libcuvs -v --gpu-arch=89-real --cmake-args=\"${cmake_extra_args}\"

echo "============================================="
echo "libcuvs ${CUVS_VERSION} installed to cugraph conda environment"
echo "============================================="

# Cleanup source to save image space
rm -rf "${CUVS_SRC_DIR}"
