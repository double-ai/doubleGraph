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

set -euo pipefail

REPO_DIR=${REPO_DIR:-/cugraph}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-cugraph}
GPU_ARCH=${GPU_ARCH:-}
PARALLEL_LEVEL=${PARALLEL_LEVEL:-}
if [[ -z "${PARALLEL_LEVEL}" ]]; then
  PARALLEL_LEVEL=$(nproc)
fi
BUILD_ARGS=${BUILD_ARGS:-}
EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS:-}
ENABLE_CCACHE=${ENABLE_CCACHE:-1}
CCACHE_DIR=${CCACHE_DIR:-/ccache}
CCACHE_BASEDIR=${CCACHE_BASEDIR:-${REPO_DIR}}
CCACHE_MAXSIZE=${CCACHE_MAXSIZE:-20G}

# Note: --pydevelop (editable install) is intentionally NOT used in Docker builds
# because scikit-build-core's metadata check may fail for README.md in container context.
# Regular install works fine and is appropriate for Docker images.

cd "${REPO_DIR}"

extra_cmake_args="${EXTRA_CMAKE_ARGS}"

if [[ -n "${GPU_ARCH}" ]]; then
  extra_cmake_args="${extra_cmake_args} -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH}"
fi

echo "PARALLEL_LEVEL=${PARALLEL_LEVEL}"
echo "GPU_ARCH=${GPU_ARCH:-<unset>}"
echo "ENABLE_CCACHE=${ENABLE_CCACHE}"
echo "CCACHE_DIR=${CCACHE_DIR}"
echo "CCACHE_BASEDIR=${CCACHE_BASEDIR}"
echo "CCACHE_MAXSIZE=${CCACHE_MAXSIZE}"

case "${ENABLE_CCACHE,,}" in
  1|true|yes|on)
    export CCACHE_DIR
    export CCACHE_BASEDIR
    export CCACHE_MAXSIZE
    export CCACHE_COMPILERCHECK=${CCACHE_COMPILERCHECK:-content}
    ccache -M "${CCACHE_MAXSIZE}" >/dev/null 2>&1 || true
    ccache -p || true
    extra_cmake_args="${extra_cmake_args} -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache"
    ;;
esac

EXTRA_CMAKE_ARGS="${extra_cmake_args}" \
  PARALLEL_LEVEL="${PARALLEL_LEVEL}" \
  /opt/conda/bin/conda run -n "${CONDA_ENV_NAME}" -v --no-capture-output ./build.sh -v ${BUILD_ARGS}

if [[ "${ENABLE_CCACHE,,}" =~ ^(1|true|yes|on)$ ]]; then
  ccache -s || true
fi
