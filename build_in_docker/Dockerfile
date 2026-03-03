# syntax=docker/dockerfile:1.7
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

ARG ENV_IMAGE

FROM ${ENV_IMAGE}

ARG CONDA_ENV_NAME=cugraph
ARG GPU_ARCH=
ARG PARALLEL_LEVEL=
ARG BUILD_ARGS=
ARG RUN_BUILD=1
ARG ENABLE_CCACHE=1
ARG EXTRA_CMAKE_ARGS=

WORKDIR /cugraph
ENV RAPIDS_DATASET_ROOT_DIR=/cugraph/datasets

# First copy only the necessary files to build cugraph. This is for caching the build if only non-related files are changed.
COPY build.sh VERSION README.md LICENSE RAPIDS_BRANCH /cugraph/
COPY thirdparty /cugraph/thirdparty 
COPY cmake /cugraph/cmake
COPY python /cugraph/python
COPY cpp /cugraph/cpp

# For L4 (sm_89): build and install libcuvs from source to override the
# pre-compiled version in the conda env (which lacks sm_89 support).
RUN --mount=type=cache,target=/ccache,sharing=locked,id=ccache if [[ "${GPU_ARCH}" == "89" ]]; then \
        CONDA_ENV_NAME="${CONDA_ENV_NAME}" PARALLEL_LEVEL="${PARALLEL_LEVEL}" ENABLE_CCACHE="${ENABLE_CCACHE}" /opt/cugraph/build_libcuvs_sm89.sh; \
    else \
        echo "Skipping libcuvs source build (GPU_ARCH=${GPU_ARCH}, not L4)"; \
    fi

# Build cugraph. All artifacts (*.o, CMakeFiles, _deps, .git) are preserved
# so the dev image supports quick edit-compile-test loops via ninja.
RUN --mount=type=cache,target=/ccache,sharing=locked,id=ccache if [[ "${RUN_BUILD}" == "1" ]]; then \
        GPU_ARCH="${GPU_ARCH}" PARALLEL_LEVEL="${PARALLEL_LEVEL}" BUILD_ARGS="${BUILD_ARGS}" CONDA_ENV_NAME="${CONDA_ENV_NAME}" ENABLE_CCACHE="${ENABLE_CCACHE}" EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS}" /opt/cugraph/build.sh; \
    else \
        echo "Skipping cugraph build (RUN_BUILD=${RUN_BUILD})"; \
    fi

# Copy the rest of the repo files unneeded by the build process.
COPY . /cugraph

CMD ["/bin/bash"]
