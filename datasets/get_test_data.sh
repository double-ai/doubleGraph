#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Modifications Copyright (c) 2025, AA-I Technologies Ltd.
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
#
# NOTICE: This file has been modified by AA-I Technologies Ltd. from the original.
set -e
set -o pipefail

# Ensure we're in the cugraph/datasets dir
cd "$( cd "$( dirname "$(realpath -m "${BASH_SOURCE[0]}")" )" && pwd )";

# Update this to add/remove/change a dataset, using the following format:
#
#  comment about the dataset
#  dataset download URL
#  destination dir to untar to
#  blank line separator
#
# FIXME: some test data needs to be extracted to "benchmarks", which is
# confusing now that there's dedicated datasets for benchmarks.
CPP_CI_DATASET_DATA="
# ~10s download
https://data.rapids.ai/cugraph/test/cpp_ci_datasets.tgz
test
"

BASE_DATASET_DATA="
# ~22s download
https://data.rapids.ai/cugraph/test/datasets.tgz
test

# ~14s download
https://data.rapids.ai/cugraph/test/ref/pagerank.tgz
test/ref

# ~1s download
https://data.rapids.ai/cugraph/test/ref/sssp.tgz
test/ref

# ~15s download
https://data.rapids.ai/cugraph/benchmark/hibench/hibench_1_large.tgz
benchmark

# ~1s download
https://data.rapids.ai/cugraph/benchmark/hibench/hibench_1_small.tgz
benchmark

# ~0.6s download
https://data.rapids.ai/cugraph/test/tsplib/datasets.tar.gz
tsplib
"

EXTENDED_DATASET_DATA="
# ~42s download - tests using this dataset are currently not run in test.sh with --quick
https://data.rapids.ai/cugraph/benchmark/hibench/hibench_1_huge.tgz
benchmark
"

BENCHMARK_DATASET_DATA="
# ~90s download - these are used for benchmarks runs (code in <cugraph root>/benchmarks)
https://data.rapids.ai/cugraph/benchmark/benchmark_csv_data.tgz
csv
"

SELF_LOOPS_DATASET_DATA="
# ~1s download
https://data.rapids.ai/cugraph/benchmark/benchmark_csv_data_self_loops.tgz
self_loops
"
################################################################################
# Do not change the script below this line if only adding/updating a dataset

NUMARGS=$#
ARGS=$*
function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

VALID_ARGS="-h --help --subset --cpp_ci_subset --benchmark --self_loops"
USAGE="Usage: $0 [$(echo "${VALID_ARGS}" | sed 's/ / | /g')]"

if hasArg -h || hasArg --help; then
    echo "${USAGE}"
    exit 0
fi

# Validate all arguments
for arg in $ARGS; do
    if ! echo " ${VALID_ARGS} " | grep -q " ${arg} "; then
        echo "Error: unknown option '${arg}'"
        echo "${USAGE}"
        exit 1
    fi
done

# Select the datasets to install
if hasArg "--benchmark"; then
    DATASET_DATA="${BENCHMARK_DATASET_DATA}"
elif hasArg "--subset"; then
    DATASET_DATA="${BASE_DATASET_DATA}"
elif hasArg "--cpp_ci_subset"; then
    DATASET_DATA="${CPP_CI_DATASET_DATA}"
elif hasArg "--self_loops"; then
    DATASET_DATA="${SELF_LOOPS_DATASET_DATA}"
# Do not include benchmark datasets by default - too big
else
    DATASET_DATA="${BASE_DATASET_DATA} ${EXTENDED_DATASET_DATA}"
fi

# shellcheck disable=SC2207
URLS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 3) print $0}'))  # extract 3rd fields to a bash array
# shellcheck disable=SC2207
DESTDIRS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 0) print $0}'))  # extract 4th fields to a bash array

echo Downloading ...

# Download all tarfiles to a tmp dir
mkdir -p tmp
cd tmp
for url in "${URLS[@]}"; do
   time wget -N --progress=dot:giga "${url}"
done
cd ..

# create the destination dirs
mkdir -p "${DESTDIRS[@]}"

# Iterate over the arrays and untar the nth tarfile to the nth dest directory.
# The tarfile name is derived from the download url.
echo Decompressing ...
# shellcheck disable=SC2016
for index in ${!DESTDIRS[*]}; do
    echo "tmp/$(basename "${URLS[$index]}") -C ${DESTDIRS[$index]}" | tr '\n' '\0'
done | xargs -0 -t -r -n1 -P"$(nproc --all)" sh -c 'tar -xzvf $0 --overwrite'

# Some tar files extract into csv/directed/ and csv/undirected/, but certain
# scripts expect CSV files directly under the datasets directory. Create
# symlinks to bridge the two layouts.
for subdir in csv/directed csv/undirected; do
    if [ -d "$subdir" ]; then
        for src in "$subdir"/*.csv; do
            [ -e "$src" ] || continue
            dest="$(basename "$src")"
            if [ ! -e "$dest" ]; then
                ln -s "$src" "$dest"
            fi
        done
    fi
done
