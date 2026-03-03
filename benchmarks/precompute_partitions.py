#!/usr/bin/env python3
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
"""
Precompute Louvain partitions for the analyze_clustering_* benchmarks.

Generates one parquet file per (dataset, weights_dtype, renumber) combo
used by bench_analyze_clustering_{modularity,edge_cut,ratio_cut} in
bench_algos.py.  Output files are written to a ``partitions/`` subdirectory
of --output-dir (default: $RAPIDS_DATASET_ROOT_DIR or /cugraph/datasets).

Usage (inside the cugraph Docker container):
    python precompute_partitions.py --output-dir /mnt/nvme/cugraph-datasets
"""
import argparse
import os
import sys
import time

import cugraph

# Add shared/python to the path so we can import cugraph_benchmarking
# (pytest gets this for free via pytest.ini's pythonpath setting).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shared", "python"))

from cugraph_benchmarking.datasets import (  # noqa: E402
    get_dataset,
    WEIGHTS_MAP,
    make_weight_processor,
)

# All datasets used across every benchmark tier.
_ALL_DATASETS = [
    "karate", "netscience", "email_Eu_core",
    "amazon0302", "cit_patents",
    "europe_osm", "hollywood", "soc_livejournal",
    "rmat_14_16", "rmat_16_16", "rmat_18_16", "rmat_20_16",
]


# ---------------------------------------------------------------------------
# Partition filename convention (must match bench_algos.py loading logic)
# ---------------------------------------------------------------------------


def partition_filename(dataset, weights_dtype, renumber):
    """Return the parquet filename for a given partition spec."""
    renum = "renum" if renumber else "norenum"
    return f"partition-{dataset}-{weights_dtype}-{renum}.parquet"


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_partition(dataset_name, weights_dtype, renumber):
    """Build a graph and run Louvain, returning (clustering_df, n_clusters)."""
    dataset = get_dataset(dataset_name)
    wt_dtype = WEIGHTS_MAP[weights_dtype]
    G = dataset.get_graph(
        download=True,
        ignore_weights=wt_dtype is None,
        create_using=cugraph.Graph,
        store_transposed=False,
        renumber=renumber,
        process_edgelist_fn=make_weight_processor(wt_dtype),
    )
    num_vertices = G.number_of_vertices()
    if num_vertices > 500_000:
        parts, _ = cugraph.louvain(G, max_level=2, threshold=1e-4)
    else:
        parts, _ = cugraph.louvain(G)

    n_clusters = parts["partition"].nunique()
    parts = parts.rename(columns={"partition": "cluster"})
    return parts, n_clusters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("RAPIDS_DATASET_ROOT_DIR", "/cugraph/datasets"),
        help="Root directory for benchmark data (partitions/ is created inside it).",
    )
    args = parser.parse_args()

    partitions_dir = os.path.join(args.output_dir, "partitions")
    os.makedirs(partitions_dir, exist_ok=True)

    # All (dataset, weights_dtype, renumber) combos used by the three
    # analyze_clustering benchmarks (union of graph_params and
    # weighted_params over _SMALL_GRAPHS).
    specs = []
    for ds in _ALL_DATASETS:
        for wt in ("none", "float32", "float64"):
            for renum in (True, False):
                specs.append((ds, wt, renum))

    print(f"Precomputing {len(specs)} Louvain partitions into {partitions_dir}/")
    for i, (ds, wt, renum) in enumerate(specs, 1):
        fname = partition_filename(ds, wt, renum)
        out_path = os.path.join(partitions_dir, fname)
        print(f"  [{i}/{len(specs)}] {fname} ...", end="", flush=True)
        t0 = time.perf_counter()
        parts, n_clusters = compute_partition(ds, wt, renum)
        parts.to_parquet(out_path)
        elapsed = time.perf_counter() - t0
        print(f" {n_clusters} clusters, {elapsed:.2f}s")

    print(f"\nDone. {len(specs)} partition files written to {partitions_dir}/")


if __name__ == "__main__":
    main()
