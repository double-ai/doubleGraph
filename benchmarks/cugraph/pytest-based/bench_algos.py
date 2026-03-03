# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import gc
import os
import sys
import time
from dataclasses import dataclass

import pytest
import numpy as np
import cupy
import cudf

import dask_cudf

import cugraph
import cugraph.dask as dask_cugraph
from cugraph.datasets import get_download_dir

from cugraph_benchmarking.datasets import (
    get_dataset,
    WEIGHTS_MAP,
    make_weight_processor,
)


###############################################################################
# GraphSpec: frozen dataclass for explicit benchmark parametrization


@dataclass(frozen=True)
class GraphSpec:
    dataset: str  # "karate", "rmat_14_16", "hollywood", etc.
    weights_dtype: str  # "none", "float32", "float64"
    renumber: bool
    directed: bool = False
    transposed: bool = False
    edge_mask_prob: float | None = None  # None = no mask

    def __repr__(self):
        wt = self.weights_dtype
        renum = "renum" if self.renumber else "norenum"
        parts = [self.dataset, wt, renum]
        if self.directed:
            parts.append("dir")
        if self.transposed:
            parts.append("trans")
        if self.edge_mask_prob is not None:
            parts.append(f"mask{self.edge_mask_prob}")
        return "-".join(parts)


###############################################################################
# Dataset lists -- grouped by graph size
#
# File datasets by edge count:
#   karate (156), netscience (5K), email_Eu_core (26K),
#   amazon0302 (1.2M), cit_patents (16M),
#   europe_osm (54M), hollywood (58M), soc_livejournal (69M), com_orkut (117M)
#
# RMAT datasets (edgefactor 16):
#   rmat_14_16 (~262K), rmat_16_16 (~1M),
#   rmat_18_16 (~4M), rmat_20_16 (~16M)

_TINY_FILES = ["karate", "netscience", "email_Eu_core"]
_MEDIUM_FILES = ["amazon0302", "cit_patents"]
_LARGE_FILES = ["europe_osm", "hollywood", "soc_livejournal"]

_TINY_GRAPHS = _TINY_FILES
_SMALL_GRAPHS = _TINY_FILES + ["rmat_14_16"]
_MEDIUM_GRAPHS = _TINY_FILES + _MEDIUM_FILES + ["rmat_16_16"]
_LARGE_GRAPHS = _TINY_FILES + _MEDIUM_FILES + _LARGE_FILES + ["rmat_18_16"]
_ALL_GRAPHS = _TINY_FILES + _MEDIUM_FILES + _LARGE_FILES + ["rmat_20_16"]

# When BENCH_EXHAUSTIVE=1, every algorithm runs on the same set of file
# datasets (netscience … soc_livejournal, com_orkut).
_EXHAUSTIVE = os.environ.get("BENCH_EXHAUSTIVE", "0") == "1"
if _EXHAUSTIVE:
    _EXHAUSTIVE_DATASETS = [
        "netscience", "email_Eu_core", "amazon0302", "cit_patents", "soc_livejournal",
        "com_orkut",
    ]
    _TINY_GRAPHS = _EXHAUSTIVE_DATASETS
    _SMALL_GRAPHS = _EXHAUSTIVE_DATASETS
    _MEDIUM_GRAPHS = _EXHAUSTIVE_DATASETS
    _LARGE_GRAPHS = _EXHAUSTIVE_DATASETS
    _ALL_GRAPHS = _EXHAUSTIVE_DATASETS

# Dataset -> size mark mapping for pytest -m filtering.
# Each dataset gets the *smallest* group it belongs to, so that
# ``pytest -m tiny`` runs only the cheapest datasets and ``-m small``
# adds slightly larger ones, etc.
_DATASET_SIZE_MARK = {}
for _ds in _TINY_FILES:
    _DATASET_SIZE_MARK[_ds] = pytest.mark.tiny
for _ds in _MEDIUM_FILES:
    _DATASET_SIZE_MARK[_ds] = pytest.mark.medium
for _ds in _LARGE_FILES:
    _DATASET_SIZE_MARK[_ds] = pytest.mark.large
_DATASET_SIZE_MARK["rmat_14_16"] = pytest.mark.small
_DATASET_SIZE_MARK["rmat_16_16"] = pytest.mark.medium
_DATASET_SIZE_MARK["rmat_18_16"] = pytest.mark.large
_DATASET_SIZE_MARK["rmat_20_16"] = pytest.mark.large


###############################################################################
# Helper functions for generating GraphSpec parametrize lists


def graph_params(
    datasets,
    weights=("none", "float32", "float64"),
    renumber=(True, False),
    directed=False,
    transposed=False,
):
    """Generate GraphSpec parametrize list with human-readable IDs.

    Each pytest.param carries the dataset's size mark (tiny/small/medium/large)
    so that ``pytest -m tiny`` selects only tiny-dataset variants.
    """
    result = []
    for ds in datasets:
        marks = [_DATASET_SIZE_MARK[ds]] if ds in _DATASET_SIZE_MARK else []
        for wt in weights:
            for renum in renumber:
                spec = GraphSpec(ds, wt, renum, directed=directed, transposed=transposed)
                result.append(pytest.param(spec, id=repr(spec), marks=marks))
    return result


def unweighted_params(datasets, renumber=(True, False)):
    """Generate unweighted GraphSpec parametrize list."""
    return graph_params(datasets, weights=("none",), renumber=renumber)


def weighted_params(datasets, renumber=(True, False)):
    """Generate weighted-only GraphSpec parametrize list."""
    return graph_params(datasets, weights=("float32", "float64"), renumber=renumber)


def transposed_params(datasets, **kw):
    """Generate transposed GraphSpec parametrize list."""
    return graph_params(datasets, transposed=True, **kw)


def with_edge_mask_variants(params, edge_mask_prob=0.3):
    """Duplicate param list with edge-mask variants for mask-supporting algos."""
    from dataclasses import replace
    result = list(params)
    for p in params:
        spec = p.values[0]
        masked = replace(spec, edge_mask_prob=edge_mask_prob)
        result.append(pytest.param(masked, id=repr(masked), marks=list(p.marks)))
    return result


###############################################################################
# Cached graph creation


_graph_cache = {}  # manual 1-entry cache: evict BEFORE creating new graph


def _make_graph(spec):
    """Create a cugraph.Graph from spec. Cached across all benchmarks.

    Uses a manual 1-entry cache instead of @lru_cache so that the old graph
    is evicted (and its GPU memory freed) BEFORE the new graph is allocated.
    lru_cache computes the new value first, causing both graphs to coexist
    briefly — enough to OOM on 24 GB GPUs with large graphs.
    """
    if spec in _graph_cache:
        return _graph_cache[spec]

    # Evict old graph and free GPU memory BEFORE creating the new one.
    # gc.collect() triggers DeviceBuffer deallocation which returns memory
    # to RMM's pool.  (cupy.get_default_memory_pool().free_all_blocks() is
    # a no-op because CuPy is configured to use rmm_cupy_allocator.)
    _graph_cache.clear()
    _mask_cache.clear()  # old graph's mask is no longer useful
    gc.collect()

    t0 = time.perf_counter()
    dataset = get_dataset(spec.dataset)
    weights_dtype = WEIGHTS_MAP[spec.weights_dtype]
    create_using = cugraph.Graph(directed=True) if spec.directed else cugraph.Graph
    G = dataset.get_graph(
        download=True,
        ignore_weights=weights_dtype is None,
        create_using=create_using,
        store_transposed=spec.transposed,
        renumber=spec.renumber,
        process_edgelist_fn=make_weight_processor(weights_dtype),
    )
    elapsed = time.perf_counter() - t0
    n_verts = G.number_of_vertices()
    n_edges = G.number_of_edges()
    print(
        f"GRAPH_LOADED: {spec!r}  vertices={n_verts:,}  edges={n_edges:,}  "
        f"load_time={elapsed:.2f}s",
        file=sys.stderr,
        flush=True,
    )
    _graph_cache[spec] = G
    return G


_mask_cache = {}  # {(base_spec, edge_mask_prob): bool_mask} – one entry, cleared on graph change

# Fused GPU kernel for deterministic symmetric edge masking.
#
# For each directed edge (src, dst), canonicalizes the pair via min/max so
# both directions hash identically, then applies a splitmix-style hash:
#   1. Combine: h = min * 0x9E3779B9 (golden ratio) + max * 0x85EBCA6B
#   2. Avalanche: XOR-shift and multiply to decorrelate similar inputs
#   3. Threshold: h < uint32(p * 2^32) yields True with probability ~p
#
# The hash is deterministic (no RNG state) but its avalanche property makes
# outputs indistinguishable from independent Bernoulli(p) draws for
# non-adversarial vertex IDs.  All computation is fused into a single GPU
# pass — the only allocation is the output boolean mask.
_symmetric_mask_kernel = cupy.ElementwiseKernel(
    'int32 src, int32 dst, uint32 threshold',
    'bool mask',
    '''
    unsigned int a = (unsigned int)min(src, dst);
    unsigned int b = (unsigned int)max(src, dst);
    unsigned int h = a * 0x9E3779B9u + b * 0x85EBCA6Bu;
    h ^= (h >> 16);
    h *= 0x45D9F3B3u;
    h ^= (h >> 13);
    mask = (h < threshold);
    ''',
    'symmetric_mask_kernel',
)


def _generate_mask(G, edge_mask_prob):
    """Generate a deterministic edge mask for *G*.

    For directed graphs a simple per-edge random draw is used.
    For undirected graphs the mask is symmetric: both directed copies of each
    undirected edge get the same value (via a canonical-pair hash computed in
    a single fused ElementwiseKernel to avoid intermediate GPU arrays).
    """
    if G.is_directed():
        num_edges = G.number_of_edges(directed_edges=True)
        rng = cupy.random.RandomState(seed=0xdeadbeef)
        return rng.rand(num_edges) < edge_mask_prob

    # Undirected: decompress CSR → fused hash kernel on canonical (min,max) pairs.
    from pylibcugraph import decompress_to_edgelist, ResourceHandle
    result = decompress_to_edgelist(ResourceHandle(), G._plc_graph, False)
    threshold = np.uint32(int(edge_mask_prob * (2**32 - 1)))
    return _symmetric_mask_kernel(result[0], result[1], threshold)


def _get_graph(spec):
    """Get graph with optional edge mask from spec."""
    from dataclasses import replace
    base = replace(spec, edge_mask_prob=None)
    G = _make_graph(base)

    # Always clean up any stale mask from a previous benchmark
    if G.has_edge_mask():
        G.detach_edge_mask()

    if spec.edge_mask_prob is not None:
        cache_key = (base, spec.edge_mask_prob)
        mask = _mask_cache.get(cache_key)
        if mask is None:
            # Free dangling GPU memory from previous benchmarks so that
            # mask generation has headroom.  CuPy uses RMM as its allocator
            # so cupy.get_default_memory_pool().free_all_blocks() is a no-op;
            # gc.collect() is what actually triggers DeviceBuffer dealloc.
            gc.collect()
            mask = _generate_mask(G, spec.edge_mask_prob)
            _mask_cache.clear()
            _mask_cache[cache_key] = mask
        G.attach_edge_mask(mask)

    return G


###############################################################################
# Helpers


def is_graph_distributed(graph):
    """Return True if graph is distributed (for use with cugraph.dask APIs)."""
    return isinstance(graph.edgelist.edgelist_df, dask_cudf.DataFrame)


def get_vertex_pairs(G, num_vertices=10):
    """Return a DataFrame containing two-hop vertex pairs deterministically sampled."""
    random_vertices = G.select_random_vertices(
        random_state=42, num_vertices=num_vertices
    )
    if isinstance(random_vertices, dask_cudf.Series):
        random_vertices = random_vertices.compute()
    vertices = random_vertices.to_arrow().to_pylist()
    return G.get_two_hop_neighbors(start_vertices=vertices)


def _select_random_vertices(G, num_vertices):
    n = min(num_vertices, G.number_of_vertices())
    return G.select_random_vertices(random_state=42, num_vertices=n)


_ALL_PAIRS_SMALL = 100
_ALL_PAIRS_LARGE = 500


def _partition_filename(spec):
    """Return the parquet filename for a cached Louvain partition."""
    renum = "renum" if spec.renumber else "norenum"
    return f"partition-{spec.dataset}-{spec.weights_dtype}-{renum}.parquet"


def _normalize_clustering(parts, col="cluster"):
    """Ensure cluster IDs are contiguous [0, k) so num_clusters == max_id + 1.

    Louvain can produce non-contiguous IDs (e.g. after renumbering), which
    causes num_clusters (from nunique) to be smaller than max_id + 1.  The
    analyze_clustering kernels allocate shared memory of size num_clusters, so
    an ID >= num_clusters triggers an illegal memory access.
    """
    codes, _ = parts[col].factorize()
    parts[col] = codes.astype("int32")
    n_clusters = int(codes.max()) + 1
    return parts, n_clusters


def _get_louvain_clustering(G, spec=None):
    """Load a precomputed Louvain partition for analyze_clustering benchmarks.

    Partitions are stored as parquet files under
    <dataset_root>/partitions/ (generated by
    benchmarks/precompute_partitions.py).  Falls back to computing
    Louvain on the fly if the cached file is not found.
    """
    if spec is not None:
        path = get_download_dir() / "partitions" / _partition_filename(spec)
        if path.is_file():
            parts = cudf.read_parquet(path)
            return _normalize_clustering(parts)

    # Fallback: compute on the fly
    num_vertices = G.number_of_vertices()
    if num_vertices > 500_000:
        parts, _ = cugraph.louvain(G, max_level=2, threshold=1e-4)
    else:
        parts, _ = cugraph.louvain(G)

    parts = parts.rename(columns={"partition": "cluster"})
    return _normalize_clustering(parts)


###############################################################################
# Benchmarks



@pytest.mark.parametrize("spec", with_edge_mask_variants(transposed_params(_ALL_GRAPHS)))
def bench_pagerank(benchmark, spec):
    graph = _get_graph(spec)
    pagerank = (
        dask_cugraph.pagerank
        if is_graph_distributed(graph)
        else cugraph.pagerank
    )
    benchmark(pagerank, graph)


@pytest.mark.parametrize("spec", with_edge_mask_variants(transposed_params(_ALL_GRAPHS)))
def bench_personalized_pagerank(benchmark, spec):
    G = _get_graph(spec)
    pagerank = (
        dask_cugraph.pagerank
        if is_graph_distributed(G)
        else cugraph.pagerank
    )
    verts = _select_random_vertices(G, 10)
    if isinstance(verts, dask_cudf.Series):
        verts = verts.compute()
    personalization = cudf.DataFrame({
        "vertex": verts,
        "values": cudf.Series([1.0 / len(verts)] * len(verts)),
    })
    benchmark(pagerank, G, personalization=personalization)


@pytest.mark.parametrize("spec", with_edge_mask_variants(graph_params(_ALL_GRAPHS)))
def bench_bfs(benchmark, spec):
    graph = _get_graph(spec)
    bfs = dask_cugraph.bfs if is_graph_distributed(graph) else cugraph.bfs
    start = graph.nodes().iloc[0]
    benchmark(bfs, graph, start)


@pytest.mark.parametrize("spec", with_edge_mask_variants(graph_params(_ALL_GRAPHS)))
def bench_bfs_direction_optimizing(benchmark, spec):
    graph = _get_graph(spec)
    bfs = dask_cugraph.bfs if is_graph_distributed(graph) else cugraph.bfs
    start = graph.nodes().iloc[0]
    benchmark(bfs, graph, start, direction_optimizing=True)


@pytest.mark.parametrize("spec", with_edge_mask_variants(weighted_params(_ALL_GRAPHS)))
def bench_sssp(benchmark, spec):
    graph = _get_graph(spec)
    sssp = (
        dask_cugraph.sssp
        if is_graph_distributed(graph)
        else cugraph.sssp
    )
    start = graph.nodes().iloc[0]
    benchmark(sssp, graph, start)


@pytest.mark.parametrize("spec", graph_params(_ALL_GRAPHS))
def bench_jaccard(benchmark, spec):
    G = _get_graph(spec)
    vert_pairs = get_vertex_pairs(G)
    jaccard = dask_cugraph.jaccard if is_graph_distributed(G) else cugraph.jaccard
    benchmark(jaccard, G, vert_pairs, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_ALL_GRAPHS))
def bench_sorensen(benchmark, spec):
    G = _get_graph(spec)
    # algo cannot compute neighbors on all nodes without running into OOM
    # this is why we will call sorensen on a subset of nodes
    vert_pairs = get_vertex_pairs(G)
    sorensen = dask_cugraph.sorensen if is_graph_distributed(G) else cugraph.sorensen
    benchmark(sorensen, G, vert_pairs, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_louvain(benchmark, spec):
    graph = _get_graph(spec)
    louvain = dask_cugraph.louvain if is_graph_distributed(graph) else cugraph.louvain
    benchmark(louvain, graph)


@pytest.mark.parametrize("spec", with_edge_mask_variants(graph_params(_ALL_GRAPHS)))
def bench_weakly_connected_components(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    if graph.is_directed():
        G = graph.to_undirected()
    else:
        G = graph
    benchmark(cugraph.weakly_connected_components, G)


@pytest.mark.parametrize("spec", graph_params(_ALL_GRAPHS))
def bench_overlap(benchmark, spec):
    G = _get_graph(spec)
    vertex_pairs = get_vertex_pairs(G)
    overlap = dask_cugraph.overlap if is_graph_distributed(G) else cugraph.overlap
    benchmark(overlap, G, vertex_pairs, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", with_edge_mask_variants(graph_params(_LARGE_GRAPHS)))
def bench_triangle_count(benchmark, spec):
    graph = _get_graph(spec)
    tc = (
        dask_cugraph.triangle_count
        if is_graph_distributed(graph)
        else cugraph.triangle_count
    )
    benchmark(tc, graph)


@pytest.mark.parametrize("spec", weighted_params(_TINY_GRAPHS))
def bench_spectralModularityMaximizationClustering(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    benchmark(cugraph.spectralModularityMaximizationClustering, graph, 2, random_state=42)


@pytest.mark.parametrize("spec", with_edge_mask_variants(graph_params(_MEDIUM_GRAPHS)))
def bench_betweenness_centrality(benchmark, spec):
    graph = _get_graph(spec)
    bc = (
        dask_cugraph.betweenness_centrality
        if is_graph_distributed(graph)
        else cugraph.betweenness_centrality
    )
    benchmark(bc, graph, k=10, random_state=123)


@pytest.mark.parametrize("spec", with_edge_mask_variants(graph_params(_SMALL_GRAPHS)))
def bench_edge_betweenness_centrality(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    benchmark(cugraph.edge_betweenness_centrality, graph, k=3, seed=123)


@pytest.mark.parametrize("spec", graph_params(_ALL_GRAPHS))
def bench_egonet(benchmark, spec):
    graph = _get_graph(spec)
    egonet = (
        dask_cugraph.ego_graph if is_graph_distributed(graph) else cugraph.ego_graph
    )
    n = 1
    radius = 2
    benchmark(egonet, graph, n, radius=radius)


@pytest.mark.parametrize("spec", with_edge_mask_variants(graph_params(_ALL_GRAPHS)))
def bench_two_hop_neighbors(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, 10)
    vertices = verts.to_arrow().to_pylist()
    benchmark(graph.get_two_hop_neighbors, start_vertices=vertices)


@pytest.mark.parametrize("spec", with_edge_mask_variants(transposed_params(_ALL_GRAPHS)))
def bench_hits(benchmark, spec):
    graph = _get_graph(spec)
    hits = (
        dask_cugraph.hits
        if is_graph_distributed(graph)
        else cugraph.hits
    )
    benchmark(hits, graph)


@pytest.mark.parametrize("spec", with_edge_mask_variants(transposed_params(_ALL_GRAPHS)))
def bench_eigenvector_centrality(benchmark, spec):
    graph = _get_graph(spec)
    ec = (
        dask_cugraph.eigenvector_centrality
        if is_graph_distributed(graph)
        else cugraph.eigenvector_centrality
    )
    benchmark(ec, graph)


@pytest.mark.parametrize("spec", with_edge_mask_variants(transposed_params(_ALL_GRAPHS)))
def bench_katz_centrality(benchmark, spec):
    graph = _get_graph(spec)
    katz = (
        dask_cugraph.katz_centrality
        if is_graph_distributed(graph)
        else cugraph.katz_centrality
    )
    kwargs = dict(alpha=None)
    if spec.edge_mask_prob is not None:
        kwargs["max_iter"] = 1000
    benchmark(katz, graph, **kwargs)


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_leiden(benchmark, spec):
    graph = _get_graph(spec)
    leiden = dask_cugraph.leiden if is_graph_distributed(graph) else cugraph.leiden
    benchmark(leiden, graph, max_iter=10, resolution=1.0, random_state=42)


@pytest.mark.parametrize("spec", with_edge_mask_variants(unweighted_params(_ALL_GRAPHS)))
def bench_core_number(benchmark, spec):
    graph = _get_graph(spec)
    core_number = (
        dask_cugraph.core_number
        if is_graph_distributed(graph)
        else cugraph.core_number
    )
    benchmark(core_number, graph)


# SCC requires directed graph with renumber=True (needs renumber_map).
@pytest.mark.parametrize(
    "spec",
    with_edge_mask_variants(graph_params(_SMALL_GRAPHS, directed=True, renumber=(True,))),
)
def bench_strongly_connected_components(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    benchmark(cugraph.strongly_connected_components, graph)


@pytest.mark.parametrize("spec", with_edge_mask_variants(unweighted_params(_ALL_GRAPHS)))
def bench_k_core(benchmark, spec):
    graph = _get_graph(spec)
    k_core = (
        dask_cugraph.k_core
        if is_graph_distributed(graph)
        else cugraph.k_core
    )
    benchmark(k_core, graph, k=None)


@pytest.mark.parametrize("spec", graph_params(_LARGE_GRAPHS))
def bench_cosine(benchmark, spec):
    graph = _get_graph(spec)
    vert_pairs = get_vertex_pairs(graph)
    cosine = (
        dask_cugraph.cosine
        if is_graph_distributed(graph)
        else cugraph.cosine
    )
    benchmark(cosine, graph, vert_pairs, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_jaccard_small(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_SMALL)
    benchmark(cugraph.all_pairs_jaccard, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_jaccard_large(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_LARGE)
    benchmark(cugraph.all_pairs_jaccard, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_sorensen_small(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_SMALL)
    benchmark(cugraph.all_pairs_sorensen, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_sorensen_large(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_LARGE)
    benchmark(cugraph.all_pairs_sorensen, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_overlap_small(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_SMALL)
    benchmark(cugraph.all_pairs_overlap, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_overlap_large(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_LARGE)
    benchmark(cugraph.all_pairs_overlap, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_cosine_small(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_SMALL)
    benchmark(cugraph.all_pairs_cosine, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", graph_params(_SMALL_GRAPHS))
def bench_all_pairs_cosine_large(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    verts = _select_random_vertices(graph, _ALL_PAIRS_LARGE)
    benchmark(cugraph.all_pairs_cosine, graph, vertices=verts, topk=100, use_weight=spec.weights_dtype != "none")


@pytest.mark.parametrize("spec", with_edge_mask_variants(unweighted_params(_MEDIUM_GRAPHS)))
def bench_k_truss(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    benchmark(cugraph.k_truss, graph, k=5)


@pytest.mark.parametrize("spec", weighted_params(_LARGE_GRAPHS))
def bench_minimum_spanning_tree(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    benchmark(cugraph.minimum_spanning_tree, graph)


@pytest.mark.parametrize("spec", graph_params(_TINY_GRAPHS))
def bench_ecg(benchmark, spec):
    graph = _get_graph(spec)
    ecg = dask_cugraph.ecg if is_graph_distributed(graph) else cugraph.ecg
    # ECG runs Louvain ensemble_size times internally, so large graphs
    # with default settings can take hours.  Keep ensemble_size small to
    # stay under 1s per call on tiny/small graphs.
    num_vertices = graph.number_of_vertices()
    if num_vertices > 500_000:
        kw = dict(ensemble_size=2, max_level=3, threshold=1e-3, random_state=42)
    else:
        kw = dict(ensemble_size=4, max_level=5, threshold=1e-4, random_state=42)
    benchmark(ecg, graph, **kw)


# Clustering quality metrics -- these evaluate an existing clustering against
# the graph structure. The louvain call in setup is NOT measured; only the
# scoring function is benchmarked.


@pytest.mark.parametrize("spec", weighted_params(_TINY_GRAPHS))
def bench_analyze_clustering_modularity(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    clustering, n_clusters = _get_louvain_clustering(graph, spec)
    benchmark(cugraph.analyzeClustering_modularity, graph, n_clusters, clustering)


@pytest.mark.parametrize("spec", graph_params(_TINY_GRAPHS))
def bench_analyze_clustering_edge_cut(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    clustering, n_clusters = _get_louvain_clustering(graph, spec)
    benchmark(cugraph.analyzeClustering_edge_cut, graph, n_clusters, clustering)


@pytest.mark.parametrize("spec", graph_params(_TINY_GRAPHS))
def bench_analyze_clustering_ratio_cut(benchmark, spec):
    graph = _get_graph(spec)
    if is_graph_distributed(graph):
        pytest.skip("distributed graphs are not supported")
    clustering, n_clusters = _get_louvain_clustering(graph, spec)
    benchmark(
        cugraph.analyzeClustering_ratio_cut, graph, n_clusters, clustering
    )
