# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
import ctypes
import gc
import hashlib
import sys
import re

import pytest

import rmm


# ---------------------------------------------------------------------------
# CUDA recovery: reset GPU after fatal CUDA errors so subsequent tests survive
# ---------------------------------------------------------------------------

_CUDA_FATAL_KEYWORDS = ("cudaError", "CUDA error", "cudaErrorIllegalAddress")
_CUDA_OOM_KEYWORDS = ("MemoryError", "out_of_memory")
_cuda_needs_recovery = False


def _invalidate_graph_caches():
    """Clear graph and mask caches so the next test rebuilds from scratch."""
    try:
        from bench_algos import _graph_cache, _mask_cache
        _graph_cache.clear()
        _mask_cache.clear()
    except Exception:
        pass
    gc.collect()


def _reset_cuda_device(config):
    """Reset the CUDA device and reinitialize RMM to recover from fatal errors.

    Must be called AFTER _invalidate_graph_caches() so that old GPU objects
    are freed while the CUDA context is still valid.
    """
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        rc = libcudart.cudaDeviceReset()
        if rc != 0:
            print(f"[CUDA_RECOVERY] cudaDeviceReset returned {rc}", file=sys.stderr, flush=True)
            return False

        managed_mem = config.getoption("managed_memory")
        pool_alloc = not config.getoption("no_pool_allocator")
        pool_size = config.getoption("rmm_pool_size")
        rmm.reinitialize(
            managed_memory=managed_mem,
            pool_allocator=pool_alloc,
            initial_pool_size=pool_size,
        )

        print("[CUDA_RECOVERY] Device reset and RMM reinitialized successfully", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[CUDA_RECOVERY] Recovery failed: {e}", file=sys.stderr, flush=True)
        return False


def pytest_addoption(parser):
    parser.addoption(
        "--shard-index",
        action="store",
        type=int,
        default=None,
        metavar="N",
        help="This worker's shard index (0-based) for parallel execution.",
    )
    parser.addoption(
        "--shard-total",
        action="store",
        type=int,
        default=None,
        metavar="N",
        help="Total number of shards for parallel execution.",
    )
    parser.addoption(
        "--print-bench-stats",
        action="store_true",
        default=False,
        help="Print pytest-benchmark stats (mean/min/max/stddev/rounds) after each benchmark test.",
    )
    parser.addoption(
        "--shard-mode",
        action="store",
        choices=["contiguous", "hash"],
        default="contiguous",
        help="Sharding strategy: 'contiguous' splits sorted list into chunks "
        "(preserves fixture grouping), 'hash' assigns by node ID hash. "
        "Default: %(default)s.",
    )
    parser.addoption(
        "--nodeids-file",
        action="store",
        default=None,
        help="File with allowed node IDs (one per line). "
        "Only tests whose nodeid appears in the file will run.",
    )
    parser.addoption(
        "--managed-memory",
        action="store_true",
        default=False,
        help="Enable RMM managed memory.",
    )
    parser.addoption(
        "--no-pool-allocator",
        action="store_true",
        default=True,
        help="Disable RMM pool allocator (disabled by default; "
        "the pool never releases memory back to CUDA, which causes OOM "
        "when cycling through many graph configurations).",
    )
    parser.addoption(
        "--pool-allocator",
        action="store_true",
        default=False,
        help="Enable RMM pool allocator (overrides --no-pool-allocator).",
    )
    parser.addoption(
        "--rmm-pool-size",
        action="store",
        type=int,
        default=2 << 27,
        metavar="bytes",
        help="Initial RMM pool size in bytes. Default is %(default)s.",
    )


def pytest_sessionstart(session):
    # Initialize RMM once at session start with CLI-configurable settings.
    # Skip during --collect-only (no GPU needed for test collection).
    if session.config.option.collectonly:
        return

    managed_mem = session.config.getoption("managed_memory")
    pool_alloc = session.config.getoption("pool_allocator") or not session.config.getoption("no_pool_allocator")
    pool_size = session.config.getoption("rmm_pool_size")

    rmm.reinitialize(
        managed_memory=managed_mem,
        pool_allocator=pool_alloc,
        initial_pool_size=pool_size,
    )
    print(
        f"RMM initialized: managed_memory={managed_mem}, pool_allocator={pool_alloc}, pool_size={pool_size}",
        file=sys.stderr,
        flush=True,
    )


def pytest_collection_modifyitems(config, items):
    """Filter by nodeids file (if given), then sort benchmarks by graph spec.

    Test IDs use GraphSpec repr: "dataset-dtype-renum[-dir][-trans]".

    Ordering priority:
      1. Graph spec (groups all benchmarks for the same graph together).
      2. RMAT datasets first (sorted by scale), then file-based alphabetically.
      3. Weight dtype: none < float32 < float64.
      4. Renumber: norenum < renum.
      5. Alphabetical test name as tiebreaker.
    """
    # --- Filter by nodeids file (runs BEFORE sort) ---
    nodeids_file = config.getoption("nodeids_file")
    if nodeids_file:
        with open(nodeids_file) as f:
            allowed = {line.strip() for line in f if line.strip()}
        items[:] = [item for item in items if item.nodeid in allowed]

    _SPEC_PATTERN = re.compile(r"\[(.+)\]$")
    _DTYPE_ORDER = {"none": 0, "float32": 1, "float64": 2}
    _RENUM_ORDER = {"norenum": 0, "renum": 1}
    _MASK_PATTERN = re.compile(r"-mask[\d.]+$")

    def sort_key(item):
        nid = item.nodeid
        spec_match = _SPEC_PATTERN.search(nid)
        if not spec_match:
            return (0, "", 0, 0, 0, nid)

        spec_str = spec_match.group(1)
        parts = spec_str.split("-")

        # Dataset name is the first part; RMAT datasets start with "rmat_"
        ds_name = parts[0] if parts else ""
        if ds_name.startswith("rmat"):
            try:
                scale = int(ds_name.split("_")[1])
            except (IndexError, ValueError):
                scale = 0
            ds_sort = (-1, scale, ds_name)
        else:
            ds_sort = (0, 0, ds_name)

        # Weight dtype is the second part
        dtype_key = _DTYPE_ORDER.get(parts[1], -1) if len(parts) > 1 else -1

        # Renumber is the third part
        renum_key = _RENUM_ORDER.get(parts[2], -1) if len(parts) > 2 else -1

        # Masked variants sort after all non-masked for the same graph spec
        mask_key = 1 if _MASK_PATTERN.search(spec_str) else 0

        return (ds_sort, dtype_key, renum_key, mask_key, nid)

    items.sort(key=sort_key)

    # --- Sharding for parallel execution (runs AFTER sort) ---
    shard_index = items[0].config.getoption("shard_index") if items else None
    shard_total = items[0].config.getoption("shard_total") if items else None

    if shard_index is not None and shard_total is not None:
        shard_mode = items[0].config.getoption("shard_mode")
        n = len(items)

        if shard_mode == "contiguous":
            base_size, remainder = divmod(n, shard_total)
            if shard_index < remainder:
                start = shard_index * (base_size + 1)
                end = start + base_size + 1
            else:
                start = remainder * (base_size + 1) + (shard_index - remainder) * base_size
                end = start + base_size
            items[:] = items[start:end]
        else:
            # Hash-based: keep items where md5(nodeid) % total == index.
            items[:] = [
                item
                for item in items
                if int(hashlib.md5(item.nodeid.encode()).hexdigest(), 16) % shard_total
                == shard_index
            ]


def pytest_runtest_logstart(nodeid, location):
    """Print test name BEFORE execution to identify stragglers in real-time."""
    global _cuda_needs_recovery
    if _cuda_needs_recovery:
        print(f"[CUDA_RECOVERY] Running after device reset: {nodeid}", file=sys.stderr, flush=True)
        _cuda_needs_recovery = False
    print(f"STARTING: {nodeid}", file=sys.stderr, flush=True)


@pytest.hookimpl(trylast=True)
def pytest_runtest_makereport(item, call):
    """Detect CUDA/OOM errors and recover so subsequent tests aren't poisoned.

    OOM errors: clear graph/mask caches so the next test rebuilds from scratch.
    Fatal CUDA errors: additionally reset the device and reinitialize RMM.
    """
    global _cuda_needs_recovery
    if call.when == "call" and call.excinfo is not None:
        exc_repr = str(call.excinfo.getrepr()) if call.excinfo else ""
        is_fatal = any(kw in exc_repr for kw in _CUDA_FATAL_KEYWORDS)
        is_oom = any(kw in exc_repr for kw in _CUDA_OOM_KEYWORDS)
        if is_oom or is_fatal:
            # Always invalidate caches FIRST (while CUDA context is still
            # alive) so that GPU objects are freed safely via RMM.
            _invalidate_graph_caches()
            print(
                f"[CUDA_RECOVERY] {'Fatal CUDA error' if is_fatal else 'OOM'} "
                f"detected in {item.nodeid}, caches invalidated",
                file=sys.stderr, flush=True,
            )
            if is_fatal:
                _cuda_needs_recovery = True
                _reset_cuda_device(item.config)


def pytest_runtest_logreport(report):
    """Print test duration immediately after execution for progressive timing data."""
    if report.when != "call":
        return

    duration = getattr(report, "duration", 0)
    if report.passed:
        status = "PASSED"
    elif report.failed:
        status = "FAILED"
    elif report.skipped:
        status = "SKIPPED"
    else:
        status = "UNKNOWN"
    print(f"FINISHED: {report.nodeid} - {status} in {duration:.2f}s", file=sys.stderr, flush=True)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    """Print pytest-benchmark stats after each test when --print-bench-stats is set."""
    if not item.config.getoption("--print-bench-stats"):
        return

    bm = item.funcargs.get("benchmark", None)
    if not bm or not getattr(bm, "stats", None):
        return

    s = bm.stats
    print(
        f"\n[BENCH] {item.nodeid}\n"
        f"  mean={s['mean'] * 1e6:.2f}us  std={s['stddev'] * 1e6:.2f}us  "
        f"min={s['min'] * 1e6:.2f}us  max={s['max'] * 1e6:.2f}us  "
        f"rounds={s.get('rounds', '?')}  iterations={s.get('iterations', '?')}\n"
    )
