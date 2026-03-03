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
Run cuGraph benchmarks locally via Docker on the current machine.

This is a local benchmark runner with no distributed orchestration concerns.
"""

import argparse
import json
import pathlib
import shlex
import subprocess
import sys
import time

from .common import (
    Bundle,
    BundleResult,
    build_collect_shell_cmd,
    build_docker_run_cmd,
    create_bundles,
    extract_results,
    find_missing_test_ids,
    merge_bundle_results,
    normalize_test_id,
    parse_collected_test_ids,
    run_streaming_command,
)


def _run_local(cmd: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run a shell command locally via bash -lc."""
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _require_local_image(image: str) -> None:
    """Raise if the requested Docker image is not available locally."""
    proc = subprocess.run(
        ["docker", "images", "-q", image],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip()
        raise RuntimeError(f"Failed to check local Docker image {image!r}: {err}")
    if not proc.stdout.strip():
        raise RuntimeError(
            f"Docker image {image!r} is not available locally. "
            "Build or pull it before running benchmarks."
        )


def _collect_test_ids(
    image: str,
    mark: str,
    pytest_args: str,
    datasets_dir: str,
    bench_overrides: str,
    exhaustive: bool,
    gpu_index: int,
) -> list[str]:
    """Collect benchmark test IDs from a local docker container."""
    collect_cmd = build_collect_shell_cmd(mark, pytest_args, exhaustive=exhaustive)
    docker_cmd = build_docker_run_cmd(
        image,
        datasets_dir,
        collect_cmd,
        bench_overrides,
        gpu_index=gpu_index,
    )
    rc, stdout, stderr = _run_local(docker_cmd, timeout=900)
    if rc not in (0, 5):
        print(f"  ERROR: collect failed: {stderr[:500]}")
        return []
    return parse_collected_test_ids(stdout)


def _run_bundle(
    image: str,
    bundle: Bundle,
    mark: str,
    pytest_args: str,
    datasets_dir: str,
    bench_overrides: str,
    timeout: int,
    exhaustive: bool,
    gpu_index: int,
) -> BundleResult:
    """Run a single benchmark bundle locally and return its parsed result."""
    pytest_args_list = [
        "python",
        "-m",
        "pytest",
        *bundle.test_ids,
        "-v",
        "--benchmark-json",
        "/tmp/results.json",
        f"--timeout={timeout}",
        "--timeout-method=thread",
    ]
    if mark:
        pytest_args_list.extend(["-m", mark])
    pytest_args_list.extend(shlex.split(pytest_args) if pytest_args else [])
    pytest_cmd = shlex.join(pytest_args_list)
    marker_lines = " && ".join(
        [
            "echo '===RESULTS_START==='",
            "cat /tmp/results.json 2>/dev/null || true",
            "echo '===RESULTS_END==='",
            "exit $PYTEST_RC",
        ]
    )
    pre_steps = [
        "source /opt/conda/bin/activate cugraph",
        "export RAPIDS_DATASET_ROOT_DIR=/cugraph/datasets",
    ]
    if exhaustive:
        pre_steps.append("export BENCH_EXHAUSTIVE=1")
    pre_steps.append("cd /cugraph/benchmarks/cugraph/pytest-based")
    bench_cmd = " && ".join(pre_steps) + f" && {{ {pytest_cmd}; PYTEST_RC=$?; }} ; " + marker_lines
    docker_cmd = build_docker_run_cmd(
        image,
        datasets_dir,
        bench_cmd,
        bench_overrides,
        gpu_index=gpu_index,
    )
    start = time.time()
    prefix = f"b{bundle.bundle_id}"
    n_tests = len(bundle.test_ids)
    retry_tag = " [retry]" if bundle.is_retry else ""
    plural = "s" if n_tests != 1 else ""
    print(f"  [{prefix}] Starting bundle {bundle.bundle_id} ({n_tests} test{plural}{retry_tag})")

    stream_timeout = max(7200, timeout * max(n_tests, 1) + 600)
    result = run_streaming_command(
        ["bash", "-lc", docker_cmd],
        prefix=prefix,
        timeout=stream_timeout,
        capture_output=True,
    )
    stdout = result.output
    benchmark_data = extract_results(stdout)
    elapsed = time.time() - start
    return BundleResult(
        bundle_id=bundle.bundle_id,
        host="local",
        exit_code=result.returncode,
        stdout=stdout,
        benchmark_data=benchmark_data,
        test_ids=bundle.test_ids,
        is_retry=bundle.is_retry,
        elapsed=elapsed,
    )


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    default_datasets = repo_root / "datasets"
    datasets_default = str(
        default_datasets
        if default_datasets.exists()
        else pathlib.Path("/mnt/nvme/cugraph-datasets")
    )

    parser = argparse.ArgumentParser(
        description="Run cuGraph benchmarks locally via Docker on the current machine",
    )
    parser.add_argument(
        "--image", required=True, help="Docker image to run benchmarks from"
    )
    parser.add_argument(
        "--output",
        default="benchmarks_combined.json",
        help="Output JSON file for benchmark results",
    )
    parser.add_argument(
        "--mark",
        default="",
        help='Pytest mark expression (default: "" = all tests). Examples: "tiny", "tiny or small", "not large".',
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help="Extra pytest arguments (e.g. '-k bfs'). Do NOT use -m here, use --mark instead.",
    )
    parser.add_argument(
        "--datasets-dir",
        default=datasets_default,
        help=f"Host path to benchmark datasets (default: {datasets_default})",
    )
    parser.add_argument(
        "--bench-overrides",
        default="",
        help="Host path to benchmark file overrides (bench_algos.py, conftest.py, pytest.ini)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-test timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index for docker --gpus device=<index> (default: 0)",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Set BENCH_EXHAUSTIVE=1 inside benchmark containers",
    )
    parser.add_argument(
        "--max-bundle-size",
        type=int,
        default=5,
        help="Max tests per bundle (default: 5)",
    )
    args = parser.parse_args()

    if args.max_bundle_size <= 0:
        print("ERROR: --max-bundle-size must be > 0")
        return 1
    if args.timeout <= 0:
        print("ERROR: --timeout must be > 0")
        return 1
    if args.gpu_index < 0:
        print("ERROR: --gpu-index must be >= 0")
        return 1

    args.datasets_dir = str(pathlib.Path(args.datasets_dir).expanduser())
    args.output = str(pathlib.Path(args.output).expanduser())
    if args.bench_overrides:
        args.bench_overrides = str(pathlib.Path(args.bench_overrides).expanduser())

    print(f"Image: {args.image}")
    print(f"Datasets: {args.datasets_dir}")
    print(f"GPU index: {args.gpu_index}")
    print(f"Exhaustive: {'yes' if args.exhaustive else 'no'}")
    print(f"Max bundle size: {args.max_bundle_size}")
    print()

    print("Step 1: Verifying Docker image is available locally...")
    _require_local_image(args.image)
    print("  Image is available locally")

    print("\nStep 2: Collecting test IDs...")
    print(f"  Mark filter: {args.mark}")
    test_ids = _collect_test_ids(
        args.image,
        args.mark,
        args.pytest_args,
        args.datasets_dir,
        args.bench_overrides,
        args.exhaustive,
        args.gpu_index,
    )
    if not test_ids:
        print("  ERROR: No tests collected. Check image and pytest args.")
        return 1
    test_ids = [normalize_test_id(tid) for tid in test_ids]
    print(f"  Collected {len(test_ids)} tests")

    bundles = create_bundles(test_ids, args.max_bundle_size)
    print(f"\nStep 3: Created {len(bundles)} bundles")

    print("\nStep 4: Running bundles locally...")
    start_time = time.time()
    all_results: list[BundleResult] = []
    pending = list(bundles)
    next_bundle_id = max(bundle.bundle_id for bundle in bundles) + 1 if bundles else 0

    while pending:
        bundle = pending.pop(0)
        result = _run_bundle(
            args.image,
            bundle,
            args.mark,
            args.pytest_args,
            args.datasets_dir,
            args.bench_overrides,
            args.timeout,
            args.exhaustive,
            args.gpu_index,
        )
        all_results.append(result)

        if not bundle.is_retry:
            missing = find_missing_test_ids(result)
            for test_id in missing:
                pending.append(Bundle(bundle_id=next_bundle_id, test_ids=[test_id], is_retry=True))
                next_bundle_id += 1

        n_collected = len(result.benchmark_data.get("benchmarks", [])) if result.benchmark_data else 0
        status = "OK" if result.exit_code == 0 and result.benchmark_data else f"exit {result.exit_code}"
        print(f"  [b{result.bundle_id}] Done: {n_collected}/{len(result.test_ids)} benchmarks [{status}]")

    elapsed = time.time() - start_time

    print("\nStep 5: Writing merged results...")
    merged = merge_bundle_results(all_results)
    with open(args.output, "w") as fh:
        json.dump(merged, fh, indent=2)

    total_benchmarks = len(merged.get("benchmarks", []))
    print(f"  Wrote {total_benchmarks} benchmarks to {args.output}")

    result_names = {
        normalize_test_id(b.get("fullname", ""))
        for b in merged.get("benchmarks", [])
    }
    permanently_failed = [
        tid for tid in test_ids if normalize_test_id(tid) not in result_names
    ]

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total tests: {len(test_ids)}")
    print(f"  Total benchmarks collected: {total_benchmarks}")
    n_original = sum(1 for r in all_results if not r.is_retry)
    n_original_ok = sum(1 for r in all_results if not r.is_retry and r.benchmark_data is not None)
    n_retries = sum(1 for r in all_results if r.is_retry)
    n_retries_ok = sum(1 for r in all_results if r.is_retry and r.benchmark_data is not None)
    print(f"  Original bundles: {n_original_ok}/{n_original} OK")
    if n_retries > 0:
        print(f"  Retries: {n_retries_ok}/{n_retries} OK")
    if permanently_failed:
        print(f"  Permanently failed: {len(permanently_failed)} tests")
    print(f"  Elapsed time: {elapsed / 60:.1f} minutes")

    stragglers = []
    for benchmark in merged.get("benchmarks", []):
        mean = benchmark.get("stats", {}).get("mean", 0)
        if mean > 240:
            stragglers.append((benchmark.get("name", "<unknown>"), mean))
    if stragglers:
        print("\n  Stragglers (>4 min):")
        for name, mean in sorted(stragglers, key=lambda item: -item[1]):
            print(f"    {name}: {mean:.1f}s")

    print(f"\nResults: {args.output}")
    return 0 if not permanently_failed else 1


if __name__ == "__main__":
    sys.exit(main())
