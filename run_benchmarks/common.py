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

"""Shared helpers for local and distributed benchmark orchestration."""

import datetime
import json
import _thread
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Sequence, TextIO


RESULTS_START_MARKER = "===RESULTS_START==="
RESULTS_END_MARKER = "===RESULTS_END==="


def build_collect_shell_cmd(
    mark: str, pytest_args: str, exhaustive: bool = False
) -> str:
    """Build the in-container collect-only command."""
    prefix = "export BENCH_EXHAUSTIVE=1 && " if exhaustive else ""
    pytest_args_list = [
        "python",
        "-m",
        "pytest",
        "bench_algos.py",
        "--collect-only",
        "-q",
    ]
    if mark:
        pytest_args_list.extend(["-m", mark])
    pytest_args_list.extend(shlex.split(pytest_args) if pytest_args else [])
    return " && ".join(
        [
            "source /opt/conda/bin/activate cugraph",
            f"{prefix}cd /cugraph/benchmarks/cugraph/pytest-based",
            shlex.join(pytest_args_list),
        ]
    )


def build_benchmark_shell_cmd(mark: str, pytest_args: str) -> str:
    """Build the in-container benchmark command with JSON result markers."""
    pytest_args_list = [
        "python",
        "-m",
        "pytest",
        "bench_algos.py",
        "-v",
        "--benchmark-json",
        "/tmp/results.json",
        "--timeout=60",
    ]
    if mark:
        pytest_args_list.extend(["-m", mark])
    pytest_args_list.extend(shlex.split(pytest_args) if pytest_args else [])
    pytest_cmd = shlex.join(pytest_args_list)
    marker_lines = " && ".join(
        [
            f"echo {shlex.quote(RESULTS_START_MARKER)}",
            "cat /tmp/results.json 2>/dev/null || true",
            f"echo {shlex.quote(RESULTS_END_MARKER)}",
            "exit $PYTEST_RC",
        ]
    )
    return (
        " && ".join(
            [
                "source /opt/conda/bin/activate cugraph",
                "export RAPIDS_DATASET_ROOT_DIR=/cugraph/datasets",
                "cd /cugraph/benchmarks/cugraph/pytest-based",
            ]
        )
        + f" && {{ {pytest_cmd}; PYTEST_RC=$?; }} ; "
        + marker_lines
    )


def build_override_mounts(bench_overrides: str) -> str:
    """Build docker mount flags for benchmark override files."""
    if not bench_overrides:
        return ""

    base = bench_overrides.rstrip("/")
    files = [
        ("bench_algos.py", "/cugraph/benchmarks/cugraph/pytest-based/bench_algos.py"),
        ("conftest.py", "/cugraph/benchmarks/cugraph/pytest-based/conftest.py"),
        ("pytest.ini", "/cugraph/benchmarks/pytest.ini"),
    ]
    return "".join(
        f" -v {shlex.quote(base + '/' + src)}:{dst}:ro" for src, dst in files
    )


def build_docker_run_cmd(
    image: str,
    datasets_dir: str,
    bench_cmd: str,
    bench_overrides: str = "",
    gpu_index: int = 0,
    extra_docker_args: str = "",
) -> str:
    """Build a docker run command for benchmark execution."""
    override_mounts = build_override_mounts(bench_overrides)
    extra = f" {extra_docker_args}" if extra_docker_args else ""
    return (
        f"docker run --gpus 'device={gpu_index}' --rm"
        f" -v {shlex.quote(datasets_dir)}:/cugraph/datasets"
        f"{override_mounts}"
        f"{extra}"
        f" {shlex.quote(image)}"
        f" bash -lc {shlex.quote(bench_cmd)}"
    )


def parse_collected_test_ids(stdout: str) -> list[str]:
    """Parse pytest --collect-only output into benchmark node IDs."""
    test_ids = []
    for line in stdout.splitlines():
        line = line.strip()
        if "::bench_" in line and not line.startswith("="):
            test_ids.append(line)
    return test_ids


def extract_results(stdout: str) -> dict | None:
    """Extract benchmark JSON from stdout using result markers."""
    start = stdout.find(RESULTS_START_MARKER)
    end = stdout.find(RESULTS_END_MARKER)
    if start < 0 or end < 0:
        return None

    json_str = stdout[start + len(RESULTS_START_MARKER) : end].strip()
    if not json_str:
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def merge_results(shard_results: dict[int, dict]) -> dict:
    """Merge per-shard pytest-benchmark JSON results."""
    merged = {"benchmarks": [], "_parallel_metadata": {"shards": []}}
    for shard_idx, data in sorted(shard_results.items()):
        benchmarks = data.get("benchmarks", [])
        merged["benchmarks"].extend(benchmarks)
        merged["_parallel_metadata"]["shards"].append(
            {"shard": shard_idx, "benchmark_count": len(benchmarks)}
        )

    if shard_results:
        first = next(iter(shard_results.values()))
        merged["machine_info"] = first.get("machine_info", {})
        merged["commit_info"] = first.get("commit_info", {})

    return merged


@dataclass(frozen=True)
class Bundle:
    """A group of benchmark test IDs run together in one container."""

    bundle_id: int
    test_ids: list[str]
    is_retry: bool = False


@dataclass(frozen=True)
class BundleResult:
    """Result for a single benchmark bundle run."""

    bundle_id: int
    host: str
    exit_code: int
    stdout: str
    benchmark_data: dict | None
    test_ids: list[str]
    is_retry: bool
    elapsed: float


def normalize_test_id(test_id: str) -> str:
    """Normalize to the ``bench_algos.py::...`` suffix when present."""
    idx = test_id.find("bench_algos.py::")
    if idx >= 0:
        return test_id[idx:]
    return test_id


def create_bundles(
    test_ids: Sequence[str], max_bundle_size: int, start_id: int = 0
) -> list[Bundle]:
    """Split test IDs into contiguous bundles."""
    if max_bundle_size <= 0:
        raise ValueError("max_bundle_size must be > 0")
    bundles: list[Bundle] = []
    for i in range(0, len(test_ids), max_bundle_size):
        bundles.append(
            Bundle(
                bundle_id=start_id + len(bundles),
                test_ids=list(test_ids[i : i + max_bundle_size]),
                is_retry=False,
            )
        )
    return bundles


def find_missing_test_ids(result: BundleResult) -> list[str]:
    """Return test IDs missing from a bundle result."""
    if result.benchmark_data is None:
        return list(result.test_ids)

    names = {
        normalize_test_id(item.get("fullname", ""))
        for item in result.benchmark_data.get("benchmarks", [])
    }
    return [tid for tid in result.test_ids if normalize_test_id(tid) not in names]


def merge_bundle_results(all_results: Sequence[BundleResult]) -> dict:
    """Merge all bundle benchmark outputs. Later results override earlier ones."""
    benchmarks_by_name: dict[str, dict] = {}
    for result in all_results:
        if result.benchmark_data:
            for benchmark in result.benchmark_data.get("benchmarks", []):
                key = normalize_test_id(benchmark.get("fullname", benchmark.get("name", "")))
                benchmarks_by_name[key] = benchmark

    merged = {
        "benchmarks": list(benchmarks_by_name.values()),
        "_parallel_metadata": {
            "bundles": [
                {
                    "bundle_id": result.bundle_id,
                    "is_retry": result.is_retry,
                    "host": result.host,
                    "exit_code": result.exit_code,
                    "test_count": len(result.test_ids),
                    "benchmark_count": (
                        len(result.benchmark_data.get("benchmarks", []))
                        if result.benchmark_data
                        else 0
                    ),
                    "elapsed": result.elapsed,
                }
                for result in sorted(all_results, key=lambda item: item.bundle_id)
            ]
        },
    }

    for result in all_results:
        if result.benchmark_data:
            merged["machine_info"] = result.benchmark_data.get("machine_info", {})
            merged["commit_info"] = result.benchmark_data.get("commit_info", {})
            break

    return merged


@dataclass(frozen=True)
class StreamRunResult:
    """Result from a streaming subprocess invocation."""

    returncode: int
    output: str
    timed_out: bool


def run_streaming_command(
    cmd: Sequence[str],
    prefix: str = "",
    timeout: int | None = None,
    capture_output: bool = True,
    stdout_stream: TextIO | None = sys.stdout,
    extra_streams: Sequence[TextIO] = (),
    line_lock: _thread.LockType | None = None,
    log_timestamp: bool = False,
    on_line: Callable[[str], None] | None = None,
    on_timeout: Callable[[], None] | None = None,
    encoding: str = "utf-8",
    errors: str = "replace",
    on_process_start: Callable[[subprocess.Popen], None] | None = None,
) -> StreamRunResult:
    """Run a command and stream merged stdout/stderr line-by-line.

    Args:
        cmd: Command and arguments passed directly to ``subprocess.Popen``.
        prefix: Optional tag prepended as ``[<prefix>] `` to each streamed line.
        timeout: Optional timeout in seconds. On timeout, the process is killed.
        capture_output: When True, accumulates raw merged output in the result.
        stdout_stream: Primary stream sink for emitted lines (defaults to ``sys.stdout``).
        extra_streams: Additional stream sinks that receive the same emitted lines.
        line_lock: Optional lock to serialize per-line writes across threads.
        log_timestamp: When True, prepend timestamps on writes to ``extra_streams`` only.
        on_line: Optional callback invoked for each raw output line.
        on_timeout: Optional callback invoked immediately before killing on timeout.
        encoding: Text decoding used by ``Popen(..., text=True)``.
        errors: Decode error handling strategy.
        on_process_start: Optional callback receiving the live ``Popen`` object.

    Returns:
        ``StreamRunResult`` with process return code, optional captured output,
        and whether timeout-triggered termination occurred.
    """

    output_lines: list[str] = []
    timed_out = False
    prefixed_tag = f"[{prefix}] " if prefix else ""
    extra_stream_ids = {id(stream) for stream in extra_streams}

    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=encoding,
        errors=errors,
    )
    if on_process_start is not None:
        on_process_start(proc)

    def _emit(line: str) -> None:
        if stdout_stream is None and not extra_streams:
            return

        prefixed = f"{prefixed_tag}{line}"
        targets: list[TextIO] = []
        if stdout_stream is not None:
            targets.append(stdout_stream)
        targets.extend(extra_streams)

        def _write() -> None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for stream in targets:
                out = prefixed
                if log_timestamp and id(stream) in extra_stream_ids:
                    out = f"{timestamp} {prefixed}"
                stream.write(out)
                stream.flush()

        if line_lock is not None:
            with line_lock:
                _write()
        else:
            _write()

    timer: threading.Timer | None = None

    if timeout is not None:

        def _kill_on_timeout() -> None:
            nonlocal timed_out
            timed_out = True
            if on_timeout is not None:
                on_timeout()
            try:
                proc.kill()
            except OSError:
                pass

        timer = threading.Timer(timeout, _kill_on_timeout)
        timer.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if capture_output:
                output_lines.append(line)
            if on_line is not None:
                on_line(line)
            _emit(line)
        returncode = proc.wait()
    finally:
        if timer is not None:
            timer.cancel()

    return StreamRunResult(
        returncode=returncode,
        output="".join(output_lines) if capture_output else "",
        timed_out=timed_out,
    )
