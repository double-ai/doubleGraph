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
"""Build libcugraph wheels with a self-contained Docker-based workflow."""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from wheel_building.common import (
    WHEEL_BUILD_ORDER,
    detect_target_gpu,
    load_target_gpu_map,
    resolve_wheel_chain,
    wheel_arch_from_target_gpu,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cugraph wheels in a manylinux_2_28-compatible Docker container."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="./wheel_building/dist")
    parser.add_argument("--raw-wheel-dir", default="/tmp/cugraph-wheel-raw")
    parser.add_argument("--ccache-dir", default="/tmp/cugraph-wheel-ccache")
    parser.add_argument("--pip-cache-dir", default="/tmp/cugraph-wheel-pip-cache")
    parser.add_argument("--cuda-version", default="13.0.2")
    parser.add_argument("--wheel-target-gpu", default="")
    parser.add_argument("--wheel-aai-algorithms", default="")
    parser.add_argument(
        "--manylinux-platform",
        default="manylinux_2_28_x86_64",
        help="PEP 600 platform tag to enforce via auditwheel repair",
    )
    parser.add_argument(
        "--no-auditwheel",
        action="store_false",
        dest="use_auditwheel",
        default=True,
        help="Skip auditwheel repair (wheel will not be manylinux-tagged)",
    )
    parser.add_argument("--builder-image", default="")
    parser.add_argument(
        "--parallel",
        "-j",
        default="",
        help="Max parallel compilation jobs inside container (default: nproc)",
    )
    parser.add_argument(
        "--wheel",
        default="cugraph",
        choices=WHEEL_BUILD_ORDER,
        help="Top-level wheel to build; prerequisites are built automatically (default: cugraph)",
    )
    return parser.parse_args()


def normalize_tag(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "", value)
    return cleaned or "default"


def ensure_repo_layout(repo_root: Path, wheel_chain: list[str]) -> None:
    required = [
        repo_root / "wheel_building" / "in_container_build.py",
        repo_root / "wheel_building" / "Dockerfile",
    ]
    for wheel_name in wheel_chain:
        required.append(repo_root / "python" / wheel_name)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required files for wheel build:\n" + "\n".join(missing))


def run_checked(cmd: list[str], *, label: str = "") -> None:
    print("+", " ".join(cmd))
    # Use sys.stdout which may be a _Tee, so subprocess needs to write through it.
    # If sys.stdout has a real fileno (or is the original), just inherit.
    stdout_target = sys.stdout if isinstance(sys.stdout, _Tee) else None
    if stdout_target:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.write(line.decode("utf-8", errors="replace"))
        proc.wait()
        returncode = proc.returncode
    else:
        returncode = subprocess.run(cmd).returncode
    if returncode != 0:
        step = label or cmd[0]
        raise SystemExit(f"\n*** {step} failed (exit code {returncode}) ***")


class _Tee:
    """Write to both a file and the original stream."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def fileno(self):
        return self._stream.fileno()


def main() -> int:
    args = parse_args()
    if not args.wheel_target_gpu:
        args.wheel_target_gpu = detect_target_gpu()
    repo_root = Path(args.repo_root).resolve()

    # Set up log file: tee all output to wheel_building/logs/<timestamp>.log
    log_dir = repo_root / "wheel_building" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Logging to {log_path}")
    target_map = load_target_gpu_map(repo_root)
    wheel_cuda_arch = wheel_arch_from_target_gpu(args.wheel_target_gpu, target_map)
    output_root = Path(args.output_dir).resolve()
    output_dir = output_root / f"arch-{normalize_tag(wheel_cuda_arch)}"
    raw_wheel_dir = Path(args.raw_wheel_dir).resolve()
    ccache_dir = Path(args.ccache_dir).resolve()
    pip_cache_dir = Path(args.pip_cache_dir).resolve()
    wheel_chain = resolve_wheel_chain(args.wheel)
    ensure_repo_layout(repo_root, wheel_chain)

    subprocess.run(["docker", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    image = args.builder_image or f"cugraph-wheel-builder:manylinux228-cu{normalize_tag(args.cuda_version)}"

    run_checked(
        [
            "docker",
            "build",
            "-f",
            str(repo_root / "wheel_building" / "Dockerfile"),
            "--build-arg",
            f"CUDA_VERSION={args.cuda_version}",
            "-t",
            image,
            str(repo_root / "wheel_building"),
        ],
        label="docker build",
    )

    uid = os.getuid()
    gid = os.getgid()
    for path in (output_dir, raw_wheel_dir):
        path.mkdir(parents=True, exist_ok=True)
    for path in (ccache_dir, pip_cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    container_cmd = [
        "python3",
        "-m",
        "wheel_building.in_container_build",
        "--output-dir",
        "/output",
        "--raw-wheel-dir",
        "/tmp/wheel-raw",
        "--wheel-target-gpu",
        args.wheel_target_gpu,
        "--wheel-aai-algorithms",
        args.wheel_aai_algorithms,
        "--manylinux-platform",
        args.manylinux_platform,
    ]
    container_cmd.extend(["--wheel", args.wheel])
    if not args.use_auditwheel:
        container_cmd.append("--no-auditwheel")
    if args.parallel:
        container_cmd.extend(["--parallel", args.parallel])

    run_checked(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{repo_root}:/src",
            "-v",
            f"{output_dir}:/output",
            "-v",
            f"{raw_wheel_dir}:/tmp/wheel-raw",
            "-v",
            f"{ccache_dir}:/ccache",
            "-v",
            f"{pip_cache_dir}:/root/.cache/pip",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "--workdir",
            "/src",
            image,
            *container_cmd,
        ],
        label="in-container wheel build",
    )

    # Fix ownership of files created by the root-running container.
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{output_dir}:/output",
            "-v", f"{raw_wheel_dir}:/tmp/wheel-raw",
            "-v", f"{ccache_dir}:/ccache",
            "-v", f"{repo_root / 'wheel_building' / '__pycache__'}:/pycache",
            image,
            "chown", "-R", f"{uid}:{gid}",
            "/output", "/tmp/wheel-raw", "/ccache", "/pycache",
        ],
        check=False,
    )

    wheels = sorted(output_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"No wheels produced in {output_dir}")
    print("=== Built wheel(s) ===")
    for wheel in wheels:
        print(f"  - {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
