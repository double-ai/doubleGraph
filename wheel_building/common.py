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
"""Shared helpers for self-contained wheel_building scripts."""

import json
import shlex
import site
import subprocess
import sysconfig
from pathlib import Path


def run_checked(cmd: list[str], *, env: dict[str, str] | None = None, label: str = "") -> None:
    print("+", shlex.join(cmd))
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        step = label or shlex.join(cmd[:3])
        raise SystemExit(f"\n*** {step} failed (exit code {result.returncode}) ***")


def site_package_roots() -> list[Path]:
    roots = [Path(p) for p in site.getsitepackages()]
    paths = sysconfig.get_paths()
    for key in ("platlib", "purelib"):
        raw = paths.get(key)
        if raw:
            roots.append(Path(raw))
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def collect_wheel_paths(wheel_dir: Path, *, context: str) -> list[Path]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"No wheels found in {wheel_dir}")
    print(f"=== {context} ===")
    for wheel in wheels:
        print(f"  - {wheel}")
    return wheels


_GPU_NAME_TO_TARGET = {
    "a100": "A100",
    "a10g": "A10G",
    "l4": "L4",
}


def detect_target_gpu() -> str:
    """Detect the target GPU from nvidia-smi output."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise SystemExit(
            "No --wheel-target-gpu specified and nvidia-smi not available. "
            "Please pass --wheel-target-gpu explicitly."
        )
    gpu_name = result.stdout.strip().splitlines()[0].lower()
    for key, target in _GPU_NAME_TO_TARGET.items():
        if key in gpu_name:
            print(f"Auto-detected target GPU: {target} (from '{result.stdout.strip()}')")
            return target
    raise SystemExit(
        f"No --wheel-target-gpu specified and could not map GPU '{result.stdout.strip()}' "
        f"to a known target. Please pass --wheel-target-gpu explicitly."
    )


WHEEL_BUILD_ORDER = ["libcugraph", "pylibcugraph", "cugraph"]


def resolve_wheel_chain(target_wheel: str) -> list[str]:
    """Return the list of wheels to build, in order, up to and including target_wheel."""
    if target_wheel not in WHEEL_BUILD_ORDER:
        raise SystemExit(
            f"Unknown wheel '{target_wheel}'. Choose from: {', '.join(WHEEL_BUILD_ORDER)}"
        )
    idx = WHEEL_BUILD_ORDER.index(target_wheel)
    return WHEEL_BUILD_ORDER[: idx + 1]


def write_constraints(path: Path, local_wheels: list[Path]) -> None:
    """Write a PIP_CONSTRAINT file mapping package names to local wheel file paths."""
    lines = []
    for whl in local_wheels:
        # Wheel filename: {name}-{version}-{tags}.whl — extract package name
        name = whl.name.split("-")[0].replace("_", "-")
        lines.append(f"{name} @ file://{whl}")
    path.write_text("\n".join(lines) + "\n" if lines else "")


def load_target_gpu_map(repo_root: Path) -> dict:
    path = repo_root / "build_in_docker" / "target_gpu_map.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wheel_arch_from_target_gpu(target_gpu: str, target_map: dict) -> str:
    key = target_gpu.strip().lower()
    if key not in target_map:
        raise SystemExit(f"Unknown wheel target GPU '{target_gpu}' in target_gpu_map.json")
    return str(target_map[key]["arch"])
