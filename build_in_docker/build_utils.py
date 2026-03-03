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

import json
import os
import subprocess


def git_short_hash(repo_path=".", *, fallback: str = "") -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "--short=10", "HEAD"],
            text=True,
        )
        return out.strip()
    except Exception:
        if fallback:
            return fallback
        raise


def load_target_gpu_map(repo_root: str) -> dict:
    path = os.path.join(repo_root, "build_in_docker", "target_gpu_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_known_target_arch(target: str, target_map: dict) -> str:
    if target not in target_map:
        raise SystemExit(f"Unknown target GPU '{target}' in target_gpu_map.json")
    return str(target_map[target]["arch"])


def wheel_arches_from_targets(target_gpus: str, target_map: dict) -> str:
    targets = [t.strip().lower() for t in target_gpus.split(",") if t.strip()]
    if not targets:
        raise SystemExit(
            "--target-gpus must contain at least one target when --build-wheel is set"
        )
    arches = []
    for target in targets:
        arch = require_known_target_arch(target, target_map)
        if arch not in arches:
            arches.append(arch)
    return ";".join(arches)


def wheel_target_gpu_from_targets(target_gpus: str, target_map: dict) -> str:
    targets = [t.strip().lower() for t in target_gpus.split(",") if t.strip()]
    if not targets:
        raise SystemExit(
            "--target-gpus must contain at least one target when --build-wheel is set"
        )
    if len(targets) != 1:
        raise SystemExit(
            "--build-wheel requires exactly one target GPU in --target-gpus "
            "(e.g. --target-gpus a10g)"
        )
    target = targets[0]
    require_known_target_arch(target, target_map)
    return target.upper()
