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

import argparse
import os
import subprocess
import sys
from typing import List

from build_utils import git_short_hash, load_target_gpu_map


def buildx_cmd_base(
    *, dockerfile: str, platform: str, provenance: str, sbom: str
) -> List[str]:
    return [
        "docker",
        "buildx",
        "build",
        "-f",
        dockerfile,
        "--platform",
        platform,
        "--provenance",
        provenance,
        "--sbom",
        sbom,
        "--load",
    ]


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    parser = argparse.ArgumentParser(
        description="Build cugraph from source inside Docker and produce a local image.",
    )
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument(
        "--target-gpu", default="a10g", help="GPU target in target_gpu_map.json"
    )
    parser.add_argument("--env-image-tag", default="cugraph-public-env:local")
    parser.add_argument(
        "--image-tag",
        default="",
        help="Final dev image tag (default: cugraph-public-dev:<git-short-hash>-<target-gpu>)",
    )
    parser.add_argument("--provenance", default="false")
    parser.add_argument("--sbom", default="false")
    parser.add_argument("--cuda-version", default="")
    parser.add_argument("--ubuntu-version", default="")
    parser.add_argument("--parallel-level", default="")
    parser.add_argument("--conda-env-file", default="")
    parser.add_argument("--enable-ccache", default="1")
    parser.add_argument(
        "--extra-cmake-args",
        default="",
        help="Extra CMake args passed through EXTRA_CMAKE_ARGS",
    )
    parser.add_argument(
        "--no-run-build",
        dest="run_build",
        action="store_false",
        help="Skip cugraph build inside the dev image",
    )
    parser.set_defaults(run_build=True)
    parser.add_argument(
        "--skip-env-build",
        action="store_true",
        help="Do not build the env image; assumes --env-image-tag already exists locally",
    )

    args = parser.parse_args()

    if "," in args.platform:
        print(
            "This public tool uses --load and supports a single platform per invocation.",
            file=sys.stderr,
        )
        print(f"Received --platform={args.platform}", file=sys.stderr)
        return 2

    target_map = load_target_gpu_map(repo_root)
    target = args.target_gpu.strip().lower()
    if target not in target_map:
        print(
            f"Unknown target gpu '{target}'. Valid values: {', '.join(sorted(target_map.keys()))}",
            file=sys.stderr,
        )
        return 2

    arch = str(target_map[target]["arch"])
    cmake_target = target.upper()
    extra_cmake_args = args.extra_cmake_args
    if "-DTARGET_GPU=" not in extra_cmake_args:
        extra_cmake_args = (
            extra_cmake_args + " " if extra_cmake_args else ""
        ) + f"-DTARGET_GPU={cmake_target}"

    image_tag = (
        args.image_tag
        or f"cugraph-public-dev:{git_short_hash(repo_root, fallback='unknown')}-{target}"
    )

    common_build_args: List[str] = []
    if args.cuda_version:
        common_build_args += ["--build-arg", f"CUDA_VERSION={args.cuda_version}"]
    if args.ubuntu_version:
        common_build_args += ["--build-arg", f"UBUNTU_VERSION={args.ubuntu_version}"]
    if args.parallel_level:
        common_build_args += ["--build-arg", f"PARALLEL_LEVEL={args.parallel_level}"]
    if args.conda_env_file:
        common_build_args += ["--build-arg", f"CONDA_ENV_FILE={args.conda_env_file}"]
    if args.enable_ccache:
        common_build_args += ["--build-arg", f"ENABLE_CCACHE={args.enable_ccache}"]

    if not args.skip_env_build:
        env_cmd = buildx_cmd_base(
            dockerfile=os.path.join(repo_root, "build_in_docker", "Dockerfile.env"),
            platform=args.platform,
            provenance=args.provenance,
            sbom=args.sbom,
        )
        env_cmd += common_build_args
        env_cmd += ["-t", args.env_image_tag, repo_root]
        subprocess.run(env_cmd, check=True)

    dev_cmd = buildx_cmd_base(
        dockerfile=os.path.join(repo_root, "build_in_docker", "Dockerfile"),
        platform=args.platform,
        provenance=args.provenance,
        sbom=args.sbom,
    )
    dev_cmd += common_build_args
    dev_cmd += [
        "--build-arg",
        f"ENV_IMAGE={args.env_image_tag}",
        "--build-arg",
        f"GPU_ARCH={arch}",
        "--build-arg",
        f"RUN_BUILD={'1' if args.run_build else '0'}",
        "--build-arg",
        f"EXTRA_CMAKE_ARGS={extra_cmake_args}",
        "-t",
        image_tag,
        repo_root,
    ]
    subprocess.run(dev_cmd, check=True)

    print("Build completed.")
    print(f"Env image: {args.env_image_tag}")
    print(f"Dev image: {image_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
