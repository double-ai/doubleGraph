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
"""Build cugraph wheels inside a dedicated wheel_building container."""

import argparse
import os
import shutil
from pathlib import Path

from wheel_building.common import (
    WHEEL_BUILD_ORDER,
    collect_wheel_paths,
    load_target_gpu_map,
    resolve_wheel_chain,
    run_checked,
    site_package_roots,
    wheel_arch_from_target_gpu,
    write_constraints,
)


def clear_dir_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cugraph wheels inside container.")
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument("--raw-wheel-dir", default="/tmp/wheel-raw")
    parser.add_argument("--wheel-target-gpu", required=True)
    parser.add_argument("--wheel-aai-algorithms", default="")
    parser.add_argument(
        "--manylinux-platform",
        default="manylinux_2_28_x86_64",
        help="PEP 600 platform tag to enforce with auditwheel repair",
    )
    parser.add_argument(
        "--no-auditwheel",
        action="store_false",
        dest="use_auditwheel",
        default=True,
        help="Skip auditwheel repair and copy raw wheel(s) as-is",
    )
    parser.add_argument(
        "--parallel",
        "-j",
        default="",
        help="Max parallel compilation jobs (default: nproc)",
    )
    parser.add_argument(
        "--wheel",
        default="cugraph",
        choices=WHEEL_BUILD_ORDER,
        help="Top-level wheel to build; prerequisites are built automatically (default: cugraph)",
    )
    return parser.parse_args()


def _libcugraph_cmake_args(
    wheel_cuda_arch: str,
    target_gpu: str,
    aai_algorithms: str,
    roots: list[Path],
    *,
    toolset_gcc: Path | None = None,
    toolset_gxx: Path | None = None,
    use_ccache: bool = False,
) -> list[str]:
    """Full CMAKE_ARGS for libcugraph (cuVS, AAI routing, etc.)."""
    args = [
        "-DCMAKE_INSTALL_LIBDIR=lib64",
        "-DCUGRAPH_COMPILE_CUVS=ON",
        "-DCUGRAPH_USE_CUVS_STATIC=ON",
        f"-DCMAKE_CUDA_ARCHITECTURES={wheel_cuda_arch}",
        f"-DCMAKE_PREFIX_PATH={';'.join(str(r) for r in roots)}",
        f"-DTARGET_GPU={target_gpu}",
    ]
    if toolset_gxx:
        args.extend(
            [
                f"-DCMAKE_C_COMPILER={toolset_gcc}",
                f"-DCMAKE_CXX_COMPILER={toolset_gxx}",
                f"-DCMAKE_CUDA_HOST_COMPILER={toolset_gxx}",
            ]
        )
    if use_ccache:
        args.extend(
            [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]
        )
    if aai_algorithms:
        args.append(f"-DAAI_ROUTED_ALGORITHMS={aai_algorithms}")
    return args


def _cython_cmake_args(
    wheel_cuda_arch: str,
    roots: list[Path],
    *,
    toolset_gcc: Path | None = None,
    toolset_gxx: Path | None = None,
    use_ccache: bool = False,
) -> list[str]:
    """Simpler CMAKE_ARGS for pylibcugraph/cugraph (Cython packages)."""
    args = [
        f"-DCMAKE_CUDA_ARCHITECTURES={wheel_cuda_arch}",
        f"-DCMAKE_PREFIX_PATH={';'.join(str(r) for r in roots)}",
    ]
    if toolset_gxx:
        args.extend(
            [
                f"-DCMAKE_C_COMPILER={toolset_gcc}",
                f"-DCMAKE_CXX_COMPILER={toolset_gxx}",
                f"-DCMAKE_CUDA_HOST_COMPILER={toolset_gxx}",
            ]
        )
    if use_ccache:
        args.extend(
            [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]
        )
    return args


def main() -> int:
    args = parse_args()
    wheel_chain = resolve_wheel_chain(args.wheel)
    target_map = load_target_gpu_map(Path(".").resolve())
    wheel_cuda_arch = wheel_arch_from_target_gpu(args.wheel_target_gpu, target_map)
    toolset_prefix = Path("/opt/rh/gcc-toolset-12/root/usr/bin")
    toolset_gcc = toolset_prefix / "gcc"
    toolset_gxx = toolset_prefix / "g++"
    use_toolset = toolset_gcc.exists() and toolset_gxx.exists()
    use_ccache = shutil.which("ccache") is not None

    # Install build dependencies.
    # cython/pylibraft/rmm are needed for pylibcugraph and cugraph builds.
    pip_deps = [
        "cmake>=3.30.4",
        "ninja",
        "rapids-build-backend>=0.4.0,<0.5.0",
        "scikit-build-core[pyproject]>=0.11.0",
        "packaging>=24.2",
        "libraft-cu13==26.2.*",
        "librmm-cu13==26.2.*",
    ]
    if any(w != "libcugraph" for w in wheel_chain):
        pip_deps.extend(
            [
                "cython>=3.1.2,<3.2.0",
                "pylibraft-cu13==26.2.*",
                "rmm-cu13==26.2.*",
            ]
        )
    run_checked(
        ["python3", "-m", "pip", "install", *pip_deps, "--extra-index-url", "https://pypi.nvidia.com"],
        label="pip install build deps",
    )
    if args.use_auditwheel:
        run_checked(["python3", "-m", "pip", "install", "auditwheel"])

    # Shared env for all builds.
    base_env = os.environ.copy()
    base_env["CCACHE_DIR"] = "/ccache"
    base_env["CCACHE_BASEDIR"] = "/src"
    if args.parallel:
        base_env["CMAKE_BUILD_PARALLEL_LEVEL"] = args.parallel
    if use_toolset:
        base_env["CC"] = str(toolset_gcc)
        base_env["CXX"] = str(toolset_gxx)

    raw_wheel_dir = Path(args.raw_wheel_dir).resolve()
    constraint_file = raw_wheel_dir / "constraints.txt"
    all_raw_wheels: list[Path] = []

    for wheel_name in wheel_chain:
        print(f"\n{'=' * 60}")
        print(f"  Building {wheel_name}")
        print(f"{'=' * 60}\n")

        # Re-read site-packages roots (they may have changed after installing a wheel).
        roots = site_package_roots()

        if wheel_name == "libcugraph":
            cmake_args = _libcugraph_cmake_args(
                wheel_cuda_arch,
                args.wheel_target_gpu,
                args.wheel_aai_algorithms,
                roots,
                toolset_gcc=toolset_gcc if use_toolset else None,
                toolset_gxx=toolset_gxx if use_toolset else None,
                use_ccache=use_ccache,
            )
        else:
            # Add package dirs with lib64/cmake/ so CMake finds installed libs
            # (e.g. libcugraph/lib64/cmake/cugraph/).
            for root in list(roots):
                for cmake_dir in root.glob("*/lib64/cmake"):
                    pkg_dir = cmake_dir.parent.parent
                    if pkg_dir not in roots:
                        roots.append(pkg_dir)
            cmake_args = _cython_cmake_args(
                wheel_cuda_arch,
                roots,
                toolset_gcc=toolset_gcc if use_toolset else None,
                toolset_gxx=toolset_gxx if use_toolset else None,
                use_ccache=use_ccache,
            )

        env = base_env.copy()
        env["CMAKE_ARGS"] = " ".join(cmake_args)

        # Write constraints pointing to previously-built local wheels.
        write_constraints(constraint_file, all_raw_wheels)
        env["PIP_CONSTRAINT"] = str(constraint_file)

        pkg_dir = Path(f"python/{wheel_name}")
        pyproject = pkg_dir / "pyproject.toml"
        pyproject_orig = pyproject.read_bytes()

        # Clear stale CMake cache so flags are taken from this invocation.
        shutil.rmtree(pkg_dir / "build", ignore_errors=True)
        # rapids-build-backend with cuda_suffixed=true renames packages (e.g.
        # libcugraph_cu13) and tries to write GIT_COMMIT there.
        (pkg_dir / f"{wheel_name}_cu13").mkdir(exist_ok=True)

        step_raw_dir = raw_wheel_dir / wheel_name
        clear_dir_contents(step_raw_dir)

        try:
            run_checked(
                [
                    "python3",
                    "-m",
                    "pip",
                    "wheel",
                    f"python/{wheel_name}",
                    "-w",
                    str(step_raw_dir),
                    "--no-build-isolation",
                    "--no-deps",
                    "--extra-index-url",
                    "https://pypi.nvidia.com",
                    "-v",
                ],
                env=env,
                label=f"pip wheel {wheel_name}",
            )
        finally:
            # Restore pyproject.toml — rapids-build-backend modifies it
            # in-place (cuda suffix, dep rewriting) and we don't want
            # those changes leaking to the host-mounted source.
            pyproject.write_bytes(pyproject_orig)

        step_wheels = collect_wheel_paths(step_raw_dir, context=f"{wheel_name} raw wheel(s)")
        all_raw_wheels.extend(step_wheels)

        # Install the just-built wheel so the next package can find it at build time.
        if wheel_name != wheel_chain[-1]:
            run_checked(
                ["python3", "-m", "pip", "install", "--no-deps", *[str(w) for w in step_wheels]],
                label=f"pip install {wheel_name}",
            )

    # Auditwheel repair (or copy) all collected wheels.
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_auditwheel:
        exclude_args = [
            "--exclude=libraft.so",
            "--exclude=libcublas.so.*",
            "--exclude=libcublasLt.so.*",
            "--exclude=libcurand.so.*",
            "--exclude=libcusolver.so.*",
            "--exclude=libcusparse.so.*",
            "--exclude=libnvJitLink.so.*",
            "--exclude=librapids_logger.so",
            "--exclude=librmm.so",
            "--exclude=libcugraph.so",
            "--exclude=libcugraph_c.so",
        ]
        for wheel in all_raw_wheels:
            run_checked(
                [
                    "python3",
                    "-m",
                    "auditwheel",
                    "repair",
                    "-w",
                    str(output_dir),
                    *exclude_args,
                    str(wheel),
                ],
                label=f"auditwheel repair {wheel.name}",
            )
    else:
        for wheel in all_raw_wheels:
            shutil.copy2(wheel, output_dir / wheel.name)

    output_wheels = collect_wheel_paths(output_dir, context="output wheel(s)")
    if args.use_auditwheel:
        # auditwheel may copy the original wheel alongside the repaired one;
        # remove non-manylinux duplicates when a manylinux version exists.
        for wheel in list(output_wheels):
            if "manylinux_" not in wheel.name:
                wheel.unlink()
                output_wheels.remove(wheel)
        if not output_wheels:
            raise SystemExit("auditwheel produced no manylinux-tagged wheels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
