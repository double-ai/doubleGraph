# wheel_building

Self-contained wheel build flow that assumes only Docker is installed.

## Wheel dependency chain

The repo produces three wheels with a linear dependency chain:

```
libcugraph  →  pylibcugraph  →  cugraph
```

- **libcugraph** — the compiled C++/CUDA shared library (`libcugraph.so`) packaged as a wheel.
- **pylibcugraph** — Cython bindings to the C API; depends on libcugraph at build time.
- **cugraph** — high-level Python graph analytics API; depends on both libcugraph and pylibcugraph at build time.

The `--wheel` flag controls which wheel to build. All prerequisites are built
automatically. For example, `--wheel pylibcugraph` builds libcugraph first,
then pylibcugraph. The default is `--wheel cugraph` (all three).

## Build wheels locally

```bash
uv run cugraph-build-wheel -j 4
```

Target GPU is auto-detected via `nvidia-smi`. To override or customize:

```bash
uv run cugraph-build-wheel \
  --wheel-target-gpu A10G \
  --wheel-aai-algorithms NONE \
  -j 4
```

Build only libcugraph:

```bash
uv run cugraph-build-wheel --wheel libcugraph -j 4
```

Artifacts are written to `--output-dir/arch-<derived-arch>/` as `.whl` files.
By default, wheels are repaired and tagged with `manylinux_2_28_x86_64` via
`auditwheel` (disable with `--no-auditwheel`).
The CUDA arch is derived from `--wheel-target-gpu` via `build_in_docker/target_gpu_map.json`.

## Install built wheels

Install in dependency order with `--force-reinstall --no-deps` (deps are
managed separately by the target environment):

```bash
pip install --force-reinstall --no-deps dist/arch-*/libcugraph_*.whl
pip install --force-reinstall --no-deps dist/arch-*/pylibcugraph_*.whl
pip install --force-reinstall --no-deps dist/arch-*/cugraph_*.whl
```
