<!-- SPDX-FileCopyrightText: Copyright (c) 2026, AA-I Technologies Ltd. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# build_in_docker

Public, local-only tooling to build cuGraph from source inside Docker.
These Dockerfiles are the canonical source used by local and internal build flows.

## License

All files in this directory are distributed under the Apache-2.0 license.

## Quick start

Build a source-based image for the default target (`a10g`):

```bash
python3 build_in_docker/build_image.py
```

Build for a specific target GPU and custom tags:

```bash
python3 build_in_docker/build_image.py \
  --target-gpu l4 \
  --env-image-tag cugraph-env:l4 \
  --image-tag cugraph-dev:l4
```

Reuse an already-built env image and only build the dev image:

```bash
python3 build_in_docker/build_image.py \
  --skip-env-build \
  --env-image-tag cugraph-env:l4 \
  --target-gpu l4 \
  --image-tag cugraph-dev:l4
```

## Notes

- The resulting dev image contains the built cuGraph artifacts.
- Builds are local (`docker buildx build --load`) and single-platform.
- GPU target-to-architecture mapping is handled by the local build tooling.
- The dev image sets `RAPIDS_DATASET_ROOT_DIR=/cugraph/datasets` so pytest data files resolve by default.
