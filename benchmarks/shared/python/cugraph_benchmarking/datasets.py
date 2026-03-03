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
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Shared dataset utilities for cuGraph benchmarks.

Provides dataset name resolution, RMAT graph generation, and weight
processing helpers used by both ``bench_algos.py`` and
``precompute_partitions.py``.
"""

import numpy as np
import cupy as cp

import dask_cudf

import cugraph
from cugraph.generators import rmat
from cugraph.datasets import (
    karate,
    netscience,
    email_Eu_core,
    hollywood,
    europe_osm,
    cit_patents,
    soc_livejournal,
    amazon0302,
    com_orkut,
)

###############################################################################
# Duck-type compatible Dataset for RMAT data


class RmatDataset:
    def __init__(self, scale=4, edgefactor=2, mg=False):
        self._scale = scale
        self._edgefactor = edgefactor
        self._edgelist = None

        self.mg = mg

    def __str__(self):
        mg_str = "mg" if self.mg else "sg"
        return f"rmat_{mg_str}_{self._scale}_{self._edgefactor}"

    def get_edgelist(self, fetch=False):
        seed = 42
        if self._edgelist is None:
            self._edgelist = rmat(
                self._scale,
                (2**self._scale) * self._edgefactor,
                0.57,  # from Graph500
                0.19,  # from Graph500
                0.19,  # from Graph500
                seed or 42,
                clip_and_flip=False,
                scramble_vertex_ids=True,
                create_using=None,  # return edgelist instead of Graph instance
                mg=self.mg,
            )
            rng = cp.random.default_rng(seed=42)
            if self.mg:
                self._edgelist["weights"] = self._edgelist.map_partitions(
                    lambda df: rng.random(size=len(df))
                )
            else:
                self._edgelist["weights"] = rng.random(size=len(self._edgelist))

        return self._edgelist

    def get_graph(
        self,
        fetch=False,
        download=False,
        create_using=cugraph.Graph,
        ignore_weights=False,
        store_transposed=False,
        renumber=True,
        process_edgelist_fn=None,
    ):
        if isinstance(create_using, cugraph.Graph):
            attrs = {"directed": create_using.is_directed()}
            G = type(create_using)(**attrs)
        elif type(create_using) is type:
            G = create_using()

        df = self.get_edgelist()
        src_col, dst_col, weight_col = "src", "dst", "weights"

        if process_edgelist_fn is not None:
            df = process_edgelist_fn(df.copy(), src_col, dst_col, weight_col)

        edge_attr = None
        if not ignore_weights and weight_col in df.columns:
            edge_attr = weight_col

        if isinstance(df, dask_cudf.DataFrame):
            G.from_dask_cudf_edgelist(
                df,
                source=src_col,
                destination=dst_col,
                edge_attr=edge_attr,
                store_transposed=store_transposed,
                renumber=renumber,
            )
        else:
            G.from_cudf_edgelist(
                df,
                source=src_col,
                destination=dst_col,
                edge_attr=edge_attr,
                store_transposed=store_transposed,
                renumber=renumber,
            )
        return G

    def get_path(self):
        return str(self)

    def unload(self):
        self._edgelist = None


###############################################################################
# Dataset name -> object mapping

FILE_DATASET_MAP = {
    "karate": karate,
    "hollywood": hollywood,
    "europe_osm": europe_osm,
    "netscience": netscience,
    "email_Eu_core": email_Eu_core,
    "amazon0302": amazon0302,
    "cit_patents": cit_patents,
    "soc_livejournal": soc_livejournal,
    "com_orkut": com_orkut,
}


def get_dataset(name):
    """Resolve a dataset name string to a Dataset object."""
    if name in FILE_DATASET_MAP:
        return FILE_DATASET_MAP[name]
    if name.startswith("rmat_"):
        parts = name.split("_")
        scale = int(parts[1])
        edgefactor = int(parts[2])
        return RmatDataset(scale=scale, edgefactor=edgefactor)
    raise ValueError(f"Unknown dataset: {name}")


###############################################################################
# Weight processing

WEIGHTS_MAP = {"none": None, "float32": np.float32, "float64": np.float64}


def make_weight_processor(weights_dtype):
    """Return a process_edgelist_fn that ensures weights exist with the given dtype.

    If the edgelist already has a weight column, cast it. Otherwise add
    deterministic random weights (seeded for reproducibility).
    Returns None if weights_dtype is None.
    """
    if weights_dtype is None:
        return None

    def process(df, src_col, dst_col, weight_col):
        if weight_col is None:
            weight_col = "weights"

        if weight_col in df.columns:
            if np.dtype(df[weight_col].dtype) != np.dtype(weights_dtype):
                df[weight_col] = df[weight_col].astype(weights_dtype)
        else:
            rng = cp.random.default_rng(seed=42)
            df[weight_col] = rng.random(size=len(df), dtype=weights_dtype)
        return df

    return process
