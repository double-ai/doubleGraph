# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Modifications Copyright (c) 2025, AA-I Technologies Ltd.
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
#
# NOTICE: This file has been modified by AA-I Technologies Ltd. from the original.
import cudf

from pylibcugraph import minimum_spanning_tree as pylibcugraph_minimum_spanning_tree
from pylibcugraph import ResourceHandle
from cugraph.structure.graph_classes import Graph


def _minimum_spanning_tree_subgraph(G):
    mst_subgraph = Graph()
    if G.is_directed():
        raise ValueError("input graph must be undirected")

    sources, destinations, edge_weights, _ = pylibcugraph_minimum_spanning_tree(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        do_expensive_check=True,
    )

    mst_df = cudf.DataFrame()
    mst_df["src"] = sources
    mst_df["dst"] = destinations
    if edge_weights is not None:
        mst_df["weight"] = edge_weights

    if G.renumbered:
        mst_df = G.unrenumber(mst_df, "src")
        mst_df = G.unrenumber(mst_df, "dst")

    mst_subgraph.from_cudf_edgelist(
        mst_df, source="src", destination="dst", edge_attr="weight"
    )
    return mst_subgraph


def _maximum_spanning_tree_subgraph(G):
    if G.is_directed():
        raise ValueError("input graph must be undirected")

    # Build a temporary graph with negated weights so that MST == max spanning tree.
    edgelist_df = G.view_edge_list()
    neg_df = edgelist_df.copy()

    # Find the weight column — it may be named "weights", "weight", "wgt", etc.
    weight_col = None
    for col in neg_df.columns:
        if col not in ("src", "dst"):
            weight_col = col
            break

    if weight_col is not None:
        neg_df[weight_col] = neg_df[weight_col].mul(-1)

    G_neg = Graph()
    G_neg.from_cudf_edgelist(
        neg_df, source="src", destination="dst", edge_attr=weight_col
    )

    sources, destinations, edge_weights, _ = pylibcugraph_minimum_spanning_tree(
        resource_handle=ResourceHandle(),
        graph=G_neg._plc_graph,
        do_expensive_check=True,
    )

    mst_df = cudf.DataFrame()
    mst_df["src"] = sources
    mst_df["dst"] = destinations
    if edge_weights is not None:
        mst_df["weight"] = cudf.Series(edge_weights).mul(-1)

    if G_neg.renumbered:
        mst_df = G_neg.unrenumber(mst_df, "src")
        mst_df = G_neg.unrenumber(mst_df, "dst")

    mst_subgraph = Graph()
    mst_subgraph.from_cudf_edgelist(
        mst_df, source="src", destination="dst", edge_attr="weight"
    )
    return mst_subgraph


def minimum_spanning_tree(
    G: Graph, weight=None, algorithm="boruvka", ignore_nan=False
) -> Graph:
    """
    Returns a minimum spanning tree (MST) or forest (MSF) on an undirected
    graph

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information.

    weight : string
        default to the weights in the graph, if the graph edges do not have a
        weight attribute a default weight of 1 will be used.

    algorithm : string
        Default to 'boruvka'. The parallel algorithm to use when finding a
        minimum spanning tree.

    ignore_nan : bool
        Default to False

    Returns
    -------
    G_mst : cuGraph.Graph
        A graph descriptor with a minimum spanning tree or forest.

    Examples
    --------
    >>> from cugraph.datasets import netscience
    >>> G = netscience.get_graph(download=True)
    >>> G_mst = cugraph.minimum_spanning_tree(G)

    """

    return _minimum_spanning_tree_subgraph(G)


def maximum_spanning_tree(G, weight=None, algorithm="boruvka", ignore_nan=False):
    """
    Returns a maximum spanning tree (MST) or forest (MSF) on an undirected
    graph. Also computes the adjacency list if G does not have one.

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information.

    weight : string
        default to the weights in the graph, if the graph edges do not have a
        weight attribute a default weight of 1 will be used.

    algorithm : string
        Default to 'boruvka'. The parallel algorithm to use when finding a
        maximum spanning tree.

    ignore_nan : bool
        Default to False

    Returns
    -------
    G_mst : cuGraph.Graph
        A graph descriptor with a maximum spanning tree or forest.

    Examples
    --------
    >>> from cugraph.datasets import netscience
    >>> G = netscience.get_graph(download=True)
    >>> G_mst = cugraph.maximum_spanning_tree(G)

    """

    return _maximum_spanning_tree_subgraph(G)
