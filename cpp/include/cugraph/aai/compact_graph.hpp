/*
 * Copyright (c) 2025, AA-I Technologies Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cugraph/graph_view.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace aai {

/**
 * Compact CSR/CSC graph representation.
 *
 * A minimal graph struct for AAI algorithm implementations.
 * All pointers are device pointers unless otherwise noted.
 *
 * @tparam vertex_t Vertex ID type (int32_t or int64_t)
 * @tparam edge_t   Edge index type (int32_t or int64_t)
 */
template <typename vertex_t, typename edge_t>
struct compact_graph_t {
  // ===== Core CSR/CSC structure =====

  const edge_t* offsets;    // Size: number_of_vertices + 1.
                            // offsets[i] = start index in indices for vertex i

  const vertex_t* indices;  // Size: number_of_edges.
                            // Neighbor vertex IDs (dst if CSR, src if CSC)

  // ===== Counts =====

  vertex_t number_of_vertices;  // Total vertices in graph.
                                // Isolated vertices valid: offsets[i] == offsets[i+1]

  edge_t number_of_edges;       // = offsets[number_of_vertices] - offsets[0]

  // ===== Graph properties =====

  bool is_symmetric;       // True if edge (u,v) implies edge (v,u)
  bool is_multigraph;      // True if duplicate edges may exist
  bool is_csc;             // false = CSR (row = src), true = CSC (row = dst)

  // ===== Degree-based vertex segmentation (HOST memory) =====
  //
  // When segment_offsets has a value, vertices are sorted by degree in descending order.
  // The degree of vertex i is: offsets[i+1] - offsets[i] (out-degree for CSR, in-degree for CSC).
  //
  // segment_offsets partitions vertices into degree-based segments for kernel optimization.
  // For single-GPU graphs, the vector has 5 elements defining 4 segments:
  //   - segment_offsets[0] to segment_offsets[1]: high-degree vertices (degree >= 1024)
  //   - segment_offsets[1] to segment_offsets[2]: mid-degree vertices (32 <= degree < 1024)
  //   - segment_offsets[2] to segment_offsets[3]: low-degree vertices (1 <= degree < 32)
  //   - segment_offsets[3] to segment_offsets[4]: zero-degree vertices (isolated)
  //
  // Range: all values are in [0, number_of_vertices], with segment_offsets[0] == 0
  //        and segment_offsets[4] == number_of_vertices.
  //
  // std::nullopt if the graph was not renumbered (vertices not sorted by degree).

  std::optional<std::vector<vertex_t>> segment_offsets;

  // ===== Edge mask (DEVICE memory) =====
  //
  // Packed bitmask over edges. Bit j of word i (i.e. edge_mask[j/32] >> (j%32) & 1)
  // indicates whether edge j is active (1) or masked out (0).
  //
  // nullptr if no edge mask is applied (all edges are active).

  const uint32_t* edge_mask = nullptr;

  // ===== Factory method =====

  /**
   * Create a compact_graph_t from a cuGraph graph_view.
   *
   * Requirements:
   * - Single-GPU graph only (multi_gpu = false)
   * - Exactly one edge partition
   *
   * Template parameter note:
   * The `is_csc_` parameter is required because cuGraph's graph_view_t encodes the storage
   * format (CSR vs CSC) as a compile-time template parameter called `store_transposed`.
   * When store_transposed=false, the graph is in CSR format (rows are sources).
   * When store_transposed=true, the graph is in CSC format (rows are destinations).
   *
   * This template parameter is automatically deduced from the graph_view_t type passed in,
   * so callers don't need to specify it explicitly. The compile-time value is then stored
   * in the runtime `is_csc` field for use by AAI algorithms.
   */
  template <bool is_csc_>
  static compact_graph_t from_graph_view(
      cugraph::graph_view_t<vertex_t, edge_t, is_csc_, false> const& gv)
  {
    if (gv.number_of_local_edge_partitions() != 1) {
      throw std::runtime_error("AAI requires single-GPU graphs with exactly one edge partition");
    }
    auto ep = gv.local_edge_partition_view(0);

    // Validate that graph structure pointers are valid
    if (ep.offsets().data() == nullptr) {
      throw std::runtime_error("AAI requires non-null graph offsets");
    }
    if (ep.number_of_edges() > 0 && ep.indices().data() == nullptr) {
      throw std::runtime_error("AAI requires non-null graph indices for non-empty graphs");
    }

    auto seg_offsets = gv.local_edge_partition_segment_offsets(0);
    if (seg_offsets.has_value() && seg_offsets->size() != 5) {
      throw std::runtime_error(
          "AAI requires exactly 5 segment offsets (4 segments). "
          "Hypersparse graphs (DCS) with " + std::to_string(seg_offsets->size()) +
          " segment offsets are not supported.");
    }

    return compact_graph_t{
        .offsets            = ep.offsets().data(),
        .indices            = ep.indices().data(),
        .number_of_vertices = gv.number_of_vertices(),
        .number_of_edges    = static_cast<edge_t>(ep.number_of_edges()),
        .is_symmetric       = gv.is_symmetric(),
        .is_multigraph      = gv.is_multigraph(),
        .is_csc             = is_csc_,
        .segment_offsets    = std::move(seg_offsets),
        .edge_mask          = gv.has_edge_mask()
                                  ? (*(gv.edge_mask_view())).value_firsts()[0]
                                  : nullptr,
    };
  }
};

// Common type aliases
using graph32_t = compact_graph_t<int32_t, int32_t>;

}  // namespace aai
