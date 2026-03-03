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
#ifdef AAI_ROUTE_BFS

#include <cugraph/aai/algorithms.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// SG instantiation - template specialization that routes to AAI implementation
template <>
void bfs<int32_t, int32_t, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    int32_t* distances,
    int32_t* predecessors,
    int32_t const* sources,
    size_t n_sources,
    bool direction_optimizing,
    int32_t depth_limit,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Common preconditions (match original bfs_impl.cuh)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "BFS requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS((n_sources == 0) || (sources != nullptr),
                  "Invalid input argument: sources cannot be null if n_sources > 0.");
  CUGRAPH_EXPECTS(n_sources > 0,
                  "Invalid input argument: input should have at least one source.");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  // =========================================================================
  // CALL AAI IMPLEMENTATION - route based on direction_optimizing
  // =========================================================================

  if (direction_optimizing) {
    // Direction-optimizing BFS requires symmetric graph (match original bfs_impl.cuh)
    CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                    "Invalid input argument: input graph should be symmetric for direction "
                    "optimizing BFS.");

    // 4-way dispatch: mask x segment
    if (compact_graph.edge_mask != nullptr) {
      if (compact_graph.segment_offsets.has_value()) {
        aai::bfs_direction_optimizing_seg_mask(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      } else {
        aai::bfs_direction_optimizing_mask(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      }
    } else {
      if (compact_graph.segment_offsets.has_value()) {
        aai::bfs_direction_optimizing_seg(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      } else {
        aai::bfs_direction_optimizing(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      }
    }
  } else {
    // Standard BFS - works on directed or undirected graphs
    // 4-way dispatch: mask x segment
    if (compact_graph.edge_mask != nullptr) {
      if (compact_graph.segment_offsets.has_value()) {
        aai::bfs_seg_mask(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      } else {
        aai::bfs_mask(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      }
    } else {
      if (compact_graph.segment_offsets.has_value()) {
        aai::bfs_seg(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      } else {
        aai::bfs(
            compact_graph, distances, predecessors, sources, n_sources, depth_limit);
      }
    }
  }

  // AAI may use different streams than handle.get_stream(); sync all device work before returning
  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

}  // namespace cugraph

#else

#include "traversal/bfs_impl.cuh"

namespace cugraph {
template void bfs<int32_t, int32_t, false>(
    raft::handle_t const&,
    graph_view_t<int32_t, int32_t, false, false> const&,
    int32_t*,
    int32_t*,
    int32_t const*,
    size_t,
    bool,
    int32_t,
    bool);
}  // namespace cugraph

#endif
