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
// Include original implementation for non-specialized cases
#include "community/triangle_count_impl.cuh"

#ifdef AAI_ROUTE_TRIANGLE_COUNT

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// Template specialization that routes to AAI implementation
// =============================================================================

template <>
void triangle_count<int32_t, int32_t, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    raft::device_span<int32_t> counts,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from triangle_count_impl.cuh, lines 139-152)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Triangle count requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(
    graph_view.is_symmetric(),
    "Invalid input arguments: triangle_count currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(
    !graph_view.is_multigraph(),
    "Invalid input arguments: triangle_count currently does not support multi-graphs.");
  if (vertices) {
    CUGRAPH_EXPECTS(counts.size() == (*vertices).size(),
                    "Invalid arguments: counts.size() does not coincide with (*vertices).size().");
  } else {
    CUGRAPH_EXPECTS(
      counts.size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
      "Invalid arguments: counts.size() does not coincide with the number of local vertices.");
  }

  // Extract raw pointers
  const int32_t* vertices_ptr = vertices ? vertices->data() : nullptr;
  std::size_t n_vertices = vertices ? vertices->size() : 0;

  // Sync stream before AAI call (AAI uses default stream)
  handle.sync_stream();

  // Call AAI function (4-way dispatch: mask x segment)
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      aai::triangle_count_seg_mask(compact_graph, counts.data(), vertices_ptr, n_vertices);
    } else {
      aai::triangle_count_mask(compact_graph, counts.data(), vertices_ptr, n_vertices);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      aai::triangle_count_seg(compact_graph, counts.data(), vertices_ptr, n_vertices);
    } else {
      aai::triangle_count(compact_graph, counts.data(), vertices_ptr, n_vertices);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

}  // namespace cugraph

#else  // !AAI_ROUTE_TRIANGLE_COUNT

namespace cugraph {
template void triangle_count<int32_t, int32_t, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<raft::device_span<int32_t const>>, raft::device_span<int32_t>, bool);
}  // namespace cugraph

#endif  // AAI_ROUTE_TRIANGLE_COUNT
