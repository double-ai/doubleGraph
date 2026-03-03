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
// Include original implementation for template instantiations
#include "traversal/k_hop_nbrs_impl.cuh"

#ifdef AAI_ROUTE_K_HOP_NBRS

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// Specialized k_hop_nbrs to route to AAI
// =============================================================================

template <>
std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<int32_t>> k_hop_nbrs<int32_t, int32_t, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  raft::device_span<int32_t const> start_vertices,
  size_t k,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (match original k_hop_nbrs_impl.cuh)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "k_hop_nbrs requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(k > 0, "Invalid input argument: k should be a positive integer.");
  CUGRAPH_EXPECTS(start_vertices.size() > 0,
                  "Invalid input argument: input should have at least one starting vertex.");

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION
  // =========================================================================

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  aai::k_hop_nbrs_result_t result;
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::k_hop_nbrs_seg_mask(
        compact_graph,
        start_vertices.data(),
        start_vertices.size(),
        k);
    } else {
      result = aai::k_hop_nbrs_mask(
        compact_graph,
        start_vertices.data(),
        start_vertices.size(),
        k);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::k_hop_nbrs_seg(
        compact_graph,
        start_vertices.data(),
        start_vertices.size(),
        k);
    } else {
      result = aai::k_hop_nbrs(
        compact_graph,
        start_vertices.data(),
        start_vertices.size(),
        k);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  // Wrap AAI result in RMM vectors (takes ownership of device memory)
  rmm::device_uvector<size_t> offsets(result.num_offsets, handle.get_stream());
  rmm::device_uvector<int32_t> neighbors(result.num_neighbors, handle.get_stream());

  cudaMemcpyAsync(offsets.data(), result.offsets,
                  result.num_offsets * sizeof(size_t), cudaMemcpyDeviceToDevice,
                  handle.get_stream());
  cudaMemcpyAsync(neighbors.data(), result.neighbors,
                  result.num_neighbors * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                  handle.get_stream());

  // Sync stream before freeing AAI-allocated memory to ensure copies complete
  handle.sync_stream();

  // Free AAI-allocated memory
  cudaFree(result.offsets);
  cudaFree(result.neighbors);

  return std::make_tuple(std::move(offsets), std::move(neighbors));
}

}  // namespace cugraph

#else

namespace cugraph {
template std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<int32_t>> k_hop_nbrs<int32_t, int32_t, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, raft::device_span<int32_t const>, size_t, bool);
}  // namespace cugraph

#endif
