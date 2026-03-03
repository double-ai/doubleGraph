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
#include "cores/core_number_impl.cuh"

#ifdef AAI_ROUTE_CORE_NUMBER

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

#include <limits>

namespace cugraph {

// =============================================================================
// Template specialization that routes to AAI implementation
// =============================================================================

template <>
void core_number<int32_t, int32_t, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    int32_t* core_numbers,
    k_core_degree_type_t degree_type,
    size_t k_first,
    size_t k_last,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from core_number_impl.cuh:70-78)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Core number requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input argument: core_number currently supports only undirected graphs.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input argument: core_number currently does not support multi-graphs.");
  CUGRAPH_EXPECTS((degree_type == k_core_degree_type_t::IN) ||
                    (degree_type == k_core_degree_type_t::OUT) ||
                    (degree_type == k_core_degree_type_t::INOUT),
                  "Invalid input argument: degree_type should be IN, OUT, or INOUT.");
  CUGRAPH_EXPECTS(k_first <= k_last, "Invalid input argument: k_first <= k_last.");

  // Convert degree type enum to int (0=IN, 1=OUT, 2=INOUT)
  int degree_type_int = static_cast<int>(degree_type);

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  // Call AAI function (4-way dispatch: mask x segment)
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      aai::core_number_seg_mask(compact_graph, core_numbers, degree_type_int, k_first, k_last);
    } else {
      aai::core_number_mask(compact_graph, core_numbers, degree_type_int, k_first, k_last);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      aai::core_number_seg(compact_graph, core_numbers, degree_type_int, k_first, k_last);
    } else {
      aai::core_number(compact_graph, core_numbers, degree_type_int, k_first, k_last);
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

#else

namespace cugraph {
template void core_number<int32_t, int32_t, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, int32_t*, k_core_degree_type_t, size_t, size_t, bool);
}  // namespace cugraph

#endif
