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
// Include original implementation for other template instantiations
#include "components/weakly_connected_components_impl.cuh"

#ifdef AAI_ROUTE_WCC

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// Specialization for int32_t vertex/edge types - routes to AAI implementation
// =============================================================================

template <>
void weakly_connected_components<int32_t, int32_t, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    int32_t* components,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from weakly_connected_components_impl.cuh:286-288)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Weakly connected components requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(
    graph_view.is_symmetric(),
    "Invalid input argument: input graph should be symmetric for weakly connected components.");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  // 4-way dispatch: mask x segment
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      aai::weakly_connected_components_seg_mask(compact_graph, components);
    } else {
      aai::weakly_connected_components_mask(compact_graph, components);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      aai::weakly_connected_components_seg(compact_graph, components);
    } else {
      aai::weakly_connected_components(compact_graph, components);
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
template void weakly_connected_components<int32_t, int32_t, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, int32_t*, bool);
}  // namespace cugraph

#endif
