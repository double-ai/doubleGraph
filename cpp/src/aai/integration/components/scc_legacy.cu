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
#include "components/legacy/scc_matrix.cuh"

#ifdef AAI_ROUTE_SCC
#include <cugraph/aai/algorithms.hpp>
#endif
#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/sequence.h>

#include <cstdint>
#include <iostream>
#include <type_traits>

namespace cugraph {
namespace detail {

/**
 * @brief Compute connected components.
 * The weak version has been eliminated in lieu of the primitive based implementation
 *
 * The strong version (for directed or undirected graphs) is based on:
 * [2] Gilbert, J. et al, 2011. "Graph Algorithms in the Language of Linear Algebra"
 *
 * C = I | A | A^2 |...| A^k
 * where matrix multiplication is via semi-ring:
 * (combine, reduce) == (&, |) (bitwise ops)
 * Then: X = C & transpose(C); and finally, apply get_labels(X);
 *
 *
 * @tparam IndexT the numeric type of non-floating point elements
 * @tparam TPB_X the threads to use per block when configuring the kernel
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param connectivity_type Ignored [in]
 * @param stream the cuda stream [in]
 */
template <typename VT, typename ET, typename WT, int TPB_X = 32>
std::enable_if_t<std::is_signed<VT>::value> connected_components_impl(
  legacy::GraphCSRView<VT, ET, WT> const& graph,
  cugraph_cc_t connectivity_type,
  VT* labels,
  cudaStream_t stream)
{
  using ByteT = unsigned char;  // minimum addressable unit

  CUGRAPH_EXPECTS(graph.offsets != nullptr, "Invalid input argument: graph.offsets is nullptr");
  CUGRAPH_EXPECTS(graph.indices != nullptr, "Invalid input argument: graph.indices is nullptr");

  VT nrows = graph.number_of_vertices;

  SCC_Data<ByteT, VT> sccd(nrows, graph.offsets, graph.indices);
  auto num_iters = sccd.run_scc(labels);
}
}  // namespace detail

template <typename VT, typename ET, typename WT>
void connected_components(legacy::GraphCSRView<VT, ET, WT> const& graph,
                          cugraph_cc_t connectivity_type,
                          VT* labels)
{
  CUGRAPH_EXPECTS(labels != nullptr, "Invalid input argument: labels parameter is NULL");

#ifdef AAI_ROUTE_SCC
  // Route to AAI implementation for int32 types
  // Note: AAI only supports strongly connected components. The legacy API parameter
  // connectivity_type is checked here to ensure we don't silently produce SCC results
  // when WCC was requested. Use weakly_connected_components() for WCC.
  if constexpr (std::is_same_v<VT, int32_t>) {
    CUGRAPH_EXPECTS(connectivity_type == cugraph_cc_t::CUGRAPH_STRONG,
                    "AAI only supports strongly connected components (CUGRAPH_STRONG). "
                    "Use weakly_connected_components() for weakly connected components.");
    CUGRAPH_EXPECTS(graph.offsets != nullptr, "Invalid input argument: graph.offsets is nullptr");
    CUGRAPH_EXPECTS(graph.indices != nullptr, "Invalid input argument: graph.indices is nullptr");

    // No _seg variant dispatch: The legacy GraphCSRView API does not carry segment_offsets
    // information. Graphs created via this legacy path always have segment_offsets = nullopt,
    // so we unconditionally call the non-seg AAI variant.
    aai::graph32_t compact_graph{
        .offsets            = graph.offsets,
        .indices            = graph.indices,
        .number_of_vertices = graph.number_of_vertices,
        .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
        .is_symmetric       = false,
        .is_multigraph      = false,
        .is_csc             = false,  // SCC requires CSR format
        .segment_offsets    = std::nullopt,
    };

    // Sync before AAI call to ensure any pending work on default stream is complete
    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }

    aai::strongly_connected_components(compact_graph, labels);

    // Sync after AAI call to ensure results are ready
    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }
  } else {
    cudaStream_t stream{nullptr};
    return detail::connected_components_impl<VT, ET, WT>(graph, connectivity_type, labels, stream);
  }
#else
  cudaStream_t stream{nullptr};
  return detail::connected_components_impl<VT, ET, WT>(graph, connectivity_type, labels, stream);
#endif
}

template void connected_components<int32_t, int32_t, float>(
  legacy::GraphCSRView<int32_t, int32_t, float> const&, cugraph_cc_t, int32_t*);
template void connected_components<int64_t, int64_t, float>(
  legacy::GraphCSRView<int64_t, int64_t, float> const&, cugraph_cc_t, int64_t*);

}  // namespace cugraph
