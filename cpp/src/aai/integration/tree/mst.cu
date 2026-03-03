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
#include <cugraph/algorithms.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/error.hpp>

#ifdef AAI_ROUTE_MST
#include <cugraph/aai/algorithms.hpp>
#include <cugraph/aai/compact_graph.hpp>
#endif

#include <raft/sparse/solver/mst.cuh>

#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/transform.h>

#include <cuda_runtime.h>

#include <ctime>
#include <memory>
#include <utility>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<legacy::GraphCOO<vertex_t, edge_t, weight_t>> mst_impl(
  raft::handle_t const& handle,
  legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  rmm::device_async_resource_ref mr)

{
  auto stream = handle.get_stream();
  rmm::device_uvector<vertex_t> colors(graph.number_of_vertices, stream);
  auto mst_edges = raft::sparse::solver::mst<vertex_t, edge_t, weight_t>(handle,
                                                                         graph.offsets,
                                                                         graph.indices,
                                                                         graph.edge_data,
                                                                         graph.number_of_vertices,
                                                                         graph.number_of_edges,
                                                                         colors.data(),
                                                                         stream);

  legacy::GraphCOOContents<vertex_t, edge_t, weight_t> coo_contents{
    graph.number_of_vertices,
    mst_edges.n_edges,
    std::make_unique<rmm::device_buffer>(mst_edges.src.release()),
    std::make_unique<rmm::device_buffer>(mst_edges.dst.release()),
    std::make_unique<rmm::device_buffer>(mst_edges.weights.release())};

  return std::make_unique<legacy::GraphCOO<vertex_t, edge_t, weight_t>>(std::move(coo_contents));
}

}  // namespace detail

// =============================================================================
// Original template definition (for int64 or other types not routed to AAI)
// =============================================================================

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<legacy::GraphCOO<vertex_t, edge_t, weight_t>> minimum_spanning_tree(
  raft::handle_t const& handle,
  legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  rmm::device_async_resource_ref mr)
{
  return detail::mst_impl(handle, graph, mr);
}

#ifdef AAI_ROUTE_MST

// =============================================================================
// Template specializations to route int32 to AAI
// =============================================================================

// Float weights specialization
template <>
std::unique_ptr<legacy::GraphCOO<int, int, float>> minimum_spanning_tree<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const& graph,
  rmm::device_async_resource_ref mr)
{
  CUGRAPH_EXPECTS(graph.offsets != nullptr, "Invalid input argument: graph.offsets is nullptr");
  CUGRAPH_EXPECTS(graph.indices != nullptr, "Invalid input argument: graph.indices is nullptr");
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "MST requires edge weights (graph.edge_data is nullptr)");

  aai::graph32_t compact_graph{
    .offsets            = graph.offsets,
    .indices            = graph.indices,
    .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
    .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
    .is_symmetric       = true,
    .is_multigraph      = false,
    .is_csc             = false,
    .segment_offsets    = std::nullopt,
  };

  // AAI MST returns symmetrized output (both directions per edge),
  // so allocate 2*V to fit all edges.
  std::size_t max_mst_edges = static_cast<std::size_t>(2 * graph.number_of_vertices);

  rmm::device_uvector<int32_t> mst_srcs(max_mst_edges, handle.get_stream());
  rmm::device_uvector<int32_t> mst_dsts(max_mst_edges, handle.get_stream());
  rmm::device_uvector<float> mst_weights(max_mst_edges, handle.get_stream());

  handle.sync_stream();

  std::size_t num_mst_edges;
  if (compact_graph.segment_offsets.has_value()) {
    num_mst_edges = aai::minimum_spanning_tree_seg(
      compact_graph,
      graph.edge_data,
      mst_srcs.data(),
      mst_dsts.data(),
      mst_weights.data(),
      max_mst_edges);
  } else {
    num_mst_edges = aai::minimum_spanning_tree(
      compact_graph,
      graph.edge_data,
      mst_srcs.data(),
      mst_dsts.data(),
      mst_weights.data(),
      max_mst_edges);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  mst_srcs.resize(num_mst_edges, handle.get_stream());
  mst_dsts.resize(num_mst_edges, handle.get_stream());
  mst_weights.resize(num_mst_edges, handle.get_stream());

  legacy::GraphCOOContents<int, int, float> coo_contents{
    graph.number_of_vertices,
    num_mst_edges,
    std::make_unique<rmm::device_buffer>(mst_srcs.release()),
    std::make_unique<rmm::device_buffer>(mst_dsts.release()),
    std::make_unique<rmm::device_buffer>(mst_weights.release())};

  return std::make_unique<legacy::GraphCOO<int, int, float>>(std::move(coo_contents));
}

// Double weights specialization
template <>
std::unique_ptr<legacy::GraphCOO<int, int, double>> minimum_spanning_tree<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const& graph,
  rmm::device_async_resource_ref mr)
{
  CUGRAPH_EXPECTS(graph.offsets != nullptr, "Invalid input argument: graph.offsets is nullptr");
  CUGRAPH_EXPECTS(graph.indices != nullptr, "Invalid input argument: graph.indices is nullptr");
  CUGRAPH_EXPECTS(graph.edge_data != nullptr, "MST requires edge weights (graph.edge_data is nullptr)");

  aai::graph32_t compact_graph{
    .offsets            = graph.offsets,
    .indices            = graph.indices,
    .number_of_vertices = static_cast<int32_t>(graph.number_of_vertices),
    .number_of_edges    = static_cast<int32_t>(graph.number_of_edges),
    .is_symmetric       = true,
    .is_multigraph      = false,
    .is_csc             = false,
    .segment_offsets    = std::nullopt,
  };

  // AAI MST returns symmetrized output (both directions per edge),
  // so allocate 2*V to fit all edges.
  std::size_t max_mst_edges = static_cast<std::size_t>(2 * graph.number_of_vertices);

  rmm::device_uvector<int32_t> mst_srcs(max_mst_edges, handle.get_stream());
  rmm::device_uvector<int32_t> mst_dsts(max_mst_edges, handle.get_stream());
  rmm::device_uvector<double> mst_weights(max_mst_edges, handle.get_stream());

  handle.sync_stream();

  std::size_t num_mst_edges;
  if (compact_graph.segment_offsets.has_value()) {
    num_mst_edges = aai::minimum_spanning_tree_seg(
      compact_graph,
      graph.edge_data,
      mst_srcs.data(),
      mst_dsts.data(),
      mst_weights.data(),
      max_mst_edges);
  } else {
    num_mst_edges = aai::minimum_spanning_tree(
      compact_graph,
      graph.edge_data,
      mst_srcs.data(),
      mst_dsts.data(),
      mst_weights.data(),
      max_mst_edges);
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  mst_srcs.resize(num_mst_edges, handle.get_stream());
  mst_dsts.resize(num_mst_edges, handle.get_stream());
  mst_weights.resize(num_mst_edges, handle.get_stream());

  legacy::GraphCOOContents<int, int, double> coo_contents{
    graph.number_of_vertices,
    num_mst_edges,
    std::make_unique<rmm::device_buffer>(mst_srcs.release()),
    std::make_unique<rmm::device_buffer>(mst_dsts.release()),
    std::make_unique<rmm::device_buffer>(mst_weights.release())};

  return std::make_unique<legacy::GraphCOO<int, int, double>>(std::move(coo_contents));
}

#else  // !AAI_ROUTE_MST

template std::unique_ptr<legacy::GraphCOO<int, int, float>> minimum_spanning_tree<int, int, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, float> const& graph,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<legacy::GraphCOO<int, int, double>> minimum_spanning_tree<int, int, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int, int, double> const& graph,
  rmm::device_async_resource_ref mr);

#endif  // AAI_ROUTE_MST

// =============================================================================
// Explicit template instantiations for int64 types (use original implementation)
// =============================================================================

template std::unique_ptr<legacy::GraphCOO<int64_t, int64_t, float>> minimum_spanning_tree<int64_t, int64_t, float>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int64_t, int64_t, float> const& graph,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<legacy::GraphCOO<int64_t, int64_t, double>> minimum_spanning_tree<int64_t, int64_t, double>(
  raft::handle_t const& handle,
  legacy::GraphCSRView<int64_t, int64_t, double> const& graph,
  rmm::device_async_resource_ref mr);

}  // namespace cugraph
