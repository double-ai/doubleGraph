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
// Include original implementation for non-specialized template instantiations
#include "community/egonet_impl.cuh"

// =============================================================================
// Original template instantiations (legacy pointer-based API)
// =============================================================================

// These are kept for backwards compatibility with the legacy API

namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const& handle,
            graph_view_t<int32_t, int32_t, false, false> const& graph_view,
            std::optional<edge_property_view_t<int32_t, float const*>>,
            int32_t* source_vertex,
            int32_t n_subgraphs,
            int32_t radius);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const&,
            graph_view_t<int32_t, int32_t, false, false> const& graph_view,
            std::optional<edge_property_view_t<int32_t, double const*>>,
            int32_t* source_vertex,
            int32_t n_subgraphs,
            int32_t radius);

}  // namespace cugraph

// =============================================================================
// Template specializations for device_span API - route to AAI
// =============================================================================

#ifdef AAI_ROUTE_EGONET

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// Float specialization
template <>
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<float>>,
           rmm::device_uvector<size_t>>
extract_ego<int32_t, int32_t, float, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  raft::device_span<int32_t const> source_vertices,
  int32_t radius,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from egonet_impl.cuh)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "extract_ego requires CSR format (store_transposed=false)");
  CUGRAPH_EXPECTS(radius > 0, "Radius should be at least 1");
  CUGRAPH_EXPECTS(radius < graph_view.number_of_vertices(), "radius is too large");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  if (edge_weight_view.has_value()) {
    const float* weights = edge_weight_view->value_firsts()[0];

    aai::extract_ego_weighted_result_float_t result;
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::extract_ego_weighted_f32_seg(
          compact_graph, weights, source_vertices.data(), source_vertices.size(), radius);
    } else {
      result = aai::extract_ego_weighted_f32(
          compact_graph, weights, source_vertices.data(), source_vertices.size(), radius);
    }

    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }

    // Wrap AAI result in RMM vectors
    rmm::device_uvector<int32_t> edge_srcs(result.num_edges, handle.get_stream());
    rmm::device_uvector<int32_t> edge_dsts(result.num_edges, handle.get_stream());
    rmm::device_uvector<float> out_weights(result.num_edges, handle.get_stream());
    rmm::device_uvector<size_t> edge_offsets(result.num_offsets, handle.get_stream());

    cudaMemcpyAsync(edge_srcs.data(), result.edge_srcs,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_dsts.data(), result.edge_dsts,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(out_weights.data(), result.edge_weights,
                    result.num_edges * sizeof(float), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_offsets.data(), result.offsets,
                    result.num_offsets * sizeof(size_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());

    handle.sync_stream();

    cudaFree(result.edge_srcs);
    cudaFree(result.edge_dsts);
    cudaFree(result.edge_weights);
    cudaFree(result.offsets);

    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts),
                           std::make_optional(std::move(out_weights)), std::move(edge_offsets));
  } else {
    aai::extract_ego_result_t result;
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::extract_ego_seg(
          compact_graph, source_vertices.data(), source_vertices.size(), radius);
    } else {
      result = aai::extract_ego(
          compact_graph, source_vertices.data(), source_vertices.size(), radius);
    }

    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }

    rmm::device_uvector<int32_t> edge_srcs(result.num_edges, handle.get_stream());
    rmm::device_uvector<int32_t> edge_dsts(result.num_edges, handle.get_stream());
    rmm::device_uvector<size_t> edge_offsets(result.num_offsets, handle.get_stream());

    cudaMemcpyAsync(edge_srcs.data(), result.edge_srcs,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_dsts.data(), result.edge_dsts,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_offsets.data(), result.offsets,
                    result.num_offsets * sizeof(size_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());

    handle.sync_stream();

    cudaFree(result.edge_srcs);
    cudaFree(result.edge_dsts);
    cudaFree(result.offsets);

    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts),
                           std::nullopt, std::move(edge_offsets));
  }
}

// Double specialization
template <>
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<double>>,
           rmm::device_uvector<size_t>>
extract_ego<int32_t, int32_t, double, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  raft::device_span<int32_t const> source_vertices,
  int32_t radius,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from egonet_impl.cuh)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "extract_ego requires CSR format (store_transposed=false)");
  CUGRAPH_EXPECTS(radius > 0, "Radius should be at least 1");
  CUGRAPH_EXPECTS(radius < graph_view.number_of_vertices(), "radius is too large");

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  if (edge_weight_view.has_value()) {
    const double* weights = edge_weight_view->value_firsts()[0];

    aai::extract_ego_weighted_result_double_t result;
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::extract_ego_weighted_f64_seg(
          compact_graph, weights, source_vertices.data(), source_vertices.size(), radius);
    } else {
      result = aai::extract_ego_weighted_f64(
          compact_graph, weights, source_vertices.data(), source_vertices.size(), radius);
    }

    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }

    // Wrap AAI result in RMM vectors
    rmm::device_uvector<int32_t> edge_srcs(result.num_edges, handle.get_stream());
    rmm::device_uvector<int32_t> edge_dsts(result.num_edges, handle.get_stream());
    rmm::device_uvector<double> out_weights(result.num_edges, handle.get_stream());
    rmm::device_uvector<size_t> edge_offsets(result.num_offsets, handle.get_stream());

    cudaMemcpyAsync(edge_srcs.data(), result.edge_srcs,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_dsts.data(), result.edge_dsts,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(out_weights.data(), result.edge_weights,
                    result.num_edges * sizeof(double), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_offsets.data(), result.offsets,
                    result.num_offsets * sizeof(size_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());

    handle.sync_stream();

    cudaFree(result.edge_srcs);
    cudaFree(result.edge_dsts);
    cudaFree(result.edge_weights);
    cudaFree(result.offsets);

    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts),
                           std::make_optional(std::move(out_weights)), std::move(edge_offsets));
  } else {
    aai::extract_ego_result_t result;
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::extract_ego_seg(
          compact_graph, source_vertices.data(), source_vertices.size(), radius);
    } else {
      result = aai::extract_ego(
          compact_graph, source_vertices.data(), source_vertices.size(), radius);
    }

    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }

    rmm::device_uvector<int32_t> edge_srcs(result.num_edges, handle.get_stream());
    rmm::device_uvector<int32_t> edge_dsts(result.num_edges, handle.get_stream());
    rmm::device_uvector<size_t> edge_offsets(result.num_offsets, handle.get_stream());

    cudaMemcpyAsync(edge_srcs.data(), result.edge_srcs,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_dsts.data(), result.edge_dsts,
                    result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());
    cudaMemcpyAsync(edge_offsets.data(), result.offsets,
                    result.num_offsets * sizeof(size_t), cudaMemcpyDeviceToDevice,
                    handle.get_stream());

    handle.sync_stream();

    cudaFree(result.edge_srcs);
    cudaFree(result.edge_dsts);
    cudaFree(result.offsets);

    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts),
                           std::nullopt, std::move(edge_offsets));
  }
}

}  // namespace cugraph

#else  // !AAI_ROUTE_EGONET

namespace cugraph {
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>, rmm::device_uvector<size_t>> extract_ego<int32_t, int32_t, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, raft::device_span<int32_t const>, int32_t, bool);
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>, rmm::device_uvector<size_t>> extract_ego<int32_t, int32_t, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, raft::device_span<int32_t const>, int32_t, bool);
}  // namespace cugraph

#endif  // AAI_ROUTE_EGONET
