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
#include "centrality/betweenness_centrality_impl.cuh"

#ifdef AAI_ROUTE_BETWEENNESS_CENTRALITY

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>
#include <thrust/sequence.h>

namespace cugraph {

// =============================================================================
// Betweenness Centrality - AAI Specializations
// =============================================================================

template <>
rmm::device_uvector<float> betweenness_centrality<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const include_endpoints,
    bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // AAI preconditions
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Betweenness centrality requires CSR format (is_csc=false)");

  rmm::device_uvector<float> centralities(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());

  // AAI kernels require sample_vertices to always be non-null with num_samples > 0.
  // When vertices is nullopt (k=None), generate all vertex IDs [0..V-1].
  rmm::device_uvector<int32_t> all_vertices(0, handle.get_stream());
  const int32_t* sample_verts;
  std::size_t num_samples;
  if (vertices) {
    sample_verts = vertices->data();
    num_samples = vertices->size();
  } else {
    int32_t V = graph_view.local_vertex_partition_range_size();
    all_vertices.resize(V, handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), all_vertices.begin(), all_vertices.end(), int32_t{0});
    sample_verts = all_vertices.data();
    num_samples = V;
  }

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  // 4-way dispatch: mask x segment
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      aai::betweenness_centrality_seg_mask(compact_graph, centralities.data(),
                                            normalized, include_endpoints, sample_verts, num_samples);
    } else {
      aai::betweenness_centrality_mask(compact_graph, centralities.data(),
                                        normalized, include_endpoints, sample_verts, num_samples);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      aai::betweenness_centrality_seg(compact_graph, centralities.data(),
                                      normalized, include_endpoints, sample_verts, num_samples);
    } else {
      aai::betweenness_centrality(compact_graph, centralities.data(),
                                  normalized, include_endpoints, sample_verts, num_samples);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
  return centralities;
}

// Double specialization: no AAI solution available, fall through to original cuGraph implementation.
template rmm::device_uvector<double> betweenness_centrality<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const include_endpoints,
    bool do_expensive_check);

// =============================================================================
// Edge Betweenness Centrality - AAI Specializations
// =============================================================================

template <>
edge_property_t<int32_t, float>
edge_betweenness_centrality<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // AAI preconditions
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Edge betweenness centrality requires CSR format (is_csc=false)");

  edge_property_t<int32_t, float> edge_centralities(handle, graph_view);

  // AAI kernels require sample_vertices to always be non-null with num_samples > 0.
  // When vertices is nullopt (k=None), generate all vertex IDs [0..V-1].
  rmm::device_uvector<int32_t> all_vertices(0, handle.get_stream());
  const int32_t* sample_verts;
  std::size_t num_samples;
  if (vertices) {
    sample_verts = vertices->data();
    num_samples = vertices->size();
  } else {
    int32_t V = graph_view.local_vertex_partition_range_size();
    all_vertices.resize(V, handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), all_vertices.begin(), all_vertices.end(), int32_t{0});
    sample_verts = all_vertices.data();
    num_samples = V;
  }

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  // 4-way dispatch: mask x segment
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      aai::edge_betweenness_centrality_seg_mask(compact_graph,
                                                 edge_centralities.mutable_view().value_firsts()[0],
                                                 normalized, sample_verts, num_samples);
    } else {
      aai::edge_betweenness_centrality_mask(compact_graph,
                                             edge_centralities.mutable_view().value_firsts()[0],
                                             normalized, sample_verts, num_samples);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      aai::edge_betweenness_centrality_seg(compact_graph,
                                           edge_centralities.mutable_view().value_firsts()[0],
                                           normalized, sample_verts, num_samples);
    } else {
      aai::edge_betweenness_centrality(compact_graph,
                                       edge_centralities.mutable_view().value_firsts()[0],
                                       normalized, sample_verts, num_samples);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
  return edge_centralities;
}

// Double specialization: no AAI solution available, fall through to original cuGraph implementation.
template edge_property_t<int32_t, double>
edge_betweenness_centrality<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const do_expensive_check);

}  // namespace cugraph

#else  // !AAI_ROUTE_BETWEENNESS_CENTRALITY

namespace cugraph {

template rmm::device_uvector<float> betweenness_centrality<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const include_endpoints,
    bool do_expensive_check);

template rmm::device_uvector<double> betweenness_centrality<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const include_endpoints,
    bool do_expensive_check);

template edge_property_t<int32_t, float>
edge_betweenness_centrality<int32_t, int32_t, float, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const do_expensive_check);

template edge_property_t<int32_t, double>
edge_betweenness_centrality<int32_t, int32_t, double, false>(
    raft::handle_t const& handle,
    graph_view_t<int32_t, int32_t, false, false> const& graph_view,
    std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
    std::optional<raft::device_span<int32_t const>> vertices,
    bool const normalized,
    bool const do_expensive_check);

}  // namespace cugraph

#endif  // AAI_ROUTE_BETWEENNESS_CENTRALITY
