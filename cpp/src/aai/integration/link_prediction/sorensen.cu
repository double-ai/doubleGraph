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
#include "link_prediction/sorensen_impl.cuh"

#ifdef AAI_ROUTE_SORENSEN

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <cuda_runtime.h>

#include <stdexcept>
#include <limits>

namespace cugraph {

// =============================================================================
// Specialized sorensen_coefficients to route to AAI
// =============================================================================

// Float specialization (handles both unweighted and float-weighted cases)
template <>
rmm::device_uvector<float> sorensen_coefficients<int32_t, int32_t, float, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>> vertex_pairs,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from sorensen_impl.cuh:39 + similarity_impl.cuh:45-48)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Sorensen similarity requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(std::get<0>(vertex_pairs).size() == std::get<1>(vertex_pairs).size(),
                  "vertex pairs have mismatched sizes");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");

  // Extract vertex pairs
  const int32_t* first_vertices = std::get<0>(vertex_pairs).data();
  const int32_t* second_vertices = std::get<1>(vertex_pairs).data();
  std::size_t num_pairs = std::get<0>(vertex_pairs).size();

  // Allocate output
  rmm::device_uvector<float> similarity_scores(num_pairs, handle.get_stream());

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION
  // =========================================================================

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  if (edge_weight_view.has_value()) {
    // Weighted Sorensen with float weights
    const float* weights = edge_weight_view->value_firsts()[0];
    if (compact_graph.segment_offsets.has_value()) {
      aai::sorensen_similarity_seg(compact_graph, weights, first_vertices, second_vertices,
                                   num_pairs, similarity_scores.data());
    } else {
      aai::sorensen_similarity(compact_graph, weights, first_vertices, second_vertices,
                               num_pairs, similarity_scores.data());
    }
  } else {
    // Unweighted Sorensen
    if (compact_graph.segment_offsets.has_value()) {
      aai::sorensen_similarity_seg(compact_graph, first_vertices, second_vertices,
                                   num_pairs, similarity_scores.data());
    } else {
      aai::sorensen_similarity(compact_graph, first_vertices, second_vertices,
                               num_pairs, similarity_scores.data());
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return similarity_scores;
}

// Double-precision specialization: supports both weighted (double weights) and
// unweighted (calls float AAI kernel + widening copy) Sorensen.
template <>
rmm::device_uvector<double> sorensen_coefficients<int32_t, int32_t, double, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>> vertex_pairs,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from sorensen_impl.cuh:39 + similarity_impl.cuh:45-48)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Sorensen similarity requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(std::get<0>(vertex_pairs).size() == std::get<1>(vertex_pairs).size(),
                  "vertex pairs have mismatched sizes");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");

  // Extract vertex pairs
  const int32_t* first_vertices = std::get<0>(vertex_pairs).data();
  const int32_t* second_vertices = std::get<1>(vertex_pairs).data();
  std::size_t num_pairs = std::get<0>(vertex_pairs).size();

  // Allocate output
  rmm::device_uvector<double> similarity_scores(num_pairs, handle.get_stream());

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  if (edge_weight_view.has_value()) {
    // Weighted Sorensen with double weights
    const double* weights = edge_weight_view->value_firsts()[0];
    if (compact_graph.segment_offsets.has_value()) {
      aai::sorensen_similarity_seg(compact_graph, weights, first_vertices, second_vertices,
                                   num_pairs, similarity_scores.data());
    } else {
      aai::sorensen_similarity(compact_graph, weights, first_vertices, second_vertices,
                               num_pairs, similarity_scores.data());
    }
  } else {
    // Unweighted Sorensen: AAI only has float output, so compute in float and widen.
    rmm::device_uvector<float> float_scores(num_pairs, handle.get_stream());
    handle.sync_stream();
    if (compact_graph.segment_offsets.has_value()) {
      aai::sorensen_similarity_seg(compact_graph, first_vertices, second_vertices,
                                   num_pairs, float_scores.data());
    } else {
      aai::sorensen_similarity(compact_graph, first_vertices, second_vertices,
                               num_pairs, float_scores.data());
    }
    cudaDeviceSynchronize();
    // Widen float -> double
    thrust::transform(rmm::exec_policy(handle.get_stream()),
                      float_scores.begin(), float_scores.end(),
                      similarity_scores.begin(),
                      [] __device__(float v) { return static_cast<double>(v); });
    handle.sync_stream();
    return similarity_scores;
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return similarity_scores;
}

// =============================================================================
// Specialized sorensen_all_pairs_coefficients to route to AAI
// =============================================================================

// Float specialization (handles both unweighted and float-weighted cases)
template <>
std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
sorensen_all_pairs_coefficients<int32_t, int32_t, float, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> vertices,
  std::optional<size_t> topk,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from sorensen_impl.cuh:61 + similarity_impl.cuh:213-219)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Sorensen all-pairs similarity requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph() || !edge_weight_view,
                  "Weighted implementation currently fails on multi-graph");

  // Extract optional vertices
  const int32_t* vertices_ptr = vertices.has_value() ? vertices->data() : nullptr;
  std::size_t num_vertices = vertices.has_value() ? vertices->size() : 0;

  // Pass topk directly to AAI (std::optional<std::size_t>, nullopt = unlimited)

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION
  // =========================================================================

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  aai::similarity_result_float_t result;
  if (edge_weight_view.has_value()) {
    const float* weights = edge_weight_view->value_firsts()[0];
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::sorensen_all_pairs_similarity_seg(
          compact_graph, weights, vertices_ptr, num_vertices, topk);
    } else {
      result = aai::sorensen_all_pairs_similarity(
          compact_graph, weights, vertices_ptr, num_vertices, topk);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::sorensen_all_pairs_similarity_seg(
          compact_graph, vertices_ptr, num_vertices, topk);
    } else {
      result = aai::sorensen_all_pairs_similarity(
          compact_graph, vertices_ptr, num_vertices, topk);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  // Copy from AAI-allocated device memory to RMM-managed device_uvector
  rmm::device_uvector<int32_t> out_first(result.count, handle.get_stream());
  rmm::device_uvector<int32_t> out_second(result.count, handle.get_stream());
  rmm::device_uvector<float> out_scores(result.count, handle.get_stream());

  if (result.count > 0) {
    cudaMemcpyAsync(out_first.data(), result.first, result.count * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, handle.get_stream());
    cudaMemcpyAsync(out_second.data(), result.second, result.count * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, handle.get_stream());
    cudaMemcpyAsync(out_scores.data(), result.scores, result.count * sizeof(float),
                    cudaMemcpyDeviceToDevice, handle.get_stream());
  }

  // Sync stream before freeing AAI-allocated memory to ensure copies complete
  handle.sync_stream();

  // Free AAI-allocated memory
  if (result.first) cudaFree(result.first);
  if (result.second) cudaFree(result.second);
  if (result.scores) cudaFree(result.scores);

  return std::make_tuple(std::move(out_first), std::move(out_second), std::move(out_scores));
}

// Double-precision all-pairs specialization: supports both weighted (double weights) and
// unweighted (calls float AAI kernel + widening copy) Sorensen.
template <>
std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
sorensen_all_pairs_coefficients<int32_t, int32_t, double, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> vertices,
  std::optional<size_t> topk,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from sorensen_impl.cuh:61 + similarity_impl.cuh:213-219)
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "Edge masks are not supported");
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "Sorensen all-pairs similarity requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph() || !edge_weight_view,
                  "Weighted implementation currently fails on multi-graph");

  // Extract optional vertices
  const int32_t* vertices_ptr = vertices.has_value() ? vertices->data() : nullptr;
  std::size_t num_vertices = vertices.has_value() ? vertices->size() : 0;

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  if (edge_weight_view.has_value()) {
    // Weighted all-pairs Sorensen with double weights
    const double* weights = edge_weight_view->value_firsts()[0];
    aai::similarity_result_double_t result;
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::sorensen_all_pairs_similarity_seg(
          compact_graph, weights, vertices_ptr, num_vertices, topk);
    } else {
      result = aai::sorensen_all_pairs_similarity(
          compact_graph, weights, vertices_ptr, num_vertices, topk);
    }

    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }

    rmm::device_uvector<int32_t> out_first(result.count, handle.get_stream());
    rmm::device_uvector<int32_t> out_second(result.count, handle.get_stream());
    rmm::device_uvector<double> out_scores(result.count, handle.get_stream());

    if (result.count > 0) {
      cudaMemcpyAsync(out_first.data(), result.first, result.count * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, handle.get_stream());
      cudaMemcpyAsync(out_second.data(), result.second, result.count * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, handle.get_stream());
      cudaMemcpyAsync(out_scores.data(), result.scores, result.count * sizeof(double),
                      cudaMemcpyDeviceToDevice, handle.get_stream());
    }

    handle.sync_stream();
    if (result.first) cudaFree(result.first);
    if (result.second) cudaFree(result.second);
    if (result.scores) cudaFree(result.scores);

    return std::make_tuple(std::move(out_first), std::move(out_second), std::move(out_scores));
  } else {
    // Unweighted all-pairs Sorensen: AAI only has float output, so compute in float and widen.
    aai::similarity_result_float_t result;
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::sorensen_all_pairs_similarity_seg(
          compact_graph, vertices_ptr, num_vertices, topk);
    } else {
      result = aai::sorensen_all_pairs_similarity(
          compact_graph, vertices_ptr, num_vertices, topk);
    }

    cudaDeviceSynchronize();
    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
      }
    }

    rmm::device_uvector<int32_t> out_first(result.count, handle.get_stream());
    rmm::device_uvector<int32_t> out_second(result.count, handle.get_stream());
    rmm::device_uvector<double> out_scores(result.count, handle.get_stream());

    if (result.count > 0) {
      cudaMemcpyAsync(out_first.data(), result.first, result.count * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, handle.get_stream());
      cudaMemcpyAsync(out_second.data(), result.second, result.count * sizeof(int32_t),
                      cudaMemcpyDeviceToDevice, handle.get_stream());
      // Widen float scores -> double
      rmm::device_uvector<float> tmp_scores(result.count, handle.get_stream());
      cudaMemcpyAsync(tmp_scores.data(), result.scores, result.count * sizeof(float),
                      cudaMemcpyDeviceToDevice, handle.get_stream());
      handle.sync_stream();
      thrust::transform(rmm::exec_policy(handle.get_stream()),
                        tmp_scores.begin(), tmp_scores.end(),
                        out_scores.begin(),
                        [] __device__(float v) { return static_cast<double>(v); });
    }

    handle.sync_stream();
    if (result.first) cudaFree(result.first);
    if (result.second) cudaFree(result.second);
    if (result.scores) cudaFree(result.scores);

    return std::make_tuple(std::move(out_first), std::move(out_second), std::move(out_scores));
  }
}

}  // namespace cugraph

#else  // !AAI_ROUTE_SORENSEN

namespace cugraph {
template rmm::device_uvector<float> sorensen_coefficients<int32_t, int32_t, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>, bool);
template rmm::device_uvector<double> sorensen_coefficients<int32_t, int32_t, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>, bool);
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<float>> sorensen_all_pairs_coefficients<int32_t, int32_t, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, std::optional<raft::device_span<int32_t const>>, std::optional<size_t>, bool);
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<double>> sorensen_all_pairs_coefficients<int32_t, int32_t, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, std::optional<raft::device_span<int32_t const>>, std::optional<size_t>, bool);
}  // namespace cugraph

#endif  // AAI_ROUTE_SORENSEN
