/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modifications Copyright (c) 2025, AA-I Technologies Ltd.
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
 *
 * NOTICE: This file has been modified by AA-I Technologies Ltd. from the original.
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <vector>

// Validate that the given centralities are a fixed point of the eigenvector
// centrality power iteration: x_new = L2_normalize((A + I) * x).
// This checks the mathematical eigenvector property rather than comparing
// against a specific reference implementation (which would depend on the
// particular initialization vector used).
template <typename vertex_t, typename weight_t>
void eigenvector_centrality_validate(vertex_t const* src,
                                     vertex_t const* dst,
                                     std::optional<weight_t const*> weights,
                                     size_t num_edges,
                                     weight_t const* centralities,
                                     vertex_t num_vertices,
                                     weight_t epsilon)
{
  if (num_vertices == 0) { return; }

  // Convert to double for precision
  std::vector<double> x(num_vertices);
  for (vertex_t i = 0; i < num_vertices; ++i) {
    x[i] = static_cast<double>(centralities[i]);
  }

  // Basic sanity: should be L2-normalized and not all zeros
  auto l2 = std::sqrt(
    std::inner_product(x.begin(), x.end(), x.begin(), double{0.0}));
  ASSERT_GT(l2, 0.0) << "Eigenvector centrality result is all zeros.";
  EXPECT_NEAR(l2, 1.0, 0.01) << "Eigenvector centrality result is not L2-normalized.";

  // Normalize to exact unit norm for the fixed-point check
  for (auto& v : x) { v /= l2; }

  // Apply one iteration: x_new = (A + I) * x
  std::vector<double> x_new(x.begin(), x.end());  // +I term (start with copy of x)
  for (size_t e = 0; e < num_edges; ++e) {
    auto w = weights ? static_cast<double>((*weights)[e]) : 1.0;
    x_new[src[e]] += x[dst[e]] * w;
  }

  // L2-normalize x_new
  auto l2_new = std::sqrt(
    std::inner_product(x_new.begin(), x_new.end(), x_new.begin(), double{0.0}));
  ASSERT_GT(l2_new, 0.0);
  for (auto& v : x_new) { v /= l2_new; }

  // Check fixed-point property: one more iteration should barely change the result.
  // The algorithm converges when diff < n * epsilon, so use a small multiple for tolerance.
  double diff_sum{0.0};
  for (vertex_t i = 0; i < num_vertices; ++i) {
    diff_sum += std::abs(x_new[i] - x[i]);
  }

  EXPECT_LT(diff_sum, num_vertices * static_cast<double>(epsilon) * 3)
    << "Eigenvector centrality result is not a valid eigenvector "
       "(one additional power iteration changed the result by " << diff_sum << ").";
}

struct EigenvectorCentrality_Usecase {
  size_t max_iterations{std::numeric_limits<size_t>::max()};
  bool test_weighted{false};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_EigenvectorCentrality
  : public ::testing::TestWithParam<std::tuple<EigenvectorCentrality_Usecase, input_usecase_t>> {
 public:
  Tests_EigenvectorCentrality() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t, typename weight_t>
  void run_current_test(EigenvectorCentrality_Usecase const& eigenvector_usecase,
                        input_usecase_t const& input_usecase)
  {
    constexpr bool renumber = true;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    auto [graph, edge_weights, d_renumber_map_labels] =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, true, false>(
        handle, input_usecase, eigenvector_usecase.test_weighted, renumber);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    auto graph_view = graph.view();
    auto edge_weight_view =
      edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (eigenvector_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    weight_t constexpr epsilon{1e-6};

    rmm::device_uvector<weight_t> d_centralities(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Eigenvector centrality");
    }

    d_centralities =
      cugraph::eigenvector_centrality(handle,
                                      graph_view,
                                      edge_weight_view,
                                      std::optional<raft::device_span<weight_t const>>{},
                                      epsilon,
                                      eigenvector_usecase.max_iterations,
                                      false);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (eigenvector_usecase.check_correctness) {
      rmm::device_uvector<vertex_t> dst_v(0, handle.get_stream());
      rmm::device_uvector<vertex_t> src_v(0, handle.get_stream());
      std::optional<rmm::device_uvector<weight_t>> opt_wgt_v{std::nullopt};

      std::tie(dst_v, src_v, opt_wgt_v, std::ignore, std::ignore) = cugraph::decompress_to_edgelist(
        handle,
        graph_view,
        edge_weight_view,
        std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>{std::nullopt});

      auto h_src     = cugraph::test::to_host(handle, src_v);
      auto h_dst     = cugraph::test::to_host(handle, dst_v);
      auto h_weights = cugraph::test::to_host(handle, opt_wgt_v);

      auto h_cugraph_centralities = cugraph::test::to_host(handle, d_centralities);

      eigenvector_centrality_validate(
        h_src.data(),
        h_dst.data(),
        h_weights ? std::make_optional<weight_t const*>(h_weights->data()) : std::nullopt,
        h_src.size(),
        h_cugraph_centralities.data(),
        graph_view.number_of_vertices(),
        epsilon);
    }
  }
};

using Tests_EigenvectorCentrality_File = Tests_EigenvectorCentrality<cugraph::test::File_Usecase>;
using Tests_EigenvectorCentrality_Rmat = Tests_EigenvectorCentrality<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_EigenvectorCentrality_File, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_EigenvectorCentrality_Rmat, CheckInt32Int32FloatFloat)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_EigenvectorCentrality_Rmat, CheckInt64Int64FloatFloat)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t, float>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  file_test_test,
  Tests_EigenvectorCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EigenvectorCentrality_Usecase{500, false, false},
                      EigenvectorCentrality_Usecase{500, false, true},
                      EigenvectorCentrality_Usecase{500, true, false},
                      EigenvectorCentrality_Usecase{500, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/karate.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  file_large_test,
  Tests_EigenvectorCentrality_File,
  ::testing::Combine(
    // enable correctness checks
    ::testing::Values(EigenvectorCentrality_Usecase{500, false, false},
                      EigenvectorCentrality_Usecase{500, false, true},
                      EigenvectorCentrality_Usecase{500, true, false},
                      EigenvectorCentrality_Usecase{500, true, true}),
    ::testing::Values(cugraph::test::File_Usecase("test/datasets/web-Google.mtx"),
                      cugraph::test::File_Usecase("test/datasets/ljournal-2008.mtx"),
                      cugraph::test::File_Usecase("test/datasets/webbase-1M.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_EigenvectorCentrality_Rmat,
  // enable correctness checks
  ::testing::Combine(
    ::testing::Values(EigenvectorCentrality_Usecase{500, false, false},
                      EigenvectorCentrality_Usecase{500, false, true},
                      EigenvectorCentrality_Usecase{500, true, false},
                      EigenvectorCentrality_Usecase{500, true, true}),
    ::testing::Values(cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_EigenvectorCentrality_Rmat,
  // disable correctness checks for large graphs
  ::testing::Combine(
    ::testing::Values(EigenvectorCentrality_Usecase{500, false, false, false},
                      EigenvectorCentrality_Usecase{500, false, true, false},
                      EigenvectorCentrality_Usecase{500, true, false, false},
                      EigenvectorCentrality_Usecase{500, true, true, false}),
    ::testing::Values(cugraph::test::Rmat_Usecase(20, 32, 0.57, 0.19, 0.19, 0, false, false))));

CUGRAPH_TEST_PROGRAM_MAIN()
