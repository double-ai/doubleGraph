# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Modifications Copyright (c) 2025, AA-I Technologies Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: This file has been modified by AA-I Technologies Ltd. from the original.
import pytest

import cugraph
from cugraph.datasets import karate


@pytest.mark.sg
def test_spectral_balanced_cut_clustering_deprecation_warning():
    """Test that spectralBalancedCutClustering emits a deprecation warning.

    Note: spectralBalancedCutClustering is deprecated in favor of
    spectralModularityMaximizationClustering. Functional tests for spectral
    clustering (including edge cut validation) are in test_modularity.py.
    """
    G = karate.get_graph(
        create_using=cugraph.Graph(directed=False)
    )
    warning_msg = (
        "spectralBalancedCutClustering is deprecated and will be removed in a future "
        "release. Use spectralModularityMaximizationClustering instead."
    )

    with pytest.warns(FutureWarning, match=warning_msg):
        cugraph.spectralBalancedCutClustering(G, num_clusters=2)
