# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time

import jax
import jax.numpy as jnp
import numpy as np
import sml.utils.emulation as emulation
from scipy.sparse.csgraph import shortest_path
from sml.manifold.dijkstra import mpc_dijkstra
from sml.manifold.floyd import floyd_opt,floyd_opt_1


def emul_cpz(mode: emulation.Mode.MULTIPROCESS):
    try:
        # bandwidth and latency only work for docker mode
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        def dijkstra_all_pairs(
            Knn,
            mpc_dist_inf,
            num_samples,
        ):

            def compute_distances_for_sample(i, Knn, num_samples, mpc_dist_inf):
                return mpc_dijkstra(Knn, num_samples, i, mpc_dist_inf)

            compute_distances = jax.vmap(
                compute_distances_for_sample, in_axes=(0, None, None, None)
            )

            indices = jnp.arange(num_samples)  # 样本索引
            mpc_shortest_paths = compute_distances(
                indices, Knn, num_samples, mpc_dist_inf
            )
            return mpc_shortest_paths

        num_samples = 200
        dist_inf = jnp.full(num_samples, np.inf)

        X = np.random.rand(num_samples, num_samples)
        X = (X + X.T) / 2
        X[X == 0] = np.inf
        np.fill_diagonal(X, 0)
        sX = emulator.seal(X)

        # dijkstra_ans = emulator.run(dijkstra_all_pairs, static_argnums=(2,))(
        #     sX, dist_inf, num_samples
        # )

        # print('dijkstra_ans: \n', dijkstra_ans)

        # floyd_opt_1_ans = emulator.run(floyd_opt_1)(sX)

        # print('floyd_ans: \n', floyd_opt_1_ans)

        floyd_opt_ans = emulator.run(floyd_opt)(sX)

        # print('floyd_opt_ans: \n', floyd_opt_ans)

        # sklearn test
        sklearn_ans = shortest_path(X, method="D", directed=False)
        # print('sklearn_ans: \n', sklearn_ans)

    finally:
        emulator.down()


if __name__ == "__main__":
    emul_cpz(emulation.Mode.MULTIPROCESS)
