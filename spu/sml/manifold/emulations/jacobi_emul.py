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
from sml.manifold.jacobi import Jacobi, Jacobi_Without_Opt
import sml.utils.emulation as emulation

def emul_cpz(mode: emulation.Mode.MULTIPROCESS):
    try:
        emulator = emulation.Emulator(
            emulation.CLUSTER_ABY3_3PC, mode, bandwidth=300, latency=20
        )
        emulator.up()

        num_samples = 100

        X = np.random.rand(num_samples, num_samples)
        X = (X + X.T) / 2

        sX = emulator.seal(X)
        values, vectors = emulator.run(Jacobi, static_argnums=(1,))(sX, num_samples)
        values2, vectors2 = emulator.run(Jacobi_Without_Opt, static_argnums=(1,))(sX, num_samples)
        # print("values: \n",values)
        # print("vectors: \n",vectors)
        # print("values2: \n",values2)
        # print("vectors2: \n",vectors2)
    
    finally:
        emulator.down()


if __name__ == "__main__":
    emul_cpz(emulation.Mode.MULTIPROCESS)
