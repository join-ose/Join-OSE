# Code for Anonymous Submission ID: 210

This repository contains the official implementation for the anonymous submission ID 210.

The codebase includes our proposed protocols built on the **SPU** framework, as well as the source code for baseline comparisons.

---

##  Project Structure

### 1. Our Implementation (Based on SPU)
Located in `Join-OSE/spu/sml/database/emulations/`.
* **`OSE`**: Implementation of the Oblivious Sort Expansion (OSE) along with correctness verification.
* **`Join-OM`**: Implements the Join-OM protocol from **AHK+** (*Secure Statistical Analysis on Multiple Datasets: Join and Group-By*) and our Join-OM protocol based on OSE. Includes checks to verify identical outputs.
* **`Join-MM`**: Our main Join-MM protocol with working examples.
* **`Join-MM-32`**: Parameterized version (32-bit inputs, adjustable input size) for benchmarks against **BDG+** (*Secret-Shared Joins with Multiplicity from Aggregation Trees*).
* **`Join-MM-64`**: Parameterized version (64-bit inputs, adjustable input size) for benchmarks against **Scape** (*Scape: Scalable Collaborative Analytics System on Private Database with Malicious Security*).

### 2. Scape
Located in `Join-OSE/scql/engine/operator/secret_join.cc`. 
* **Note on Modification**: We have modified the `secret_join.cc` and `secret_join_test.cc` file specifically to support runtime profiling and performance testing for the Scape.

---

##  Environment Setup

We strongly recommend using the official SPU Docker environment to ensure all dependencies (Bazel, C++ toolchains) are correctly configured.

1. **Pull and run the SPU Docker image:**
   ```bash
   docker pull secretflow/ubuntu-base-ci:latest
   docker run -d -it --name spu-dev-$(whoami) \
         --mount type=bind,source="$(pwd)",target=/home/admin/dev/ \
         -w /home/admin/dev \
         --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
         --cap-add=NET_ADMIN \
         --privileged=true \
         secretflow/ubuntu-base-ci:latest \
         bash
    docker exec -it spu-dev-$(whoami) bash
   ```

2. **Mount/Place the code:**
   Copy this entire repository into the container

---

##  Required Source Code Modifications (Crucial)

To reproduce the exact results reported in the paper (for the **Join-MM-64** benchmarks), the following manual change to the underlying framework kernel is required:

Please open the following file: `Join-OSE/spu/src/libspu/kernel/hal/permute.cc` (around Line 1026)
* **Change from:** `bv_size = 32;`
* **Change to:** `bv_size = 64;`

---

##  How to Run

### 1. Running Our Protocols
All core benchmarks are managed via Bazel. Run the following commands from inside the `Join-OSE/spu/` directory:

| Protocol | Command | Comparison Target |
| :--- | :--- | :--- |
| **OSE** | `bazel run -c opt //sml/database/emulations:OSE` | Base |
| **Join-OM** | `bazel run -c opt //sml/database/emulations:Join-OM` | AHK+ |
| **Join-MM** | `bazel run -c opt //sml/database/emulations:Join-MM` | |
| **Join-MM (32-bit)** | `bazel run -c opt //sml/database/emulations:Join-MM-32` | BDG+ |
| **Join-MM (64-bit)** | `bazel run -c opt //sml/database/emulations:Join-MM-64` | Scape (semi2k) |

### 2. Running Scape Runtime Benchmark
To test the runtime of the Scape baseline, navigate to the SCQL operator directory and execute the modified test:

```bash
cd ./Join-OSE/scql/engine/operator/
bazel test secret_join_test
```
