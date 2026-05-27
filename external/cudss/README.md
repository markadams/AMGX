# cuDSS External Plugin for AMGX

This directory contains the cuDSS coarse-grid solver plugin for AMGX.
It is built only when `-DAMGX_USE_CUDSS=ON` is passed to CMake.

## Files

| File | Description |
|------|-------------|
| `cudss_solver.h` | Class declaration for `CudssSolver<T_Config>` and `CudssSolverFactory` |
| `cudss_solver.cu` | Full implementation: symbolic analysis, numeric factorization, triangular solve |
| `cudss_solver_test.cu` | GTest unit tests (compiled into `amgx_tests_launcher`) |
| `bench_cudss_vs_dense_lu.c` | C benchmark comparing `CUDSS_SOLVER` vs `DENSE_LU_SOLVER` |
| `CMakeLists.txt` | Plugin build logic: `find_package(cudss)`, source wiring, link libraries |

## cuDSS Version Requirements

- cuDSS **0.3.0** or later (API: `cudssCreate`, `cudssMatrixCreateCsr`, `cudssExecute`)
- CUDA Toolkit **12.0** or later
- Tested with cuDSS 0.4.0 on CUDA 12.4 / Hopper (sm_90)

## Building with cuDSS

### 1. Install cuDSS

Download from the [NVIDIA cuDSS page](https://developer.nvidia.com/cudss) or install
via the CUDA network repository:

```bash
# Ubuntu 22.04 example
apt-get install libcudss-dev
```

Or extract the tarball and set `CUDSS_ROOT`:

```bash
export CUDSS_ROOT=/path/to/cudss
```

### 2. Configure AMGX with cuDSS enabled

```bash
cmake -B build \
      -DAMGX_USE_CUDSS=ON \
      -DCUDSS_ROOT=/path/to/cudss \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      /path/to/amgx
cmake --build build -j$(nproc)
```

### 3. Use `CUDSS_SOLVER` in an AMGX config

```json
{
  "config_version": 2,
  "solver": {
    "algorithm": "AGGREGATION",
    "solver": "PCG",
    "preconditioner": {
      "solver": "AMG",
      "algorithm": "AGGREGATION",
      "selector": "SIZE_2",
      "coarse_solver": "CUDSS_SOLVER",
      "cudss_matrix_type": "GENERAL",
      "cudss_reorder": 1
    }
  }
}
```

See `src/configs/PCG_SA_CUDSS.json` for a complete example.

### 4. Run the benchmark

```bash
cd build
./bench_cudss_vs_dense_lu -n 300 -mode dDDI
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cudss_matrix_type` | string | `"GENERAL"` | Matrix symmetry: `"GENERAL"`, `"SYMMETRIC"`, or `"SPD"` |
| `cudss_reorder` | int | `1` | Reordering: `0`=none, `1`=AMD (default), `2`=METIS |
| `exact_coarse_solve` | int | `0` | `1`=gather global matrix across MPI ranks and solve globally |

## Architecture Notes

This plugin follows the **Interface + Plugin** pattern:

- **When `AMGX_USE_CUDSS=OFF`** (default): `src/solvers/cudss_solver_stub.cu` is
  compiled into the main library. It registers `"CUDSS_SOLVER"` in the solver factory
  but prints a clear error message if the solver is actually instantiated.

- **When `AMGX_USE_CUDSS=ON`**: `external/cudss/CMakeLists.txt` is included via
  `add_subdirectory`. It adds `cudss_solver.cu` to `amgx_libs`, links cuDSS, and
  the stub is excluded from the build (the real implementation takes its place).

The solver factory key `"CUDSS_SOLVER"` maps to `cudss_solver::CudssSolverFactory`
in both cases — the stub factory produces an error-throwing solver, while the real
factory produces a fully functional `CudssSolver`.
