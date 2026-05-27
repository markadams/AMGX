# Feature: Smoothed Aggregation AMG with MIS-k selectors (prototype available)

## Overview

I have implemented **Smoothed Aggregation (SA) AMG** on top of AMGx's existing
aggregation framework. The goal is to bring AMGx closer to feature parity with
PETSc's GAMG solver for GPU-accelerated algebraic multigrid. A working prototype
is available at: https://github.com/markadams/AMGX/tree/feature/smoothed-aggregation-amg

## What Was Added

### 1. SA Prolongator Smoothing

The standard SA step that transforms a tentative prolongator P_tent into a
smoothed prolongator P:

```
P = (I - ω D⁻¹ A) P_tent,   ω = (4/3) / ρ(D⁻¹A)
```

- Spectral radius `ρ(D⁻¹A)` estimated via power iteration (100 iterations)
- Supports scalar (`block_size=1`) and block problems
- Uses AMGx's existing SpGEMM infrastructure for the `S * P_tent` multiply
- Near-null space vector forwarded through the solver to the AMG hierarchy

**Key files**: `src/aggregation/aggregation_amg_level.cu`,
`include/aggregation/aggregation_amg_level.h`

### 2. Batched QR for Tentative Prolongator

Batched Modified Gram-Schmidt QR to construct P_tent from near-null space
vectors, one QR per aggregate.

**Key file**: `src/aggregation/batched_qr.cu`

### 3. MIS-k Aggregation Selectors

Two MIS-2 algorithms selectable via `"mis2_algorithm"`:

**Algorithm 0** (default): MIS-2 via 2× Galerkin coarsening loop — composes
two MIS-1 passes through a Galerkin coarse graph. Produces ~9.5 avg aggregate
size on 5-point Poisson.

**Algorithm 1** (recommended): True MIS-2 directly on the original graph
without forming A²:
- Phase 1: each node checks all distance-2 neighbors; root if highest weight
- Phase 2: distance-1 assignment to nearest root
- Phase 3: propagation for remaining unassigned nodes

Produces ~7.1 avg aggregate size, matching PETSc GAMG's square-graph
aggregation quality.

**Key file**: `src/aggregation/selectors/mis_selector.cu`

### 4. Chebyshev Eigenvalue Reuse (Mode 4)

New `chebyshev_lambda_estimate_mode=4` reuses `ρ(D⁻¹A)` computed during SA
prolongator smoothing, avoiding a redundant eigenvalue computation:

```json
{
  "solver": "CHEBYSHEV",
  "chebyshev_polynomial_order": 1,
  "chebyshev_lambda_estimate_mode": 4,
  "chebyshev_lmin_denom": 11.0
}
```

`lmin_denom=11` matches PETSc GAMG's default Chebyshev interval:
`emax = 1.1*ρ`, `emin = 0.1*ρ = emax/11`.

### 5. Bug Fix: Chebyshev solve_init zeros x when xIsZero=true

`solve_init()` now explicitly zeros `x` when `xIsZero=true`. Without this,
stale data from the previous outer PCG iteration accumulates via
`x += gamma*p`, causing divergence. (Also submitted as a standalone PR.)

## Verification Results

### 100×100 2D Poisson (exact match with PETSc GAMG)

Using PETSc-exported aggregates imported into AMGx:

| Metric | PETSc GAMG | AMGx SA |
|--------|-----------|---------|
| Iterations | 71 | 71 |
| P nnz | 23,705 | 23,705 |
| ‖P_tent‖_F | 3.780211634287129e+01 | 3.780211634287129e+01 |

AMGx SA matches PETSc GAMG **exactly** when given the same aggregates.

### 200×200 2D Poisson (40,000 DOFs)

| | PETSc GAMG | AMGx SA (MULTI_PAIRWISE) |
|---|-----------|---------|
| Levels | 5 | 4 |
| Iterations | 14 | 23 |
| Conv. rate | 0.47 | 0.59 |
| Grid complexity | 1.17 | 1.15 |
| Operator complexity | 1.42 | 1.38 |

### 400×400 2D Poisson (160,000 DOFs)

| | PETSc GAMG | AMGx SA (MIS-2, agg_levels=1) |
|---|-----------|-------------------------------|
| Levels | 5 | 5 |
| Iterations | 16 | 82 |
| Conv. rate | 0.49 | 0.86 |
| L0→L1 ratio | 7.1× | 9.9× |
| Op complexity | 1.43 | 1.24 |

The convergence gap at 400×400 is primarily due to aggregate size non-uniformity:
AMGx's parallel MIS with random hash weights produces aggregates up to size 26,
while PETSc's greedy sequential MIS caps at 13. Capping max aggregate size or
using a greedy ordering is the main remaining work item.

## Configuration Example

```json
{
  "config_version": 2,
  "solver": {
    "solver": "PCG",
    "preconditioner": {
      "solver": "AMG",
      "algorithm": "AGGREGATION",
      "selector": "MIS",
      "mis_k": 2,
      "aggressive_levels": 1,
      "smoother": {
        "solver": "CHEBYSHEV",
        "chebyshev_polynomial_order": 1,
        "chebyshev_lambda_estimate_mode": 4,
        "chebyshev_lmin_denom": 11.0,
        "preconditioner": {
          "solver": "BLOCK_JACOBI",
          "relaxation_factor": 1.0
        }
      },
      "coarse_solver": "DENSE_LU_SOLVER",
      "min_coarse_rows": 32
    }
  }
}
```

## Current / Future Work

### cuDSS Coarse Solver *(in progress — not yet tested)*

A GPU-native sparse direct coarse solver (`CUDSS_SOLVER`) is being developed to replace `DENSE_LU_SOLVER`. It uses NVIDIA's cuDSS library (CUDA 12.4+) for sparse LU or Cholesky factorization entirely on the GPU, avoiding the CPU round-trip. Code scaffolding is in place (`src/solvers/cudss_solver.cu`, `include/solvers/cudss_solver.h`) but has **not yet been tested or validated**.

### Multi-GPU (MPI) Support *(in progress — not yet tested)*

Initial code scaffolding for multi-GPU is in place:
- The MIS-k selector includes halo exchange calls for distributed aggregation
- The cuDSS coarse solver includes a multi-GPU gather/scatter path (`exact_coarse_solve=1`)

**None of this has been tested or validated** in multi-GPU configurations. Full MPI testing on Perlmutter is a primary remaining work item.

### Other Remaining Work

- Cap max aggregate size (target: max ≤ 13, match PETSc GAMG quality)
- Test block_size > 1 (elasticity problems with rigid body modes)
- Adaptive SA (multiple near-null vectors via randomized eigenvectors)

The full implementation with build instructions and verification scripts is at
https://github.com/markadams/AMGX/tree/main
