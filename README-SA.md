# AMGx-SA: Smoothed Aggregation AMG Extensions for AMGx

This fork adds **Smoothed Aggregation (SA) AMG** capabilities to NVIDIA's AMGx library, bringing it closer to feature parity with PETSc's GAMG solver for GPU-accelerated algebraic multigrid.

## What AMGx-SA Adds to AMGx

The original NVIDIA AMGx uses handshake-based pairwise matching for aggregation (SIZE_2/4/8, MULTI_PAIRWISE selectors). AMGx-SA adds:

1. **MIS-based aggregation** — Maximal Independent Set selectors (MIS-1, MIS-2) that produce well-shaped aggregates suitable for SA prolongator smoothing
2. **SA prolongator smoothing** — The standard `P = (I - ω D⁻¹A) P_tent` smoothing step
3. **Chebyshev smoother eigenvalue reuse** — Avoids redundant spectral radius computation
4. **MIS-2 (implicit)** — A new algorithm matching PETSc GAMG's aggregation quality

## New Features

### MIS-k Aggregation Selector

Two MIS-2 algorithms are available, selected via `"mis2_algorithm"`:

#### Algorithm 0: MIS-2 (2xMIS-1 + Galerkin) — default

Implements MIS-2 via a Galerkin coarsening loop:
```
R_out = I
for pass = 0 to k-1:
    R = MIS-1(A_cur)           // true MIS on current graph
    R_out = R ∘ R_out          // compose via gather
    A_cur = R * A_cur * R^T    // Galerkin coarse graph
```

Each pass produces a true MIS-1 on the current graph. The composition produces a distance-2 independent set on the original graph, but it is **not necessarily maximal** — the Galerkin coarse graph has denser connectivity than the original, so the second MIS-1 produces fewer roots than a true MIS-2 would. This results in larger aggregates (~9.5 avg for 5-point Poisson).

#### Algorithm 1: MIS-2 implicit (recommended)

Produces a **true MIS-2** (maximal independent set at distance 2) directly on the original graph without forming A^T*A:
```
Phase 1: MIS-2 selection
    Each node checks all neighbors AND neighbors-of-neighbors
    Root if highest weight within distance 2; dominated if any root within distance 2
    
Phase 2: Distance-1 assignment
    Each non-root joins its strongest ROOT neighbor at distance 1 only
    
Phase 3: Propagation
    Unassigned nodes join their strongest assigned neighbor
```

This naturally produces smaller, well-connected aggregates (~7.1 avg) because distance-1 assignment limits how many nodes each root can grab. The result matches PETSc GAMG's square graph aggregation quality.

**Key file:** `src/aggregation/selectors/mis_selector.cu`

### SA Prolongator Smoothing

The core SA step that transforms a tentative prolongator P_tent into a smoothed prolongator P:

```
P = (I - omega * D^{-1} * A) * P_tent
```

where `omega = (4/3) / rho(D^{-1}A)` is the damping factor computed from the spectral radius of the Jacobi iteration matrix.

**Key files:**
- `src/aggregation/aggregation_amg_level.cu` — `smoothProlongator()`, `estimateSADampingFactor()`
- `include/aggregation/aggregation_amg_level.h` — declarations

**Implementation details:**
- Spectral radius estimated via power iteration (100 iterations by default)
- Supports scalar (block_size=1) and block problems
- Uses AMGx's existing SpGEMM infrastructure for the S*P_tent multiply
- Near-null space vector forwarded through the PCG solver to the AMG hierarchy

### Chebyshev Smoother Eigenvalue Reuse (Mode 4)

A new `chebyshev_lambda_estimate_mode=4` that reuses the spectral radius computed during SA prolongator smoothing, avoiding a redundant eigenvalue computation:

```json
{
  "solver": "CHEBYSHEV",
  "chebyshev_polynomial_order": 1,
  "chebyshev_lambda_estimate_mode": 4,
  "chebyshev_lmin_denom": 11.0,
  "preconditioner": {
    "solver": "BLOCK_JACOBI",
    "relaxation_factor": 1.0
  }
}
```

The `lmin_denom=11` setting matches PETSc GAMG's default Chebyshev eigenvalue interval: `emax = 1.1*rho`, `emin = 0.1*rho = emax/11`.

**Smoother choice**: Chebyshev/Jacobi damps optimally when you have a decent and conservative estimate of the max eigenvalue of `D⁻¹A`. When such an estimate is unavailable or unreliable, Richardson with Jacobi-L1 (`solver=JACOBI_L1`) is more robust — it requires no eigenvalue estimate and is unconditionally stable, at the cost of slightly slower convergence.

## Bug Fixes

### Chebyshev Solver: Zero Initial Guess

Fixed a bug in `src/solvers/cheb_solver.cu` where `solve_init()` did not zero the output vector when `xIsZero=true`. This caused PCG to diverge when using Chebyshev as a smoother inside an AMG preconditioner, because stale data from the previous PCG iteration accumulated via `x += gamma*p`.

## Performance Comparison

All tests: PCG + Chebyshev(1)+Jacobi smoother, V(1,1) cycle, rtol=1e-8, unpreconditioned residual norm.
MIS-2 implicit aggregation with `aggressive_levels=1`. Run on NERSC Perlmutter (NVIDIA A100 40GB).

**Work Complexity** = CG Iterations × Operator Complexity (total floating-point work relative to a single MatVec).

### 2D Q1 Poisson (5-point stencil, ~1M DOFs)

| Solver | Threshold | Grid Cx | Op Cx | CG Iters | Work Cx |
|--------|:---------:|:-------:|:-----:|:--------:|:-------:|
| PETSc GAMG | 0.0 | 1.091 | 1.149 | 13 | **14.9** |
| PETSc GAMG | 0.02 | 1.095 | 1.174 | 13 | **15.3** |
| AMGx-SA | 0.0 | 1.166 | 1.436 | 20 | **28.7** |
| AMGx-SA | 0.02 | 1.171 | 1.496 | 19 | **28.4** |

### 3D Q2 Poisson (hex FEM, 60 nnz/row, ~250K DOFs)

| Solver | Threshold | Grid Cx | Op Cx | CG Iters | Work Cx |
|--------|:---------:|:-------:|:-----:|:--------:|:-------:|
| PETSc GAMG | 0.02 | 1.027 | 1.094 | 16 | **17.5** |
| PETSc GAMG | 0.05 | 1.074 | 1.963 | 13 | **25.5** |
| AMGx-SA | 0.02 | 1.027 | 1.096 | 17 | **18.6** |
| AMGx-SA | 0.03 | 1.044 | 1.267 | 16 | **20.3** |
| AMGx-SA | 0.05 | 1.466 | 9.797 | 12 | **117.6** |
| AMGx-SA | 0.0 | — | — | crash | — |

### Analysis

**2D (5 nnz/row):** PETSc GAMG achieves ~2× lower work complexity than AMGx-SA. The gap is primarily in operator complexity (1.15 vs 1.44) — PETSc produces sparser coarse operators. Both solvers are insensitive to threshold on this problem since the 5-point stencil has uniform edge weights.

**3D Q2 (60 nnz/row):** AMGx-SA with `threshold=0.02` nearly matches PETSc GAMG (Work Cx 18.6 vs 17.5, within 6%). The strength threshold is critical here — it controls which edges participate in the MIS-2 distance computation, directly affecting aggregate size and coarse operator density.

**Threshold sensitivity on 3D Q2:**
- `threshold=0.02`: Optimal for AMGx-SA. Produces ~5,600 aggregates (avg 44), matching PETSc's coarsening ratio.
- `threshold=0.03`: Slightly more aggregates (8,457), higher Op Cx but still reasonable.
- `threshold=0.05`: Catastrophic for AMGx-SA — creates 102K tiny aggregates (avg 2.4), exploding Op Cx to 9.8. PETSc handles this better (Op Cx 1.96) because its square graph approach naturally limits coarse operator fill.
- `threshold=0.0`: Crashes on 3D Q2 (see Known Issues below).

### Known Issues

**threshold=0 crashes on 3D Q2 (60 nnz/row):** With no strength thresholding, the SA prolongation smoothing step `P = (I - ω·D⁻¹·A)·P_tent` creates extremely dense rows when A has 60 nnz/row and aggregates are large (>50 nodes). The SpGEMM for P overflows GPU memory or hits an out-of-bounds error. This requires further investigation in the SA prolongation code — possible mitigations include:
1. Truncating weak entries in P after smoothing (as PETSc does with `-pc_gamg_sa_nsmooths_filter`)
2. Limiting aggregate size when threshold=0
3. Using unsmoothed aggregation (P = P_tent) as fallback for dense problems

## Validation

The SA-AMG implementation has been cross-validated against PETSc GAMG using identical aggregates (imported from PETSc). All solver/smoother combinations produce **identical iteration counts** when using the same aggregates, confirming correctness of the SA prolongator smoothing, Galerkin coarse operator construction, and Chebyshev smoother.

The code runs clean under `compute-sanitizer --tool memcheck` with zero memory errors on all test cases (2D and 3D, single-GPU).

## Configuration Example

```json
{
  "config_version": 2,
  "solver": {
    "solver": "PCG",
    "preconditioner": {
      "algorithm": "AGGREGATION",
      "solver": "AMG",
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
      "presweeps": 1,
      "postsweeps": 1,
      "selector": "MIS",
      "mis_k": 2,
      "mis2_algorithm": 1,
      "aggressive_levels": 1,
      "coarse_solver": "DENSE_LU_SOLVER",
      "min_coarse_rows": 10,
      "max_levels": 20,
      "cycle": "V"
    },
    "convergence": "RELATIVE_INI",
    "tolerance": 1e-8,
    "norm": "L2"
  }
}
```

### MIS Selector Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `mis_k` | 1 | MIS distance (1=standard, 2=aggressive coarsening) |
| `mis2_algorithm` | 0 | MIS-2 algorithm: 0=Galerkin loop, 1=implicit |
| `aggressive_levels` | 1 | Number of levels to use mis_k (0=all, N=first N levels) |
| `strength_threshold` | 0.0 | Drop edges with normalized weight < threshold (absolute comparison on \|a_ij\|/\|a_ii\|; values 0–0.05 are typical; 0.02 recommended for Q2 FEM) |
| `merge_singletons` | 1 | Merge isolated nodes into neighbor aggregates |
| `max_aggregate_size` | 0 | Max aggregate size for quality refinement (0=disabled) |
| `refine_threshold` | 0.0 | Weak edge threshold fraction for refinement (0.05 recommended) |

## Test Drivers

| File | Description |
|------|-------------|
| `examples/test_multilevel_cheby.c` | PCG + Chebyshev(1)+Jacobi SA-AMG, multi-level |
| `examples/test_multilevel_jacobi_l1.c` | PCG + Jacobi-L1 SA-AMG, multi-level |
| `examples/test_cheby_pcg_sa.c` | PCG + Chebyshev SA-AMG (2-level, validation) |
| `examples/test_cheby_sa.c` | Richardson + Chebyshev SA-AMG (2-level, validation) |
| `examples/test_cg_sa.c` | PCG + Jacobi-L1 SA-AMG (2-level, validation) |
| `examples/test_mg_diag.c` | Richardson + Jacobi-L1 SA-AMG (2-level, validation) |
| `examples/gen_poisson2d.py` | Generate 2D Poisson matrices in MatrixMarket format |

## Architecture Notes

### Differences from Original AMGx

| Feature | Original AMGx | AMGx-SA |
|---------|--------------|---------|
| Aggregation | Handshake pairwise matching (SIZE_2/4/8) | MIS-based (MIS-1, MIS-2) |
| Prolongator | Tentative only (P = P_tent) | Smoothed (P = (I-ωD⁻¹A)P_tent) |
| Coarsening control | Fixed aggregate size | MIS distance + aggressive levels |
| Aggressive coarsening | Not available | MIS-2 (Galerkin or implicit) |

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
make -j8 amgxsh
```

Build the test drivers separately (they link against the shared library):
```bash
cc -o test_multilevel_cheby ../examples/test_multilevel_cheby.c \
   -I../include -L. -lamgxsh -lcudart -Wl,-rpath,$PWD
```

On Cray systems (e.g., NERSC Perlmutter), use the provided Makefile:
```bash
cd examples && make -f Makefile.cray test_multilevel_cheby
```

## Future Work

- **Block-size > 1 (systems/elasticity):** The SA framework supports block problems in principle, but block-size > 1 is not yet tested. This requires a rigid body mode API (near-null space with multiple vectors per node), utility functions for block tentative prolongator construction, and validation against PETSc GAMG's systems solver.
- **Multi-GPU (MPI):** The MIS-k selector includes halo exchange calls but has not been validated in multi-GPU configurations.
