# AMGx SA-AMG Extensions

This fork adds **Smoothed Aggregation (SA) AMG** capabilities to AMGx, bringing it closer to feature parity with PETSc's GAMG solver for GPU-accelerated algebraic multigrid.

## New Features

### SA Prolongator Smoothing

The core addition is the SA prolongator smoothing step that transforms a tentative prolongator P_tent into a smoothed prolongator P:

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

### MIS-k Aggregation Selector

AMGx's MIS (Maximal Independent Set) selector extended with distance-k support:

- `"selector": "MIS"` with `"mis_k": 2` for MIS-2 (aggressive) coarsening
- `"aggressive_levels": N` controls how many levels use MIS-2 vs standard MIS-1
- Produces well-shaped aggregates suitable for SA prolongator smoothing

**Key file:** `src/aggregation/selectors/mis_selector.cu`

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

## Bug Fixes

### Chebyshev Solver: Zero Initial Guess

Fixed a bug in `src/solvers/cheb_solver.cu` where `solve_init()` did not zero the output vector when `xIsZero=true`. This caused PCG to diverge when using Chebyshev as a smoother inside an AMG preconditioner, because stale data from the previous PCG iteration accumulated via `x += gamma*p`.

## Validation

The SA-AMG implementation has been cross-validated against PETSc GAMG using identical aggregates (imported from PETSc). All solver/smoother combinations produce **identical iteration counts**:

| Configuration | Iterations |
|---------------|:----------:|
| Richardson + Jacobi-L1 | 71 |
| CG + Jacobi-L1 | 22 |
| Richardson + Chebyshev(1)+Jacobi | 42 |
| CG + Chebyshev(1)+Jacobi | 17 |

## Performance Comparison (1000×1000 2D Poisson, 1M DOFs)

| Solver | Agg Levels | Method | CG Iters | Grid Cx | Op Cx |
|--------|:---:|--------|:---:|:---:|:---:|
| PETSc GAMG | 1 | Square graph | 15 | 1.166 | 1.430 |
| PETSc GAMG | 1 | MIS-2 | 17 | 1.176 | 1.493 |
| AMGx (this fork) | 1 | MIS-2 | 27 | 1.124 | 1.337 |
| AMGx (this fork) | 2 | MIS-2 | 39 | 1.110 | 1.251 |

AMGx achieves lower grid/operator complexity but higher iteration counts due to more aggressive coarsening on the finest level. Improving aggregation quality is the focus of ongoing MIS-k development.

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
