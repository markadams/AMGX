# AMGx SA-AMG Project Status

**Date**: 2026-05-11
**Phase**: Non-aggregation validation COMPLETE — transitioning to MIS-k development

## Completed Milestones

### ✅ SA Smoothed Aggregation (Serial)
- `smoothProlongator()` implemented: builds S = I - omega*D^{-1}*A, computes P = S * P_tent
- Block-size > 1 expansion to scalar CSR
- P_tent construction via `buildTentativeProlongator()` with QR normalization
- Spectral radius estimation via power iteration (`estimateSADampingFactor`)
- Omega = (4/3) / rho(D^{-1}A) damping

### ✅ Chebyshev Bug Fix
- `cheb_solver.cu` `solve_init()`: added `fill(x, 0)` when `xIsZero=true`
- Callers (e.g. PCG) pass `xIsZero=true` but may not have zeroed the buffer
- Without this fix, `solve_iteration`'s "x += gamma*p" accumulates onto stale data

### ✅ Cross-Validation with PETSc GAMG (100×100 Poisson, imported aggregates)
- All 4 solver/smoother combinations match PETSc exactly:
  - Richardson + Jacobi-L1: 71 iterations
  - CG + Jacobi-L1: 22 iterations
  - Richardson + Chebyshev(1)+Jacobi: 42 iterations
  - CG + Chebyshev(1)+Jacobi: 17 iterations
- Validation code (aggregate import/export, diagnostics) removed; preserved in git history

## Current Phase: MIS-k Development

### Multi-Level Baseline (400×400 Poisson, 160,000 DOFs, CG + Chebyshev(1)+Jacobi)

| Config | Agg Levels | Method | L0 | L1 | L2 | L3 | L4 | L5 | Iters | Grid Cx | Op Cx |
|--------|:---:|--------|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| PETSc | 1 | Square graph | 160,000 | 22,463 | 3,836 | 340 | 27 | 4 | **20** | 1.167 | 1.428 |
| PETSc | 1 | MIS-2 | 160,000 | 23,509 | 4,295 | 382 | 25 | 3 | **24** | 1.176 | 1.486 |
| PETSc | 2 | Square graph | 160,000 | 22,463 | 1,348 | 181 | 17 | 3 | **24** | 1.150 | 1.333 |
| PETSc | 2 | MIS-2 | 160,000 | 23,509 | 1,420 | 205 | 16 | 2 | **30** | 1.157 | 1.364 |
| **AMGx** | **all** | **MIS-2** | **160,000** | **16,143** | **2,736** | **215** | **17** | — | **27** | **1.119** | **1.315** |

Key observations:
- AMGx MIS-2 all levels (27 iters, Cx=1.119) is the baseline for MIS-k development
- PETSc's best (20 iters) uses square-graph aggressive on finest + standard MIS on coarser
- AMGx coarsens more aggressively on L0→L1 (10:1 vs PETSc's 7:1)
- Goal: improve AMGx aggregation quality to approach PETSc's 20-iteration convergence

### 🔲 MIS-k MPI-Parallel Implementation
- See `plans/mis_k_mpi_parallel_plan.md`
- Multi-level Galerkin approach: R_out = MIS-1(A^k) composed k times
- Requires `exchange_halo` in MIS-1 loop for MPI correctness

### 🔲 Production Readiness
- Reduce `max_iter` in `estimateSADampingFactor` from 100 to a practical value (20-30)
- Test with block_size > 1 (elasticity problems)
- Test multi-level hierarchy (3+ levels) — done (5 levels on 400×400)

## Key Files

| Component | File |
|-----------|------|
| SA smoother | `src/aggregation/aggregation_amg_level.cu` |
| SA header | `include/aggregation/aggregation_amg_level.h` |
| Power iteration | `estimateSADampingFactor()` in the .cu file |
| Chebyshev fix | `src/solvers/cheb_solver.cu` (solve_init) |
| MIS selector | `src/aggregation/selectors/mis_selector.cu` |
| Test drivers | `examples/test_multilevel_cheby.c`, `test_multilevel_jacobi_l1.c` |
| Matrix generator | `examples/gen_poisson2d.py` |
| MIS-k plan | `plans/mis_k_mpi_parallel_plan.md` |
