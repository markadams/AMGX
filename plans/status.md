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

### ✅ GAMG Aggregate Import (validation tool, now removed)
- AMGx read aggregate file via `AMGX_IMPORT_AGGREGATES` env var
- File format: header lines + one aggregate ID per line (10,000 nodes)
- Enabled exact comparison with PETSc GAMG
- **Removed after validation complete** (preserved in git history)

### ✅ PETSc Aggregate Export (stashed)
- Added to `~/petsc/src/ksp/pc/impls/gamg/agg.c`
- Triggered by `PETSC_EXPORT_AGGREGATES` env var
- Extracts aggregate IDs from P_tent column indices
- **Stashed** via `git stash save "GAMG aggregate export for AMGx validation"`

### ✅ Chebyshev Bug Fix
- `cheb_solver.cu` `solve_init()`: added `fill(x, 0)` when `xIsZero=true`
- Callers (e.g. PCG) pass `xIsZero=true` but may not have zeroed the buffer
- Without this fix, `solve_iteration`'s "x += gamma*p" accumulates onto stale data
- This is a real bug fix, kept in production code

### ✅ Validation: Richardson + Jacobi-L1 (71 iterations, PETSc = AMGx)
- See `plans/testing_setup.md` for complete configuration
- Identical: A (49,600 nnz), P_tent (10,000 nnz), P (23,705 nnz), Ac (15,049 nnz)
- Both converge in 71 Richardson iterations to rtol=1e-8

### ✅ Validation: CG + Jacobi-L1 (22 iterations, PETSc = AMGx)
- Same SA-AMG preconditioner (Jacobi-L1, 1 sweep, imported GAMG aggregates)
- Outer solver changed from Richardson to CG (PCG in AMGx, `-ksp_type cg` in PETSc)
- Both converge in 22 CG iterations to rtol=1e-8
- Test driver: `examples/test_cg_sa.c`; script: `scripts/run_cg_compare.sh`

### ✅ Validation: Richardson + Chebyshev(1)+Jacobi (42 iterations, PETSc = AMGx)
- Smoother changed from Jacobi-L1 to Chebyshev(1)+Jacobi, V(1,1) cycle
- Outer solver: Richardson (standalone AMG), `-ksp_norm_type unpreconditioned`
- AMGx config: `chebyshev_lambda_estimate_mode=4`, `chebyshev_lmin_denom=11`,
  `BLOCK_JACOBI` with `relaxation_factor=1.0`
- Key insight: AMGx `lmin_denom=11` matches PETSc's esteig `[0, 0.1, 0, 1.1]`
  because PETSc uses `emin = 0.1*rho` while AMGx uses `lmin = lmax/denom = 1.1*rho/11 = 0.1*rho`
- Both converge in 42 iterations to rtol=1e-8
- Test driver: `examples/test_cheby_sa.c`; script: `scripts/run_cheby_compare.sh`

### ✅ Validation: CG + Chebyshev(1)+Jacobi (17 iterations, PETSc = AMGx)
- Outer solver: CG (PCG in AMGx, `-ksp_type cg` in PETSc)
- Smoother: Chebyshev(1)+Jacobi, V(1,1) cycle
- Required Chebyshev bug fix (zero x in solve_init when xIsZero=true)
- Both converge in 17 CG iterations to rtol=1e-8
- Test driver: `examples/test_cheby_pcg_sa.c`; script: `scripts/run_pcg_test.sh`

## Current Phase: MIS-k Development

### ✅ Native MIS Aggregation Baseline (2026-05-11)
- All solvers converge with AMGx native MIS-2 aggregation (no GAMG import)
- Iteration counts (native MIS-2 vs GAMG-imported):

| Test | Solver | Smoother | Native MIS-2 | GAMG Import |
|------|--------|----------|:------------:|:-----------:|
| test_mg_diag | Richardson | Jacobi-L1 | **136** | 71 |
| test_cg_sa | CG | Jacobi-L1 | **30** | 22 |
| test_cheby_sa | Richardson | Chebyshev(1)+Jacobi | **82** | 42 |
| test_cheby_pcg_sa | CG | Chebyshev(1)+Jacobi | **23** | 17 |

- Higher iteration counts expected: MIS-2 produces different (larger) aggregates than GAMG
- All converge to rtol=1e-8, confirming SA-AMG implementation is correct
- These are the baseline for MIS-k development

### 🔲 MIS-k MPI-Parallel Implementation
- See `plans/mis_k_mpi_parallel_plan.md`
- Multi-level Galerkin approach: R_out = MIS-1(A^k) composed k times
- Requires `exchange_halo` in MIS-1 loop for MPI correctness

### 🔲 Production Readiness
- Reduce `max_iter` in `estimateSADampingFactor` from 100 to a practical value (20-30)
  that balances accuracy vs cost for typical problems
- Test with block_size > 1 (elasticity problems)
- Test multi-level hierarchy (3+ levels)

### 🔲 Performance
- Power iteration cost: 100 iterations of D^{-1}*A*x is expensive for large problems
- Consider Chebyshev-based rho estimate (fewer iterations needed)
- SpGEMM uses AMGx's hash-based multiply (confirmed correct, well-optimized)

## Key Files

| Component | File |
|-----------|------|
| SA smoother | `src/aggregation/aggregation_amg_level.cu` |
| SA header | `include/aggregation/aggregation_amg_level.h` |
| Power iteration | `estimateSADampingFactor()` in the .cu file |
| Build SA kernel | `build_SA_smoother_kernel` (scalar), `build_SA_smoother_block_kernel` (block) |
| Chebyshev fix | `src/solvers/cheb_solver.cu` (solve_init) |
| Test drivers | `examples/test_mg_diag.c`, `test_cg_sa.c`, `test_cheby_sa.c`, `test_cheby_pcg_sa.c` |
| Comparison scripts | `scripts/run_mg_diag_compare.sh`, `run_cg_compare.sh`, `run_cheby_compare.sh` |
| Verification | `scripts/verify_sa_matrices.py` |
| AMGx patch tool | `scripts/patch_amgx_mtx.py` |

## Git History

1. **First commit** (with debugging): SA-AMG validation complete, includes all diagnostic code
2. **Second commit** (clean): Removed GAMG aggregate import, diagnostic printf; kept SA-AMG + Chebyshev fix
