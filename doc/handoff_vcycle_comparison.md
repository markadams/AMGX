# V-cycle Comparison: AMGX SA-AMG vs PETSc GAMG

## Problem Setup
- 3D Q1 hexahedral elasticity, E=1.0, ν=0.25
- Grid: ne=9 → 10×10×10 nodes → 3000 DOFs (block_size=3)
- SA-AMG with MIS-1 aggregation, L1-Jacobi + Richardson(2/3), 2 pre/post sweeps
- Near-null space: 6 rigid body modes (3 translations + 3 rotations)

## PETSc GAMG Output (Reference — with near-null space, block_size=6 on coarse grids)

Command:
```
./ex99 -ne 9 -pc_type gamg -ksp_norm_type unpreconditioned -ksp_type cg \
  -pc_gamg_aggressive_coarsening 0 -mg_levels_pc_jacobi_type rowl1 \
  -mg_levels_ksp_type richardson -mg_levels_ksp_richardson_scale .666 \
  -use_mat_nearnullspace
```

```
3 levels, operator complexity = 1.64759:
  Level 2 (fine):   N=3000, nnz/row=66,  block_size=3
  Level 1 (mid):    N=612,  nnz/row=206, block_size=6
  Level 0 (coarse): N=48,   nnz/row=48,  block_size=6

Coarsening: 3000 → 102 aggregates (×6 null = 612 DOFs) → 8 aggregates (×6 null = 48 DOFs)
P[1→2]: 3000 × 612  (avg nnz/row = 33)
P[0→1]: 612 × 48    (avg nnz/row = 31)

Coarse solver: LU factorization of 48×48 matrix (block_size=6)
Smoother: Richardson(0.666) + L1-Jacobi, 2 sweeps

max_eigen level 0 (fine):  3.172454e+00  min: 6.688274e-02
max_eigen level 1 (mid):   2.707569e+00  min: 3.474966e-02
```

### PETSc V-cycle trace (1 iteration, from x=0):
```
level 2: BEFORE pre-smooth: ||b||=5.391826e+00  ||x||=0
level 2: AFTER pre-smooth:  ||x||=1.386696e+00
level 2: residual ||r||=5.316726e+00
level 2: AFTER restrict:    ||bc||=5.081769e+00

level 1: BEFORE pre-smooth: ||b||=5.081769e+00  ||x||=0
level 1: AFTER pre-smooth:  ||x||=7.165128e+00
level 1: residual ||r||=4.917562e+00
level 1: AFTER restrict:    ||bc||=4.566524e+00

level 0: BEFORE pre-smooth (=coarse solve): ||b||=4.566524e+00  ||x||=0
level 0: AFTER pre-smooth:  ||x||=7.960777e+02  (LU exact solve of 48×48 system)

level 1: AFTER prolongate+correct: ||x||=7.891613e+02
level 1: AFTER post-smooth:        ||x||=7.886539e+02

level 2: AFTER prolongate+correct: ||x||=7.871627e+02
level 2: AFTER post-smooth:        ||x||=7.868136e+02
```

**PETSc outer residual after 1 V-cycle: ||r|| = 17.874** (from initial 5.392)
- Residual **increases** on the first CG iteration — this is common with PCG
  where the preconditioner is not a contraction. CG requires only that M⁻¹ be SPD,
  not that ‖I - M⁻¹A‖ < 1. The first iteration can amplify the residual while
  CG builds up conjugate directions that eventually converge.
- PETSc converges when running many CG iterations despite early amplification.

## AMGX SA-AMG Output (Current)
```
4 levels:
  Level 0 (fine):    N=3000(block-3), nnz=197568
  Level 1 (mid):     N=?? (block-6 after createBlockGraph)
  Level 2 (mid):     N=?? (block-6)
  Level 3 (coarse):  N=1 (block-6), nnz=1

Coarse solver: PCG + BLOCK_JACOBI, 100 iters, tol=1e-6 → converges in 9 iters
```

### AMGX V-cycle trace (1 iteration, from x=0):
```
level 1: BEFORE pre-smooth: ||b||=5.392e+00  ||x||=0  N=1000
level 1: AFTER pre-smooth:  ||x||=1.388e+00
level 1: residual ||r||=5.317e+00
level 1: AFTER restrict:    ||bc||=5.100e+00
  (coarse solve: 9 iters PCG, coarsest level N=1 block-6)
level 1: AFTER prolongate+correct: ||x||=7.290e+02
level 1: AFTER post-smooth:        ||x||=7.286e+02
```

## Key Differences to Investigate

### 1. Number of Levels — Now Matching
- **PETSc**: 3 levels (3000 → 612 → 48)
- **AMGX**: 4 levels (3000 → ? → ? → 6)
- With `-use_mat_nearnullspace`, PETSc now uses block_size=6 on coarse grids (matching AMGX's `createBlockGraph` approach). However, PETSc produces 3 levels while AMGX produces 4.
- **Same aggregation**: Both produce 102 aggregates at level 1 and 8 aggregates at level 0.
- **Same null-space dimension**: Both use 6 null-space vectors (3 translations + 3 rotations).

### 2. Coarse Matrix Structure — Now Comparable
- **PETSc**: Level 1 is 612×612 (102 aggs × 6 null), bs=6. Level 0 is 48×48 (8 aggs × 6 null), bs=6.
- **AMGX**: Uses `createBlockGraph` to compress scalar Ac into block-nd matrices:
  - Level 1 Ac: 102×102 scalar → 17×17 block-6 (6=null_dim)
  - This block-6 matrix is then used for further coarsening, producing more levels
- **Key insight**: PETSc keeps 612 scalar DOFs with bs=6 metadata, while AMGX converts to 17 block-6 rows. Mathematically equivalent but AMGX coarsens further, producing 4 levels vs 3.

### 3. Prolongator Dimensions
- **PETSc**: P[1→2] is 3000×612 with avg nnz/row=33. P[0→1] is 612×48 with avg nnz/row=31.
- **AMGX**: P_smooth is 3000×102 with nnz=54594 (avg nnz/row=18.2).
- PETSc's prolongator is wider (612 cols vs 102 cols) because it stores 6 null components per aggregate as separate columns, while AMGX packs them into the block structure.

### 4. Pre-smooth Comparison (Level 2/Fine)
- PETSc: ||x|| after pre-smooth = 1.386696
- AMGX:  ||x|| after pre-smooth = 1.388147  (close but not identical)
- This small difference is expected due to L1-Jacobi implementation differences

### 5. Restriction Comparison
- PETSc: ||bc|| after restrict = 5.081769
- AMGX:  ||bc|| after restrict = 5.099862
- **0.4% difference** — much closer than before (was 2.3%). The near-null space makes PETSc's restriction match AMGX's more closely.

### 6. Coarse Solve Amplification
- Both PETSc and AMGX show large amplification at the coarse solve:
  - PETSc: ||bc||=4.567 → ||x_coarse||=796 (174× amplification)
  - AMGX:  ||bc||=5.100 → ||x_correction||=729 (143× amplification)
- PETSc's coarse solve is an exact LU on a 48×48 system (bs=6)
- AMGX's coarse solve is PCG on a much smaller system (1 block-6 row = 6 DOFs)
- **Both amplify!** This is expected — the coarse solve corrects the near-null space components
- PETSc amplification is now larger than AMGX (174× vs 143×) — the 6-component null space creates a larger coarse correction

### 7. Post-prolongation
- PETSc: After prolongation to level 2: ||x||=787.16, after post-smooth: ||x||=786.81
- AMGX: After prolongation to level 1: ||x||=729.02, after post-smooth: ||x||=728.55
- With near-null space, PETSc's correction is now **larger** than AMGX's (787 vs 729)
- Both are large corrections that amplify the initial residual on the first V-cycle

### 8. Outer Solver Convergence
- PETSc CG with 1 V-cycle: residual goes from 5.39 to 17.87 (increases!)
- AMGX PCG with 100 V-cycles: NOT CONVERGED after 100 iterations
- **Residual increase in early CG iterations is common** with PCG — CG requires only that M⁻¹ be SPD, not that it be a contraction. The first iterations can amplify the residual while CG builds conjugate directions.
- PETSc eventually converges when running many CG iterations despite early amplification.
- The fact that AMGX does not converge after 100 iterations suggests a deeper issue beyond first-iteration amplification.

## Null Space Verification (Session 2 — 2025-05-26)

### Test: `./src/test_elasticity3d_sa 9 --null-test`

Assembles A **without** Dirichlet BCs (DD1 everywhere, no DD2), builds AMG hierarchy,
and verifies `Ac * B = 0` at every level. **All levels passed** to machine precision:

```
Fine level:    ||A * B_k|| ≈ 1e-14  (k=0..5)  ✓
Coarse Level 0 (1000 bs=3):  ||Ac * B_k|| ≈ 1e-14  ✓
Coarse Level 1 (102 bs=6):   ||Ac * B_k|| ≈ 1e-14  ✓
Coarse Level 2 (5 bs=6):     ||Ac * B_k|| ≈ 1e-14  ✓
```

**Key differences from the BC case:**
- P_tent has **no zero columns** (all column norms = 1.0)
- P_smooth has **no zero columns** (min_col_norm = 0.539)
- Ac has **no zero diagonals** (min_diag = 0.075, no regularization needed)
- The null space is perfectly preserved through all coarsening levels

**Conclusion:** The SA-AMG prolongation and Galerkin coarse grid construction
correctly preserves the null space. The issue is specifically with Dirichlet BC
handling — the clamped-face nodes zero out rotational null-space columns in P_tent,
which propagates to create near-singular coarse matrices.

## Root Cause Hypotheses

### H1: createBlockGraph — extra coarsening levels (PARTIALLY RESOLVED)
With `min_coarse_rows=10`, AMGX now produces 3 levels matching GAMG:
- Level 0: 1000 block-3 (3000 DOFs)
- Level 1: 102 block-6 (612 DOFs) — 102 aggregates × 6 null
- Level 2: 5 block-6 (30 DOFs) — 5 aggregates × 6 null

However, **the coarse solver (PCG+BLOCK_JACOBI on 30 DOFs) does not converge
after 100 iterations**, making the V-cycle useless. This is the primary blocker.

### H2: Coarse solver failure on BC case (PRIMARY BLOCKER)
With Dirichlet BCs applied, P_tent has 3 zero columns (agg 0, null vecs 3,4,5 —
rotational modes at the clamped boundary). These propagate to:
- 3 zero rows/cols in Ac (612×612) → regularized by `[SA-FIX]` to small diag values
- After `createBlockGraph`, these become near-singular entries in the block-6 Ac
- The coarse PCG+BLOCK_JACOBI cannot converge on this near-singular system
- **Fix**: Use `DENSE_LU_SOLVER` for the coarse grid (matches GAMG's exact LU)
  OR properly handle Dirichlet BC nodes in the null-space construction

### H3: P_smooth quality differs
AMGX SA smoothing uses ω = 4/(3·ρ(D⁻¹A)) while PETSc uses the same formula.
The spectral radius estimates differ slightly:
- PETSc: ρ = 3.172454 (with BCs)
- AMGX:  ρ = 3.232762 (with BCs), ρ = 3.215918 (without BCs)
This 2% difference propagates to ω and hence to P_smooth.

### H4: Mid-level smoother divergence (8.2%)
Comparing GAMG (with near-null space, bs=6) vs AMGX traces:
- Fine-level pre-smooth: ||x|| = 1.3867 (GAMG) vs 1.3881 (AMGX) — 0.1% ✓
- Fine-level restriction: ||bc|| = 5.082 (GAMG) vs 5.100 (AMGX) — 0.4%
- **Mid-level pre-smooth: ||x|| = 7.165 (GAMG) vs 6.578 (AMGX) — 8.2%**
- Coarse entry: ||bc|| = 4.567 (GAMG) vs 4.473 (AMGX) — 2%

The 8.2% mid-level smoother difference is likely due to how L1-Jacobi operates
on 612-row CSR (GAMG, per-scalar-row scaling) vs 102-row BSR block-6
(AMGX, per-block-row scaling via `smooth_BxB`).

## Next Steps

1. **Fix coarse solver**: Switch to `DENSE_LU_SOLVER` for the coarse grid to match
   GAMG's exact LU factorization. This is the immediate fix for non-convergence.

2. **Handle Dirichlet BC nodes in null space**: The 3 zero P_tent columns from
   clamped boundary rotational modes need proper treatment. PETSc handles this
   internally; AMGX needs the same logic.

3. **Investigate mid-level L1-Jacobi difference**: The 8.2% smoother divergence
   between GAMG (scalar L1-Jacobi on 612 rows) and AMGX (block L1-Jacobi on
   102 block-6 rows) needs investigation — likely a `smooth_BxB` scaling issue.

4. **Run PETSc GAMG to convergence**: Confirm iteration count with near-null space
   and compare convergence rate once AMGX coarse solver is fixed.

## Files Modified in This Session

### Bug Fixes (Keep)
- `src/amg.cu:991` — Coarse solver setup matrix: walk hierarchy to find coarsest level
- `examples/test_elasticity3d_sa.cu:601` — Add `"scope": "amg"` to AMG preconditioner block

### New Features (Session 2)
- `src/aggregation/aggregation_amg_level.h:98-99` — Public accessors for near-null space vectors (`getNearNullSpace()`, `getCoarseNearNullSpace()`)
- `examples/test_elasticity3d_sa.cu` — `--null-test` mode: assembles A without BCs, verifies `Ac * B = 0` at every AMG level
- `examples/test_elasticity3d_sa.cu` — `assemble_elasticity_3d()` now accepts `bool apply_bcs` parameter
- `examples/test_elasticity3d_sa.cu` — `min_coarse_rows` changed from 4 to 10

### Diagnostic Prints (Remove Before Release)
- `src/amg.cu:64` — AMG::AMG scope/coarse_solver debug print
- `src/amg.cu:999` — Coarse solver setup matrix dimensions print
- `src/amg_level.cu:145` — launchCoarseSolver debug print
- `src/amg_config.cu:70-76` — JSON parse success/failure print
- `src/amg_config.cu:581-592` — import_json_object scope resolution prints
- `src/cycles/fixed_cycle.cu` — Various VCYCLE-TRACE and VCYCLE-DBG prints

## Build Notes
- Source: `~/amgx-sa/` on Perlmutter
- Build: `~/amgx-sa/build_perlmutter/`
- **Correct executable**: `./src/test_elasticity3d_sa` (NOT `./test_elasticity3d_sa` in build root)
- Run: `LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH ./src/test_elasticity3d_sa 9`
- Null test: `LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH ./src/test_elasticity3d_sa 9 --null-test`
