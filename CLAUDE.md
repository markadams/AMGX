# CLAUDE.md — amgx-sa Development Notes

## Repository Overview

This is a fork of [NVIDIA/AMGX](https://github.com/NVIDIA/AMGX) with an
implementation of **Smoothed Aggregation (SA) AMG** added on top of the
existing aggregation AMG framework.

Remote: `git@gitlab.com:markadams4/amgx-sa.git`

---

## Build (Perlmutter @ NERSC)

```bash
cd ~/amgx-sa/build_perlmutter
make -j8 amgxsh
```

The build directory was configured with:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_ARCH="80" \
         -DAMGX_NO_RPATH=ON
```

---

## Verification Procedure (Validated 2026-05-11)

**Status**: ✅ AMGx SA matches PETSc GAMG exactly (71 iterations, identical P nnz)

Full details: see `plans/testing_setup.md`

### Quick Run (Perlmutter)

```bash
salloc -N 1 -C gpu -q interactive -t 00:30:00 -A m1516_g
cd ~/amgx-sa/build_perlmutter
bash ~/amgx-sa/scripts/run_mg_diag_compare.sh
```

This runs PETSc (exports aggregates) then AMGx (imports same aggregates), comparing:
- 100×100 2D Poisson, MIS-k(2) aggregation, 1429 coarse DOFs
- Richardson + Jacobi-L1 smoother, 2-level V-cycle, coarse LU
- SA smoothing: omega = (4/3)/rho(D^{-1}A) with 100 power iterations

### Key Result

| Metric | PETSc | AMGx |
|--------|-------|------|
| Iterations | 71 | 71 |
| P nnz | 23,705 | 23,705 |
| P_tent ‖·‖_F | 3.780211634287129e+01 | 3.780211634287129e+01 |

### Matrix Verification

```bash
module load python
python3 ~/amgx-sa/scripts/verify_sa_matrices.py .
```

---

## Test

```bash
cd ~/amgx-sa/build_perlmutter
# Interactive GPU node required:
salloc -N1 -C gpu --gpus=1 -A m1516_g --qos=debug -t 30:00
./test_mg_diag poisson2d.mtx
```

Test binary source: `examples/test_mg_diag.c`

---

## SA AMG Implementation

### Key Files

| File | Purpose |
|------|---------|
| `src/aggregation/aggregation_amg_level.cu` | Main SA path: `buildTentativeProlongator()`, `smoothProlongator()`, Galerkin product, SA→Chebyshev eigenvalue wiring |
| `src/aggregation/aggregation_amg_level.h` | Level class: `m_sa_rho`, `m_null_dim`, `m_P_tent`, `m_P_tent_T` |
| `src/aggregation/batched_qr.cu` | Batched QR for P_tent construction |
| `include/solvers/cheb_solver.h` | Chebyshev solver header; `setSAEigenvalue()` method |
| `src/solvers/cheb_solver.cu` | Chebyshev solver; `lambda_mode=4`; `chebyshev_lmin_denom` config param |
| `src/core.cu` | Config parameter registration |
| `src/amg.cu` | AMG hierarchy setup; `min_coarse_rows` stopping logic |
| `src/configs/AGGREGATION_SA_CHEBY.json` | SA config: Chebyshev(2)+Jacobi, MULTI_PAIRWISE, lambda_mode=4 |

### SA Path in `createCoarseMatrices()`

The SA path is gated by `m_null_dim > 0` (near-null space set) and
`sizeof(ValueTypeB) == sizeof(PODType)` (real-valued only, not complex):

1. **Aggregate ID compaction** — remaps non-contiguous aggregate IDs to
   contiguous 0..N-1 using thrust sort/unique/lower_bound.

2. **`buildTentativeProlongator()`** — calls `batched_qr()` to compute
   P_tent via Modified Gram-Schmidt QR on the near-null space vectors.

3. **`smoothProlongator()`** — builds `S = I - ω D⁻¹ A` explicitly, then
   computes `P_smooth = S * P_tent` via SpGEMM. Damping factor
   `ω = 1.4 / ρ(D⁻¹A)` estimated by power iteration.
   - **block_size == 1**: S has same sparsity as A (scalar CSR).
   - **block_size > 1**: A is expanded from block-CSR to scalar CSR using
     `build_SA_smoother_block_rowlen_kernel` + `build_SA_smoother_block_kernel`.
   - Stores `m_sa_rho = ρ(D⁻¹A)` for the Chebyshev smoother.

4. **SA→Chebyshev wiring** — after `smoothProlongator()`, `dynamic_cast`s
   the smoother to `Chebyshev_Solver*` and calls `setSAEigenvalue(m_sa_rho)`.

5. **Galerkin product** — `Ac = P^T A P` via
   `CSR_Multiply::csr_galerkin_product`.

6. **Near-null space propagation** — coarse near-null space (R factors from
   QR) propagated to the next level via `next_agg->setNearNullSpace(...)`.

### MULTI_PAIRWISE Aggregation

The `MULTI_PAIRWISE` selector uses pairwise matching (heavy-edge matching)
to build aggregates. Key config parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aggregation_passes` | 1 | Number of pairwise matching passes. Each pass roughly doubles aggregate size. 3 passes → ~8× coarsening ratio. |
| `weight_formula` | 0 | Edge weight formula (0=default, 1=alternative) |
| `merge_singletons` | 1 | How to handle unaggregated nodes (0=leave, 1=merge with strongest, 2=merge with size limit) |
| `max_matching_iterations` | 15 | Max iterations per matching pass |
| `max_unassigned_percentage` | 0.05 | Stop matching when this fraction of nodes remain unassigned |
| `handshaking_phases` | 1 | 1 or 2 phase handshaking |
| `filter_weights` | 0 | Filter weak edges (0=no, 1=yes) |
| `filter_weights_alpha` | 0.25 | Threshold for weight filtering |

For SA verification, use `aggregation_passes=3` to get ~8× coarsening
(similar to GAMG with `aggressive_coarsening=1`).

### Chebyshev Smoother Eigenvalue Modes

`chebyshev_lambda_estimate_mode` (config, default 0):
- **0**: eigensolver for both lmin and lmax
- **1**: eigensolver for lmax; lmin = lmax / lmin_denom
- **2**: row-sum estimate for lmax; lmin = lmax / lmin_denom
- **3**: user-provided `cheby_max_lambda` / `cheby_min_lambda`
- **4**: reuse SA-computed `rho(D⁻¹A)` from `estimateSADampingFactor()`;
  lmax = rho × 1.1; lmin = lmax / lmin_denom. Matches PETSc GAMG
  `-pc_gamg_use_sa_esteig`.

`chebyshev_lmin_denom` (config, default 10.0):
- lmin = lmax / lmin_denom
- PETSc GAMG uses 10.0 (transform [0,0.1;0,1.1])

### Coarsening Control

`min_coarse_rows` in the JSON config is a **target** — coarsening continues
until a grid falls below the target, but that grid is **kept** (not thrown
away). The check at the top of the setup loop stops further coarsening.

### `[SA-VIEW]` Diagnostics

Permanent printfs in `aggregation_amg_level.cu`:
1. After Galerkin product — per-level `#eqs` and `avg_nnz/row`
2. In `estimateSADampingFactor()` — `rho(D^{-1}A)` and `omega`

`[Chebyshev]` printf in `cheb_solver.cu` `solver_setup()`:
- `level=#rows lambda_mode lmax lmin lmin_denom sa_eig_set`

### Fixed Bugs

1. **`estimateSADampingFactor`**: Changed `!A.hasProps(DIAG)` →
   `A.diag.size() == 0` so power iteration actually runs.

2. **Full SA smoothing via SpGEMM**: Replaced pattern-preserving kernel
   with explicit `S = I - ω D⁻¹ A` + SpGEMM.

3. **`build_agg_row_lists` off-by-one** (`batched_qr.cu`): Changed
   `thrust::exclusive_scan(..., offsets_ptr + 1)` →
   `thrust::exclusive_scan(..., offsets_ptr)`. Root cause of zero columns
   in P and SA divergence.

4. **`min_coarse_rows` stopping logic** (`amg.cu`): Changed from "don't
   create level if below target" to "create level, then stop". The grid
   that falls below the target is kept as the coarsest level.

---

## Verified Results

### 200×200 2D Poisson (40,000 DOFs, Perlmutter)

Test binary: `~/amgx-sa/build_perlmutter/test_sa_phase1`
Matrix: `~/amgx-sa/build_perlmutter/poisson2d_200.mtx`
Config: Chebyshev(2)+Jacobi, lambda_mode=4, lmin_denom=10, DENSE_LU coarse solver

#### PETSc GAMG reference (aggressive_coarsening=1, rtol=1e-5) — runs locally

| Level | #equations | nnz/row | ρ(D⁻¹A) | λ_max (target) | λ_min (target) |
|-------|-----------|---------|----------|----------------|----------------|
| 4 (finest) | 40000 | 5 | 1.974 | 2.172 | 0.197 |
| 3 | 5602 | 11 | 1.436 | 1.580 | 0.144 |
| 2 | 976 | 23 | 1.634 | 1.797 | 0.163 |
| 1 | 98 | 26 | 1.649 | 1.814 | 0.165 |
| 0 (coarsest) | 9 | 9 | — | — (LU) | — |

- **14 iterations**, rate 0.47, grid cx 1.17, op cx 1.42

#### AMGX SA, MULTI_PAIRWISE (3 passes), DENSE_LU (rtol=1e-5)

| Level | #equations | nnz/row | ρ(D⁻¹A) | λ_max (ρ×1.1) | λ_min (λ_max/10) |
|-------|-----------|---------|----------|----------------|-------------------|
| 0 (finest) | 40000 | 5.0 | 1.846 | 2.030 | 0.203 |
| 1 | 5213 | 11.6 | 1.480 | 1.627 | 0.163 |
| 2 | 650 | 20.6 | 1.504 | 1.654 | 0.165 |
| 3 (coarsest) | 79 | 28.5 | — | — (LU) | — |

- **23 iterations**, rate 0.59, grid cx 1.15, op cx 1.38

#### Comparison (200×200)

| | PETSc GAMG | AMGX SA |
|---|-----------|---------|
| Levels | 5 | 4 |
| Iterations | 14 | 23 |
| Conv. rate | 0.47 | 0.59 |
| Grid complexity | 1.17 | 1.15 |
| Operator complexity | 1.42 | 1.38 |
| Coarsest grid | 9 (LU) | 79 (LU) |
| All ρ < 2 | ✓ | ✓ |

---

### 400×400 2D Poisson (160,000 DOFs)

Matrix: `~/amgx-sa/build_perlmutter/poisson2d_400.mtx` (generated via `./examples/generate_poisson -p 5 400 400`)

#### PETSc GAMG reference (aggressive_coarsening=1, rtol=1e-5) — runs locally

```bash
cd ~/Codes/petsc/src/ksp/ksp/tutorials
./ex2 -m 400 -n 400 -ksp_type richardson -pc_type gamg \
      -pc_gamg_aggressive_coarsening 1 \
      -ksp_monitor -ksp_view -ksp_rtol 1e-5
```

| Level | #equations | nnz/row | ρ(D⁻¹A) | λ_max (target) | λ_min (target) |
|-------|-----------|---------|----------|----------------|----------------|
| 4 (finest) | 160,000 | 5 | 1.974 | 2.171 | 0.197 |
| 3 | 22,463 | 11 | 1.446 | 1.591 | 0.145 |
| 2 | 3,836 | 23 | 1.642 | 1.806 | 0.164 |
| 1 | 340 | 28 | 1.552 | 1.707 | 0.155 |
| 0 (coarsest) | 27 | 17 | — | — (LU) | — |

- **16 iterations**, grid cx 1.167, op cx 1.428

#### AMGX SA, MULTI_PAIRWISE (3 passes), DENSE_LU (rtol=1e-5)

| Level | #equations | nnz/row | ρ(D⁻¹A) | λ_max (ρ×1.1) | λ_min (λ_max/10) |
|-------|-----------|---------|----------|----------------|-------------------|
| 0 (finest) | 160,000 | 5.0 | 1.846 | 2.030 | 0.203 |
| 1 | 20,433 | 11.6 | 1.506 | 1.656 | 0.166 |
| 2 | 2,569 | 21.0 | 1.486 | 1.634 | 0.163 |
| 3 | 321 | 34.5 | 1.731 | 1.905 | 0.190 |
| 4 (coarsest) | 39 | — | — | — (LU) | — |

- **57 iterations**, rate 0.809, grid cx 1.146, op cx 1.380

#### AMGX SA, MIS-1 selector, DENSE_LU (rtol=1e-5) — Perlmutter, May 10 2026

| Level | #equations | nnz/row | ρ(D⁻¹A) | λ_max | λ_min |
|-------|-----------|---------|----------|-------|-------|
| 0 (finest) | 160,000 | 5.0 | 1.846 | 2.030 | 0.203 |
| 1 | 57,872 | 16.9 | 1.674 | 1.841 | 0.184 |
| 2 | 6,596 | 31.2 | 1.654 | 1.819 | 0.182 |
| 3 | 413 | 29.0 | 1.441 | 1.585 | 0.159 |
| 4 (coarsest) | 28 | — | — | — (LU) | — |

- **DID NOT CONVERGE** — hit 100-iteration limit, final residual 6.93 (tolerance ~4e-3)
- Avg convergence rate: **0.9603** (vs 0.809 for MULTI_PAIRWISE)
- Grid complexity: 1.406, Operator complexity: 2.496
- Coarsening ratio L0→L1: only **2.76×** (57,872/160,000) — far too fine
- Residual oscillates wildly (rates 0.30–3.16) — coarse-grid correction is destabilizing

**Root cause**: MIS-1 (distance-1 neighborhoods) on a 5-point stencil gives ~1/3 coarsening
ratio per level. Level 1 has 57,872 rows (36% of fine grid) vs MULTI_PAIRWISE's 20,433 (13%).
The prolongation/restriction operators from distance-1 aggregates are not smooth enough for
effective SA coarse-grid correction. Need MIS-2 or MIS-3 for adequate coarsening.

#### AMGX SA, MIS-2 selector (aggressive_levels=1), DENSE_LU (rtol=1e-5) — Perlmutter, May 10 2026

Config: `mis_k=2, aggressive_levels=1` (MIS-2 on level 0, MIS-1 on levels 1+)

| Level | #equations | nnz/row | ρ(D⁻¹A) | λ_max | λ_min |
|-------|-----------|---------|----------|-------|-------|
| 0 (finest) | 160,000 | 5.0 | 1.846 | 2.030 | 0.203 |
| 1 | 16,143 | 11.0 | 1.467 | 1.614 | 0.161 |
| 2 | 2,736 | — | — | — | — |
| 3 | 217 | — | — | — | — |
| 4 (coarsest) | 17 | — | — | — (LU) | — |

- **82 iterations**, rate **0.8637** — converges (was diverging without aggressive_levels)
- Grid complexity: ~1.11, Operator complexity: ~1.24

Aggregate size histograms:
```
[SA-AGG] L0: 160000 → 16143: min=1 avg=9.9 max=26 | 1:114 2:225 3:142 4:933 5-8:4823 9-16:8929 17-32:977
[SA-AGG] L1: 16143 → 2736:  min=1 avg=5.9 max=13 | 1:13 2:72 3:167 4:332 5-8:1924 9-16:228
[SA-AGG] L2: 2736 → 217:    min=2 avg=12.6 max=20
[SA-AGG] L3: 217 → 17:      min=2 avg=12.8 max=19
```

PETSc GAMG comparison (same problem, `-pc_gamg_aggressive_coarsening 1`):
```
[GAMG-AGG] L0: 160000 → 22463: min=1 avg=7.1 max=13 | 1:2219 2:1863 3:1759 4:1612 5-8:5714 9-16:9296
[GAMG-AGG] L1: 22463 → 3836:  min=1 avg=5.9 max=15
[GAMG-AGG] L2: 3836 → 340:    min=1 avg=11.3 max=31
[GAMG-AGG] L3: 340 → 27:      min=1 avg=12.6 max=36
16 iterations, rate ~0.49
```

**Key difference**: AMGX MIS-2 L0 max aggregate size = **26** (977 aggs of size 17-32),
PETSc GAMG L0 max = **13** (zero aggs >16). AMGX's parallel MIS with random hash weights
produces more variable aggregate sizes than PETSc's greedy sequential MIS.

#### Comparison (400×400)

| | PETSc GAMG | SIZE_4 | MULTI_PW | **MIS-2 (agg=1)** | MIS-2 (all) | MIS-1 |
|---|-----------|--------|----------|-------------------|-------------|-------|
| Levels | 5 | 7 | 5 | **5** | 4 | 5 |
| Iterations | **16** | 35 | 57 | **82** | >100 (DIV) | >100 |
| Conv. rate | 0.49 | 0.70 | 0.81 | **0.86** | 1.02 | 0.96 |
| L0→L1 ratio | 7.1× | 4.2× | 7.8× | **9.9×** | 9.9× | 2.8× |
| L0 max agg | 13 | 10 | 10 | **26** | 26 | 5 |
| Op cx | 1.43 | 2.70 | 1.38 | **1.24** | 1.23 | 2.50 |
| Status | ✓ | ✓ | ✓ | **✓** | DIVERGE | DNF |

---

## Remaining Work

- [x] Build on Perlmutter and run 200×200 verification
- [x] Test with MULTI_PAIRWISE selector — best match to PETSc GAMG
- [x] Fix coarse solver: DENSE_LU_SOLVER instead of JACOBI_L1
- [x] Fix min_coarse_rows stopping logic: keep grid that falls below target
- [x] Implement MIS-1 selector (Steps 1+2 of mis_k_mpi_parallel_plan.md)
- [x] Verify MIS-1 on 400×400: DNF (rate 0.960) — coarsening too fine (2.76×)
- [x] Implement MIS-2 (mis_k=2) via Galerkin coarsening loop
- [x] Test MIS-2 on 400×400: diverges without aggressive_levels
- [x] Add aggressive_levels config: MIS-2 on level 0, MIS-1 on rest → **82 iters, rate 0.864**
- [x] Add aggregate size histogram diagnostic [SA-AGG]
- [x] Compare with PETSc GAMG aggregate histograms [GAMG-AGG]
- [ ] Improve MIS aggregate uniformity: cap max aggregate size or use greedy ordering (target: max≤13, match PETSc)
- [ ] Step 3: MPI-parallel MIS-k (exchange_halo between passes)
- [ ] Test block_size > 1 (elasticity problem)
- [ ] Adaptive SA (multiple near-null vectors via randomized eigenvectors)
