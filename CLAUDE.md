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

## Verification Procedure

The goal is to match the AMGX SA solver algorithm as closely as possible to
PETSc GAMG, then compare per-level grid sizes, eigenvalue estimates, and
convergence rates on the same problem.

### Problem: 200×200 2D Poisson (40,000 DOFs)

### Step 1: Run PETSc GAMG reference

```bash
cd ~/Codes/petsc/src/ksp/ksp/tutorials
./ex2 -m 200 -n 200 \
      -ksp_type richardson -pc_type gamg \
      -pc_gamg_aggressive_coarsening 1 \
      -ksp_monitor -ksp_view -ksp_rtol 1e-5
```

GAMG defaults used:
- Smoother: Chebyshev(2) + Jacobi (not L1 Jacobi), pre+post
- Eigenvalue estimate: CG-based, transform [0, 0.1; 0, 1.1]
  → lmin = 0.1 × rho(D⁻¹A), lmax = 1.1 × rho(D⁻¹A)
  → equivalent to lmin_denom = 10
- Aggressive coarsening: 1 level (to match AMGX MULTI_PAIRWISE coarsening ratio)
- Tolerance: rtol = 1e-5

### Step 2: Run AMGX SA reference

```bash
cd ~/amgx-sa
export LD_LIBRARY_PATH=~/amgx-sa/build_perlmutter:$LD_LIBRARY_PATH
srun -n1 --gpus-per-task=1 -A m1516_g --qos=debug -C gpu -t 5:00 \
  ./build_perlmutter/test_sa_phase1 build_perlmutter/poisson2d_200.mtx
```

AMGX config (`AGGREGATION_SA_CHEBY.json`):
- Smoother: CHEBYSHEV + BLOCK_JACOBI, order=2, pre=1, post=1
- Eigenvalue: lambda_mode=4 (reuse SA rho), lmin_denom=10
- Selector: MULTI_PAIRWISE (aggregation_passes=3)
- Coarse solver: DENSE_LU_SOLVER
- min_coarse_rows: 10
- Tolerance: 1e-5

### Step 3: Compare

For each level, compare:
1. **#equations** — should be similar
2. **nnz/row** — proxy for operator cost per level
3. **rho(D⁻¹A)** — AMGX `[SA-VIEW]` prints this; GAMG `-ksp_view` shows "eigenvalues provided"
4. **λ_max, λ_min** — AMGX `[Chebyshev]` prints these; GAMG shows "eigenvalue targets used"
5. **Convergence rate** — compute from last few residual norms: r_{k+1}/r_k
6. **Iteration count** — should be within ~50% of each other

### Matching the smoother algorithm

| Parameter | PETSc GAMG default | AMGX SA config |
|-----------|-------------------|----------------|
| Smoother type | Chebyshev | CHEBYSHEV |
| Preconditioner | Jacobi (diagonal) | BLOCK_JACOBI |
| Chebyshev degree | 2 (mg_levels_ksp_max_it) | chebyshev_polynomial_order=2 |
| Pre-smoothing | 1 call × 2 iters | presweeps=1 × order=2 |
| Post-smoothing | 1 call × 2 iters | postsweeps=1 × order=2 |
| λ_max estimate | CG-based rho(D⁻¹A) | SA power iteration rho(D⁻¹A) |
| λ_max scaling | ×1.1 | ×1.1 (lambda_mode=4) |
| λ_min | λ_max_provided × 0.1 | lmax / lmin_denom (=10) |
| Coarse solver | LU | DENSE_LU_SOLVER |
| Aggregation | MIS-k | MULTI_PAIRWISE (3 passes) |

### Diagnostic output to check

**AMGX `[SA-VIEW]` lines** (from `aggregation_amg_level.cu`):
```
[SA-VIEW] Level 0: #eqs=40000  avg_nnz/row=5.0
[SA-VIEW] Level 0: rho(D^{-1}A)=1.846  omega=0.759
```

**AMGX `[Chebyshev]` lines** (from `cheb_solver.cu`):
```
[Chebyshev] level=40000  lambda_mode=4  lmax=2.030  lmin=0.203  lmin_denom=10  sa_eig_set=1
```
- `sa_eig_set=1` confirms the SA rho was passed to the smoother
- `lmax = rho × 1.1`, `lmin = lmax / 10`

---

## Test

```bash
cd ~/amgx-sa/build_perlmutter
# Interactive GPU node required:
salloc -N1 -C gpu --gpus=1 -A m1516_g --qos=debug -t 30:00
srun -n1 --gpus-per-task=1 ./test_sa_phase1 poisson2d_200.mtx
```

Test binary source: `examples/test_sa_phase1.c`

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

#### AMGX SA, MIS-2 selector (Galerkin loop), DENSE_LU (rtol=1e-5) — Perlmutter, May 10 2026

MIS-2 Galerkin loop diagnostics:
```
[MIS-k] Pass 0/2: n_cur=160000, nnz=798400, avg_nnz/row=5.0
[MIS-k] Pass 0: MIS-1 converged in 9 iters, 57872 roots (36.2%), 0 forced-root
[MIS-k] Pass 0: 57872 aggregates from 160000 nodes (coarsening ratio 2.76x)
[MIS-k] Pass 0: Galerkin coarse matrix: 57872 rows, 350536 nnz, avg_nnz/row=6.1
[MIS-k] Pass 1/2: n_cur=57872, nnz=350536, avg_nnz/row=6.1
[MIS-k] Pass 1: MIS-1 converged in 9 iters, 16143 roots (27.9%), 0 forced-root
[MIS-k] Pass 1: 16143 aggregates from 57872 nodes (coarsening ratio 3.58x)
[MIS-k] Final: 160000 fine nodes -> 16143 aggregates (net coarsening ratio 9.91x)
```

| Level | #equations | nnz/row | ρ(D⁻¹A) | λ_max | λ_min |
|-------|-----------|---------|----------|-------|-------|
| 0 (finest) | 160,000 | 5.0 | 1.846 | 2.030 | 0.203 |
| 1 | 16,143 | 11.0 | 1.467 | 1.614 | 0.161 |
| 2 | 605 | 12.6 | 1.471 | 1.618 | 0.162 |
| 3 (coarsest) | 24 | — | — | — (LU) | — |

- **DIVERGED** — hit 100-iteration limit, final residual 3,185 (initial 400)
- Avg convergence rate: **1.0210** (diverging)
- Grid complexity: 1.105, Operator complexity: 1.232
- L0→L1 coarsening: **9.91×** (good — matches MULTI_PAIRWISE's 7.83×)

**Root cause**: The Galerkin-composed MIS-2 aggregates achieve the target coarsening ratio
(9.91×) and produce low grid/operator complexity (1.105/1.232), but the V-cycle **diverges**.
The composed aggregates from two MIS-1 passes produce prolongation operators that fail the
SA approximation property. Possible issues:
1. **Aggregate shape**: Two-pass MIS-1 composition may produce irregular aggregate shapes
   (chains of chains) that don't span the near-null space well
2. **Edge weight loss**: The Galerkin product `R·A·R^T` sums fine-grid values, but the
   absolute-value edge weights used for pass-1 aggregate assignment may not reflect true
   coupling strength in the coarse graph
3. **SA smoothing on composed aggregates**: The SA prolongator `P = (I - ω D⁻¹A) P_tent`
   is built from the composed aggregates, but the smoothing stencil (distance-1 in A) may
   not reach across the larger MIS-2 aggregates, leaving P_tent essentially unsmoothed
   for nodes far from aggregate boundaries

**Key insight**: SIZE_4 (26 iters, rate 0.626) and SIZE_8 (62 iters, rate 0.827) both
converge with similar coarsening ratios. The difference is aggregate shape quality — SIZE_4
produces compact 2×2 blocks, while MIS-2 via Galerkin composition produces irregular shapes.

#### Comparison (400×400)

| | PETSc GAMG | SIZE_4 | SIZE_8 | MULTI_PAIRWISE | MIS-1 | **MIS-2** |
|---|-----------|--------|--------|----------------|-------|-----------|
| Levels | 5 | 7 | 5 | 5 | 5 | **4** |
| Iterations | 16 | 26 | 62 | 57 | >100 | **>100** |
| Conv. rate | ~0.49 | 0.626 | 0.827 | 0.809 | 0.960 | **1.021** |
| Grid cx | 1.167 | 1.314 | 1.166 | 1.146 | 1.406 | **1.105** |
| Op cx | 1.428 | 2.702 | 1.481 | 1.380 | 2.496 | **1.232** |
| L0→L1 ratio | ~7× | 4.20× | 6.88× | 7.83× | 2.76× | **9.91×** |
| Coarsest | 27 | 31 | 49 | 39 | 28 | **24** |
| Status | ✓ | ✓ | ✓ | ✓ | DNF | **DIVERGE** |

Gap (57 vs 16 iters) is larger than at 200×200 (23 vs 14). Likely causes:
1. GAMG uses MIS-k aggregation (better aggregate shapes) vs MULTI_PAIRWISE
2. GAMG eigenvalue estimate via CG vs AMGX power iteration
3. AMGX convergence rate degrades with problem size (0.59→0.81) while GAMG stays flat (~0.49)

Note: AMGX SIZE_4 selector gives **26 iters** at 400×400 (rate 0.626, op cx 2.70) — closest
to GAMG's 16 iters but at much higher operator cost.

---

## Remaining Work

- [x] Build on Perlmutter and run 200×200 verification
- [x] Test with MULTI_PAIRWISE selector — best match to PETSc GAMG
- [x] Fix coarse solver: DENSE_LU_SOLVER instead of JACOBI_L1
- [x] Fix min_coarse_rows stopping logic: keep grid that falls below target
- [x] Implement MIS-1 selector (Steps 1+2 of mis_k_mpi_parallel_plan.md)
- [x] Verify MIS-1 on 400×400: DNF (rate 0.960) — coarsening too fine (2.76×), need MIS-2+
- [x] Implement MIS-2 (mis_k=2) via Galerkin coarsening loop
- [x] Test MIS-2 on 400×400: **DIVERGES** (rate 1.021) — coarsening ratio good (9.91×) but aggregate quality poor
- [ ] Diagnose MIS-2 divergence: compare aggregate shapes, try direct MIS-2 kernel, check SA smoothing reach
- [ ] Step 3: MPI-parallel MIS-k (exchange_halo between passes)
- [ ] Test block_size > 1 (elasticity problem)
- [ ] Adaptive SA (multiple near-null vectors via randomized eigenvectors)
