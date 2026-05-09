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
- Aggressive coarsening: 1 level (to match AMGX SIZE_8 coarsening ratio)
- Tolerance: rtol = 1e-5

**Reference results (200×200, aggressive_coarsening=1):**

| Level | #equations | rho(D⁻¹A) provided | lmin target | lmax target |
|-------|-----------|---------------------|-------------|-------------|
| 4 (finest) | 40000 | 1.97426 | 0.197426 | 2.17169 |
| 3 | 5602 | 1.43638 | 0.143638 | 1.58002 |
| 2 | 976 | 1.63406 | 0.163406 | 1.79747 |
| 1 | 98 | 1.64906 (min 0.171299) | 0.164906 | 1.81397 |
| 0 (coarsest) | 9 | — (LU direct) | — | — |

- **14 iterations** to rtol=1e-5
- Asymptotic convergence rate: ~0.47
- Grid complexity: 1.167, operator complexity: 1.423

### Step 2: Run AMGX SA reference

```bash
cd ~/amgx-sa
export LD_LIBRARY_PATH=~/amgx-sa/build_perlmutter:$LD_LIBRARY_PATH
./build_perlmutter/test_sa_phase1 -m 200 -n 200 \
    -c src/configs/AGGREGATION_SA_CHEBY.json
```

AMGX config (`AGGREGATION_SA_CHEBY.json`):
- Smoother: CHEBYSHEV + BLOCK_JACOBI, order=2, pre=1, post=1
- Eigenvalue: lambda_mode=4 (reuse SA rho), lmin_denom=10
- Selector: SIZE_8
- min_coarse_rows: 25
- Tolerance: 1e-5

### Step 3: Compare

For each level, compare:
1. **#equations** — should be similar (AMGX SIZE_8 ≈ GAMG aggressive_coarsening=1)
2. **rho(D⁻¹A)** — AMGX `[SA-VIEW]` prints this; GAMG `-ksp_view` shows "eigenvalues provided"
3. **lmax, lmin** — AMGX `[Chebyshev]` prints these; GAMG shows "eigenvalue targets used"
4. **Convergence rate** — compute from last few residual norms: r_{k+1}/r_k
5. **Iteration count** — should be within ~20% of each other

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
| Coarse solver | LU | 6 sweeps (coarsest_sweeps=6) |

### Diagnostic output to check

**AMGX `[SA-VIEW]` lines** (from `aggregation_amg_level.cu`):
```
[SA-VIEW] Level 0: #eqs=40000  avg_nnz/row=...
[SA-VIEW] Level 0: rho(D^{-1}A)=1.974  omega=0.709
[SA-VIEW] Level 1: #eqs=...
```

**AMGX `[Chebyshev]` lines** (from `cheb_solver.cu`):
```
[Chebyshev] level=40000  lambda_mode=4  lmax=2.172  lmin=0.217  lmin_denom=10  sa_eig_set=1
```
- `sa_eig_set=1` confirms the SA rho was passed to the smoother
- `lmax ≈ rho × 1.1`, `lmin ≈ lmax / 10`

---

## SA Verification Tests (quick)

### 100×100 2D Poisson

```bash
cd ~/amgx-sa
export LD_LIBRARY_PATH=~/amgx-sa/build_perlmutter:$LD_LIBRARY_PATH

# SA + Chebyshev(2)+Jacobi smoother (matches PETSc GAMG default)
./build_perlmutter/test_sa_phase1 -m 100 -n 100 -c src/configs/AGGREGATION_SA_CHEBY.json

# SA + Jacobi smoother
./build_perlmutter/test_sa_phase1 -m 100 -n 100 -c src/configs/AGGREGATION_SA_JACOBI.json
```

Expected: 4 levels (10000 → ~1411 → ~185 → ~26), convergence rate < 0.5.

### PETSc GAMG reference (100×100)

```bash
cd ~/Codes/petsc/src/ksp/ksp/tutorials
./ex2 -m 100 -n 100 -ksp_type richardson -pc_type gamg \
      -pc_gamg_aggressive_coarsening 1 \
      -ksp_monitor -ksp_view -ksp_rtol 1e-5
```

---

## Test

```bash
cd ~/amgx-sa/build_perlmutter
# Interactive GPU node required:
salloc -N1 -C gpu --gpus=1 -A m4267_g --qos=debug -t 30:00
srun -n1 --gpus-per-task=1 ./test_sa_phase1 -m 200 -n 200 \
     -c ../src/configs/AGGREGATION_SA_CHEBY.json
```

Test binary source: `examples/test_sa_phase1.c`

---

## SA AMG Implementation

### Key Files

| File | Purpose |
|------|---------|
| `src/aggregation/aggregation_amg_level.cu` | Main SA path: compaction, `buildTentativeProlongator()`, `smoothProlongator()`, Galerkin product, SA→Chebyshev eigenvalue wiring |
| `src/aggregation/aggregation_amg_level.h` | Level class: `m_sa_rho`, `m_null_dim`, `m_P_tent`, `m_P_tent_T` |
| `src/aggregation/batched_qr.cu` | Batched QR for P_tent construction (one CUDA block per aggregate) |
| `include/solvers/cheb_solver.h` | Chebyshev solver header; `setSAEigenvalue()` method |
| `src/solvers/cheb_solver.cu` | Chebyshev solver; `lambda_mode=4`; `chebyshev_lmin_denom` config param |
| `src/core.cu` | Config parameter registration |
| `src/configs/AGGREGATION_SA_CHEBY.json` | SA config: Chebyshev(2)+Jacobi, lambda_mode=4, lmin_denom=10 |
| `src/configs/AGGREGATION_SA_JACOBI.json` | SA config: Jacobi smoother, min_coarse_rows=25 |

### SA Path in `createCoarseMatrices()`

The SA path is gated by `m_null_dim > 0` (near-null space set) and
`sizeof(ValueTypeB) == sizeof(PODType)` (real-valued only, not complex):

1. **Aggregate ID compaction** — remaps non-contiguous aggregate IDs to
   contiguous 0..N-1 using thrust sort/unique/lower_bound.

2. **`buildTentativeProlongator()`** — calls `batched_qr()` to compute
   P_tent via Modified Gram-Schmidt QR on the near-null space vectors,
   one aggregate at a time.

3. **`smoothProlongator()`** — builds `S = I - ω D⁻¹ A` explicitly, then
   computes `P_smooth = S * P_tent` via SpGEMM. Damping factor
   `ω = 1.4 / ρ(D⁻¹A)` estimated by power iteration.
   - **block_size == 1**: S has same sparsity as A (scalar CSR).
   - **block_size > 1**: A is expanded from block-CSR to scalar CSR using
     `build_SA_smoother_block_rowlen_kernel` + `build_SA_smoother_block_kernel`.
   - Stores `m_sa_rho = ρ(D⁻¹A)` for the Chebyshev smoother.

4. **SA→Chebyshev wiring** — after `smoothProlongator()`, `dynamic_cast`s
   the smoother to `Chebyshev_Solver*` and calls `setSAEigenvalue(m_sa_rho)`.
   This happens before `setup_smoother()` is called in `amg.cu`.

5. **Galerkin product** — `Ac = P^T A P` via
   `CSR_Multiply::csr_galerkin_product`.

6. **Near-null space propagation** — coarse near-null space (R factors from
   QR) propagated to the next level via `next_agg->setNearNullSpace(...)`.

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
- For D-dimensional problems: D³ (8 for 2D, 27 for 3D) is another option

### `[SA-VIEW]` Diagnostics

Permanent printfs in `aggregation_amg_level.cu`:
1. After Galerkin product — per-level `#eqs` and `avg_nnz/row`
2. In `estimateSADampingFactor()` — `rho(D^{-1}A)` and `omega`

`[Chebyshev]` printf in `cheb_solver.cu` `solver_setup()`:
- `level=#rows lambda_mode lmax lmin lmin_denom sa_eig_set`

### Coarsening Control

`min_coarse_rows` in the JSON config controls when coarsening stops.
Set to `5^D` where D is the problem dimension (25 for 2D, 125 for 3D).
With SIZE_8 selector (~7× coarsening ratio per level):
- 10000 → 1411 → 185 → ~26 (4 levels)

### Fixed Bugs

1. **`estimateSADampingFactor`**: Changed `!A.hasProps(DIAG)` →
   `A.diag.size() == 0` so power iteration actually runs.

2. **Full SA smoothing via SpGEMM**: Replaced pattern-preserving kernel
   with explicit `S = I - ω D⁻¹ A` + SpGEMM.

3. **`build_agg_row_lists` off-by-one** (`batched_qr.cu`): Changed
   `thrust::exclusive_scan(..., offsets_ptr + 1)` →
   `thrust::exclusive_scan(..., offsets_ptr)`. Root cause of zero columns
   in P and SA divergence.

### Verified Results (100×100 2D Poisson, 1 MG level)

- P: 10000×1411, nnz=25436, zero-norm columns: 0
- Ac: 1411×1411, nnz=17155, zero diagonal: 0, diag min=0.385
- Galerkin check `‖Ac − PᵀAP‖_F / ‖Ac‖_F` = 2.6e-16 (machine precision)
- SA convergence rate: 0.69–0.73 (PETSc GAMG reference: ~0.32 with 4 levels)

---

## Remaining Work

- [ ] Build on Perlmutter and run 200×200 verification (compare vs PETSc GAMG reference above)
- [ ] Test with SIZE_4 selector
- [ ] Test block_size > 1 (elasticity problem)
- [ ] Adaptive SA (multiple near-null vectors via randomized eigenvectors)
