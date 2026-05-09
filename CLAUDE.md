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

## SA Verification Tests

Run these on an interactive GPU node (`salloc -N1 -C gpu --gpus=1 -A m4267_g --qos=debug -t 30:00`).

### Standard test: 100×100 2D Poisson

```bash
cd ~/amgx-sa
export LD_LIBRARY_PATH=~/amgx-sa/build_perlmutter:$LD_LIBRARY_PATH

# SA + Chebyshev(1)+Jacobi smoother (matches PETSc GAMG default)
./build_perlmutter/test_sa_phase1 -m 100 -n 100 -c src/configs/AGGREGATION_SA_CHEBY.json

# SA + Jacobi smoother
./build_perlmutter/test_sa_phase1 -m 100 -n 100 -c src/configs/AGGREGATION_SA_JACOBI.json
```

### Expected output (with min_coarse_rows=25, SIZE_8 selector)

Look for `[SA-VIEW]` lines and `Number of Levels`:

```
[SA-VIEW] Level 0: #eqs=10000  avg_nnz/row=4.9
[SA-VIEW] Level 0: rho(D^{-1}A)=...  omega=...
[SA-VIEW] Level 1: #eqs=1411   avg_nnz/row=...
[SA-VIEW] Level 2: #eqs=185    avg_nnz/row=...
Number of Levels: 4
```

4 levels expected: 10000 → ~1411 → ~185 → ~26 (coarsest < 25 = 5^D, D=2).

Also look for `[Chebyshev]` lines showing eigenvalue estimates per level:
```
[Chebyshev] level=10000  lambda_mode=4  lmax=...  lmin=...  lmin_denom=10  sa_eig_set=1
[Chebyshev] level=1411   lambda_mode=4  lmax=...  lmin=...  lmin_denom=10  sa_eig_set=1
```
`sa_eig_set=1` confirms the SA rho was passed to the smoother.

### Reference: PETSc GAMG (200×200, aggressive_coarsening=1)

```bash
# On local machine (PETSC_ARCH=arch-macosx-gnu-O):
cd ~/Codes/petsc/src/ksp/ksp/tutorials
./ex2 -m 200 -n 200 -ksp_type richardson -pc_type gamg \
      -pc_gamg_aggressive_coarsening 1 \
      -mg_levels_ksp_type chebyshev \
      -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.1 \
      -mg_levels_pc_type jacobi \
      -mg_levels_ksp_max_it 1 \
      -ksp_monitor -ksp_view
```

PETSc GAMG reference (200×200, aggressive_coarsening=1, Chebyshev(1)+Jacobi):
- 5 levels: 40000 → 5602 → 976 → 98 → 9
- λ_max per level: ~1.974, ~1.436, ~1.634, ~1.649
- 21 iterations to convergence

### Convergence rate check

In the AMGX output, look for the per-iteration residual norms and compute
the asymptotic ratio `r_{k+1}/r_k` over the last few iterations.
Target: rate < 0.5 with SA+Chebyshev on 4 levels.

---

## Test

```bash
cd ~/amgx-sa/build_perlmutter
# Interactive GPU node required:
srun -n1 --gpus-per-task=1 -A m4267_g --qos=debug -C gpu \
     ./test_sa_phase1 poisson2d_amgx.mtx
```

Test binary source: `examples/test_sa_phase1.c`
Test matrix: `build_perlmutter/poisson2d_amgx.mtx` (100×100 2D Poisson)

---

## SA AMG Implementation

### Key Files

| File | Purpose |
|------|---------|
| `src/aggregation/aggregation_amg_level.cu` | Main SA path: compaction, `buildTentativeProlongator()`, `smoothProlongator()`, Galerkin product |
| `src/aggregation/batched_qr.cu` | Batched QR for P_tent construction (one CUDA block per aggregate) |
| `src/aggregation/near_null_space.cu` | Near-null space storage and propagation |
| `include/aggregation/batched_qr.h` | Header for batched QR |
| `include/aggregation/near_null_space.h` | Header for near-null space |
| `include/solvers/cheb_solver.h` | Chebyshev solver header; `setSAEigenvalue()` method |
| `src/solvers/cheb_solver.cu` | Chebyshev solver; `lambda_mode=4`; `chebyshev_lmin_denom` config param |
| `src/core.cu` | Config parameter registration |
| `src/configs/AGGREGATION_SA_JACOBI.json` | SA config: Jacobi smoother, min_coarse_rows=25 |
| `src/configs/AGGREGATION_SA_CHEBY.json` | SA config: Chebyshev(1)+Jacobi, lambda_mode=4, lmin_denom=10 |

### SA Path in `createCoarseMatrices()`

The SA path is gated by `m_null_dim > 0` (near-null space set) and
`sizeof(ValueTypeB) == sizeof(PODType)` (real-valued only, not complex):

1. **Aggregate ID compaction** — remaps non-contiguous aggregate IDs to
   contiguous 0..N-1 using thrust sort/unique/lower_bound. Defensive; the
   SIZE_8 selector currently produces contiguous IDs for the test problem.

2. **`buildTentativeProlongator()`** — calls `batched_qr()` to compute
   P_tent via Modified Gram-Schmidt QR on the near-null space vectors,
   one aggregate at a time.

3. **`smoothProlongator()`** — builds `S = I - ω D⁻¹ A` explicitly, then
   computes `P_smooth = S * P_tent` via SpGEMM
   (`CSR_Multiply::csr_multiply`). Damping factor `ω = 1.4 / ρ(D⁻¹A)`
   estimated by power iteration in `estimateSADampingFactor()`.
   - **block_size == 1**: S has same sparsity as A (scalar CSR).
   - **block_size > 1**: A is expanded from block-CSR to scalar CSR using
     `build_SA_smoother_block_rowlen_kernel` + `build_SA_smoother_block_kernel`.
     Each block-row i → block_size scalar rows; each block-nz → block_size²
     scalar entries.
   - After `smoothProlongator()`, `m_sa_rho = 1.4 / omega` is stored on the
     level and passed to the Chebyshev smoother via `setSAEigenvalue(m_sa_rho)`
     before `setup_smoother()` is called.

4. **Galerkin product** — `Ac = P^T A P` via
   `CSR_Multiply::csr_galerkin_product`.

5. **Near-null space propagation** — after the Galerkin product, the coarse
   near-null space (R factors from QR) is propagated to the next level via
   `next_agg->setNearNullSpace(m_null_dim, coarse_dofs, ...)`. This enables
   SA on all coarse levels.

### Chebyshev Smoother Eigenvalue Modes

`chebyshev_lambda_estimate_mode` (config, default 0):
- **0**: eigensolver for both lmin and lmax
- **1**: eigensolver for lmax; lmin = lmax / lmin_denom
- **2**: row-sum estimate for lmax; lmin = lmax / lmin_denom
- **3**: user-provided `cheby_max_lambda` / `cheby_min_lambda`
- **4**: reuse SA-computed `rho(D⁻¹A)` from `estimateSADampingFactor()`;
  lmax = rho × 1.1; lmin = lmax / lmin_denom. Avoids a second power
  iteration. Matches PETSc GAMG `-pc_gamg_use_sa_esteig`.

`chebyshev_lmin_denom` (config, default 10.0):
- lmin = lmax / lmin_denom
- PETSc GAMG uses 10.0 (transform [0,0.1;0,1.1])
- For D-dimensional problems: D³ (8 for 2D, 27 for 3D) is another option

### `[SA-VIEW]` Diagnostics

Two permanent (non-debug) printfs in `aggregation_amg_level.cu`:

1. After Galerkin product — prints per-level `#eqs` and `avg_nnz/row`
2. In `estimateSADampingFactor()` — prints `rho(D^{-1}A)` and `omega`

`[Chebyshev]` printf in `cheb_solver.cu` `solver_setup()`:
- Prints `level=#rows lambda_mode lmax lmin lmin_denom sa_eig_set`
- `sa_eig_set=1` confirms the SA rho was passed via `setSAEigenvalue()`

### Coarsening Control

`min_coarse_rows` in the JSON config controls when coarsening stops.
Set to `5^D` where D is the problem dimension (25 for 2D, 125 for 3D).
With SIZE_8 selector (~7× coarsening ratio per level):
- 10000 → 1411 → 185 → ~26 (stops: 26 < 25 would be next, so 4 levels)

### Fixed Bugs

1. **`estimateSADampingFactor`**: Changed `!A.hasProps(DIAG)` →
   `A.diag.size() == 0` so power iteration actually runs.

2. **Full SA smoothing via SpGEMM**: Replaced pattern-preserving kernel
   with explicit `S = I - ω D⁻¹ A` + SpGEMM.

3. **`build_agg_row_lists` off-by-one** (`batched_qr.cu`): Changed
   `thrust::exclusive_scan(..., offsets_ptr + 1)` →
   `thrust::exclusive_scan(..., offsets_ptr)`. This was the root cause of
   zero columns in P and SA divergence (convergence rate 1.004 → 0.69–0.73).

### Verified Results (100×100 2D Poisson, 1 MG level)

- P: 10000×1411, nnz=25436, zero-norm columns: 0
- Ac: 1411×1411, nnz=17155, zero diagonal: 0, diag min=0.385
- Galerkin check `‖Ac − PᵀAP‖_F / ‖Ac‖_F` = 2.6e-16 (machine precision)
- SA convergence rate: 0.69–0.73 (PETSc GAMG reference: ~0.32 with 4 levels)

---

## Remaining Work

- [ ] Build on Perlmutter and test 4-level hierarchy (min_coarse_rows=25, SIZE_8 selector, 100×100)
- [ ] Verify `[Chebyshev] sa_eig_set=1` at each level with lambda_mode=4
- [ ] Compare convergence rate vs PETSc GAMG reference (200×200, 5 levels, 21 iters)
- [ ] Test with SIZE_4 selector
- [ ] Test block_size > 1 (elasticity problem)
- [ ] Adaptive SA (multiple near-null vectors via randomized eigenvectors)
