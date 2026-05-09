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

# SA + Chebyshev smoother (best convergence)
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

### Reference: PETSc GAMG (local, no GPU)

```bash
# On local machine (PETSC_ARCH=arch-macosx-gnu-O):
cd ~/Codes/petsc/src/ksp/ksp/tutorials
./ex2 -m 100 -n 100 -ksp_type richardson -pc_type gamg \
      -pc_gamg_aggressive_coarsening 0 \
      -ksp_monitor -ksp_view
```

PETSc GAMG reference: 4 levels (10000→3683→519→38), ~11 iterations,
asymptotic rate ~0.31.

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
| `src/configs/AGGREGATION_SA_JACOBI.json` | SA config: Jacobi smoother, min_coarse_rows=25 |
| `src/configs/AGGREGATION_SA_CHEBY.json` | SA config: Chebyshev smoother, min_coarse_rows=25 |

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

4. **Galerkin product** — `Ac = P^T A P` via
   `CSR_Multiply::csr_galerkin_product`.

5. **Near-null space propagation** — after the Galerkin product, the coarse
   near-null space (R factors from QR) is propagated to the next level via
   `next_agg->setNearNullSpace(m_null_dim, coarse_dofs, ...)`. This enables
   SA on all coarse levels.

### `[SA-VIEW]` Diagnostics

Two permanent (non-debug) printfs in `aggregation_amg_level.cu`:

1. After Galerkin product — prints per-level `#eqs` and `avg_nnz/row`
2. In `estimateSADampingFactor()` — prints `rho(D^{-1}A)` and `omega`

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

- [ ] Verify 4-level hierarchy with min_coarse_rows=25
- [ ] Test with SIZE_4 selector
- [ ] Block-size > 1 support in `smoothProlongator()`
- [ ] Adaptive SA (multiple near-null vectors via randomized eigenvectors)
