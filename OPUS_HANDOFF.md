# SA-AMG Non-SPD Bug — Debugging Handoff for Claude Opus

## Repository
`/Users/markadams/Codes/amgx` — NVIDIA AmgX GPU-accelerated algebraic multigrid library (CUDA/C++).

Remote access to Perlmutter (NERSC) for running tests:
```
ssh -i ~/.ssh/nersc -A -Y madams@perlmutter-p1.nersc.gov
scp -i ~/.ssh/nersc localfile madams@perlmutter-p1.nersc.gov:/path/
scp -i ~/.ssh/nersc madams@perlmutter-p1.nersc.gov:/path/file ./
```

---

## The Bug

**Symptom**: For certain grid sizes (e.g., ne=20), the SA-AMG V-cycle preconditioner
is **not symmetric** (positive but asymmetric), causing PCG to diverge. FGMRES
converges, confirming non-SPD preconditioner.

**Root cause**: UNKNOWN. The asymmetry is deterministic (exactly 1.9913e-02 relative
difference for ne=20 on every run) and hierarchy-dependent.

### Symmetry Test Results (x^T M^{-1} y vs y^T M^{-1} x)

| ne | Levels | Coarsest block-rows | Symmetry rel_diff | PCG result |
|----|--------|---------------------|-------------------|------------|
| 10 | 2 | ~20 | 2.9e-10 (PASS) | 18 iters ✓ |
| 15 | 3 | 21 | 1.1e-08 (PASS) | 18 iters ✓ |
| 16 | 3 | ~22 | 5.2e-09 (PASS) | 20 iters ✓ |
| 19 | 3 | 33 | 8.7e-09 (PASS) | 19 iters ✓ |
| **20** | **3** | **32** | **2.0e-02 (FAIL)** | **diverges** ✗ |
| 20 | 2 | 767 | 1.0e-14 (PASS) | 17 iters ✓ |
| 25 | 4 | 1 | 6.4e-09 (PASS) | 24 iters ✓ |
| 30 | 4 | 4 | 3.7e-03 (FAIL) | 21 iters ✓* |

*ne=30 converges despite mild asymmetry (0.37%).

### What's been ruled out
- A symmetry (always 1e-15, perfect)
- R ≠ P^T (transpose computed after smoothing, verified in code)
- Zero P_smooth rows (fixed by singleton merge: 0/27783 zero rows)
- Galerkin product structure (P^T * A * P uses correct P_smooth)
- V-cycle pre/post sweep counts (both JACOBI_L1, 2+2)
- CUDA warp boundary at 32 (ne=30 has 4 coarse block-rows, also fails)
- Positivity (x^T M^{-1} x > 0 always passes)

### Bugs already fixed
1. SA-DIAG buffer overflow: A_vals_h sized A_block_nnz not A_block_nnz*bs^2
2. Singleton merge: singletons reassigned to strongest neighbor aggregate
   (was: agg=-1 → zero P_tent rows → zero P_smooth rows → non-SPD)

**Config**: 3D elasticity, `m_null_dim=6` (6 rigid body mode near-null vectors),
block-3 BSR at level 0, block-6 BSR at levels 1+.

---

## Architecture Overview

### Key Files
- **`src/aggregation/aggregation_amg_level.cu`** — Main SA-AMG level: builds P_tent, smooths it, computes Galerkin product, block-compresses Ac
- **`src/solvers/jacobi_l1_solver.cu`** — Jacobi-L1 smoother used at all levels
- **`src/cycles/fixed_cycle.cu`** — V-cycle implementation
- **`src/multiply.cu`** — Matrix-vector multiply
- **`src/solvers/dense_lu_solver.cu`** — Dense LU for coarsest level

### SA-AMG Build Sequence (per level)
1. `buildTentativeProlongator()` — QR on near-null space per aggregate → P_tent (scalar 1×1 CSR)
2. `smoothProlongator()` — builds S = I - ω D⁻¹ A (scalar), computes P_smooth = S * P_tent via SpGEMM, then **`m_P_tent.swap(P_smooth)`** so `m_P_tent` IS the smoothed prolongator
3. `transpose(m_P_tent, m_P_tent_T)` — computed AFTER swap, so `m_P_tent_T = P_smooth^T`
4. Galerkin: `Ac = P_smooth^T * A_scalar * P_smooth` (scalar CSR result)
5. `createBlockGraph(Ac, m_null_dim)` — packs scalar Ac into block-6 BSR format
6. Near-null space propagated to next level with `m_null_dim=6` vectors

### V-Cycle Structure (`fixed_cycle.cu`)
- Pre-smooth with Jacobi-L1 (block-6 at all levels)
- Restrict: `multiply(m_P_tent_T, r_scalar, rr)` — r temporarily reinterpreted as scalar
- Recurse to coarser level
- Prolongate: `multiply(m_P_tent, e_scalar, prolongated)` — e is scalar
- Post-smooth

### Critical: `build_SA_smoother_block_kernel` (line 4456)
For block-6 A, the smoother S uses the **point diagonal** `A[node,node][li,li]` (NOT block Jacobi):
```cpp
ValueType d_li = A_values[diag_bnz * block_size * block_size + li * block_size + li];
ValueType inv_d = 1.0 / d_li;
s_ij = -omega * inv_d * a_ij;  // off-diagonal
s_ij += 1.0;                    // diagonal
```
This means S is NOT symmetric in general (row i uses `1/A[i,i]`, row j uses `1/A[j,j]`), so **P_smooth = S * P_tent is NOT the transpose of P_smooth^T** in the sense needed for SPD. However, this is the standard SA construction and should still give an SPD preconditioner IF the smoother is self-adjoint w.r.t. A.

### `jacobi_l1_postsmooth` (line 27)
```cpp
x[i] += omega * (b[i] - y[i]) / d[i]
```
where `d[i]` is the L1 norm (sum of |row entries|). This IS self-adjoint w.r.t. A if `d[i]` is the same for pre- and post-smoothing. The Jacobi-L1 smoother IS symmetric.

---

## What Has Been Ruled Out

| Source | Status | Evidence |
|--------|--------|----------|
| **A: R ≠ P^T** | ✅ Ruled out | `m_P_tent_T = transpose(m_P_tent)` computed AFTER `m_P_tent.swap(P_smooth)`, so R = P_smooth^T exactly |
| **B: DENSE_LU non-symmetric** | ✅ Ruled out | `[DENSE-LU-DIAG]` diagnostic shows `n_negative_pivots=0`, no negative pivots |
| **C: Coarse matrix ill-conditioned** | ⚠️ Suspected | Min U diagonal = `1.053e-03`, coarse correction `‖e_coarse‖=4918` from `‖bc‖=2.426` (2027× amplification) |
| **D: Smoother not self-adjoint** | ✅ Ruled out | `jacobi_l1_postsmooth` uses same `d[i]` for all iterations — symmetric |
| **E: Block dim mismatch in multiply** | ✅ Ruled out | `multiply()` at line 122 overwrites `C.set_block_dimy(A.get_block_dimx())` after the call |

---

## V-Cycle Trace (from `[VCYCLE-DBG]` diagnostics)

```
Level 3 pre-smooth: ||x||=47.38 from zero with ||b||=2.688  ← 17.6× amplification (DIVERGING)
Coarse correction:  ||e_coarse||=4918 from ||bc||=2.426     ← 2027× amplification
```

The level 3 smoother **diverges** (produces `‖x‖=47.38` from zero initial guess with `‖b‖=2.688`). This is the primary symptom.

---

## Current Hypothesis

**The `build_SA_smoother_block_kernel` uses the point diagonal `A[node,node][li,li]` for damping, but the coarse-level block-6 matrices (produced by `createBlockGraph`) may have near-zero or negative point diagonals.** Specifically:

- `createBlockGraph` packs scalar Ac into block-6 BSR. The scalar Ac diagonal entries become the block diagonal entries `A[node,node][li,li]`.
- If the scalar Ac has near-zero diagonal entries (e.g., from ill-conditioned aggregates), then `inv_d = 1/A[node,node][li,li]` blows up, making S have huge entries, making P_smooth = S * P_tent have huge entries, making the next Galerkin product ill-conditioned.
- This cascades: level 2→3 produces a near-singular Ac_3 (24 DOFs), the coarse solve produces a huge correction, and the V-cycle diverges.

**Alternative hypothesis**: The `[SA-FIX]` zero-diagonal fix (lines 2947-2953) adds a shift to zero diagonals of the scalar Ac BEFORE `createBlockGraph`. But the shift is `avg_diag * 1e-6` — possibly too small to prevent near-singularity.

---

## Diagnostics Already Added

### In `smoothProlongator()` (line ~4649):
```
[DEBUG smoothProlongator] P_smooth: count=N NaN=0 Inf=0
[DEBUG smoothProlongator] S: count=N NaN=0 Inf=0
[DEBUG smoothProlongator] P_tent: count=N NaN=0 Inf=0
[SA-EIGEN] level=L  rho(D^{-1}A)=X  omega=(4/3)/rho=Y  dofs=N  block=6
```

### In Galerkin product (line ~2844):
```
[DEBUG RAP] Ac: rows=R cols=C nnz=N NaN=0 Inf=0 max_val=X min_diag=Y max_diag=Z cond_diag=W
```

### In `createBlockGraph()` (line ~4952):
```
[SA-BLOCK] createBlockGraph: scalar Ac (NxN, nnz=K) → block-6 Ac (MxM, block_nnz=J)
```

### In `dense_lu_solver.cu`:
```
[DENSE-LU-DIAG] n=24 n_negative_pivots=0 min_U_diag=1.053441e-03 max_U_diag=X
```

### In `fixed_cycle.cu` (V-cycle trace):
```
[VCYCLE-DBG] restrictResidual: ||r_fine||=X  r.size=N  r.num_rows=M  r.block_dimy=6
[VCYCLE-DBG] restrictResidual: ||rr_coarse||=Y  rr.size=K  rr.num_rows=J  rr.block_dimy=1
[VCYCLE-DBG] prolongateAndCorrect: ||e_coarse||=X  e.size=N  e.num_rows=M  e.block_dimy=1
[VCYCLE-DBG] prolongateAndCorrect: ||x_before||=Y  x.size=K  x.num_rows=J  x.block_dimy=6
```

### In `createCoarseVertices()` (line ~2274):
```
[SA-DBG] createCoarseVertices: num_rows=N, block_dimy=6
```

---

## What Needs to Be Done Next

### Step 1: Add targeted diagnostics (NOT YET DONE)

Add to `smoothProlongator()` in `src/aggregation/aggregation_amg_level.cu` after the SpGEMM (after line 4647):

```cpp
// Print min/max column norm of P_smooth and P_tent
{
    int P_rows = m_P_tent.get_num_rows();  // after swap, m_P_tent IS P_smooth
    int P_cols = m_P_tent.get_num_cols();
    int P_nnz  = m_P_tent.get_num_nz();
    // Download P_smooth values
    std::vector<ValueTypeA> P_vals(P_nnz);
    std::vector<int> P_row(P_rows+1), P_col(P_nnz);
    cudaMemcpy(P_vals.data(), m_P_tent.values.raw(), P_nnz*sizeof(ValueTypeA), cudaMemcpyDeviceToHost);
    cudaMemcpy(P_row.data(), m_P_tent.row_offsets.raw(), (P_rows+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(P_col.data(), m_P_tent.col_indices.raw(), P_nnz*sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<double> col_norms(P_cols, 0.0);
    for (int r = 0; r < P_rows; ++r)
        for (int j = P_row[r]; j < P_row[r+1]; ++j)
            col_norms[P_col[j]] += (double)std::abs((float)P_vals[j]) * (double)std::abs((float)P_vals[j]);
    double min_cn = 1e300, max_cn = 0.0;
    for (int c = 0; c < P_cols; ++c) { col_norms[c] = std::sqrt(col_norms[c]); min_cn = std::min(min_cn, col_norms[c]); max_cn = std::max(max_cn, col_norms[c]); }
    fprintf(stderr, "[SA-PCOL] level=%d P_smooth: rows=%d cols=%d nnz=%d min_col_norm=%.4e max_col_norm=%.4e\n",
            this->getLevelIndex(), P_rows, P_cols, P_nnz, min_cn, max_cn);
}
```

Also add to `build_SA_smoother_block_kernel` call site: print min/max of `d_li` (point diagonal used for damping).

### Step 2: Run on Perlmutter

```bash
ssh -i ~/.ssh/nersc -A -Y madams@perlmutter-p1.nersc.gov
cd /path/to/amgx/build
./amgx_capi_test --min-coarse 50 2>&1 | grep -E '\[SA-|VCYCLE|DENSE-LU|DEBUG\]'
```

### Step 3: Confirm diagnosis

If `[SA-PCOL]` shows `min_col_norm ≈ 0` at level 2→3, that confirms near-zero columns in P_smooth_3 → near-singular Ac_4.

If `[DEBUG RAP]` shows `min_diag ≈ 0` at level 2, that confirms the scalar Ac before `createBlockGraph` is near-singular.

### Step 4: Fix (only after confirming diagnosis)

**Do NOT fix until diagnosis is confirmed.** The fix depends on the root cause:
- If near-zero P_smooth columns: add column scaling or drop near-zero columns
- If near-zero Ac diagonal: increase the `[SA-FIX]` shift from `avg_diag * 1e-6` to `avg_diag * 1e-3`
- If smoother omega too large: reduce omega or use Chebyshev smoother

---

## Key Code Locations

| Function | File | Lines |
|----------|------|-------|
| `smoothProlongator()` | `src/aggregation/aggregation_amg_level.cu` | 4527–4814 |
| `build_SA_smoother_block_kernel` | `src/aggregation/aggregation_amg_level.cu` | 4456–4508 |
| `createBlockGraph()` | `src/aggregation/aggregation_amg_level.cu` | 4864–4964 |
| `buildTentativeProlongator()` | `src/aggregation/aggregation_amg_level.cu` | 4020–4151 |
| `restrictResidual()` | `src/aggregation/aggregation_amg_level.cu` | 1129–1230 |
| `prolongateAndApplyCorrection()` | `src/aggregation/aggregation_amg_level.cu` | 960–1099 |
| Near-null space propagation | `src/aggregation/aggregation_amg_level.cu` | 3144–3252 |
| Galerkin product + `[SA-FIX]` | `src/aggregation/aggregation_amg_level.cu` | 2801–2968 |
| `compute_d_BxB_kernel` | `src/solvers/jacobi_l1_solver.cu` | 95–141 |
| `smooth_BxB` | `src/solvers/jacobi_l1_solver.cu` | 506–523 |
| `jacobi_l1_postsmooth` | `src/solvers/jacobi_l1_solver.cu` | 27–44 |
| `solver_setup` | `src/solvers/jacobi_l1_solver.cu` | 347–359 |
| `multiply()` | `src/multiply.cu` | 74–123 |
| `cycle()` V-cycle | `src/cycles/fixed_cycle.cu` | 1–399 |
| `isASolvable()` | `src/cycles/cycle.cu` | 18–25 |

---

## Important Invariants (from `.clinerules`)

1. **SA smoothing is full SpGEMM**: P = S * P_tent where S = I − ω D⁻¹ A. The sparsity of P_smooth GROWS beyond P_tent's pattern. This is NOT pattern-preserving SA.
2. **Do not reference pattern-preserving SA** anywhere in code or comments.
3. Before writing any file or executing any multi-step task: explicitly plan the complete list of steps.
4. If a task involves more than one file or more than ~50 lines of code, break it into explicit subtasks.
5. Never generate more than 200 lines in a single tool call.

---

## Open Questions for Opus

1. **Is the `build_SA_smoother_block_kernel` using the correct diagonal for damping?** It uses `A[node,node][li,li]` (point diagonal), not the L1 norm. For the SA prolongator smoother this is correct (it's the standard point Jacobi damping), but is it correct for block-6 matrices where the off-diagonal block entries couple different DOFs?

2. **Is the `[SA-FIX]` zero-diagonal shift sufficient?** The shift is `avg_diag * 1e-6`. If `avg_diag` is small (e.g., 1e-3), the shift is 1e-9 — essentially zero. Should it be `max_diag * 1e-6` or a fixed floor?

3. **Why does the level 3 smoother diverge (17.6× amplification)?** The Jacobi-L1 smoother should be convergent if `omega < 2/rho(D^{-1}A)`. The `[SA-EIGEN]` diagnostic prints `rho(D^{-1}A)` and `omega=(4/3)/rho`. If `rho` is estimated incorrectly (too small), then `omega` is too large and the smoother diverges.

4. **Is `estimateSADampingFactor()` reliable for coarse levels?** It uses a power iteration. If the coarse matrix is nearly singular, the power iteration may not converge or may give a wrong estimate.
