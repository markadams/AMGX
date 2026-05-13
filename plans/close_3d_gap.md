# Plan: Close the Gap Between AMGx-SA and PETSc GAMG on 3D Q2 FEM

## Problem Statement

On the 3D Q2 hex FEM Laplacian (250K DOFs, 60 nnz/row), PETSc GAMG with MIS-2 aggressive + square graph converges in 15 iterations (Work Cx ≈ 15×1.06 = 15.9), while AMGx-SA MIS-2 implicit does NOT converge.

AMGx-SA MIS-1 converges in 13 iterations but with higher Op Cx (1.34), giving Work Cx = 17.4.

## Root Cause

The AMGx MIS-2 implicit algorithm produces only 2,098 aggregates from 250K nodes (119:1 coarsening) because:
1. The Q2 matrix has 60 nnz/row
2. The distance-2 neighborhood on this graph is enormous (~hundreds of nodes)
3. The `strength_threshold=0.05` filters edge weights but does NOT reduce the graph connectivity for the MIS-2 distance computation
4. Result: very few nodes qualify as MIS-2 roots

PETSc's approach: form the square graph (A^T*A with thresholding), then run MIS-1 on it. The thresholding on A^T*A naturally limits connectivity.

## Proposed Fix

### Option A: Apply strength threshold to distance-2 neighborhood (simplest)

In `find_mis_k2_implicit_kernel` in `src/aggregation/selectors/mis_selector.cu`:
- When checking distance-2 neighbors (neighbors-of-neighbors), skip edges where the edge weight is below the strength threshold
- This effectively reduces the distance-2 neighborhood to only strongly-connected paths
- For Q2 with threshold=0.05: only face connections (weight ~16) pass, edge/corner connections (weight ~1-4) are filtered → effective distance-2 neighborhood is ~36 nodes instead of hundreds

### Option B: Implement square graph MIS-2 (matches PETSc exactly)

Add a new `mis2_algorithm=2` that:
1. Forms A_filtered (drop entries below threshold)
2. Computes A_sq = A_filtered^T * A_filtered (square graph)
3. Runs MIS-1 on A_sq
4. Uses the MIS-1 roots as MIS-2 aggregation seeds

This matches PETSc GAMG's `-pc_gamg_aggressive_square_graph true` exactly.

### Option C: Use Galerkin MIS-2 (algorithm 0) with threshold

The existing Galerkin loop (algorithm 0) already forms a coarse graph. Apply the strength threshold to the Galerkin coarse graph before the second MIS-1 pass.

## Recommended Approach

**Option A** is the simplest and most impactful:
- Single kernel modification in `find_mis_k2_implicit_kernel`
- Add a `strength_threshold` check when iterating over neighbors-of-neighbors
- No new data structures or SpGEMM operations needed

## Test Matrix

`poisson3d_q2.mtx` on Perlmutter (250,047 rows, 15M nnz) — exported from PETSc ex13 with Q2 hex elements on 32³ mesh.

## Success Criteria

- AMGx-SA MIS-2 implicit converges on `poisson3d_q2.mtx` with threshold=0.05
- Iteration count within 50% of PETSc GAMG (target: ≤ 22 iterations)
- Work Complexity competitive with PETSc (target: ≤ 20)

## Reference Results

| Solver | Config | Aggregates | CG Iters | Op Cx | Work Cx |
|--------|--------|:----------:|:--------:|:-----:|:-------:|
| PETSc GAMG | MIS-2 aggressive + square graph | ~6K (est) | 15 | 1.06 | 15.9 |
| AMGx-SA | MIS-1, threshold=0.05 | 16,675 | 13 | 1.34 | 17.4 |
| AMGx-SA | MIS-2 implicit, threshold=0.05 | 2,098 | ∞ (diverges) | — | — |
