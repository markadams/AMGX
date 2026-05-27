# SA-AMG V-cycle preconditioner asymmetry causes PCG divergence

## Summary

The SA-AMG V-cycle preconditioner is **positive but not symmetric** for certain 3D elasticity problem sizes, causing PCG to diverge while FGMRES converges.

## Problem

SA-AMG with JACOBI_L1 smoother, DENSE_LU coarse solver, and MIS-1 aggregation produces a V-cycle preconditioner M⁻¹ that fails the symmetry test:

```
x^T M^{-1} y  ≠  y^T M^{-1} x
```

for certain AMG hierarchy configurations. The asymmetry is deterministic and hierarchy-dependent.

## Reproducing

```bash
# Build
cd build_perlmutter && make -j8 test_elasticity3d_sa

# Failing case (ne=20, 3 levels)
srun -n1 -G1 ./src/test_elasticity3d_sa 20 --min-coarse 50

# Passing case (ne=20, forced 2 levels)
srun -n1 -G1 ./src/test_elasticity3d_sa 20 --min-coarse 800

# Passing case (ne=19, 3 levels)
srun -n1 -G1 ./src/test_elasticity3d_sa 19 --min-coarse 50
```

Config: Q1 hexahedral FEM, E=1.0, ν=0.25, Dirichlet BC on z=0 face, 6 rigid body mode near-null space vectors, presweeps=2, postsweeps=2.

## Symmetry Test Data

Built-in symmetry test (`[SYMM-TEST]`) generates random vectors x, y and checks `x^T M⁻¹ y` vs `y^T M⁻¹ x`:

| ne | Levels | Coarsest (block-rows × bs) | Symmetry rel_diff | PCG |
|----|--------|---------------------------|-------------------|-----|
| 10 | 2 | ~20 × 6 | 2.9e-10 ✓ | 18 iters |
| 15 | 3 | 21 × 6 | 1.1e-08 ✓ | 18 iters |
| 19 | 3 | 33 × 6 | 8.7e-09 ✓ | 19 iters |
| **20** | **3** | **32 × 6** | **2.0e-02 ✗** | **diverges** |
| 20 | 2 | 767 × 6 | 1.0e-14 ✓ | 17 iters |
| 25 | 4 | 1 × 6 | 6.4e-09 ✓ | 24 iters |
| **30** | **4** | **4 × 6** | **3.7e-03 ✗** | 21 iters* |

*ne=30 converges despite mild asymmetry.

## What has been ruled out

- **A symmetry**: Always passes at machine precision (1e-15)
- **R ≠ Pᵀ**: Transpose is computed after prolongator smoothing — verified in code
- **Zero P_smooth rows**: Fixed by singleton merge (0 zero rows for all tested sizes)
- **Galerkin product**: Uses P_smooth correctly: Ac = P_smooth^T × A × P_smooth
- **V-cycle structure**: Pre/post sweeps use same JACOBI_L1 smoother (2+2)
- **Positivity**: x^T M⁻¹ x > 0 always passes

## Key observations

1. The asymmetry appears only with ≥3 AMG levels for specific hierarchy configurations
2. Same problem size (ne=20) with 2 levels → perfect symmetry (1e-14)
3. The failure value is exactly deterministic (1.9913e-02 every run for ne=20)
4. ne=19 with 3 levels (33 coarse block-rows) passes; ne=20 with 3 levels (32 coarse block-rows) fails
5. ne=30 with 4 levels (4 coarse block-rows) shows mild asymmetry but still converges
