# PR: Fix Chebyshev solver — zero x in solve_init when xIsZero=true

## Summary

`Chebyshev_Solver::solve_init()` accepts an `xIsZero` flag that callers use to
signal that the solution vector `x` is already zeroed and no initial residual
computation is needed. However, the implementation never actually zeroed `x`,
so callers that passed `xIsZero=true` without having zeroed the buffer
themselves would get incorrect results.

## Root Cause

When Chebyshev is used as a **smoother inside AMG** with an outer Krylov solver
(e.g. PCG), the call sequence per outer iteration is:

```
PCG iteration k:
  smoother->solve_init(b, x, /*xIsZero=*/true)   // x still holds data from iter k-1
  smoother->solve_iteration(b, x, ...)            // x += gamma * p  ← accumulates onto stale x
```

`solve_iteration` computes `x += gamma * p` unconditionally. If `x` is not
actually zero at entry, the smoother accumulates onto stale data from the
previous outer iteration. This causes the smoother output to diverge, which
in turn causes the outer PCG to diverge.

## Fix

Add an explicit zero-fill at the top of `solve_init` when `xIsZero=true`:

```cpp
if (xIsZero)
{
    fill(x, types::util<ValueTypeB>::get_zero());
}
```

**File changed**: `src/solvers/cheb_solver.cu` (+9 lines, 0 deletions)

## How to Reproduce

Use Chebyshev as a smoother inside an AMG V-cycle preconditioner for PCG:

```json
{
  "solver": "PCG",
  "preconditioner": {
    "solver": "AMG",
    "smoother": {
      "solver": "CHEBYSHEV",
      "chebyshev_polynomial_order": 2,
      "preconditioner": { "solver": "BLOCK_JACOBI" }
    }
  }
}
```

Without the fix, PCG diverges immediately. With the fix, it converges normally.

## Testing

Verified on a 100×100 2D Poisson problem (10,000 DOFs) using PCG + AMG with
Chebyshev(2)/Jacobi smoother. Without the fix the solver diverges; with the
fix it converges in the expected number of iterations.
