# Preparing AMGx-SA Branch for NVIDIA Review

## Current State Summary

The branch adds Smoothed Aggregation AMG to AMGx with:
- **MIS-k aggregation selector** (`src/aggregation/selectors/mis_selector.cu`, ~1489 lines)
- **SA prolongator smoothing** (`src/aggregation/aggregation_amg_level.cu`)
- **Chebyshev eigenvalue reuse mode 4** (`src/solvers/cheb_solver.cu`)
- **Chebyshev zero-init bug fix** (`src/solvers/cheb_solver.cu`)
- **Near-null space support** (`src/aggregation/near_null_space.cu`)
- **Test drivers** (`examples/test_multilevel_cheby.c`, etc.)

**Quality indicators already in place:**
- ✅ Zero memory leaks (compute-sanitizer verified)
- ✅ Cross-validated against PETSc GAMG (identical iteration counts)
- ✅ SPDX license headers on new files
- ✅ Performance matches PETSc quality (20 iters vs 16 for true MIS-2)

---

## Preparation Checklist

### 1. Code Quality and Style

- [ ] **Match NVIDIA coding style**: Review AMGx style conventions (indentation, brace placement, naming). The existing code uses 4-space indent, opening brace on same line for functions, camelCase for methods, snake_case for variables.
- [ ] **Remove development artifacts**: Delete `plans/` directory or move to a separate branch (internal dev notes, not for upstream)
- [ ] **Remove `CLAUDE.md`**: Development notes file not appropriate for upstream
- [ ] **Clean up `.clinerules`**: Remove AI-assistant-specific configuration
- [ ] **Audit `amgx_printf` verbosity**: The MIS selector prints diagnostic info by default. Gate behind `m_verbose` or a print level config parameter to match AMGx convention (most selectors are silent unless `print_grid_stats=1`)

### 2. Testing

- [ ] **Add unit test for MIS selector**: Create `src/tests/mis_selector_test.cu` following the pattern of `src/tests/classical_pmis.cu`:
  - Test that all nodes are assigned to an aggregate
  - Test that MIS roots are distance-k separated
  - Test that aggregate sizes are reasonable
  - Test both algorithm 0 (Galerkin) and algorithm 1 (implicit)
- [ ] **Add SA prolongator test**: Verify P has correct sparsity pattern and that P^T * ones = ones (partition of unity)
- [ ] **Verify existing test suite still passes**: Run `amgx_tests_launcher` to ensure no regressions in existing AMGx functionality
- [ ] **Test with block_size > 1**: Elasticity problems (block_size=2,3) — noted as TODO in status.md
- [ ] **Multi-GPU test**: Even if MIS-k>1 falls back to MIS-1 for multi-GPU, verify the fallback works correctly

### 3. Documentation

- [ ] **Update main README.md**: Add a brief mention of SA-AMG capability in the feature list
- [ ] **API documentation**: Document new config parameters in the style of `doc/AMGX_Reference.pdf` (or at minimum in README-SA.md)
- [ ] **Inline code documentation**: Ensure all new public methods have doxygen-style comments
- [ ] **Clean up README-SA.md**: Remove PETSc-specific comparison details that are more relevant to development than to users. Focus on what AMGx-SA provides and how to use it.

### 4. Build System

- [ ] **Verify CMake integration**: Ensure new files are properly listed in `src/CMakeLists.txt`
- [ ] **CI compatibility**: Test that `ci/test.sh` passes with the new code (cmake .. && make -j8 all && ./tests/amgx_tests_launcher)
- [ ] **No new external dependencies**: Confirm no new libraries or headers were added beyond what AMGx already uses

### 5. Git History

- [ ] **Clean commit history**: Squash or rebase into logical commits:
  1. MIS-k aggregation selector (algorithm 0: Galerkin)
  2. SA prolongator smoothing
  3. Chebyshev eigenvalue reuse (mode 4)
  4. Chebyshev zero-init bug fix
  5. MIS-2 implicit algorithm (algorithm 1)
  6. Near-null space infrastructure
  7. Test drivers and documentation
- [ ] **Remove any large binary files** from git history (matrix files, build artifacts)
- [ ] **Ensure no credentials or paths** leak in commits (e.g., NERSC paths in comments)

### 6. Known Limitations to Document

- [ ] **Multi-GPU MIS-k>1**: Currently falls back to MIS-1 per level for multi-GPU. Document this limitation clearly with the TODO at line 765 of mis_selector.cu.
- [ ] **Block size > 1**: SA smoothing works but needs more testing with elasticity problems
- [ ] **Power iteration count**: `estimateSADampingFactor` uses 100 iterations by default — may want to reduce to 20-30 for production (status.md notes this)

### 7. Performance Validation

- [ ] **Benchmark on multiple problem types**: Beyond 2D Poisson, test on:
  - 3D Poisson (7-point stencil)
  - Anisotropic diffusion
  - Elasticity (block_size=2,3)
  - Unstructured meshes
- [ ] **Scaling study**: Show iteration counts are mesh-independent (h-independent convergence)
- [ ] **GPU timing comparison**: Wall-clock time vs existing AMGx aggregation methods

---

## Priority Order for Review Readiness

1. **Must-have**: Unit tests, CI passes, clean git history, remove dev artifacts
2. **Should-have**: Multi-problem benchmarks, style conformance audit, API docs
3. **Nice-to-have**: Multi-GPU MIS-k support, block_size>1 validation, timing benchmarks

---

## Files to Remove Before PR

| File | Reason |
|------|--------|
| `CLAUDE.md` | AI development notes |
| `.clinerules` | AI assistant config |
| `plans/` directory | Internal development plans |
| `examples/gen_poisson2d.py` | Test utility, not part of library |

## Files to Add

| File | Purpose |
|------|---------|
| `src/tests/mis_selector_test.cu` | Unit test for MIS-k selector |
| `src/tests/sa_prolongator_test.cu` | Unit test for SA smoothing |
