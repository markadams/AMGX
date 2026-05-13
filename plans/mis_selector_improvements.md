# MIS Selector Code Review & Improvement Plan

## Summary

Analysis of `src/aggregation/selectors/mis_selector.cu` (1479 lines), `include/aggregation/selectors/mis_selector.h`, and related files for code quality, AMGx style compliance, and performance opportunities.

---

## 1. Excessive Verbose Output (High Priority)

**Issue:** The selector unconditionally prints ~15 `amgx_printf` diagnostic messages per level per solve. AMGx selectors typically produce zero output unless explicitly requested via a `print_aggregation_info` or verbosity parameter. Compare with `multi_pairwise.cu` and `size4_selector.cu` which have no unconditional printf calls.

**Affected lines:** 789, 794, 852, 869, 937, 1003, 1037, 1139, 1219, 1320, 1402, 1410, 1449, 1451, 1457

**Fix:**
- Gate all `amgx_printf` calls behind a verbosity check, e.g. `if (this->m_verbose)` or use AMGx's existing `print_aggregation_info` parameter
- Keep only the final summary line (aggregates count) at default verbosity
- Move detailed per-pass/per-iteration diagnostics behind a debug flag

---

## 2. Aggregate Statistics Block Copies D→H Every Call (High Priority - Performance)

**Issue:** Lines 1416-1453 unconditionally compute aggregate size statistics by copying the entire `agg_sizes` array from device to host (`cudaMemcpy` of `num_aggregates * sizeof(int)`). This is a synchronization point and memory transfer that happens on every AMG level, every setup call.

**Fix:**
- Gate behind a `print_aggregation_info` or verbosity flag
- Or remove entirely and rely on AMGx's built-in grid stats printing

---

## 3. Kernel Launch Configuration (Medium Priority - Performance)

**Issue:** All kernels use a fixed `threads_per_block = 256` with grid-stride loops. The `find_mis_k2_kernel` is register-heavy (nested loops, many local variables) and would benefit from `__launch_bounds__` to help the compiler optimize register allocation. Compare with `size8_selector.cu` which uses `__launch_bounds__(256, 4)`.

**Fix:**
- Add `__launch_bounds__(256, 4)` to `find_mis_k2_kernel` (high register pressure)
- Add `__launch_bounds__(256, 8)` to simpler kernels like `assign_node_weights_kernel`, `make_singletons_kernel`
- Consider `cudaFuncSetCacheConfig(..., cudaFuncCachePreferL1)` for the MIS kernels which are latency-bound (random memory access patterns)

---

## 4. Redundant nullptr Checks in Hot Kernel Loops (Medium Priority - Performance)

**Issue:** In `find_mis_k1_kernel` and `find_mis_k2_kernel`, the strength threshold check includes:
```cuda
if (row_thresh > 0.0f && edge_weights != nullptr && edge_weights[j] < row_thresh)
```
The `edge_weights != nullptr` check is evaluated for every edge in the inner loop but is invariant per kernel launch. When `strength_threshold == 0`, the entire check is dead code but still occupies registers and instruction cache.

**Fix:**
- Use template specialization or a compile-time flag to generate two kernel variants: one with threshold filtering and one without
- Or at minimum, hoist the nullptr check outside the loop (the compiler may already do this, but explicit is better)
- Simplest approach: since `row_thresh` is already 0.0f when threshold is inactive, the `row_thresh > 0.0f` short-circuits correctly — just remove the redundant `edge_weights != nullptr` check since edge_weights is always non-null when threshold > 0

---

## 5. `find_mis_k2_kernel` Distance-2 Expansion is Serialized (Medium Priority - Performance)

**Issue:** The k2 kernel has a nested loop (distance-1 neighbors × distance-2 neighbors). For high-degree nodes on coarse levels (nnz/row > 20), this creates O(degree²) work per thread, causing severe warp divergence and long-running threads.

**Fix:**
- Consider a warp-cooperative approach where each warp processes one row (32 threads share the distance-1 loop)
- Or use shared memory to cache distance-1 neighbor statuses before expanding to distance-2
- Long-term: the Galerkin loop path (`mis2_algorithm=0`) avoids this entirely by using MIS-1 on the squared graph — consider deprecating the k2 kernel

---

## 6. Missing `cudaFuncSetCacheConfig` for MIS Kernels (Low Priority - Performance)

**Issue:** Only `computeEdgeWeightsBlockDiaCsr_V2` has `cudaFuncSetCacheConfig`. The MIS kernels (`find_mis_k1_kernel`, `find_mis_k2_kernel`) access CSR data with irregular patterns (random neighbor access) and would benefit from preferring L1 cache.

**Fix:**
```cuda
cudaFuncSetCacheConfig(find_mis_k1_kernel<IndexType>, cudaFuncCachePreferL1);
cudaFuncSetCacheConfig(find_mis_k2_kernel<IndexType>, cudaFuncCachePreferL1);
```

---

## 7. Code Duplication Between Implicit MIS-2 and Galerkin Paths (Medium Priority - Structure)

**Issue:** The `setAggregates_common_sqblocks` function has two large code paths (lines 840-1007 for implicit, 1009-1462 for Galerkin) that share significant structure:
- Edge weight computation
- Node weight assignment + halo exchange
- MIS iteration loop
- Aggregate assignment + propagation
- Renumbering

This makes the function ~700 lines long with duplicated logic.

**Fix:**
- Extract common phases into private helper methods:
  - `computeEdgeWeights(A, edge_weights)`
  - `runMIS1(A_cur, edge_weights, status, strength_thresh)` 
  - `assignAndPropagate(A_cur, edge_weights, status, aggregates, strength_thresh)`
- Keep the top-level function as an orchestrator that selects the algorithm path

---

## 8. `m_call_count` Member is Unused (Low Priority - Cleanup)

**Issue:** `m_call_count` is declared in the header (line 41) and initialized to 0 in the constructor but never incremented or read. The level index is obtained from `A.getParameter<int>("amg_level_index")` instead.

**Fix:** Remove `m_call_count` from the header and constructor.

---

## 9. Missing Factory Registration Pattern (Low Priority - Style)

**Issue:** The selector uses `AMGX_FORALL_BUILDS` for explicit instantiation but the factory registration (connecting the "MIS" string to `MISSelectorFactory`) is not visible in this file. It should be registered somewhere (likely `agg_selector.cu` or a registration file).

**Status:** Need to verify this is handled elsewhere. If not, the selector won't be discoverable at runtime.

---

## 10. `propagate_aggregates_kernel` Doesn't Use Strength Threshold (Design Decision)

**Issue:** The propagation kernel (which assigns orphan nodes to any assigned neighbor) does NOT filter by strength threshold. This is intentional — orphan nodes need to be assigned somewhere — but it means the threshold's effect on assignment is partially undone during propagation.

**Consideration:** This is actually correct behavior. Propagation is a fallback for nodes that couldn't find a strong root. No change needed, but worth documenting.

---

## 11. Host Specialization is a Fatal Error (Low Priority - Style)

**Issue:** The host specialization at line 675 throws `FatalError`. AMGx convention is to either not provide the specialization (let the linker catch it) or provide a no-op. Since this is a GPU-only selector, this is acceptable but could use a more informative message.

---

## 12. Test Driver Hardcodes Config (Low Priority - Testing)

**Issue:** `examples/test_multilevel_cheby.c` uses `snprintf` with `%s` format specifiers for numeric values. This works but is fragile — passing non-numeric strings would produce invalid JSON silently.

**Fix:** Use `%f` / `%d` format specifiers with `atof()`/`atoi()` conversion, or validate inputs.

---

## Recommended Priority Order

1. **Gate verbose output** behind a flag (biggest user-facing issue)
2. **Remove unconditional D→H copy** for aggregate stats (performance)
3. **Remove redundant nullptr checks** in kernel inner loops (easy perf win)
4. **Add `__launch_bounds__`** to k2 kernel (perf)
5. **Add `cudaFuncSetCacheConfig`** for MIS kernels (perf)
6. **Extract helper methods** to reduce function length (maintainability)
7. **Remove unused `m_call_count`** (cleanup)
8. **Validate test driver inputs** (robustness)

---

## Non-Issues (Things Done Well)

- ✅ Proper use of `cudaCheckError()` after every kernel launch
- ✅ Correct stream usage throughout (`amgx::thrust::global_thread_handle::get_stream()`)
- ✅ Proper MPI halo exchange pattern (exchange after each MIS iteration)
- ✅ Deterministic propagation via candidate buffer + join pattern
- ✅ Named functors instead of device lambdas (CUDA compatibility)
- ✅ Proper SPDX license headers
- ✅ Template instantiation via `AMGX_FORALL_BUILDS` macro
- ✅ Consistent 4-space indentation matching AMGx style
- ✅ Allman brace style matching AMGx convention
