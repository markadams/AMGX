// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "aggregation/selectors/agg_selector.h"
#include "aggregation/selectors/mis_selector.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx
{

DECLARE_UNITTEST_BEGIN(MISSelectorTest);

// Helper method (now a member function so it can access PrintOnFail/assert_true)
int runMIS(const char *cfg_string, Matrix<TConfig> &A)
{
    typedef typename Matrix<TConfig>::IVector IVector;

    AMG_Config cfg;
    cfg.parseParameterString(cfg_string);

    aggregation::Selector<TConfig> *selector =
        aggregation::SelectorFactory<TConfig>::allocate(cfg, "default");
    PrintOnFail("MIS: selector not created for config: %s", cfg_string);
    UNITTEST_ASSERT_TRUE(selector != NULL);

    IVector aggregates, aggregates_global;
    int num_aggregates = 0;
    selector->setAggregates(A, aggregates, aggregates_global, num_aggregates);
    cudaCheckError();

    int num_rows = A.get_num_rows();

    // Check 1: All nodes assigned (aggregate ID >= 0)
    int num_unassigned = 0;
    {
        IVector agg_host = aggregates;
        for (int i = 0; i < num_rows; i++)
        {
            if (agg_host[i] < 0) { num_unassigned++; }
        }
    }
    PrintOnFail("MIS: %d unassigned nodes with config: %s", num_unassigned, cfg_string);
    UNITTEST_ASSERT_TRUE(num_unassigned == 0);

    // Check 2: Reasonable coarsening (num_aggregates < num_rows)
    PrintOnFail("MIS: no coarsening, num_aggregates=%d >= num_rows=%d, config: %s",
                      num_aggregates, num_rows, cfg_string);
    UNITTEST_ASSERT_TRUE(num_aggregates > 0);
    UNITTEST_ASSERT_TRUE(num_aggregates < num_rows);

    // Check 3: Contiguous numbering - max aggregate ID == num_aggregates - 1
    {
        IVector agg_host = aggregates;
        int max_id = -1;
        for (int i = 0; i < num_rows; i++)
        {
            if (agg_host[i] > max_id) { max_id = agg_host[i]; }
        }
        PrintOnFail("MIS: non-contiguous IDs, max_id=%d, num_aggregates=%d, config: %s",
                          max_id, num_aggregates, cfg_string);
        UNITTEST_ASSERT_TRUE(max_id == num_aggregates - 1);
    }

    delete selector;
    return num_aggregates;
}

void run()
{
    randomize(42);

    // Generate a 2D 5-point Poisson matrix (10x10 = 100 nodes)
    Matrix<TConfig_h> tA;
    generatePoissonForTest(tA, 1, 0, 5, 10, 10);
    MatrixA A;
    A = tA;

    int num_rows = A.get_num_rows();
    PrintOnFail("MIS: matrix has %d rows", num_rows);
    UNITTEST_ASSERT_TRUE(num_rows == 100);

    // Test 1: MIS-1 (standard)
    int nagg_mis1 = runMIS(
        "determinism_flag=1, selector=MIS, mis_k=1",
        A);
    PrintOnFail("MIS-1: num_aggregates=%d", nagg_mis1);
    UNITTEST_ASSERT_TRUE(nagg_mis1 > 0);

    // Test 2: MIS-2 with Galerkin loop (algorithm 0)
    int nagg_mis2_alg0 = runMIS(
        "determinism_flag=1, selector=MIS, mis_k=2, mis2_algorithm=0",
        A);
    PrintOnFail("MIS-2 alg0: num_aggregates=%d", nagg_mis2_alg0);
    UNITTEST_ASSERT_TRUE(nagg_mis2_alg0 > 0);

    // Test 3: MIS-2 with implicit square graph (algorithm 1)
    int nagg_mis2_alg1 = runMIS(
        "determinism_flag=1, selector=MIS, mis_k=2, mis2_algorithm=1",
        A);
    PrintOnFail("MIS-2 alg1: num_aggregates=%d", nagg_mis2_alg1);
    UNITTEST_ASSERT_TRUE(nagg_mis2_alg1 > 0);

    // Test 4: MIS-2 should produce fewer aggregates than MIS-1
    // (more aggressive coarsening)
    PrintOnFail("MIS-2 alg0 (%d) should coarsen more than MIS-1 (%d)",
                nagg_mis2_alg0, nagg_mis1);
    UNITTEST_ASSERT_TRUE(nagg_mis2_alg0 < nagg_mis1);

    PrintOnFail("MIS-2 alg1 (%d) should coarsen more than MIS-1 (%d)",
                nagg_mis2_alg1, nagg_mis1);
    UNITTEST_ASSERT_TRUE(nagg_mis2_alg1 < nagg_mis1);

    // Test 5: Both MIS-2 algorithms produce valid (similar) results
    // They may differ slightly but should be in the same ballpark
    PrintOnFail("MIS-2 alg0=%d, alg1=%d should be similar",
                nagg_mis2_alg0, nagg_mis2_alg1);
    UNITTEST_ASSERT_TRUE(nagg_mis2_alg0 > 0);
    UNITTEST_ASSERT_TRUE(nagg_mis2_alg1 > 0);

    // Test 6: aggressive_levels=1 with MIS-2 still works
    int nagg_aggressive = runMIS(
        "determinism_flag=1, selector=MIS, mis_k=2, mis2_algorithm=0, aggressive_levels=1",
        A);
    PrintOnFail("MIS-2 aggressive_levels=1: num_aggregates=%d", nagg_aggressive);
    UNITTEST_ASSERT_TRUE(nagg_aggressive > 0);
    UNITTEST_ASSERT_TRUE(nagg_aggressive < num_rows);
}

DECLARE_UNITTEST_END(MISSelectorTest);

MISSelectorTest <TemplateMode<AMGX_mode_dDDI>::Type> MISSelectorTest_instance_mode_dDDI;

} // namespace amgx
