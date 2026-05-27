// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Standalone 3D elasticity SA-AMG driver.
//
// Assembles a Q1 hexahedral FEM stiffness matrix (E=1.0, nu=0.25) on an
// ne x ne x ne element grid, following PETSc ksp/tutorials/ex99.c.
// Dirichlet BC on the z=0 face (clamped end) via the DD2 softened element
// matrix approach: rows/cols 0-11 zeroed, diagonal scaled by 0.1.
// Solves with GMRES + SA-AMG (MIS2 selector) + JACOBI_L1 smoother using
// 6 rigid body mode near-null space vectors.
//
// Build: added to src/CMakeLists.txt as target test_elasticity3d_sa
// Run:   srun -A m1516_g -C gpu -q debug -t 5 -n 1 --gpus-per-task=1 \
//              ./test_elasticity3d_sa [ne]
//   ne defaults to 9 (10^3 = 1000 nodes, 3000 DOFs)

#include "amg_solver.h"
#include <aggregation/near_null_space.h>
#include <solvers/algebraic_multigrid_solver.h>
#include <amg.h>
#include <amg_level.h>
#include <aggregation/aggregation_amg_level.h>
#include <solvers/pcg_solver.h>
#include <matrix.h>
#include <vector.h>
#include <resources.h>
#include <error.h>
#include <multiply.h>

#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Q1 hex element stiffness matrix for E=1.0, nu=0.25.
// 24x24 = (8 nodes) x (3 DOFs/node).  Row-major, symmetric.
// Data from PETSc ksp/tutorials/ex56.c :: elem_3d_elast_v_25().
// ---------------------------------------------------------------------------
static const double DD1[576] = {
  0.18981481481481474,       5.27777777777777568E-002,  5.27777777777777568E-002,
 -5.64814814814814659E-002, -1.38888888888889072E-002, -1.38888888888889089E-002,
 -8.24074074074073876E-002, -5.27777777777777429E-002,  1.38888888888888725E-002,
  4.90740740740740339E-002,  1.38888888888889124E-002,  4.72222222222222071E-002,
  4.90740740740740339E-002,  4.72222222222221932E-002,  1.38888888888888968E-002,
 -8.24074074074073876E-002,  1.38888888888888673E-002, -5.27777777777777429E-002,
 -7.87037037037036785E-002, -4.72222222222221932E-002, -4.72222222222222071E-002,
  1.20370370370370180E-002, -1.38888888888888742E-002, -1.38888888888888829E-002,
  5.27777777777777568E-002,  0.18981481481481474,       5.27777777777777568E-002,
  1.38888888888889124E-002,  4.90740740740740269E-002,  4.72222222222221932E-002,
 -5.27777777777777637E-002, -8.24074074074073876E-002,  1.38888888888888725E-002,
 -1.38888888888889037E-002, -5.64814814814814728E-002, -1.38888888888888985E-002,
  4.72222222222221932E-002,  4.90740740740740478E-002,  1.38888888888888968E-002,
 -1.38888888888888673E-002,  1.20370370370370058E-002, -1.38888888888888742E-002,
 -4.72222222222221932E-002, -7.87037037037036785E-002, -4.72222222222222002E-002,
  1.38888888888888742E-002, -8.24074074074073598E-002, -5.27777777777777568E-002,
  5.27777777777777568E-002,  5.27777777777777568E-002,  0.18981481481481474,
  1.38888888888889055E-002,  4.72222222222222002E-002,  4.90740740740740269E-002,
 -1.38888888888888829E-002, -1.38888888888888829E-002,  1.20370370370370180E-002,
  4.72222222222222002E-002,  1.38888888888888985E-002,  4.90740740740740339E-002,
 -1.38888888888888985E-002, -1.38888888888888968E-002, -5.64814814814814520E-002,
 -5.27777777777777568E-002,  1.38888888888888777E-002, -8.24074074074073876E-002,
 -4.72222222222222002E-002, -4.72222222222221932E-002, -7.87037037037036646E-002,
  1.38888888888888794E-002, -5.27777777777777568E-002, -8.24074074074073598E-002,
 -5.64814814814814659E-002,  1.38888888888889124E-002,  1.38888888888889055E-002,
  0.18981481481481474,      -5.27777777777777568E-002, -5.27777777777777499E-002,
  4.90740740740740269E-002, -1.38888888888889072E-002, -4.72222222222221932E-002,
 -8.24074074074073876E-002,  5.27777777777777568E-002, -1.38888888888888812E-002,
 -8.24074074074073876E-002, -1.38888888888888742E-002,  5.27777777777777499E-002,
  4.90740740740740269E-002, -4.72222222222221863E-002, -1.38888888888889089E-002,
  1.20370370370370162E-002,  1.38888888888888673E-002,  1.38888888888888742E-002,
 -7.87037037037036785E-002,  4.72222222222222002E-002,  4.72222222222222071E-002,
 -1.38888888888889072E-002,  4.90740740740740269E-002,  4.72222222222222002E-002,
 -5.27777777777777568E-002,  0.18981481481481480,       5.27777777777777568E-002,
  1.38888888888889020E-002, -5.64814814814814728E-002, -1.38888888888888951E-002,
  5.27777777777777637E-002, -8.24074074074073876E-002,  1.38888888888888881E-002,
  1.38888888888888742E-002,  1.20370370370370232E-002, -1.38888888888888812E-002,
 -4.72222222222221863E-002,  4.90740740740740339E-002,  1.38888888888888933E-002,
 -1.38888888888888812E-002, -8.24074074074073876E-002, -5.27777777777777568E-002,
  4.72222222222222071E-002, -7.87037037037036924E-002, -4.72222222222222140E-002,
 -1.38888888888889089E-002,  4.72222222222221932E-002,  4.90740740740740269E-002,
 -5.27777777777777499E-002,  5.27777777777777568E-002,  0.18981481481481477,
 -4.72222222222222071E-002,  1.38888888888888968E-002,  4.90740740740740131E-002,
  1.38888888888888812E-002, -1.38888888888888708E-002,  1.20370370370370267E-002,
  5.27777777777777568E-002,  1.38888888888888812E-002, -8.24074074074073876E-002,
  1.38888888888889124E-002, -1.38888888888889055E-002, -5.64814814814814589E-002,
 -1.38888888888888812E-002, -5.27777777777777568E-002, -8.24074074074073737E-002,
  4.72222222222222002E-002, -4.72222222222222002E-002, -7.87037037037036924E-002,
 -8.24074074074073876E-002, -5.27777777777777637E-002, -1.38888888888888829E-002,
  4.90740740740740269E-002,  1.38888888888889020E-002, -4.72222222222222071E-002,
  0.18981481481481480,       5.27777777777777637E-002, -5.27777777777777637E-002,
 -5.64814814814814728E-002, -1.38888888888889037E-002,  1.38888888888888951E-002,
 -7.87037037037036785E-002, -4.72222222222222002E-002,  4.72222222222221932E-002,
  1.20370370370370128E-002, -1.38888888888888725E-002,  1.38888888888888812E-002,
  4.90740740740740408E-002,  4.72222222222222002E-002, -1.38888888888888951E-002,
 -8.24074074074073876E-002,  1.38888888888888812E-002,  5.27777777777777637E-002,
 -5.27777777777777429E-002, -8.24074074074073876E-002, -1.38888888888888829E-002,
 -1.38888888888889072E-002, -5.64814814814814728E-002,  1.38888888888888968E-002,
  5.27777777777777637E-002,  0.18981481481481480,      -5.27777777777777568E-002,
  1.38888888888888916E-002,  4.90740740740740339E-002, -4.72222222222222210E-002,
 -4.72222222222221932E-002, -7.87037037037036924E-002,  4.72222222222222002E-002,
  1.38888888888888742E-002, -8.24074074074073876E-002,  5.27777777777777429E-002,
  4.72222222222222002E-002,  4.90740740740740269E-002, -1.38888888888888951E-002,
 -1.38888888888888846E-002,  1.20370370370370267E-002,  1.38888888888888916E-002,
  1.38888888888888725E-002,  1.38888888888888725E-002,  1.20370370370370180E-002,
 -4.72222222222221932E-002, -1.38888888888888951E-002,  4.90740740740740131E-002,
 -5.27777777777777637E-002, -5.27777777777777568E-002,  0.18981481481481480,
 -1.38888888888888968E-002, -4.72222222222221932E-002,  4.90740740740740339E-002,
  4.72222222222221932E-002,  4.72222222222222071E-002, -7.87037037037036646E-002,
 -1.38888888888888742E-002,  5.27777777777777499E-002, -8.24074074074073737E-002,
  1.38888888888888933E-002,  1.38888888888889020E-002, -5.64814814814814589E-002,
  5.27777777777777568E-002, -1.38888888888888794E-002, -8.24074074074073876E-002,
  4.90740740740740339E-002, -1.38888888888889037E-002,  4.72222222222222002E-002,
 -8.24074074074073876E-002,  5.27777777777777637E-002,  1.38888888888888812E-002,
 -5.64814814814814728E-002,  1.38888888888888916E-002, -1.38888888888888968E-002,
  0.18981481481481480,      -5.27777777777777499E-002,  5.27777777777777707E-002,
  1.20370370370370180E-002,  1.38888888888888812E-002, -1.38888888888888812E-002,
 -7.87037037037036785E-002,  4.72222222222222002E-002, -4.72222222222222071E-002,
 -8.24074074074073876E-002, -1.38888888888888742E-002, -5.27777777777777568E-002,
  4.90740740740740616E-002, -4.72222222222222002E-002,  1.38888888888888846E-002,
  1.38888888888889124E-002, -5.64814814814814728E-002,  1.38888888888888985E-002,
  5.27777777777777568E-002, -8.24074074074073876E-002, -1.38888888888888708E-002,
 -1.38888888888889037E-002,  4.90740740740740339E-002, -4.72222222222221932E-002,
 -5.27777777777777499E-002,  0.18981481481481480,      -5.27777777777777568E-002,
 -1.38888888888888673E-002, -8.24074074074073598E-002,  5.27777777777777429E-002,
  4.72222222222222002E-002, -7.87037037037036785E-002,  4.72222222222222002E-002,
  1.38888888888888708E-002,  1.20370370370370128E-002,  1.38888888888888760E-002,
 -4.72222222222222002E-002,  4.90740740740740478E-002, -1.38888888888888951E-002,
  4.72222222222222071E-002, -1.38888888888888985E-002,  4.90740740740740339E-002,
 -1.38888888888888812E-002,  1.38888888888888881E-002,  1.20370370370370267E-002,
  1.38888888888888951E-002, -4.72222222222222210E-002,  4.90740740740740339E-002,
  5.27777777777777707E-002, -5.27777777777777568E-002,  0.18981481481481477,
  1.38888888888888829E-002,  5.27777777777777707E-002, -8.24074074074073598E-002,
 -4.72222222222222140E-002,  4.72222222222222140E-002, -7.87037037037036646E-002,
 -5.27777777777777707E-002, -1.38888888888888829E-002, -8.24074074074073876E-002,
 -1.38888888888888881E-002,  1.38888888888888881E-002, -5.64814814814814589E-002,
  4.90740740740740339E-002,  4.72222222222221932E-002, -1.38888888888888985E-002,
 -8.24074074074073876E-002,  1.38888888888888742E-002,  5.27777777777777568E-002,
 -7.87037037037036785E-002, -4.72222222222221932E-002,  4.72222222222221932E-002,
  1.20370370370370180E-002, -1.38888888888888673E-002,  1.38888888888888829E-002,
  0.18981481481481469,       5.27777777777777429E-002, -5.27777777777777429E-002,
 -5.64814814814814659E-002, -1.38888888888889055E-002,  1.38888888888889055E-002,
 -8.24074074074074153E-002, -5.27777777777777429E-002, -1.38888888888888760E-002,
  4.90740740740740408E-002,  1.38888888888888968E-002, -4.72222222222222071E-002,
  4.72222222222221932E-002,  4.90740740740740478E-002, -1.38888888888888968E-002,
 -1.38888888888888742E-002,  1.20370370370370232E-002,  1.38888888888888812E-002,
 -4.72222222222222002E-002, -7.87037037037036924E-002,  4.72222222222222071E-002,
  1.38888888888888812E-002, -8.24074074074073598E-002,  5.27777777777777707E-002,
  5.27777777777777429E-002,  0.18981481481481477,      -5.27777777777777499E-002,
  1.38888888888889107E-002,  4.90740740740740478E-002, -4.72222222222221932E-002,
 -5.27777777777777568E-002, -8.24074074074074153E-002, -1.38888888888888812E-002,
 -1.38888888888888846E-002, -5.64814814814814659E-002,  1.38888888888888812E-002,
  1.38888888888888968E-002,  1.38888888888888968E-002, -5.64814814814814520E-002,
  5.27777777777777499E-002, -1.38888888888888812E-002, -8.24074074074073876E-002,
  4.72222222222221932E-002,  4.72222222222222002E-002, -7.87037037037036646E-002,
 -1.38888888888888812E-002,  5.27777777777777429E-002, -8.24074074074073598E-002,
 -5.27777777777777429E-002, -5.27777777777777499E-002,  0.18981481481481474,
 -1.38888888888888985E-002, -4.72222222222221863E-002,  4.90740740740740339E-002,
  1.38888888888888829E-002,  1.38888888888888777E-002,  1.20370370370370249E-002,
 -4.72222222222222002E-002, -1.38888888888888933E-002,  4.90740740740740339E-002,
 -8.24074074074073876E-002, -1.38888888888888673E-002, -5.27777777777777568E-002,
  4.90740740740740269E-002, -4.72222222222221863E-002,  1.38888888888889124E-002,
  1.20370370370370128E-002,  1.38888888888888742E-002, -1.38888888888888742E-002,
 -7.87037037037036785E-002,  4.72222222222222002E-002, -4.72222222222222140E-002,
 -5.64814814814814659E-002,  1.38888888888889107E-002, -1.38888888888888985E-002,
  0.18981481481481474,      -5.27777777777777499E-002,  5.27777777777777499E-002,
  4.90740740740740339E-002, -1.38888888888889055E-002,  4.72222222222221932E-002,
 -8.24074074074074153E-002,  5.27777777777777499E-002,  1.38888888888888829E-002,
  1.38888888888888673E-002,  1.20370370370370058E-002,  1.38888888888888777E-002,
 -4.72222222222221863E-002,  4.90740740740740339E-002, -1.38888888888889055E-002,
 -1.38888888888888725E-002, -8.24074074074073876E-002,  5.27777777777777499E-002,
  4.72222222222222002E-002, -7.87037037037036785E-002,  4.72222222222222140E-002,
 -1.38888888888889055E-002,  4.90740740740740478E-002, -4.72222222222221863E-002,
 -5.27777777777777499E-002,  0.18981481481481469,      -5.27777777777777499E-002,
  1.38888888888889072E-002, -5.64814814814814659E-002,  1.38888888888889003E-002,
  5.27777777777777429E-002, -8.24074074074074153E-002, -1.38888888888888812E-002,
 -5.27777777777777429E-002, -1.38888888888888742E-002, -8.24074074074073876E-002,
 -1.38888888888889089E-002,  1.38888888888888933E-002, -5.64814814814814589E-002,
  1.38888888888888812E-002,  5.27777777777777429E-002, -8.24074074074073737E-002,
 -4.72222222222222071E-002,  4.72222222222222002E-002, -7.87037037037036646E-002,
  1.38888888888889055E-002, -4.72222222222221932E-002,  4.90740740740740339E-002,
  5.27777777777777499E-002, -5.27777777777777499E-002,  0.18981481481481474,
  4.72222222222222002E-002, -1.38888888888888985E-002,  4.90740740740740339E-002,
 -1.38888888888888846E-002,  1.38888888888888812E-002,  1.20370370370370284E-002,
 -7.87037037037036785E-002, -4.72222222222221932E-002, -4.72222222222222002E-002,
  1.20370370370370162E-002, -1.38888888888888812E-002, -1.38888888888888812E-002,
  4.90740740740740408E-002,  4.72222222222222002E-002,  1.38888888888888933E-002,
 -8.24074074074073876E-002,  1.38888888888888708E-002, -5.27777777777777707E-002,
 -8.24074074074074153E-002, -5.27777777777777568E-002,  1.38888888888888829E-002,
  4.90740740740740339E-002,  1.38888888888889072E-002,  4.72222222222222002E-002,
  0.18981481481481477,       5.27777777777777429E-002,  5.27777777777777568E-002,
 -5.64814814814814659E-002, -1.38888888888888846E-002, -1.38888888888888881E-002,
 -4.72222222222221932E-002, -7.87037037037036785E-002, -4.72222222222221932E-002,
  1.38888888888888673E-002, -8.24074074074073876E-002, -5.27777777777777568E-002,
  4.72222222222222002E-002,  4.90740740740740269E-002,  1.38888888888889020E-002,
 -1.38888888888888742E-002,  1.20370370370370128E-002, -1.38888888888888829E-002,
 -5.27777777777777429E-002, -8.24074074074074153E-002,  1.38888888888888777E-002,
 -1.38888888888889055E-002, -5.64814814814814659E-002, -1.38888888888888985E-002,
  5.27777777777777429E-002,  0.18981481481481469,       5.27777777777777429E-002,
  1.38888888888888933E-002,  4.90740740740740339E-002,  4.72222222222222071E-002,
 -4.72222222222222071E-002, -4.72222222222222002E-002, -7.87037037037036646E-002,
  1.38888888888888742E-002, -5.27777777777777568E-002, -8.24074074074073737E-002,
 -1.38888888888888951E-002, -1.38888888888888951E-002, -5.64814814814814589E-002,
 -5.27777777777777568E-002,  1.38888888888888760E-002, -8.24074074074073876E-002,
 -1.38888888888888760E-002, -1.38888888888888812E-002,  1.20370370370370249E-002,
  4.72222222222221932E-002,  1.38888888888889003E-002,  4.90740740740740339E-002,
  5.27777777777777568E-002,  5.27777777777777429E-002,  0.18981481481481474,
  1.38888888888888933E-002,  4.72222222222222071E-002,  4.90740740740740339E-002,
  1.20370370370370180E-002,  1.38888888888888742E-002,  1.38888888888888794E-002,
 -7.87037037037036785E-002,  4.72222222222222071E-002,  4.72222222222222002E-002,
 -8.24074074074073876E-002, -1.38888888888888846E-002,  5.27777777777777568E-002,
  4.90740740740740616E-002, -4.72222222222222002E-002, -1.38888888888888881E-002,
  4.90740740740740408E-002, -1.38888888888888846E-002, -4.72222222222222002E-002,
 -8.24074074074074153E-002,  5.27777777777777429E-002, -1.38888888888888846E-002,
 -5.64814814814814659E-002,  1.38888888888888933E-002,  1.38888888888888933E-002,
  0.18981481481481477,      -5.27777777777777568E-002, -5.27777777777777637E-002,
 -1.38888888888888742E-002, -8.24074074074073598E-002, -5.27777777777777568E-002,
  4.72222222222222002E-002, -7.87037037037036924E-002, -4.72222222222222002E-002,
  1.38888888888888812E-002,  1.20370370370370267E-002, -1.38888888888888794E-002,
 -4.72222222222222002E-002,  4.90740740740740478E-002,  1.38888888888888881E-002,
  1.38888888888888968E-002, -5.64814814814814659E-002, -1.38888888888888933E-002,
  5.27777777777777499E-002, -8.24074074074074153E-002,  1.38888888888888812E-002,
 -1.38888888888888846E-002,  4.90740740740740339E-002,  4.72222222222222071E-002,
 -5.27777777777777568E-002,  0.18981481481481477,       5.27777777777777637E-002,
 -1.38888888888888829E-002, -5.27777777777777568E-002, -8.24074074074073598E-002,
  4.72222222222222071E-002, -4.72222222222222140E-002, -7.87037037037036924E-002,
  5.27777777777777637E-002,  1.38888888888888916E-002, -8.24074074074073876E-002,
  1.38888888888888846E-002, -1.38888888888888951E-002, -5.64814814814814589E-002,
 -4.72222222222222071E-002,  1.38888888888888812E-002,  4.90740740740740339E-002,
  1.38888888888888829E-002, -1.38888888888888812E-002,  1.20370370370370284E-002,
 -1.38888888888888881E-002,  4.72222222222222071E-002,  4.90740740740740339E-002,
 -5.27777777777777637E-002,  5.27777777777777637E-002,  0.18981481481481477,
};

namespace amgx
{

// ---------------------------------------------------------------------------
// Build the Q1 hex FEM stiffness matrix in block CSR format (block_size=3).
//
// Grid: ne x ne x ne elements, nn = ne+1 nodes per side.
// Node numbering: node(i,j,k) = i + j*nn + k*nn*nn.
//
// Dirichlet BC: k=0 face (clamped end) applied POST-ASSEMBLY via
//   MatZeroRowsColumns equivalent: for all z=0 face nodes (indices 0..nn*nn-1),
//   zero all off-diagonal entries in that block row AND column, then set
//   diagonal block to 0.1 * original_diagonal.  This matches PETSc ex99.c's
//   MatZeroRowsColumns(Amat, nn_x*nn_y, rows, 0.1, NULL, NULL) call.
//   The result is a symmetric matrix.
//
// RHS: vv[i] = h^2 (x-comp), 2*h^2 (y-comp), 0 (z-comp) per node.
//      BC elements (ek==0) only load top 4 nodes (local indices 12-23).
//      BC node RHS entries are zeroed post-assembly.
//
// Output (host, TConfig_h):
//   A_out  — block CSR matrix with block_size=3
//   b_out  — RHS vector, length num_nodes*3
//   coords — node coordinates, length 3*num_nodes (interleaved x,y,z)
// ---------------------------------------------------------------------------
template <class TConfig_h>
static void assemble_elasticity_3d(int ne,
                                   Matrix<TConfig_h> &A_out,
                                   std::vector<double> &b_out,
                                   std::vector<double> &coords,
                                   bool apply_bcs = true)
{
    const int nn        = ne + 1;
    const int num_nodes = nn * nn * nn;
    const double h      = 1.0 / ne;
    const int bs        = 3;

    // --- node coordinates ---
    coords.resize(3 * num_nodes);
    for (int k = 0; k < nn; ++k)
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i)
            {
                int n = i + j * nn + k * nn * nn;
                coords[3 * n + 0] = h * i;
                coords[3 * n + 1] = h * j;
                coords[3 * n + 2] = h * k;
            }

    // --- build node-level sparsity (which nodes share an element) ---
    std::vector<std::vector<int>> adj(num_nodes);
    for (int ek = 0; ek < ne; ++ek)
        for (int ej = 0; ej < ne; ++ej)
            for (int ei = 0; ei < ne; ++ei)
            {
                int idx[8];
                idx[0] = (ei  ) + (ej  ) * nn + (ek  ) * nn * nn;
                idx[1] = (ei+1) + (ej  ) * nn + (ek  ) * nn * nn;
                idx[2] = (ei+1) + (ej+1) * nn + (ek  ) * nn * nn;
                idx[3] = (ei  ) + (ej+1) * nn + (ek  ) * nn * nn;
                idx[4] = (ei  ) + (ej  ) * nn + (ek+1) * nn * nn;
                idx[5] = (ei+1) + (ej  ) * nn + (ek+1) * nn * nn;
                idx[6] = (ei+1) + (ej+1) * nn + (ek+1) * nn * nn;
                idx[7] = (ei  ) + (ej+1) * nn + (ek+1) * nn * nn;

                for (int a = 0; a < 8; ++a)
                    for (int b = 0; b < 8; ++b)
                        adj[idx[a]].push_back(idx[b]);
            }

    // deduplicate and sort each adjacency list
    for (int n = 0; n < num_nodes; ++n)
    {
        std::sort(adj[n].begin(), adj[n].end());
        adj[n].erase(std::unique(adj[n].begin(), adj[n].end()), adj[n].end());
    }

    // --- build CSR row_offsets and col_indices (node level) ---
    int total_nnz = 0;
    for (int n = 0; n < num_nodes; ++n)
        total_nnz += (int)adj[n].size();

    std::vector<int> row_off(num_nodes + 1, 0);
    std::vector<int> col_idx(total_nnz);
    {
        int pos = 0;
        for (int n = 0; n < num_nodes; ++n)
        {
            row_off[n] = pos;
            for (int c : adj[n])
                col_idx[pos++] = c;
        }
        row_off[num_nodes] = pos;
    }

    // --- allocate values array (block_size=3, so 9 doubles per entry) ---
    std::vector<double> vals(total_nnz * bs * bs, 0.0);

    // helper: find position of col c in row r (binary search, adj lists sorted)
    auto find_pos = [&](int r, int c) -> int {
        int lo = row_off[r], hi = row_off[r + 1];
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (col_idx[mid] == c) return mid;
            else if (col_idx[mid] < c) lo = mid + 1;
            else hi = mid;
        }
        return -1;
    };

    // --- element load vectors (matching PETSc ex99.c) ---
    // vv: full element load — [h^2, 2h^2, 0] per node (all 8 nodes)
    // v2: BC element load   — only top 4 nodes (local indices 12-23) get load
    double vv[24], v2[24];
    for (int i = 0; i < 24; ++i)
    {
        if      (i % 3 == 0) vv[i] = h * h;
        else if (i % 3 == 1) vv[i] = 2.0 * h * h;
        else                 vv[i] = 0.0;
    }
    for (int i = 0; i < 24; ++i)
    {
        if      (i % 3 == 0 && i >= 12) v2[i] = h * h;
        else if (i % 3 == 1 && i >= 12) v2[i] = 2.0 * h * h;
        else                             v2[i] = 0.0;
    }

    // --- RHS vector ---
    b_out.assign(num_nodes * bs, 0.0);

    // --- element assembly using DD1 for all elements ---
    // BCs are applied post-assembly via MatZeroRowsColumns equivalent below.
    for (int ek = 0; ek < ne; ++ek)
        for (int ej = 0; ej < ne; ++ej)
            for (int ei = 0; ei < ne; ++ei)
            {
                int idx[8];
                idx[0] = (ei  ) + (ej  ) * nn + (ek  ) * nn * nn;
                idx[1] = (ei+1) + (ej  ) * nn + (ek  ) * nn * nn;
                idx[2] = (ei+1) + (ej+1) * nn + (ek  ) * nn * nn;
                idx[3] = (ei  ) + (ej+1) * nn + (ek  ) * nn * nn;
                idx[4] = (ei  ) + (ej  ) * nn + (ek+1) * nn * nn;
                idx[5] = (ei+1) + (ej  ) * nn + (ek+1) * nn * nn;
                idx[6] = (ei+1) + (ej+1) * nn + (ek+1) * nn * nn;
                idx[7] = (ei  ) + (ej+1) * nn + (ek+1) * nn * nn;

                // Use DD1 for all elements; BCs applied post-assembly.
                // RHS: BC elements (ek==0) only load top 4 nodes (v2),
                //      interior elements load all 8 nodes (vv).
                const double *lv = (ek == 0) ? v2 : vv;

                // Accumulate element stiffness into global matrix
                for (int a = 0; a < 8; ++a)
                    for (int b = 0; b < 8; ++b)
                    {
                        int pos = find_pos(idx[a], idx[b]);
                        for (int di = 0; di < bs; ++di)
                            for (int dj = 0; dj < bs; ++dj)
                            {
                                int elem_row = a * bs + di;
                                int elem_col = b * bs + dj;
                                vals[pos * bs * bs + di * bs + dj] +=
                                    DD1[elem_row * 24 + elem_col];
                            }
                    }

                // Accumulate element load into global RHS
                for (int a = 0; a < 8; ++a)
                    for (int di = 0; di < bs; ++di)
                        b_out[idx[a] * bs + di] += lv[a * bs + di];
            }

    // --- Post-assembly BC: MatZeroRowsColumns equivalent ---
    // Zero rows AND columns of all z=0 face nodes (node indices 0..nn*nn-1),
    // set diagonal block to 0.1 * diag(DD1_diag) matching PETSc ex99.c.
    // This is the authoritative BC enforcement (symmetric).
    if (apply_bcs)
    {
        const int n_bc_nodes = nn * nn; // all nodes with k=0
        // Step 1: save diagonal values of BC nodes before zeroing
        std::vector<double> bc_diag(n_bc_nodes * bs, 0.0);
        for (int r = 0; r < n_bc_nodes; ++r)
        {
            int pos = find_pos(r, r);
            for (int di = 0; di < bs; ++di)
                bc_diag[r * bs + di] = vals[pos * bs * bs + di * bs + di];
        }

        // Step 2: zero all entries in BC rows and BC columns
        for (int r = 0; r < num_nodes; ++r)
        {
            for (int p = row_off[r]; p < row_off[r + 1]; ++p)
            {
                int c = col_idx[p];
                bool bc_row = (r < n_bc_nodes);
                bool bc_col = (c < n_bc_nodes);
                if (bc_row || bc_col)
                {
                    for (int di = 0; di < bs; ++di)
                        for (int dj = 0; dj < bs; ++dj)
                            vals[p * bs * bs + di * bs + dj] = 0.0;
                }
            }
        }

        // Step 3: restore diagonal of BC nodes scaled by 0.1
        for (int r = 0; r < n_bc_nodes; ++r)
        {
            int pos = find_pos(r, r);
            for (int di = 0; di < bs; ++di)
                vals[pos * bs * bs + di * bs + di] = 0.1 * bc_diag[r * bs + di];
        }

        // Step 4: zero RHS entries for BC nodes
        for (int r = 0; r < n_bc_nodes; ++r)
            for (int di = 0; di < bs; ++di)
                b_out[r * bs + di] = 0.0;
    }

    // --- symmetry check: sample a few off-diagonal entries ---
    {
        double max_asym = 0.0;
        int nsamp = 0;
        for (int r = 0; r < num_nodes && nsamp < 500; ++r)
        {
            for (int p = row_off[r]; p < row_off[r + 1] && nsamp < 500; ++p)
            {
                int c = col_idx[p];
                if (c == r) continue;
                int q = find_pos(c, r);
                if (q < 0) continue;
                for (int di = 0; di < bs; ++di)
                    for (int dj = 0; dj < bs; ++dj)
                    {
                        double aij = vals[p * bs * bs + di * bs + dj];
                        double aji = vals[q * bs * bs + dj * bs + di];
                        double diff = std::fabs(aij - aji);
                        if (diff > max_asym) max_asym = diff;
                    }
                ++nsamp;
            }
        }
        printf("Symmetry check (sampled %d off-diag block pairs): max|A[i,j]-A[j,i]| = %.3e\n",
               nsamp, max_asym);
        if (max_asym < 1e-12)
            printf("  => Matrix is SYMMETRIC (to machine precision)\n");
        else
            printf("  => WARNING: Matrix is NOT symmetric!\n");
        fflush(stdout);
    }

    // --- fill AMGx host matrix ---
    A_out.set_initialized(0);
    A_out.addProps(CSR);
    A_out.set_block_dimy(bs);
    A_out.set_block_dimx(bs);
    A_out.resize(num_nodes, num_nodes, total_nnz, bs, bs, 1);

    for (int i = 0; i <= num_nodes; ++i)
        A_out.row_offsets[i] = row_off[i];
    for (int i = 0; i < total_nnz; ++i)
        A_out.col_indices[i] = col_idx[i];
    for (int i = 0; i < total_nnz * bs * bs; ++i)
        A_out.values[i] = static_cast<typename Matrix<TConfig_h>::value_type>(vals[i]);

    A_out.set_initialized(1);
}

} // namespace amgx

// ---------------------------------------------------------------------------
// Host-side block CSR SpMV: y = A * x  (block_size = bs)
// A is stored as: row_offsets[num_nodes+1], col_indices[nnz], values[nnz*bs*bs]
// x, y are scalar vectors of length num_nodes * bs.
// ---------------------------------------------------------------------------
static void host_block_spmv(int num_nodes, int bs,
                            const int *row_off, const int *col_idx,
                            const double *vals,
                            const double *x, double *y)
{
    for (int r = 0; r < num_nodes; ++r)
    {
        for (int di = 0; di < bs; ++di)
            y[r * bs + di] = 0.0;

        for (int p = row_off[r]; p < row_off[r + 1]; ++p)
        {
            int c = col_idx[p];
            for (int di = 0; di < bs; ++di)
                for (int dj = 0; dj < bs; ++dj)
                    y[r * bs + di] += vals[p * bs * bs + di * bs + dj] * x[c * bs + dj];
        }
    }
}

// ---------------------------------------------------------------------------
// Null-space test: assemble A without BCs, verify A*B=0 at fine level,
// build AMG hierarchy, verify Ac*Bc=0 at each coarse level.
// Returns 0 on success (all norms < tol), 1 on failure.
// ---------------------------------------------------------------------------
static int run_null_test(int ne)
{
    using namespace amgx;

    typedef TemplateMode<AMGX_mode_dDDI>::Type TConfig;
    typedef TConfig::template setMemSpace<AMGX_host>::Type   TConfig_h;
    typedef TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;
    typedef Matrix<TConfig_h> Matrix_h;
    typedef Matrix<TConfig_d> MatrixA;
    typedef Vector<TConfig_d> VVector;
    typedef Vector<TConfig_h> VVector_h;

    const int nn        = ne + 1;
    const int bs        = 3;
    const int num_nodes = nn * nn * nn;
    const int num_dofs  = num_nodes * bs;

    printf("=== NULL SPACE TEST (no BCs) ===\n");
    printf("ne=%d  nodes=%dx%dx%d  M=%d\n", ne, nn, nn, nn, num_dofs);

    // --- Assemble A WITHOUT BCs ---
    Matrix_h A_h;
    std::vector<double> b_host, coords;
    assemble_elasticity_3d<TConfig_h>(ne, A_h, b_host, coords, /*apply_bcs=*/false);

    // --- Compute 6 rigid body mode near-null space vectors ---
    int null_dim = 0;
    std::vector<double> B(6 * num_dofs, 0.0);
    aggregation::computeRigidBodyModes(3, bs, num_nodes,
                                       coords.data(), null_dim, B.data());

    // --- Fine-level check: A * B_k = 0 for each null vector k ---
    printf("\n--- Fine-level null space check: A_noBCs * B ---\n");
    {
        // Extract CSR arrays from host matrix
        std::vector<int> row_off(num_nodes + 1);
        std::vector<int> col_idx(A_h.row_offsets[num_nodes]);
        std::vector<double> vals(A_h.row_offsets[num_nodes] * bs * bs);
        for (int i = 0; i <= num_nodes; ++i)
            row_off[i] = A_h.row_offsets[i];
        for (int i = 0; i < A_h.row_offsets[num_nodes]; ++i)
            col_idx[i] = A_h.col_indices[i];
        for (int i = 0; i < (int)vals.size(); ++i)
            vals[i] = (double)A_h.values[i];

        std::vector<double> y(num_dofs, 0.0);
        bool fine_ok = true;
        for (int k = 0; k < null_dim; ++k)
        {
            const double *bk = &B[k * num_dofs];
            host_block_spmv(num_nodes, bs, row_off.data(), col_idx.data(),
                            vals.data(), bk, y.data());
            double norm = 0.0;
            for (int i = 0; i < num_dofs; ++i) norm += y[i] * y[i];
            norm = std::sqrt(norm);
            printf("  ||A * B_%d|| = %.6e\n", k, norm);
            if (norm > 1e-10) fine_ok = false;
        }
        if (!fine_ok) {
            printf("  FAIL: fine-level null space not in kernel of A_noBCs!\n");
            return 1;
        }
        printf("  PASS: fine-level A_noBCs * B = 0 (to machine precision)\n");
    }

    // --- Build AMG hierarchy from A_noBCs ---
    MatrixA A_d = A_h;
    A_d.computeDiagonal();
    A_d.set_initialized(1);

    AMGX_ERROR init_err = amgx::initialize();
    if (init_err != AMGX_OK) { fprintf(stderr, "init failed\n"); return 1; }

    // Simple AMG config for null-test (setup only, no solve needed)
    const char *cfg_json =
        "{\n"
        "  \"config_version\": 2,\n"
        "  \"solver\": {\n"
        "    \"solver\": \"PCG\",\n"
        "    \"max_iters\": 1,\n"
        "    \"scope\": \"main\",\n"
        "    \"preconditioner\": {\n"
        "      \"scope\": \"amg\",\n"
        "      \"solver\": \"AMG\",\n"
        "      \"max_iters\": 1,\n"
        "      \"algorithm\": \"AGGREGATION\",\n"
        "      \"selector\": \"MIS\",\n"
        "      \"mis_k\": 1,\n"
        "      \"mis_strength_threshold\": 0.0,\n"
        "      \"max_levels\": 10,\n"
        "      \"min_coarse_rows\": 10,\n"
        "      \"cycle\": \"V\",\n"
        "      \"presweeps\": 2,\n"
        "      \"postsweeps\": 2,\n"
        "      \"smoother\": {\n"
        "        \"solver\": \"JACOBI_L1\",\n"
        "        \"relaxation_factor\": 1.0\n"
        "      },\n"
        "      \"coarse_solver\": {\n"
        "        \"scope\": \"coarse_lu\",\n"
        "        \"solver\": \"DENSE_LU_SOLVER\"\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n";

    AMG_Configuration cfg;
    if (cfg.parseParameterString(cfg_json) != AMGX_OK) {
        fprintf(stderr, "config parse failed\n"); return 1;
    }

    Resources res;
    AMG_Solver<TConfig> solver(&res, cfg);
    solver.setNearNullSpace(null_dim, num_dofs, B.data());
    if (solver.setup(A_d) != AMGX_OK) {
        fprintf(stderr, "setup failed\n"); return 1;
    }

    // --- Walk AMG hierarchy and check null space at each coarse level ---
    printf("\n--- Coarse-level null space check ---\n");
    typedef AMG<TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> AMG_Class;
    typedef typename TConfig_d::MemSpace device_memory;

    // Get the AMG object from the solver hierarchy:
    //   AMG_Solver -> PCG_Solver (outer) -> AlgebraicMultigrid_Solver (prec)
    Solver<TConfig> *solverObj = solver.getSolverObject();
    auto *pcgSolver = dynamic_cast<PCG_Solver<TConfig>*>(solverObj);
    if (!pcgSolver) {
        fprintf(stderr, "ERROR: outer solver is not PCG_Solver\n");
        return 1;
    }
    Solver<TConfig> *precObj = pcgSolver->getPreconditioner();
    auto *amgSolver = dynamic_cast<AlgebraicMultigrid_Solver<TConfig>*>(precObj);
    if (!amgSolver) {
        fprintf(stderr, "ERROR: preconditioner is not AlgebraicMultigrid_Solver\n");
        return 1;
    }
    const AMG_Class &amg = amgSolver->getAMG();

    // Walk levels: finest → coarsest
    typedef AMG_Level<TConfig_d> Level_d;
    Level_d *level = amg.getFinestLevel(device_memory());
    int lvl_idx = 0;
    bool all_ok = true;

    while (level != nullptr)
    {
        // Get coarse matrix Ac at this level
        MatrixA &Ac_d = level->getA();
        int nrows  = Ac_d.get_num_rows();
        int bsy    = Ac_d.get_block_dimy();
        int ndofs  = nrows * bsy;
        int nnz    = Ac_d.row_offsets[nrows];

        printf("  Level %d: N=%d (bs=%d, %d DOFs), nnz=%d\n",
               lvl_idx, nrows, bsy, ndofs, nnz);

        // Try to get the near-null space from the aggregation level
        auto *aggLevel = dynamic_cast<
            aggregation::Aggregation_AMG_Level_Base<TConfig_d>*>(level);

        if (aggLevel && aggLevel->getNullDim() > 0)
        {
            int nd = aggLevel->getNullDim();
            const VVector &ns_d = aggLevel->getNearNullSpace();
            int ns_size = ns_d.size();

            if (ns_size > 0 && ns_size == nd * ndofs)
            {
                // Copy Ac and null space to host
                Matrix_h Ac_h = Ac_d;
                VVector_h ns_h = ns_d;

                // Extract host CSR
                std::vector<int> ro(nrows + 1), ci(nnz);
                std::vector<double> va(nnz * bsy * bsy);
                for (int i = 0; i <= nrows; ++i) ro[i] = Ac_h.row_offsets[i];
                for (int i = 0; i < nnz; ++i) ci[i] = Ac_h.col_indices[i];
                for (int i = 0; i < (int)va.size(); ++i) va[i] = (double)Ac_h.values[i];

                // Check Ac * B_k for each null vector
                std::vector<double> bk(ndofs), yk(ndofs);
                for (int k = 0; k < nd; ++k)
                {
                    for (int i = 0; i < ndofs; ++i)
                        bk[i] = (double)ns_h[k * ndofs + i];

                    host_block_spmv(nrows, bsy, ro.data(), ci.data(),
                                    va.data(), bk.data(), yk.data());
                    double norm = 0.0;
                    for (int i = 0; i < ndofs; ++i) norm += yk[i] * yk[i];
                    norm = std::sqrt(norm);
                    printf("    ||Ac * B_%d|| = %.6e%s\n", k, norm,
                           norm > 1e-8 ? "  *** FAIL ***" : "");
                    if (norm > 1e-8) all_ok = false;
                }
            }
            else
            {
                printf("    near-null space size mismatch: ns_size=%d, expected=%d\n",
                       ns_size, nd * ndofs);
            }
        }
        else
        {
            printf("    (no near-null space at this level)\n");
        }

        // Advance to next level
        Level_d *next = level->getNextLevel(device_memory());
        if (!next) break;
        level = next;
        ++lvl_idx;
    }

    printf("\n%s\n", all_ok ? "=== NULL SPACE TEST PASSED ===" : "=== NULL SPACE TEST FAILED ===");
    return all_ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
    using namespace amgx;

    // Template mode: device double, double, int
    typedef TemplateMode<AMGX_mode_dDDI>::Type TConfig;
    typedef TConfig::template setMemSpace<AMGX_host>::Type   TConfig_h;
    typedef TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;

    typedef Matrix<TConfig_h> Matrix_h;
    typedef Matrix<TConfig_d> MatrixA;
    typedef Vector<TConfig_d> VVector;

    // --- Parse command line ---
    // mis_mode: 0 = MIS-1 (default), 1 = MIS-2 Galerkin, 2 = MIS-2 implicit
    int ne = 9;
    int mis_mode = 0;
    int min_coarse_rows = 10;
    bool null_test = false;
    bool use_fgmres = false;
    const char *agg_file = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--null-test") == 0) {
            null_test = true;
        } else if (strcmp(argv[i], "--agg-file") == 0 && i + 1 < argc) {
            agg_file = argv[++i];
        } else if (strcmp(argv[i], "--mis2-galerkin") == 0) {
            mis_mode = 1;
        } else if (strcmp(argv[i], "--mis2-implicit") == 0) {
            mis_mode = 2;
        } else if (strcmp(argv[i], "--min-coarse") == 0 && i + 1 < argc) {
            min_coarse_rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fgmres") == 0) {
            use_fgmres = true;
        } else if (strcmp(argv[i], "--jacobi-l1") == 0) {
            /* no-op: JACOBI_L1 smoother is always used (hardcoded in JSON config) */
        } else {
            ne = atoi(argv[i]);
        }
    }

    // --- Null-space test mode ---
    if (null_test) {
        return run_null_test(ne);
    }

    const int nn        = ne + 1;
    const int bs        = 3;
    const int num_nodes = nn * nn * nn;
    const int num_dofs  = num_nodes * bs;

    printf("ne=%d  nodes=%dx%dx%d  M=%d\n",
           ne, nn, nn, nn, num_dofs);
    fflush(stdout);

    // --- Assemble FEM stiffness matrix and RHS on host ---
    Matrix_h A_h;
    std::vector<double> b_host, coords;
    assemble_elasticity_3d<TConfig_h>(ne, A_h, b_host, coords);

    // --- Diagnostic: NNZ counts ---
    {
        int total_block_nnz  = A_h.row_offsets[num_nodes];
        int total_scalar_nnz = total_block_nnz * bs * bs;
        printf("Fine level: %d DOFs, %.0f avg nnz/row\n",
               num_dofs,
               (double)total_scalar_nnz / num_dofs);
        fflush(stdout);
    }

    // --- Compute 6 rigid body mode near-null space vectors ---
    int null_dim = 0;
    std::vector<double> B(6 * num_nodes * bs, 0.0);
    aggregation::computeRigidBodyModes(3, bs, num_nodes,
                                       coords.data(), null_dim, B.data());
    fflush(stdout);

    // --- Transfer matrix to device ---
    MatrixA A_d = A_h;
    A_d.computeDiagonal();
    A_d.set_initialized(1);

    // --- RHS vector to device ---
    VVector b_d(num_dofs), x_d(num_dofs, 0.0);
    b_d.set_block_dimx(1);
    b_d.set_block_dimy(bs);
    x_d.set_block_dimx(1);
    x_d.set_block_dimy(bs);
    for (int i = 0; i < num_dofs; ++i)
        b_d[i] = static_cast<typename VVector::value_type>(b_host[i]);

    // --- Initialize AMGX (registers all solver/smoother factories) ---
    {
        AMGX_ERROR init_err = amgx::initialize();
        if (init_err != AMGX_OK)
        {
            fprintf(stderr, "ERROR: amgx::initialize() failed (err=%d)\n", (int)init_err);
            return 1;
        }
    }

    // --- Configure PCG + SA-AMG ---
    // Smoother: JACOBI_L1 (L1-Jacobi + undamped Richardson, ω=1.0)
    // presweeps=2, postsweeps=2
    // coarse_solver: DENSE_LU_SOLVER
    // MIS selector:
    //   mis_mode=0 (default): MIS-1, standard coarsening (~5-8x per level)
    //   mis_mode=1: MIS-2 Galerkin (mis2_algorithm=0), aggressive_levels=1
    //   mis_mode=2: MIS-2 implicit (mis2_algorithm=1), aggressive_levels=1
    const char *mis_label = (mis_mode == 1) ? "MIS-2 Galerkin (aggressive_levels=1)"
                          : (mis_mode == 2) ? "MIS-2 implicit (aggressive_levels=1)"
                          :                   "MIS-1 (standard)";
    printf("MIS mode: %s\n", mis_label);
    fflush(stdout);

    // Build the MIS-specific portion of the JSON config
    char mis_json_fragment[256];
    if (mis_mode == 0) {
        snprintf(mis_json_fragment, sizeof(mis_json_fragment),
                 "      \"mis_k\": 1,\n");
    } else {
        snprintf(mis_json_fragment, sizeof(mis_json_fragment),
                 "      \"mis_k\": 2,\n"
                 "      \"mis2_algorithm\": %d,\n"
                 "      \"aggressive_levels\": 1,\n",
                 (mis_mode == 1) ? 0 : 1);
    }

    // Assemble full JSON config string
    // Use FGMRES if --fgmres flag was given (FGMRES works with non-SPD preconditioners;
    // if FGMRES converges but PCG doesn't, the preconditioner is non-SPD).
    const char *outer_solver = use_fgmres ? "FGMRES" : "PCG";
    printf("Outer solver: %s\n", outer_solver);
    fflush(stdout);

    char cfg_json_buf[4096];
    snprintf(cfg_json_buf, sizeof(cfg_json_buf),
        "{\n"
        "  \"config_version\": 2,\n"
        "  \"solver\": {\n"
        "    \"solver\": \"%s\",\n"
        "    \"max_iters\": 200,\n"
        "    \"convergence\": \"RELATIVE_INI\",\n"
        "    \"tolerance\": 1e-8,\n"
        "    \"norm\": \"L2\",\n"
        "    \"monitor_residual\": 1,\n"
        "    \"print_solve_stats\": 1,\n"
        "    \"obtain_timings\": 1,\n"
        "    \"scope\": \"main\",\n"
        "    \"preconditioner\": {\n"
        "      \"scope\": \"amg\",\n"
        "      \"solver\": \"AMG\",\n"
        "      \"max_iters\": 1,\n"
        "      \"algorithm\": \"AGGREGATION\",\n"
        "      \"selector\": \"MIS\",\n"
        "%s"
        "      \"mis_strength_threshold\": 0.0,\n"
        "      \"max_levels\": 10,\n"
        "      \"min_coarse_rows\": %d,\n"
        "      \"cycle\": \"V\",\n"
        "      \"presweeps\": 2,\n"
        "      \"postsweeps\": 2,\n"
        "      \"print_grid_stats\": 1,\n"
        "      \"smoother\": {\n"
        "        \"solver\": \"JACOBI_L1\",\n"
        "        \"relaxation_factor\": 1.0\n"
        "      },\n"
        "      \"coarse_solver\": {\n"
        "        \"scope\": \"coarse\",\n"
        "        \"solver\": \"DENSE_LU_SOLVER\"\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n",
        outer_solver, mis_json_fragment, min_coarse_rows);
    const char *cfg_json = cfg_json_buf;
    // Write config to file then load
    {
        FILE *f = fopen("/tmp/amgx_sa_config.json", "w");
        if (!f) { fprintf(stderr, "ERROR: cannot write config file\n"); return 1; }
        fputs(cfg_json, f);
        fclose(f);
    }
    AMG_Configuration cfg;
    AMGX_ERROR cfg_err = cfg.parseParameterString(cfg_json);

    if (cfg_err != AMGX_OK)
    {
        fprintf(stderr, "ERROR: Failed to parse AMG configuration (err=%d)\n",
                (int)cfg_err);
        return 1;
    }

    // --- Create solver, set near-null space, setup, solve ---
    Resources res;
    AMG_Solver<TConfig> solver(&res, cfg);
    solver.setNearNullSpace(null_dim, num_nodes * bs, B.data());

    // Inject aggregate override before setup (no-op if agg_file == nullptr)
    if (agg_file)
    {
        printf("Using aggregate override file: %s\n", agg_file);
        fflush(stdout);
        amgx::aggregation::setAggregateOverrideFile(agg_file);
    }

    AMGX_ERROR setup_err = solver.setup(A_d);
    if (setup_err != AMGX_OK)
    {
        fprintf(stderr, "ERROR: AMG setup failed (err=%d)\n", (int)setup_err);
        return 1;
    }

    // ---------------------------------------------------------------
    // Symmetry tests:  x^T M y  ==  y^T M x   for random x, y
    // Tests: (1) A itself, (2) preconditioner M^{-1}, (3) smoother
    // ---------------------------------------------------------------
    {
        printf("\n=== Symmetry Tests ===\n");
        fflush(stdout);

        typedef typename VVector::value_type VT;

        // Deterministic pseudo-random fill (no curand dependency)
        auto fill_pseudo_random = [](VVector &v, unsigned seed) {
            int n = v.size();
            std::vector<VT> h(n);
            for (int i = 0; i < n; i++) {
                seed = seed * 1103515245u + 12345u;
                double val = ((double)((seed >> 16) & 0x7FFF)) / 32767.0 - 0.5;
                h[i] = static_cast<VT>(val);
            }
            cudaMemcpy(v.raw(), h.data(), n * sizeof(VT), cudaMemcpyHostToDevice);
        };

        // Dot product (download to host, compute)
        auto dot_product = [](const VVector &a, const VVector &b_vec) -> double {
            int n = a.size();
            std::vector<VT> ah(n), bh(n);
            cudaMemcpy(ah.data(), a.raw(), n * sizeof(VT), cudaMemcpyDeviceToHost);
            cudaMemcpy(bh.data(), b_vec.raw(), n * sizeof(VT), cudaMemcpyDeviceToHost);
            double result = 0.0;
            for (int i = 0; i < n; i++)
                result += (double)ah[i] * (double)bh[i];
            return result;
        };

        VVector rx(num_dofs), ry(num_dofs);
        rx.set_block_dimx(1); rx.set_block_dimy(bs);
        ry.set_block_dimx(1); ry.set_block_dimy(bs);

        fill_pseudo_random(rx, 42u);
        fill_pseudo_random(ry, 137u);

        // --- Test 1: A symmetry ---
        {
            VVector Ax_vec(num_dofs), Ay_vec(num_dofs);
            Ax_vec.set_block_dimx(1); Ax_vec.set_block_dimy(bs);
            Ay_vec.set_block_dimx(1); Ay_vec.set_block_dimy(bs);
            multiply(A_d, rx, Ax_vec, OWNED);
            multiply(A_d, ry, Ay_vec, OWNED);
            double xTAy = dot_product(rx, Ay_vec);
            double yTAx = dot_product(ry, Ax_vec);
            double max_val = std::max(fabs(xTAy), fabs(yTAx));
            double rel_diff = (max_val > 0.0) ? fabs(xTAy - yTAx) / max_val : 0.0;
            printf("[SYMM-TEST] A symmetry: x^T A y = %.15e   y^T A x = %.15e   rel_diff = %.4e  %s\n",
                   xTAy, yTAx, rel_diff, (rel_diff < 1e-12) ? "PASS" : "FAIL");
            fflush(stdout);
        }

        // --- Test 2: Preconditioner (V-cycle) symmetry ---
        {
            // Get the preconditioner from the outer solver
            Solver<TConfig> *outer = solver.getSolverObject();
            PCG_Solver<TConfig> *pcg = dynamic_cast<PCG_Solver<TConfig>*>(outer);
            Solver<TConfig> *prec = nullptr;
            if (pcg) prec = pcg->getPreconditioner();

            if (prec) {
                VVector Mrx(num_dofs, 0.0), Mry(num_dofs, 0.0);
                Mrx.set_block_dimx(1); Mrx.set_block_dimy(bs);
                Mry.set_block_dimx(1); Mry.set_block_dimy(bs);

                prec->solve(rx, Mrx, true);   // Mrx = M^{-1} rx
                cudaDeviceSynchronize();
                prec->solve(ry, Mry, true);   // Mry = M^{-1} ry
                cudaDeviceSynchronize();

                double xTMy = dot_product(rx, Mry);
                double yTMx = dot_product(ry, Mrx);
                double max_val = std::max(fabs(xTMy), fabs(yTMx));
                double rel_diff = (max_val > 0.0) ? fabs(xTMy - yTMx) / max_val : 0.0;
                printf("[SYMM-TEST] M^{-1} symmetry: x^T M^{-1} y = %.15e   y^T M^{-1} x = %.15e   rel_diff = %.4e  %s\n",
                       xTMy, yTMx, rel_diff, (rel_diff < 1e-6) ? "PASS" : "FAIL");
                fflush(stdout);

                // Also check positivity: x^T M^{-1} x > 0
                VVector Mrx2(num_dofs, 0.0);
                Mrx2.set_block_dimx(1); Mrx2.set_block_dimy(bs);
                prec->solve(rx, Mrx2, true);
                cudaDeviceSynchronize();
                double xTMx = dot_product(rx, Mrx2);
                printf("[SYMM-TEST] M^{-1} positivity: x^T M^{-1} x = %.15e  %s\n",
                       xTMx, (xTMx > 0.0) ? "PASS" : "FAIL");
                fflush(stdout);
            } else {
                printf("[SYMM-TEST] WARNING: could not get preconditioner (not PCG?)\n");
                fflush(stdout);
            }
        }

        printf("=== End Symmetry Tests ===\n\n");
        fflush(stdout);
    }

    AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
    AMGX_ERROR solve_err = solver.solve(b_d, x_d, solve_status);

    if (solve_err != AMGX_OK)
    {
        fprintf(stderr, "ERROR: Solve returned error %d\n", (int)solve_err);
        return 1;
    }

    int num_iters = solver.get_num_iters();

    printf("\nIterations: %d\n", num_iters);
    if (solve_status == AMGX_ST_CONVERGED)
        printf("Status: CONVERGED\n");
    else
        printf("Status: NOT CONVERGED (status=%d)\n", (int)solve_status);

    return (solve_status == AMGX_ST_CONVERGED) ? 0 : 1;
}
