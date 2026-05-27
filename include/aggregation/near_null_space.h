// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif

namespace amgx
{
namespace aggregation
{

/**
 * @brief Compute rigid body mode vectors from node coordinates.
 *
 * Fills a host double array with the near-null space vectors for
 * linear elasticity (or any PDE whose null space consists of rigid
 * body motions).  The output is column-major:
 *   null vector k occupies rows [k * num_nodes * block_size ..
 *                                (k+1) * num_nodes * block_size - 1]
 *
 * Supported cases:
 *   block_size == 1 : null_dim = 1, B = [1, 1, ..., 1]^T
 *   block_size == 2 : null_dim = 3, B = [e1, e2, rotation(-y, x)]
 *   block_size == 3 : null_dim = 6, B = [e1, e2, e3,
 *                                        rot1(0,-z,y),
 *                                        rot2(z,0,-x),
 *                                        rot3(-y,x,0)]
 *
 * @param dim        Spatial dimension (1, 2, or 3).  Must equal block_size.
 * @param block_size Number of DOFs per node (must equal dim).
 * @param num_nodes  Number of mesh nodes.
 * @param coords     Host array of node coordinates, length num_nodes * dim,
 *                   interleaved: [x0,y0,z0, x1,y1,z1, ...].
 * @param null_dim   Output: number of null vectors produced.
 * @param B          Output: host double array of size null_dim * num_nodes * block_size.
 *                   Caller must pre-allocate with sufficient capacity.
 *                   Maximum size needed: 6 * num_nodes * 3.
 */
void computeRigidBodyModes(int dim,
                           int block_size,
                           int num_nodes,
                           const double *coords,
                           int &null_dim,
                           double *B);

/**
 * @brief Global MGS orthonormalization of rigid body modes (matches PETSc).
 *
 * Must be called after computeRigidBodyModes().  Performs:
 *   1. Scale translations by 1/sqrt(N_global).
 *   2. MGS-orthonormalize each rotation mode against all prior modes,
 *      using MPI_Allreduce for global dot products and norms.
 *
 * @param dim       Spatial dimension (2 or 3).
 * @param null_dim  Total number of null vectors (3 for 2D, 6 for 3D).
 * @param stride    Local DOF count = num_nodes * block_size.
 * @param B         Column-major host array [null_dim * stride], modified in-place.
 * @param mpi_comm  Pointer to MPI communicator; may be nullptr for single-rank.
 */
void orthonormalizeRigidBodyModes(int dim,
                                  int null_dim,
                                  int stride,
                                  double *B
#ifdef AMGX_WITH_MPI
                                  , MPI_Comm *mpi_comm
#endif
                                  );

} // namespace aggregation
} // namespace amgx
