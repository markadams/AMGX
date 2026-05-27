// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <aggregation/near_null_space.h>
#include <error.h>
#include <cmath>    // sqrt
#include <cstring>  // memset

#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif

namespace amgx
{
namespace aggregation
{

// -----------------------------------------------------------------------
// Helper routines for global orthonormalization (Modified Gram-Schmidt)
// -----------------------------------------------------------------------

// Dot product of two vectors of length n
static inline double vec_dot(const double *a, const double *b, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// a[i] += alpha * b[i]  for i in [0, n)
static inline void vec_axpy(double *a, double alpha, const double *b, int n)
{
    for (int i = 0; i < n; ++i) a[i] += alpha * b[i];
}

// Normalize a vector in-place; returns the norm
static inline double vec_normalize(double *a, int n)
{
    double nrm = std::sqrt(vec_dot(a, a, n));
    if (nrm > 0.0)
        for (int i = 0; i < n; ++i) a[i] /= nrm;
    return nrm;
}

// -----------------------------------------------------------------------
// computeRigidBodyModes — CPU implementation
//
// Output array B is column-major:
//   null vector k at B + k * (num_nodes * block_size)
//
// For block_size == 1:
//   1 vector: constant 1
//
// For block_size == 2 (2D, coords = [x,y] per node):
//   3 vectors:
//     k=0: translation in x  -> [1,0, 1,0, ...]
//     k=1: translation in y  -> [0,1, 0,1, ...]
//     k=2: rotation          -> [-y,x, -y,x, ...]
//
// For block_size == 3 (3D, coords = [x,y,z] per node):
//   6 vectors:
//     k=0: translation in x  -> [1,0,0, ...]
//     k=1: translation in y  -> [0,1,0, ...]
//     k=2: translation in z  -> [0,0,1, ...]
//     k=3: rotation about x  -> [0,-z, y, ...]
//     k=4: rotation about y  -> [z, 0,-x, ...]
//     k=5: rotation about z  -> [-y, x, 0, ...]
// -----------------------------------------------------------------------

void computeRigidBodyModes(int dim,
                           int block_size,
                           int num_nodes,
                           const double *coords,
                           int &null_dim,
                           double *B)
{
    if (dim != block_size)
    {
        FatalError("computeRigidBodyModes: dim must equal block_size", AMGX_ERR_BAD_PARAMETERS);
    }
    if (dim < 1 || dim > 3)
    {
        FatalError("computeRigidBodyModes: dim must be 1, 2, or 3", AMGX_ERR_BAD_PARAMETERS);
    }
    if (num_nodes <= 0 || coords == nullptr || B == nullptr)
    {
        FatalError("computeRigidBodyModes: invalid arguments", AMGX_ERR_BAD_PARAMETERS);
    }

    const int stride = num_nodes * block_size; // length of one null vector

    if (dim == 1)
    {
        null_dim = 1;
        // Zero the output first
        memset(B, 0, null_dim * stride * sizeof(double));
        // k=0: constant 1
        double *col0 = B;
        for (int n = 0; n < num_nodes; ++n)
            col0[n] = 1.0;
    }
    else if (dim == 2)
    {
        null_dim = 3;
        memset(B, 0, null_dim * stride * sizeof(double));

        double *col0 = B + 0 * stride; // translation x
        double *col1 = B + 1 * stride; // translation y
        double *col2 = B + 2 * stride; // rotation

        for (int n = 0; n < num_nodes; ++n)
        {
            double x = coords[n * 2 + 0];
            double y = coords[n * 2 + 1];

            // translation x: DOF 0 = 1, DOF 1 = 0
            col0[n * 2 + 0] = 1.0;
            col0[n * 2 + 1] = 0.0;

            // translation y: DOF 0 = 0, DOF 1 = 1
            col1[n * 2 + 0] = 0.0;
            col1[n * 2 + 1] = 1.0;

            // rotation: DOF 0 = -y, DOF 1 = x
            col2[n * 2 + 0] = -y;
            col2[n * 2 + 1] =  x;
        }
    }
    else // dim == 3
    {
        null_dim = 6;
        memset(B, 0, null_dim * stride * sizeof(double));

        double *col0 = B + 0 * stride; // translation x
        double *col1 = B + 1 * stride; // translation y
        double *col2 = B + 2 * stride; // translation z
        double *col3 = B + 3 * stride; // rotation about x: (0, -z,  y)
        double *col4 = B + 4 * stride; // rotation about y: (z,  0, -x)
        double *col5 = B + 5 * stride; // rotation about z: (-y, x,  0)

        for (int n = 0; n < num_nodes; ++n)
        {
            double x = coords[n * 3 + 0];
            double y = coords[n * 3 + 1];
            double z = coords[n * 3 + 2];

            // translation x
            col0[n * 3 + 0] = 1.0;
            col0[n * 3 + 1] = 0.0;
            col0[n * 3 + 2] = 0.0;

            // translation y
            col1[n * 3 + 0] = 0.0;
            col1[n * 3 + 1] = 1.0;
            col1[n * 3 + 2] = 0.0;

            // translation z
            col2[n * 3 + 0] = 0.0;
            col2[n * 3 + 1] = 0.0;
            col2[n * 3 + 2] = 1.0;

            // rotation about x: (0, -z, y)
            col3[n * 3 + 0] =  0.0;
            col3[n * 3 + 1] = -z;
            col3[n * 3 + 2] =  y;

            // rotation about y: (z, 0, -x)
            col4[n * 3 + 0] =  z;
            col4[n * 3 + 1] =  0.0;
            col4[n * 3 + 2] = -x;

            // rotation about z: (-y, x, 0)
            col5[n * 3 + 0] = -y;
            col5[n * 3 + 1] =  x;
            col5[n * 3 + 2] =  0.0;
        }
    }
}

// -----------------------------------------------------------------------
// orthonormalizeRigidBodyModes — global MGS orthonormalization
//
// Matches PETSc MatNullSpaceCreateRigidBody (Vaněk et al. 1995):
//   1. Scale translations by 1/sqrt(N_global) where N_global = total DOFs
//      across all MPI ranks.
//   2. For each rotation mode i (i = dim .. null_dim-1):
//        project out components of modes 0..i-1 using global dot products,
//        then normalize using global norm.
//
// Parameters:
//   dim       - spatial dimension (2 or 3); translations are modes 0..dim-1
//   null_dim  - total number of modes (3 for 2D, 6 for 3D)
//   stride    - local DOF count = num_nodes * block_size
//   B         - column-major host array [null_dim * stride], modified in-place
//   mpi_comm  - pointer to MPI communicator (may be nullptr for single-rank)
// -----------------------------------------------------------------------
void orthonormalizeRigidBodyModes(int dim,
                                  int null_dim,
                                  int stride,
                                  double *B
#ifdef AMGX_WITH_MPI
                                  , MPI_Comm *mpi_comm
#endif
                                  )
{
    if (dim < 2 || null_dim <= dim || stride <= 0 || B == nullptr)
        return; // nothing to do for dim==1 or degenerate inputs

    // ------------------------------------------------------------------
    // 1. Compute global N = total DOFs across all ranks
    // ------------------------------------------------------------------
    long long local_N  = static_cast<long long>(stride);
    long long global_N = local_N;
#ifdef AMGX_WITH_MPI
    if (mpi_comm && *mpi_comm != MPI_COMM_NULL)
        MPI_Allreduce(&local_N, &global_N, 1, MPI_LONG_LONG_INT, MPI_SUM, *mpi_comm);
#endif
    double sN = 1.0 / std::sqrt(static_cast<double>(global_N));

    // ------------------------------------------------------------------
    // 2. Scale translation modes by 1/sqrt(N_global)
    // ------------------------------------------------------------------
    for (int k = 0; k < dim; ++k)
    {
        double *col = B + k * stride;
        for (int i = 0; i < stride; ++i) col[i] *= sN;
    }

    // ------------------------------------------------------------------
    // 3. MGS: orthonormalize each rotation mode against all prior modes
    // ------------------------------------------------------------------
    // cols[k] points to the k-th null vector (local portion)
    double *cols[6];
    for (int k = 0; k < null_dim; ++k)
        cols[k] = B + k * stride;

    for (int i = dim; i < null_dim; ++i)
    {
        // Project out components of modes 0..i-1
        for (int j = 0; j < i; ++j)
        {
            // Global dot product: dot(cols[i], cols[j])
            double local_dot = vec_dot(cols[i], cols[j], stride);
            double global_dot = local_dot;
#ifdef AMGX_WITH_MPI
            if (mpi_comm && *mpi_comm != MPI_COMM_NULL)
                MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, *mpi_comm);
#endif
            vec_axpy(cols[i], -global_dot, cols[j], stride);
        }

        // Normalize: global norm of cols[i]
        double local_sq  = vec_dot(cols[i], cols[i], stride);
        double global_sq = local_sq;
#ifdef AMGX_WITH_MPI
        if (mpi_comm && *mpi_comm != MPI_COMM_NULL)
            MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, *mpi_comm);
#endif
        double nrm = std::sqrt(global_sq);
        if (nrm > 0.0)
            for (int k = 0; k < stride; ++k) cols[i][k] /= nrm;
    }
}

} // namespace aggregation
} // namespace amgx
