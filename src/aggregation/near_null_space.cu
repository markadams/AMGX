// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <aggregation/near_null_space.h>
#include <error.h>
#include <cstring>  // memset

namespace amgx
{
namespace aggregation
{

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

} // namespace aggregation
} // namespace amgx
