// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <aggregation/aggregation_amg_level.h>
#include <aggregation/batched_qr.h>
#include <matrix_analysis.h>
#include <solvers/cheb_solver.h>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif
#ifdef _WIN32
#pragma warning (pop)
#endif

#include <basic_types.h>
#include <util.h>
#include <fstream>
#include <cutil.h>
#include <multiply.h>
#include <transpose.h>
#include <csr_multiply.h>
#include <blas.h>
#include <string>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <amgx_timer.h>

#include <amgx_types/util.h>

#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust_wrapper.h>

namespace amgx
{

namespace aggregation
{

// ---------------------------------------------------------------------------
// Static global: path to aggregate override file (empty = disabled).
// Set via Aggregation_AMG_Level_Base<T>::setAggregateOverrideFile() before
// solver.setup() to inject aggregates from a PETSc GAMG export file.
// ---------------------------------------------------------------------------
static std::string s_agg_override_file;

void setAggregateOverrideFile(const char *path)
{
    s_agg_override_file = (path ? path : "");
}

// ----------------------
// Kernels
// ----------------------

// Reshape R factors from per-aggregate column-major layout to global column-major B_c layout.
// R_out layout: R_out[a * nd*nd + col*nd + row] (column-major nd×nd per aggregate)
// B_coarse layout: B_coarse[k * num_aggs*nd + a*nd + j] (global column-major)
// where k = null vector index, a = aggregate, j = local DOF within aggregate
template <typename ValueType>
__global__
void reshape_R_to_coarse_B_kernel(
    int num_aggs,
    int null_dim,
    const ValueType * __restrict__ R_out,     // [num_aggs * nd * nd], per-agg column-major
    ValueType * __restrict__ B_coarse)         // [num_aggs * nd * nd], global column-major
{
    int total = num_aggs * null_dim * null_dim;
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x)
    {
        // Decode: which coarse DOF and which null vector?
        int k = idx / (num_aggs * null_dim);   // null vector index
        int c = idx % (num_aggs * null_dim);   // coarse DOF index
        int a = c / null_dim;                   // aggregate
        int j = c % null_dim;                   // local DOF within aggregate

        // R_out index: a * nd*nd + k*nd + j (column-major within aggregate)
        B_coarse[idx] = R_out[a * null_dim * null_dim + k * null_dim + j];
    }
}

template <typename IndexType, typename ValueType>
__global__
void set_to_one_kernel(IndexType start, IndexType end, IndexType *ind, ValueType *v)
{
    for (int tid = start + blockDim.x * blockIdx.x + threadIdx.x; tid < end; tid += gridDim.x * blockDim.x)
    {
        v[ind[tid]] = types::util<ValueType>::get_one();
    }
}

// --- Kernels for SA distributed column compression (mirrors classical path) ---

// Flag which halo columns (col >= nrow) are referenced in the owned rows of Ac.
template <typename IndexType>
__global__
void sa_flag_halo_columns_kernel(IndexType nrow, const IndexType *row_offsets,
                                  const IndexType *col_indices, IndexType *flags)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nrow;
         row += gridDim.x * blockDim.x)
    {
        int s = row_offsets[row];
        int e = row_offsets[row + 1];
        for (int j = s; j < e; j++)
        {
            IndexType col = col_indices[j];
            if (col >= nrow)
            {
                flags[col - nrow] = 1;
            }
        }
    }
}

// Remap halo column indices using the prefix-summed flags array.
template <typename IndexType>
__global__
void sa_compress_halo_columns_kernel(IndexType nrow, const IndexType *row_offsets,
                                      IndexType *col_indices, const IndexType *flags)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nrow;
         row += gridDim.x * blockDim.x)
    {
        int s = row_offsets[row];
        int e = row_offsets[row + 1];
        for (int j = s; j < e; j++)
        {
            IndexType col = col_indices[j];
            if (col >= nrow)
            {
                col_indices[j] = nrow + flags[col - nrow];
            }
        }
    }
}

// Compress local_to_global_map: keep only entries where flags[i] != flags[i+1].
template <typename IndexType, typename Int64Type>
__global__
void sa_compress_l2g_kernel(IndexType nl2g, const Int64Type *l2g_in,
                             Int64Type *l2g_out, const IndexType *flags)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nl2g;
         i += gridDim.x * blockDim.x)
    {
        if (flags[i] != flags[i + 1])
        {
            l2g_out[flags[i]] = l2g_in[i];
        }
    }
}

// Count how many nodes belong to each aggregate (for singleton detection).
__global__
void count_aggregate_sizes_kernel(const int *aggregates, int *agg_sizes,
                                   const int num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        if (aggregates[tid] >= 0)
            atomicAdd(&agg_sizes[aggregates[tid]], 1);
        tid += gridDim.x * blockDim.x;
    }
}

template <typename IndexType>
__global__
void renumberAggregatesKernel(const IndexType *renumbering, const int interior_offset, const int bdy_offset, IndexType *aggregates, const int num_aggregates, const int n_interior, const int renumbering_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < num_aggregates)
    {
        IndexType new_agg_id;

        if (renumbering_size == 0)
        {
            new_agg_id = aggregates[tid];
        }
        else
        {
            new_agg_id = renumbering[aggregates[tid]];
        }

        //if (aggregates[tid] > num_aggregates)
        //{
        //printf("ID %d old %d + %d = %d\n", tid, new_agg_id, ((new_agg_id >= n_interior) ? bdy_offset : interior_offset), new_agg_id + ((new_agg_id >= n_interior) ? bdy_offset : interior_offset));
        //}
        new_agg_id +=  ((new_agg_id >= n_interior) ? bdy_offset : interior_offset);
        aggregates[tid] = new_agg_id;
        tid += gridDim.x * blockDim.x;
    }
}

// Kernel to restrict residual using csr_format
template <typename IndexType, typename ValueType>
__global__
void restrictResidualKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueType *r, ValueType *rr, const int num_aggregates)
{
    int jmin, jmax;

    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_aggregates; tid += gridDim.x * blockDim.x)
    {
        ValueType temp(types::util<ValueType>::get_zero());
        jmin = row_offsets[tid];
        jmax = row_offsets[tid + 1];

        for (int j = jmin; j < jmax; j++)
        {
            int j_col = column_indices[j];
            temp = temp + r[j_col];
        }

        rr[tid] = temp;
    }
}

// Kernel to restrict residual using block_dia_csr_format
template <typename IndexType, typename ValueType, int bsize>
__global__
void restrictResidualBlockDiaCsrKernel(const IndexType *row_offsets, const IndexType *column_indices, const ValueType *r, ValueType *rr, const int num_aggregates)
{
    ValueType rr_temp[bsize];
    int offset, jmin, jmax;

    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_aggregates; tid += gridDim.x * blockDim.x)
    {
        // Initialize to zero
#pragma unroll
        for (int m = 0; m < bsize; m++)
        {
            rr_temp[m] = types::util<ValueType>::get_zero();
        }

        jmin = row_offsets[tid];
        jmax = row_offsets[tid + 1];

        for (int j = jmin; j < jmax; j++)
        {
            int jcol = column_indices[j];
            offset = jcol * bsize;
#pragma unroll

            for (int m = 0; m < bsize; m++)
            {
                rr_temp[m] = rr_temp[m] + r[offset + m];
            }
        }

        offset = tid * bsize;
#pragma unroll

        for (int m = 0; m < bsize; m++)
        {
            rr[offset + m] = rr_temp[m];
        };
    }
}

// Kernel to prolongate and apply the correction for csr format
template <typename IndexType, typename ValueType>
__global__
void prolongateAndApplyCorrectionKernel(const ValueType alpha, const int num_rows, ValueType *x, const ValueType *e, const IndexType *aggregates, IndexType num_aggregates)
{
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_rows; tid += gridDim.x * blockDim.x)
    {
        IndexType I = aggregates[tid];
        x[tid] = x[tid] + alpha * e[I];
    }
}

// Kernel to prolongate and apply the correction for block-dia-csr format
template <typename IndexType, typename ValueType>
__global__
void prolongateAndApplyCorrectionBlockDiaCsrKernel(const ValueType alpha, const int num_block_rows, ValueType *x, const ValueType *e, const IndexType *aggregates, IndexType num_aggregates, const int bsize)
{
    for (int tid = blockDim.x * blockIdx.x + threadIdx.x; tid < num_block_rows; tid += gridDim.x * blockDim.x)
    {
        IndexType I = aggregates[tid];

        for (int  m = 0; m < bsize; m++)
        {
            x[tid * bsize + m] = x[tid * bsize + m] + alpha * e[I * bsize + m];
        }
    }
}

template <typename IndexType, typename ValueType>
__global__
void prolongateVector(const IndexType *aggregates, const ValueType *in, ValueType *out, IndexType fine_rows, IndexType coarse_rows, int blocksize)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < fine_rows * blocksize )
    {
        int i = tid / blocksize;
        int e = tid % blocksize;
        IndexType I = aggregates[i];
        out[tid] = in[ I * blocksize + e ];
        tid += gridDim.x * blockDim.x;
    }
}

template <typename IndexType, typename ValueType>
__global__
void applyCorrection(ValueType lambda, const ValueType *e, ValueType *x, IndexType numRows )
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    while ( tid < numRows )
    {
        x[tid] = x[tid] + lambda * e[tid];
        tid += gridDim.x * blockDim.x;
    }
}

// -------------------------------
//  Methods
// ------------------------------

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::transfer_level(AMG_Level<TConfig1> *ref_lvl)
{
    Aggregation_AMG_Level_Base<TConfig1> *ref_agg_lvl = dynamic_cast<Aggregation_AMG_Level_Base<TConfig1>*>(ref_lvl);
    this->scale_counter = ref_agg_lvl->scale_counter;
    this->scale = ref_agg_lvl->scale;
    this->m_R_row_offsets.copy(ref_agg_lvl->m_R_row_offsets);
    this->m_R_column_indices.copy(ref_agg_lvl->m_R_column_indices);
    this->m_aggregates.copy(ref_agg_lvl->m_aggregates);
    this->m_aggregates_fine_idx.copy(ref_agg_lvl->m_aggregates_fine_idx);
    this->m_num_aggregates = ref_agg_lvl->m_num_aggregates;
    this->m_num_all_aggregates = ref_agg_lvl->m_num_all_aggregates;
}


typedef std::pair<int, int> mypair;
bool comparator ( const mypair &l, const mypair &r) { return l.first < r.first; }

// Method to compute R
// General path
// TODO: this could be merged with selector to save some computations
template <typename T_Config>
void Aggregation_AMG_Level_Base<T_Config>::computeRestrictionOperator_common()
{
    m_R_row_offsets.resize(m_num_all_aggregates + 1); //create one more row for the pseudo aggregate
    IVector R_row_indices(m_aggregates);
#if AMGX_ASYNCCPU_PROOF_OF_CONCEPT
    bool use_cpu = m_aggregates.size() < 4096;

    if (use_cpu)
    {
        struct computeRestrictionTask : public task
        {
            Aggregation_AMG_Level_Base<T_Config> *self;
            IVector *R_row_indices;

            void run()
            {
                int N = self->m_aggregates.size();
                IVector_h R_row_indices_host(self->m_aggregates);
                std::vector<mypair> pairs(N);

                for (int i = 0; i < N; i++)
                {
                    pairs[i].first = R_row_indices_host[i];
                    pairs[i].second = i;
                }

                std::stable_sort(pairs.begin(), pairs.end(), comparator);
                IVector_h R_column_indices(self->A->get_num_rows());

                for (int i = 0; i < N; i++)
                {
                    R_column_indices[i] = pairs[i].second;
                    R_row_indices_host[i] = pairs[i].first;
                }

                self->m_R_column_indices = R_column_indices;
                *R_row_indices = R_row_indices_host;
            }
        };
        computeRestrictionTask *t = new computeRestrictionTask();
        t->self = this;
        t->R_row_indices = &R_row_indices;
        t->run();
        delete t;
    }
    else
#endif
    {
        m_R_column_indices.resize(this->A->get_num_rows());
        thrust_wrapper::sequence<TConfig::memSpace>(m_R_column_indices.begin(), m_R_column_indices.end());
        cudaCheckError();
        amgx::thrust::sort_by_key(R_row_indices.begin(), R_row_indices.end(), m_R_column_indices.begin());
        cudaCheckError();
    }

    amgx::thrust::lower_bound(R_row_indices.begin(),
                        R_row_indices.end(),
                        amgx::thrust::counting_iterator<typename IVector::value_type>(0),
                        amgx::thrust::counting_iterator<typename IVector::value_type>(m_R_row_offsets.size()),
                        m_R_row_offsets.begin());
    cudaCheckError();
}


// two methods below could be merged
// Method to compute R on HOST using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_1x1()
{
    this->m_R_row_offsets.resize(this->m_num_all_aggregates + 1);
    this->m_R_column_indices.resize(this->A->get_num_rows());
    this->fillRowOffsetsAndColIndices(this->A->get_num_rows());
}

// Method to compute R on HOST using block dia-csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_4x4()
{
    this->m_R_row_offsets.resize(this->m_num_all_aggregates + 1);
    this->m_R_column_indices.resize(this->A->get_num_rows());
    this->fillRowOffsetsAndColIndices(this->A->get_num_rows());
}

// Method to create R_row_offsest and R_column_indices array on HOST using csr or block dia-csr format
template <typename T_Config>
void Aggregation_AMG_Level_Base<T_Config>::fillRowOffsetsAndColIndices(const int R_num_cols)
{
    for (int i = 0; i < m_num_all_aggregates + 1; i++)
    {
        m_R_row_offsets[i] = 0;
    }

    // Count number of neighbors for each row
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = m_aggregates[i];
        m_R_row_offsets[I]++;
    }

    m_R_row_offsets[m_num_all_aggregates] = R_num_cols;

    for (int i = m_num_all_aggregates - 1; i >= 0; i--)
    {
        m_R_row_offsets[i] = m_R_row_offsets[i + 1] - m_R_row_offsets[i];
    }

    /* Set column indices. */
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = m_aggregates[i];
        int Ip = m_R_row_offsets[I]++;
        m_R_column_indices[Ip] = i;
    }

    /* Reset r[i] to start of row memory. */
    for (int i = m_num_all_aggregates - 1; i > 0; i--)
    {
        m_R_row_offsets[i] = m_R_row_offsets[i - 1];
    }

    m_R_row_offsets[0] = 0;
}

// Method to compute R on DEVICE using block dia-csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_4x4()
{
    this->computeRestrictionOperator_common();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeRestrictionOperator_1x1()
{
    this->computeRestrictionOperator_common();
}

// Method to restrict Residual on host using csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_1x1(const VVector &r, VVector &rr)
{
    ValueTypeB temp;

    for (int i = 0; i < this->m_num_aggregates; i++)
    {
        temp = types::util<ValueTypeB>::get_zero();

        for (int j = this->m_R_row_offsets[i]; j < this->m_R_row_offsets[i + 1]; j++)
        {
            int j_col = this->m_R_column_indices[j];
            temp = temp + r[j_col];
        }

        rr[i] = temp;
    }
}

// Method to restrict Residual on host using block_dia_csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_4x4(const VVector &r, VVector &rr)
{
    IndexType bsize = this->A->get_block_dimy();
    ValueTypeB *temp = new ValueTypeB[bsize];

    for (int i = 0; i < this->m_num_aggregates; i++)
    {
        // Initialize temp to 0
        for (int k = 0; k < bsize; k++)
        {
            temp[k]  =  types::util<ValueTypeB>::get_zero();
        }

        // Add contributions from each fine point
        for (int j = this->m_R_row_offsets[i]; j < this->m_R_row_offsets[i + 1]; j++)
        {
            int j_col = this->m_R_column_indices[j];

            for (int k = 0; k < bsize; k++)
            {
                temp[k] = temp[k] + r[j_col * bsize + k];
            }
        }

        // Store result
        for (int k = 0; k < bsize; k++)
        {
            rr[i * bsize + k] = temp[k];
        }
    }
}

// Method to restrict Residual on device using csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_1x1(const VVector &r, VVector &rr)
{
    int block_size = 128;
    int max_threads;

    if (!this->isConsolidationLevel())
    {
        max_threads = this->m_num_aggregates;
    }
    else
    {
        max_threads = this->m_num_all_aggregates;
    }

    int num_blocks = max_threads / block_size + 1;
    const IndexType *R_row_offsets_ptr = this->m_R_row_offsets.raw();
    const IndexType *R_column_indices_ptr = this->m_R_column_indices.raw();
    const ValueTypeB *r_ptr = r.raw();
    ValueTypeB *rr_ptr = rr.raw();
    restrictResidualKernel <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
    cudaCheckError();
}

// Method to restrict Residual on device using block_dia_csr_matrix format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::restrictResidual_4x4(const VVector &r, VVector &rr)
{
    int block_size = 128;
    int max_threads;

    if (!this->isConsolidationLevel())
    {
        max_threads = this->m_num_aggregates;
    }
    else
    {
        max_threads = this->m_num_all_aggregates;
    };

    const int num_blocks = max_threads / block_size + 1;
    const IndexType *R_row_offsets_ptr = this->m_R_row_offsets.raw();

    const IndexType *R_column_indices_ptr = this->m_R_column_indices.raw();

    const ValueTypeB *r_ptr = r.raw();

    ValueTypeB *rr_ptr = rr.raw();

    cudaCheckError();

    switch ( this->getA().get_block_dimy() )
    {
        case 2:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 2> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            cudaCheckError();
            break;

        case 3:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 3> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            cudaCheckError();
            break;

        case 4:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 4> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            cudaCheckError();
            break;

        case 5:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 5> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            cudaCheckError();
            break;

        case 8:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 8> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            cudaCheckError();
            break;

        case 10:
            restrictResidualBlockDiaCsrKernel<IndexType, ValueTypeB, 10> <<< num_blocks, block_size>>>(R_row_offsets_ptr, R_column_indices_ptr, r_ptr, rr_ptr, max_threads);
            cudaCheckError();
            break;

        default:
            FatalError( "Unsupported block size in restrictResidual_4x4!!!", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE );
    }

    cudaCheckError();
}

__inline__ float getAlpha(float &nom, float &denom)
{
    float alpha;

    if (nom * denom <= 0. || std::abs(nom) < std::abs(denom))
    {
        alpha = 1.;
    }
    else if (std::abs(nom) > 2.*std::abs(denom))
    {
        alpha = 2.;
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

__inline__ double getAlpha(double &nom, double &denom)
{
    double alpha;

    if (nom * denom <= 0. || std::abs(nom) < std::abs(denom))
    {
        alpha = 1.;
    }
    else if (std::abs(nom) > 2.*std::abs(denom))
    {
        alpha = 2.;
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

__inline__ cuComplex getAlpha(cuComplex &nom, cuComplex &denom)
{
    cuComplex alpha;

    if (types::util<cuComplex>::abs(nom) < types::util<cuComplex>::abs(denom))
    {
        alpha = make_cuComplex(1.f, 0.f);
    }
    else if (types::util<cuComplex>::abs(nom) > 2.*types::util<cuComplex>::abs(denom))
    {
        alpha = make_cuComplex(2.f, 0.f);
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

__inline__ cuDoubleComplex getAlpha(cuDoubleComplex &nom, cuDoubleComplex &denom)
{
    cuDoubleComplex alpha;

    if (types::util<cuDoubleComplex>::abs(nom) < types::util<cuDoubleComplex>::abs(denom))
    {
        alpha = make_cuDoubleComplex(1., 0.);
    }
    else if (types::util<cuDoubleComplex>::abs(nom) > 2.*types::util<cuDoubleComplex>::abs(denom))
    {
        alpha = make_cuDoubleComplex(2., 0.);
    }
    else
    {
        alpha = nom / denom;
    }

    return alpha;
}

template< class T_Config>
typename T_Config::VecPrec Aggregation_AMG_Level_Base<T_Config>::computeAlpha(const Vector<T_Config> &e, const Vector<T_Config> &bc, const Vector<T_Config> &tmp)
{
    typename T_Config::VecPrec alpha =  types::util<ValueTypeB>::get_one();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    int size = Ac.get_num_rows();
    VVector v(2,  types::util<ValueTypeB>::get_zero());
    v[0] = amgx::thrust::inner_product(e.begin(), e.begin() + size, bc.begin(),  types::util<ValueTypeB>::get_zero());
    v[1] = amgx::thrust::inner_product(e.begin(), e.begin() + size, tmp.begin(),  types::util<ValueTypeB>::get_zero());
    cudaCheckError();
    return getAlpha(v[0], v[1]);
}

// Method to prolongate the error on HOST using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec>  >::prolongateAndApplyCorrection_1x1(Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &e, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &bc, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &tmp)
{
    Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &A = this->getA();
    Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &C = this->next_h->getA();

    if ( this->m_error_scaling >= 2 )
    {
        FatalError("error_scaling=2,3 is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    }

    ValueTypeB alpha = types::util<ValueTypeB>::get_one();

    if (this->m_error_scaling)
    {
        multiply(this->next_h->getA(), e, tmp);
        alpha = this->computeAlpha (e, bc, tmp);
    }

    // Apply correction on all (interior and exterior) equations.
    for (int i = 0; i < A.get_num_cols(); i++)
    {
        int I = this->m_aggregates[i];
        x[i] = x[i] + alpha * e[I];
    }
}

// Method to prolongate the error on HOST using block_dia_csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::prolongateAndApplyCorrection_4x4(Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &e, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &bc, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &x, Vector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > &tmp)
{
    if (this->A->get_block_dimy() != this->A->get_block_dimx())
    {
        FatalError("Aggregation_AMG_Level not implemented for non square blocks, exiting", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    if ( this->m_error_scaling >= 2 )
    {
        FatalError("error_scaling=2,3 is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED );
    }

    Matrix<TConfig> &C = this->next_h->getA();
    ValueTypeB alpha = types::util<ValueTypeB>::get_one();

    if (this->m_error_scaling)
    {
        multiply(this->next_h->getA(), e, tmp);
        alpha = this->computeAlpha (e, bc, tmp);
    }

    // Apply correction on all equations.
    for (int i = 0; i < this->A->get_num_rows(); i++)
    {
        int I = this->m_aggregates[i];

        for (int k = 0; k < this->A->get_block_dimy(); k++)
        {
            x[i * this->A->get_block_dimy() + k] =  x[i * this->A->get_block_dimy() + k] + alpha * e[I * this->A->get_block_dimy() + k];
        }
    }
}

// Prolongate the error on DEVICE using csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::prolongateAndApplyCorrection_1x1(Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &e, Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &bc, Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &x, Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &tmp)
{
    ValueTypeB alpha = types::util<ValueTypeB>::get_one();
    const int block_size = 128;
    const int num_blocks = this->A->get_num_rows() / block_size + 1;
    const IndexType *aggregates_ptr = this->m_aggregates.raw();
    ValueTypeB *x_ptr = x.raw();
    const ValueTypeB *e_ptr = e.raw();

    if (this->m_error_scaling)
    {
        FatalError("error_scaling=1 is deprecated", AMGX_ERR_NOT_IMPLEMENTED );
    }

    prolongateAndApplyCorrectionKernel <<< num_blocks, block_size>>>(alpha, (int)this->A->get_num_rows(), x_ptr, e_ptr, aggregates_ptr, this->m_num_aggregates);
    cudaCheckError();
}

// Prolongate the error on DEVICE using block dia-csr format
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Aggregation_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::prolongateAndApplyCorrection_4x4(Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &ec,
        Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &bf,
        Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &xf,
        Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > &rf)
{
    if ( this->m_error_scaling >= 2 )
    {
        if ( this->scale_counter > 0 )
        {
            const IndexType *aggregates_ptr = this->m_aggregates.raw();
            ValueTypeB *x_ptr = xf.raw();
            const ValueTypeB *e_ptr = ec.raw();
            const int block_size = 128;
            const int num_blocks = (this->A->get_num_rows() - 1) / block_size + 1;
            prolongateAndApplyCorrectionBlockDiaCsrKernel <<< num_blocks, block_size>>>(this->scale, (int)this->getA().get_num_rows(), x_ptr, e_ptr, aggregates_ptr, this->m_num_aggregates, this->getA().get_block_dimy());
            cudaCheckError();
            this->scale_counter--;
            return;
        }

        bool vanek_scaling = this->m_error_scaling > 3;
        IndexType numRowsCoarse = this->next_d->getA().get_num_rows();
        IndexType numRowsFine = this->A->get_num_rows();
        IndexType blockdim = this->A->get_block_dimx();

        if ( blockdim != this->A->get_block_dimy() )
        {
            FatalError("Unsupported dimension for aggregation amg level", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
        }

        VVector ef( rf.size() );
        VVector Aef( rf.size() );
        ef.set_block_dimy( blockdim );
        Aef.set_block_dimy( blockdim );
        // prolongate e
        const int threads_per_block = 256;
        const int num_block_values = std::min( AMGX_GRID_MAX_SIZE, (numRowsFine * blockdim - 1) / threads_per_block + 1);
        const cudaStream_t stream = nullptr;
        prolongateVector <<< num_block_values, threads_per_block, 0, stream>>>( this->m_aggregates.raw(), ec.raw(), ef.raw(), numRowsFine, numRowsCoarse, blockdim );
        cudaCheckError();
        ef.dirtybit = 1;
        cudaStreamSynchronize(stream);
        cudaCheckError();
        int preSmooth;

        if ( vanek_scaling )
        {
            preSmooth = this->amg->getNumPostsweeps();
        }
        else
        {
            preSmooth = this->scaling_smoother_steps;
        }

        //smooth error
        this->smoother->setTolerance( 0.0 );
        this->smoother->set_max_iters( preSmooth );

        if ( vanek_scaling )
        {
            thrust_wrapper::fill<TConfig::memSpace>( Aef.begin(), Aef.end(), types::util<ValueTypeB>::get_zero() );
            cudaCheckError();
            this->smoother->solve( Aef, ef, false ); //smooth correction with rhs 0
            this->smoother->solve( bf, xf, false ); // smooth x with rhs residual
            //recompute residual
            int offset, size;
            this->getA().getOffsetAndSizeForView(OWNED, &offset, &size);
            axmb( this->getA(), xf, bf, rf, offset, size );
        }
        else
        {
            this->smoother->solve( rf, ef, false ); //smooth correction with rhs residual
        }

        // multiply for lambda computation
        multiply(this->getA(), ef, Aef, OWNED);
        ValueTypeB nominator, denominator;
        int offset = 0, size = 0;
        this->A->getOffsetAndSizeForView(OWNED, &offset, &size);

        if ( this->m_error_scaling == 2 || this->m_error_scaling == 4 )
        {
            // compute lambda=<rf,Aef>/<Aef,Aef>
            nominator = amgx::thrust::inner_product( rf.begin(), rf.end(), Aef.begin(), types::util<ValueTypeB>::get_zero() );
            denominator = amgx::thrust::inner_product( Aef.begin(), Aef.end(), Aef.begin(), types::util<ValueTypeB>::get_zero() );
            cudaCheckError();
        }

        if ( this->m_error_scaling == 3 || this->m_error_scaling == 5)
        {
            // compute lambda=<rf,ef>/<ef,Aef>
            nominator = amgx::thrust::inner_product( rf.begin(), rf.begin() + size * blockdim, ef.begin(), types::util<ValueTypeB>::get_zero() );
            denominator = amgx::thrust::inner_product( ef.begin(), ef.begin() + size * blockdim, Aef.begin(), types::util<ValueTypeB>::get_zero() );

            if (!this->A->is_matrix_singleGPU())
            {
                this->A->getManager()->global_reduce_sum(&nominator);
                this->A->getManager()->global_reduce_sum(&denominator);
            }

            cudaCheckError();
        }

        if (types::util<ValueTypeB>::abs(denominator) == 0.0)
        {
            nominator = denominator = types::util<ValueTypeB>::get_one();
        }

        // apply correction x <- x + lambda*e
        const int num_block_fine = std::min( AMGX_GRID_MAX_SIZE, (numRowsFine * blockdim - 1) / threads_per_block + 1 );
        ValueTypeB alpha = nominator / denominator;

        if ( types::util<ValueTypeB>::abs(alpha) < .3 )
        {
            alpha = (alpha / types::util<ValueTypeB>::abs(alpha)) * .3;    // it was this before: alpha = .3, which is not 100% equal
        }

        if ( types::util<ValueTypeB>::abs(alpha) > 10 )
        {
            alpha = (alpha / types::util<ValueTypeB>::abs(alpha)) * 10.;    // it was this before: alpha = 10., which is not 100% equal
        }

        applyCorrection <<< num_block_fine, threads_per_block, 0, stream>>>( alpha, ef.raw(), xf.raw(), numRowsFine * blockdim );
        cudaCheckError();
        this->scale_counter = this->reuse_scale; //reuse this scale scale_counter times
        this->scale = alpha;
        return;
    }

    ValueTypeB alpha = types::util<ValueTypeB>::get_one();
    const int block_size = 128;
    const int num_blocks = this->A->get_num_rows() / block_size + 1;
    const IndexType *aggregates_ptr = this->m_aggregates.raw();
    ValueTypeB *x_ptr = xf.raw();
    const ValueTypeB *e_ptr = ec.raw();

    if (this->m_error_scaling == 1)
    {
        FatalError("error_scaling=1 is deprecated", AMGX_ERR_NOT_IMPLEMENTED );
    }

    prolongateAndApplyCorrectionBlockDiaCsrKernel <<< num_blocks, block_size>>>(alpha, (int)this->A->get_num_rows(), x_ptr, e_ptr, aggregates_ptr, this->m_num_aggregates, this->A->get_block_dimy());
    cudaCheckError();
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config >::prolongateAndApplyCorrection(VVector &e, VVector &bf, VVector &x, VVector &tmp)
{
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();

    // SA path: use the smoothed prolongator P for grid transfer.
    // x += P * e  (alpha = 1, no error scaling for SA)
    if (m_null_dim > 0 && m_P_tent.get_num_rows() > 0)
    {
        // Diagnostic: norm of coarse error e before prolongation
        {
            int n_e = e.size();
            double norm_e = 0.0;
            std::vector<ValueTypeB> e_h(n_e);
            cudaMemcpy(e_h.data(), e.raw(), n_e * sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n_e; i++) { double a = (double)types::util<ValueTypeB>::abs(e_h[i]); norm_e += a * a; }
            norm_e = sqrt(norm_e);
            fprintf(stderr, "[VCYCLE-DBG] prolongateAndCorrect: ||e_coarse||=%e  e.size=%d  e.num_rows=%d  e.block_dimy=%d\n",
                    norm_e, n_e, e.get_num_rows(), (int)e.get_block_dimy());
        }
        // Diagnostic: verify coarse solve: compute ||Ac*e - rr|| / ||rr||
        // This checks whether the dense LU produced a correct solution.
        // bf holds the coarse RHS (rr) passed into this function.
        // Only run on first call (static guard) to avoid flooding output.
        {
            static bool coarse_check_done = false;
            int n_e = e.size();
            int n_b = bf.size();
            if (!coarse_check_done && n_e > 0 && n_b > 0 && n_e == n_b && Ac.get_num_rows() == n_e)
            {
                coarse_check_done = true;
                typedef typename types::PODTypes<ValueTypeA>::type PodA;
                typedef typename types::PODTypes<ValueTypeB>::type PodB;
                int Ac_nrows = Ac.get_num_rows();
                int Ac_nnz   = Ac.get_num_nz();
                std::vector<int>      Ac_row_h(Ac_nrows + 1);
                std::vector<int>      Ac_col_h(Ac_nnz);
                std::vector<PodA>     Ac_val_h(Ac_nnz);
                std::vector<PodB>     e_h(n_e), b_h(n_b);
                // Copy Ac values as POD scalars (works for real; for complex takes real part)
                {
                    std::vector<ValueTypeA> tmp_a(Ac_nnz);
                    cudaMemcpy(tmp_a.data(), Ac.values.raw(), Ac_nnz*sizeof(ValueTypeA), cudaMemcpyDeviceToHost);
                    for (int i = 0; i < Ac_nnz; ++i) Ac_val_h[i] = static_cast<PodA>(types::util<ValueTypeA>::abs(tmp_a[i]));
                    // Note: abs() loses sign. For real symmetric positive definite Ac, diagonal is positive.
                    // We need signed values. Use memcpy trick: POD type is first member of complex.
                    memcpy(Ac_val_h.data(), tmp_a.data(), Ac_nnz * sizeof(PodA));
                }
                {
                    std::vector<ValueTypeB> tmp_b(n_e);
                    cudaMemcpy(tmp_b.data(), e.raw(), n_e*sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
                    memcpy(e_h.data(), tmp_b.data(), n_e * sizeof(PodB));
                    cudaMemcpy(tmp_b.data(), bf.raw(), n_b*sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
                    memcpy(b_h.data(), tmp_b.data(), n_b * sizeof(PodB));
                }
                cudaMemcpy(Ac_row_h.data(), Ac.row_offsets.raw(), (Ac_nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(Ac_col_h.data(), Ac.col_indices.raw(), Ac_nnz*sizeof(int), cudaMemcpyDeviceToHost);
                // Compute Ac*e (CPU SpMV)
                std::vector<double> Ace(Ac_nrows, 0.0);
                for (int row = 0; row < Ac_nrows; ++row)
                    for (int j = Ac_row_h[row]; j < Ac_row_h[row+1]; ++j)
                        if (Ac_col_h[j] < n_e)
                            Ace[row] += (double)Ac_val_h[j] * (double)e_h[Ac_col_h[j]];
                double norm_res = 0.0, norm_rr = 0.0;
                for (int i = 0; i < Ac_nrows; ++i)
                {
                    double res = Ace[i] - (double)b_h[i];
                    norm_res += res * res;
                    norm_rr  += (double)b_h[i] * (double)b_h[i];
                }
                norm_res = sqrt(norm_res);
                norm_rr  = sqrt(norm_rr);
                fprintf(stderr, "[VCYCLE-DBG] coarse solve check: ||Ac*e - rr||=%.4e  ||rr||=%.4e  rel=%.4e\n",
                        norm_res, norm_rr, (norm_rr > 0.0 ? norm_res/norm_rr : -1.0));
            }
        }
        // Diagnostic: norm of x before correction
        {
            int n_x = x.size();
            double norm_x = 0.0;
            std::vector<ValueTypeB> x_h(n_x);
            cudaMemcpy(x_h.data(), x.raw(), n_x * sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n_x; i++) { double a = (double)types::util<ValueTypeB>::abs(x_h[i]); norm_x += a * a; }
            norm_x = sqrt(norm_x);
            fprintf(stderr, "[VCYCLE-DBG] prolongateAndCorrect: ||x_before||=%e  x.size=%d  x.num_rows=%d  x.block_dimy=%d\n",
                    norm_x, n_x, x.get_num_rows(), (int)x.get_block_dimy());
        }

        // Halo exchange on the coarse error vector e before the SA prolongation SpMV.
        // Columns of m_P_tent reference coarse DOFs owned by other MPI ranks; their
        // values must be present in e's halo before multiply() is called.
        if (!Ac.is_matrix_singleGPU() && !this->isConsolidationLevel() && e.delayed_send == 0)
        {
            e.dirtybit = 1;
            if (e.in_transfer & RECEIVING) { Ac.manager->exchange_halo_wait(e, e.tag); }
            Ac.manager->exchange_halo_async(e, e.tag);
            Ac.manager->exchange_halo_wait(e, e.tag);
        }

        // m_P_tent is scalar (1x1 blocks): rows=num_fine_dofs, cols=num_coarse_dofs.
        // When A has block_size > 1, x is a block vector (block_dimy=bs, num_rows=num_nodes)
        // but its raw storage holds num_nodes*bs scalars — same as num_fine_dofs.
        // Allocate prolongated at DOF level (scalar) so multiply() sees matching block dims.
        // axpy() uses raw element count (x.size()), so it works correctly regardless of
        // block metadata as long as prolongated.size() == x.size().
        int sa_bs = this->A->get_block_dimy();
        int num_fine_dofs = this->A->get_num_rows() * sa_bs;
        VVector prolongated(num_fine_dofs);
        prolongated.set_block_dimy(1);
        prolongated.set_block_dimx(1);
        prolongated.set_num_rows(num_fine_dofs);
        fill(prolongated, types::util<ValueTypeB>::get_zero());
        multiply(m_P_tent, e, prolongated, OWNED);

        // Diagnostic: norm of prolongated correction
        {
            int n_p = prolongated.size();
            double norm_p = 0.0;
            std::vector<ValueTypeB> p_h(n_p);
            cudaMemcpy(p_h.data(), prolongated.raw(), n_p * sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n_p; i++) { double a = (double)types::util<ValueTypeB>::abs(p_h[i]); norm_p += a * a; }
            norm_p = sqrt(norm_p);
            fprintf(stderr, "[VCYCLE-DBG] prolongateAndCorrect: ||P*e||=%e  prolongated.size=%d\n",
                    norm_p, n_p);
        }

        ValueTypeB one = types::util<ValueTypeB>::get_one();
        axpy(prolongated, x, one);

        // Diagnostic: norm of x after correction
        {
            int n_x = x.size();
            double norm_x = 0.0;
            std::vector<ValueTypeB> x_h(n_x);
            cudaMemcpy(x_h.data(), x.raw(), n_x * sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n_x; i++) { double a = (double)types::util<ValueTypeB>::abs(x_h[i]); norm_x += a * a; }
            norm_x = sqrt(norm_x);
            fprintf(stderr, "[VCYCLE-DBG] prolongateAndCorrect: ||x_after||=%e\n", norm_x);
        }
    }
    //this is dirty, but error scaling 2 and 3 do not have a specialized version. Instead, the general version sits in the 4x4 function
    else if ( this->m_error_scaling >= 2 )
    {
        prolongateAndApplyCorrection_4x4(e, bf, x, tmp);
    }
    else if (this->A->get_block_size() == 1)
    {
        prolongateAndApplyCorrection_1x1(e, bf, x, tmp);
    }
    else if (this->A->get_block_dimx() == this->A->get_block_dimy() )
    {
        prolongateAndApplyCorrection_4x4(e, bf, x, tmp);
    }
    else
    {
        FatalError("Unsupported dimension for aggregation amg level", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    x.dirtybit = 1;

    if (!this->A->is_matrix_singleGPU() && x.delayed_send == 0)
    {
        if (x.in_transfer & RECEIVING) { this->A->manager->exchange_halo_wait(x, x.tag); }

        this->A->manager->exchange_halo_async(x, x.tag);
    }
}


template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::restrictResidual(VVector &r, VVector &rr)
{
    // SA path: use the pre-computed transpose P^T for restriction.
    // rr = P^T * r  (CSR SpMV with the stored transpose)
    if (m_null_dim > 0 && m_P_tent_T.get_num_rows() > 0)
    {
        // Halo exchange on fine residual r before SA restriction SpMV.
        // Columns of m_P_tent_T reference fine DOFs owned by other MPI ranks;
        // their values must be present in r's halo before multiply() is called.
        if (!this->A->is_matrix_singleGPU() && !this->isConsolidationLevel() && r.delayed_send == 0)
        {
            r.dirtybit = 1;
            if (r.in_transfer & RECEIVING) { this->A->manager->exchange_halo_wait(r, r.tag); }
            this->A->manager->exchange_halo_async(r, r.tag);
            this->A->manager->exchange_halo_wait(r, r.tag);
        }

        // Diagnostic: norm of fine residual before restriction
        {
            int n_r = r.size();
            double norm_r = 0.0;
            std::vector<ValueTypeB> r_h(n_r);
            cudaMemcpy(r_h.data(), r.raw(), n_r * sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n_r; i++) { double a = (double)types::util<ValueTypeB>::abs(r_h[i]); norm_r += a * a; }
            norm_r = sqrt(norm_r);
            fprintf(stderr, "[VCYCLE-DBG] restrictResidual: ||r_fine||=%e  r.size=%d  r.num_rows=%d  r.block_dimy=%d\n",
                    norm_r, n_r, r.get_num_rows(), (int)r.get_block_dimy());
        }

        // m_P_tent_T is scalar (1x1 blocks): rows=num_coarse_dofs, cols=num_fine_dofs.
        // When A has block_size > 1, r is a block vector (block_dimy=bs, num_rows=num_nodes).
        // Temporarily reinterpret r as a scalar DOF-level vector so multiply() sees
        // matching block dimensions.  The raw data layout is identical.
        //
        // After multiply(), rr.block_dimy is set to 1 (m_P_tent_T.block_dimx=1) but
        // rr.num_rows may still reflect the old block layout (num_coarse_nodes with bs=3).
        // We must fix rr.num_rows to num_coarse_dofs so the coarse solver sees the full RHS.
        int sa_bs = this->A->get_block_dimy();
        if (sa_bs > 1)
        {
            int saved_num_rows = r.get_num_rows();
            short saved_dimy   = r.get_block_dimy();
            short saved_dimx   = r.get_block_dimx();
            r.set_num_rows(saved_num_rows * sa_bs);
            r.set_block_dimy(1);
            r.set_block_dimx(1);
            multiply(m_P_tent_T, r, rr, OWNED);
            r.set_num_rows(saved_num_rows);
            r.set_block_dimy(saved_dimy);
            r.set_block_dimx(saved_dimx);
            // Fix rr metadata: multiply() sets block_dimy=1 but num_rows may be wrong.
            rr.set_num_rows(m_P_tent_T.get_num_rows());
            rr.set_block_dimy(1);
            rr.set_block_dimx(1);
        }
        else
        {
            multiply(m_P_tent_T, r, rr, OWNED);
        }

        // Diagnostic: norm of coarse RHS after restriction
        {
            int n_rr = rr.size();
            double norm_rr = 0.0;
            std::vector<ValueTypeB> rr_h(n_rr);
            cudaMemcpy(rr_h.data(), rr.raw(), n_rr * sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
            for (int i = 0; i < n_rr; i++) { double a = (double)types::util<ValueTypeB>::abs(rr_h[i]); norm_rr += a * a; }
            norm_rr = sqrt(norm_rr);
            fprintf(stderr, "[VCYCLE-DBG] restrictResidual: ||rr_coarse||=%e  rr.size=%d  rr.num_rows=%d  rr.block_dimy=%d\n",
                    norm_rr, n_rr, rr.get_num_rows(), (int)rr.get_block_dimy());
        }
    }
    else if (this->A->get_block_size() == 1)
    {
        restrictResidual_1x1(r, rr);
    }
    else if (this->A->get_block_dimx() == this->A->get_block_dimy() )
    {
        restrictResidual_4x4(r, rr);
    }
    else
    {
        FatalError("Unsupported dimension for aggregation amg level", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }

    //TODO: check level transfer between host and device for multiGPU
    if (!this->A->is_matrix_singleGPU())
    {
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
        rr.dirtybit = 1;

        if (!Ac.is_matrix_singleGPU() && !this->isConsolidationLevel() && rr.delayed_send == 0)
        {
            Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA(); //TODO problem in memoryspace transfer is here

            if (rr.in_transfer & RECEIVING) { Ac.manager->exchange_halo_wait(rr, rr.tag); }

            Ac.manager->exchange_halo_async(rr, rr.tag);
        }
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::computeRestrictionOperator()
{
    if (this->A->get_block_size() == 1)
    {
        computeRestrictionOperator_1x1();
    }
    else if (this->A->get_block_dimx() == 4 && this->A->get_block_dimy() == 4)
    {
        computeRestrictionOperator_4x4();
    }
    else
    {
        this->computeRestrictionOperator_common();
    }
}

template <typename IndexType>
__global__ void coarse_to_global(IndexType *aggregates, IndexType *aggregates_global, IndexType *renumbering, IndexType num_elements, int64_t offset)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;

    while (element < num_elements)
    {
        renumbering[aggregates[element]] = aggregates_global[element] + offset; //this won't be a problem, because we are overwriting the same thing
        element += blockDim.x * gridDim.x;
    }
}

template <typename T, typename IndexType>
__global__ void export_matrix_elements(IndexType *row_offsets, IndexType *col_indices, T *values, IndexType *maps, IndexType *renumbering, IndexType *new_row_offsets, IndexType *new_col_indices, T *new_values, IndexType bsize, IndexType size)
{
    int idx = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int coopIdx = threadIdx.x % 32;

    while (idx < size)
    {
        int row = maps[idx];
        INDEX_TYPE src_base = row_offsets[row];
        INDEX_TYPE dst_base = new_row_offsets[idx];

        for (int m = coopIdx; m < row_offsets[row + 1]*bsize - src_base * bsize; m += 32)
        {
            new_values[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        for (int m = coopIdx; m < row_offsets[row + 1] - src_base; m += 32)
        {
            new_col_indices[dst_base + m] = renumbering[col_indices[src_base + m]];
        }

        idx += gridDim.x * blockDim.x / 32;
    }
}

template <class T>
__global__ void export_matrix_diagonal(T *values, INDEX_TYPE bsize, INDEX_TYPE *maps, T *output, INDEX_TYPE size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size)
    {
        int row = maps[idx];
        INDEX_TYPE src_base = row;
        INDEX_TYPE dst_base = idx;

        for (int m = 0; m < bsize; m++)
        {
            output[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        idx += gridDim.x * blockDim.x;
    }
}

__global__ void remove_boundary(INDEX_TYPE *flags, INDEX_TYPE *maps, INDEX_TYPE size)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;

    while (element < size)
    {
        flags[maps[element]] = 0; //this won't be a problem, because we are overwriting the same thing
        element += blockDim.x * gridDim.x;
    }
}

__global__ void calc_inverse_renumbering(INDEX_TYPE *renum, INDEX_TYPE *irenum, INDEX_TYPE *renum_gbl, INDEX_TYPE base_index, INDEX_TYPE max_element)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < max_element)
    {
        irenum[renum[idx]] = renum_gbl[idx] - base_index;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void create_halo_mapping(INDEX_TYPE *mapping, INDEX_TYPE *node_list, INDEX_TYPE base_index, INDEX_TYPE map_offset, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        int idx = node_list[row] - base_index;
        mapping[idx] = map_offset + row;
        row += blockDim.x * gridDim.x;
    }
}

__global__ void map_col_indices_and_count_rowlen(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, INDEX_TYPE *row_length, INDEX_TYPE *renumbering, INDEX_TYPE *mapping, INDEX_TYPE *map_offsets, int64_t *index_ranges, INDEX_TYPE part_id, INDEX_TYPE my_id, INDEX_TYPE base_index, INDEX_TYPE my_range, INDEX_TYPE num_neighbors, INDEX_TYPE num_rows)
{
    extern __shared__ volatile int reduction[];
    int row = blockIdx.x * blockDim.x / 4 + threadIdx.x / 4;
    int coopIdx = threadIdx.x % 4;

    while (row < num_rows)
    {
        int valid = 0;

        for (int idx = row_offsets[row] + coopIdx; idx < row_offsets[row + 1]; idx += 4) //this may look horrible, but I expect low branch divergence, because col indices in a row usually belong to the same partition (or at most one more)
        {
            int colIdx = col_indices[idx];
            int part = -2;

            if (colIdx >= index_ranges[2 * part_id] && colIdx < index_ranges[2 * part_id + 1]) //the col index probably belongs to the partition I am working on
            {
                part = part_id;
            }
            else if (colIdx >= base_index && colIdx < base_index + my_range)     //or points back to the owned partition
            {
                part = -1;
            }
            else        //or else it points to a third partition
            {
                for (int i = 0; i < num_neighbors; i++)
                {
                    if (colIdx >= index_ranges[2 * i] && colIdx < index_ranges[2 * i + 1])
                    {
                        part = i;
                    }
                }
            }

            if (part == -2)
            {
                col_indices[idx] = -1;
#ifdef DEBUG
                printf("Column index encountered that does not belong to any of my neighbors!! %d\n", colIdx);
#endif
            }
            else
            {
                if (part == -1)
                {
                    col_indices[idx] = renumbering[colIdx - base_index];
                    valid++;
                }
                else
                {
                    int new_col_idx = mapping[map_offsets[part] + colIdx - index_ranges[2 * part]];

                    if (new_col_idx >= 0)
                    {
                        valid++;
                        col_indices[idx] = new_col_idx;
                    }
                    else
                    {
                        col_indices[idx] = -1;
                    }
                }
            }
        }

        reduction[threadIdx.x] = valid;

        for (int s = 2; s > 0; s >>= 1)
        {
            if (coopIdx < s)
            {
                reduction[threadIdx.x] += reduction[threadIdx.x + s];
            }

            __syncthreads();
        }

        if (coopIdx == 0)
        {
            row_length[row] = reduction[threadIdx.x];
        }

        row += gridDim.x * blockDim.x / 4;
    }
}

__global__ void map_col_indices(INDEX_TYPE *row_offsets, INDEX_TYPE *col_indices, int64_t *halo_ranges, INDEX_TYPE *halo_renumbering, INDEX_TYPE *halo_rows, INDEX_TYPE *global_renumbering, INDEX_TYPE num_neighbors, INDEX_TYPE num_rows, INDEX_TYPE num_rows_processed)
{
    int row = blockIdx.x * blockDim.x / 4 + threadIdx.x / 4;
    int coopIdx = threadIdx.x % 4;

    while (row < num_rows_processed)
    {
        for (int idx = row_offsets[row] + coopIdx; idx < row_offsets[row + 1]; idx += 4)
        {
            int colIdx = col_indices[idx];
            int part = 0;

            if (colIdx < num_rows)
            {
                part = -1;
            }
            else
            {
                colIdx = global_renumbering[colIdx];

                for (int i = 0; i < num_neighbors; i++)
                {
                    if (colIdx >= halo_ranges[2 * i] && colIdx < halo_ranges[2 * i + 1])
                    {
                        part = i;
                        break;
                    }
                }
            }

            if (part == -1)
            {
                col_indices[idx] = colIdx;
            }
            else
            {
                col_indices[idx] = halo_renumbering[halo_rows[part] + colIdx - halo_ranges[2 * part]];
            }
        }

        row += gridDim.x * blockDim.x / 4;
    }
}

template <class T>
__global__ void reorder_whole_matrix(INDEX_TYPE *old_rows, INDEX_TYPE *old_cols, T *old_vals, INDEX_TYPE *rows, INDEX_TYPE *cols, T *vals, INDEX_TYPE bsize, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < num_rows)
    {
        INDEX_TYPE dst_row = row;
        INDEX_TYPE src_base = old_rows[row];
        INDEX_TYPE dst = rows[dst_row];

        for (int i = 0; i < old_rows[row + 1] - src_base; i++)
        {
            INDEX_TYPE colIdx = old_cols[src_base + i];

            if (colIdx >= 0)
            {
                cols[dst] = colIdx;

                for (int j = 0; j < bsize; j++) { vals[dst * bsize + j] = old_vals[(src_base + i) * bsize + j]; }

                dst++;
            }
        }

        row += blockDim.x * gridDim.x;
    }
}

__global__ void calc_gbl_renumbering(INDEX_TYPE *inv_renum, INDEX_TYPE *gbl_renum, INDEX_TYPE size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        gbl_renum[inv_renum[idx]] = idx;
        idx += blockDim.x * gridDim.x;
    }
}

template <typename ValueType>
__global__ void write_diagonals(ValueType *values, INDEX_TYPE *diag, INDEX_TYPE *map, ValueType *output, INDEX_TYPE bsize, INDEX_TYPE size)
{
    int nzPerBlock = blockDim.x / bsize;
    int row = blockIdx.x * nzPerBlock + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= (blockDim.x / bsize)*bsize) { return; }

    while (row < size)
    {
        output[row * bsize + vecIdx] = values[diag[map[row]] * bsize + vecIdx];
        row += gridDim.x * nzPerBlock;
    }
}

template <typename ValueType>
__global__ void write_diagonals_back(ValueType *values, INDEX_TYPE *diag, ValueType *source, INDEX_TYPE bsize, INDEX_TYPE size)
{
    int nzPerBlock = blockDim.x / bsize;
    int row = blockIdx.x * nzPerBlock + threadIdx.x / bsize;
    int vecIdx = threadIdx.x % bsize;

    if (threadIdx.x >= (blockDim.x / bsize)*bsize) { return; }

    while (row < size)
    {
        values[diag[row]*bsize + vecIdx] = source[row * bsize + vecIdx];
        row += gridDim.x * nzPerBlock;
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix_full(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (A.is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Aggregation AMG Not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        int c_size = Ac.get_num_rows();
        int f_size = A.get_num_rows();
        int diag = Ac.hasProps(DIAG);

        if (A.manager->B2L_rings[0].size() > 2) { FatalError("Aggregation_AMG_Level prepareNextLevelMatrix not implemented >1 halo rings", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE); }

        //get coarse -> fine global renumbering
        IVector renumbering(c_size);
        int num_blocks = std::min(4096, (c_size + 127) / 128);
        coarse_to_global <<< num_blocks, 128>>>(this->m_aggregates.raw(), this->m_aggregates_fine_idx.raw(), renumbering.raw(), f_size, 0);
        cudaCheckError();
        //
        // Step 0 - form halo matrices that are exported to neighbors
        //
        std::vector<Matrix<TConfig> > halo_rows(num_neighbors);
        std::vector<DistributedManager<TConfig> > halo_btl(num_neighbors);

        for (int i = 0; i < num_neighbors; i++ )
        {
            int num_unique = Ac.manager->B2L_rings[i][1];
            //prepare export halo matrices
            halo_btl[i].resize(1, 1);
            halo_btl[i].set_global_id(Ac.manager->global_id());
            halo_btl[i].B2L_maps[0].resize(num_unique);
            halo_btl[i].B2L_rings[0].resize(2);
            halo_btl[i].B2L_rings[0][0] = 0;
            halo_btl[i].B2L_rings[0][1] = num_unique;
            halo_btl[i].set_index_range(A.manager->index_range());
            halo_btl[i].set_base_index(A.manager->base_index());
            //global indices of rows of the halo matrix
            amgx::thrust::copy(amgx::thrust::make_permutation_iterator( renumbering.begin(), Ac.manager->B2L_maps[i].begin()),
                         amgx::thrust::make_permutation_iterator( renumbering.begin(), Ac.manager->B2L_maps[i].begin() + num_unique),
                         halo_btl[i].B2L_maps[0].begin());
            cudaCheckError();
            halo_rows[i].addProps(CSR);

            if (diag) { halo_rows[i].addProps(DIAG); }

            //calculate row length and row_offsets
            halo_rows[i].row_offsets.resize(num_unique + 1);
            thrust_wrapper::transform<TConfig::memSpace>(amgx::thrust::make_permutation_iterator(Ac.row_offsets.begin() + 1, Ac.manager->B2L_maps[i].begin()),
                              amgx::thrust::make_permutation_iterator(Ac.row_offsets.begin() + 1, Ac.manager->B2L_maps[i].end()),
                              amgx::thrust::make_permutation_iterator(Ac.row_offsets.begin(), Ac.manager->B2L_maps[i].begin()),
                              halo_rows[i].row_offsets.begin(),
                              amgx::thrust::minus<IndexType>());
            cudaCheckError();
            thrust_wrapper::exclusive_scan<TConfig::memSpace>(halo_rows[i].row_offsets.begin(), halo_rows[i].row_offsets.end(), halo_rows[i].row_offsets.begin());
            cudaCheckError();
            //resize halo matrix
            IndexType num_nz = halo_rows[i].row_offsets[num_unique];
            halo_rows[i].resize(num_unique, num_unique, num_nz, Ac.get_block_dimy(), Ac.get_block_dimx(), 1);
            //copy relevant rows and renumber their column indices
            num_blocks = std::min(4096, (num_unique + 127) / 128);
            export_matrix_elements <<< num_blocks, 128>>>(Ac.row_offsets.raw(), Ac.col_indices.raw(), Ac.values.raw(), Ac.manager->B2L_maps[i].raw(), renumbering.raw(), halo_rows[i].row_offsets.raw(), halo_rows[i].col_indices.raw(), halo_rows[i].values.raw(), A.get_block_size(), num_unique);
            cudaCheckError();

            if (diag)
            {
                export_matrix_diagonal <<< num_blocks, 128>>>(Ac.values.raw() + Ac.row_offsets[Ac.get_num_rows()]*Ac.get_block_size(), Ac.get_block_size(), Ac.manager->B2L_maps[i].raw(), halo_rows[i].values.raw() + halo_rows[i].row_offsets[halo_rows[i].get_num_rows()]*Ac.get_block_size(), num_unique);
                cudaCheckError();
            }
        }

        Ac.manager->getComms()->exchange_matrix_halo(halo_rows, halo_btl, Ac);
        //--------------------- renumbering/reordering matrix, integrating halo -----------------------------
        Ac.set_initialized(0);
        //number of owned rows
        c_size = Ac.manager->halo_offsets[0];
        f_size = A.manager->halo_offsets[0];
        num_blocks = std::min(4096, (c_size + 511) / 512);
        int rings = 1;
        //
        // Step 1 - calculate inverse renumbering (to global indices - base_index)
        //
        Ac.manager->inverse_renumbering.resize(c_size);
        thrust_wrapper::transform<TConfig::memSpace>(renumbering.begin(),
                          renumbering.begin() + c_size,
                          amgx::thrust::constant_iterator<IndexType>(A.manager->base_index()),
                          Ac.manager->inverse_renumbering.begin(),
                          amgx::thrust::minus<IndexType>());
        cudaCheckError();
        //big renumbering table for going from global index to owned local index
        IVector global_to_coarse_local(Ac.manager->index_range());
        thrust_wrapper::fill<TConfig::memSpace>(global_to_coarse_local.begin(), global_to_coarse_local.begin() + Ac.manager->index_range(), -1);
        cudaCheckError();
        calc_gbl_renumbering <<< num_blocks, 512>>>(Ac.manager->inverse_renumbering.raw(), global_to_coarse_local.raw(), c_size);
        cudaCheckError();
        Ac.manager->set_num_halo_rows(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1] - c_size);
        cudaCheckError();
        //
        // Step 2 - create big mapping table of all halo indices we received (this may use a little too much memory sum(fine nodes per neighbor)
        //
        amgx::thrust::host_vector<INDEX_TYPE> neighbor_rows(num_neighbors + 1);
        int max_num_rows = 0;

        for (int i = 0; i < num_neighbors; i++)
        {
            neighbor_rows[i] = halo_rows[i].manager->index_range();
            max_num_rows = max_num_rows > halo_rows[i].get_num_rows() ? max_num_rows : halo_rows[i].get_num_rows();
        }

        thrust_wrapper::exclusive_scan<TConfig::memSpace>(neighbor_rows.begin(), neighbor_rows.end(), neighbor_rows.begin());
        cudaCheckError();
        int total_rows_of_neighbors = neighbor_rows[num_neighbors];
        IVector halo_mapping(total_rows_of_neighbors);
        thrust_wrapper::fill<TConfig::memSpace>(halo_mapping.begin(), halo_mapping.end(), -1);
        cudaCheckError();

        for (int ring = 0; ring < rings; ring++)
        {
            for (int i = 0; i < num_neighbors; i++)
            {
                int size = halo_btl[i].B2L_rings[0][ring + 1] - halo_btl[i].B2L_rings[0][ring];
                int num_blocks = std::min(4096, (size + 127) / 128);
                create_halo_mapping <<< num_blocks, 128>>>(halo_mapping.raw() + neighbor_rows[i],
                        halo_btl[i].B2L_maps[0].raw() + halo_btl[i].B2L_rings[0][ring],
                        halo_btl[i].base_index(),
                        Ac.manager->halo_offsets[ring * num_neighbors + i], size);
                        cudaCheckError();
            }
        }

        cudaCheckError();
        //
        // Step 3 - renumber halo matrices and calculate row length (to eventually append to the big matrix)
        //
        INDEX_TYPE owned_nnz = Ac.row_offsets[c_size];
        IVector neighbor_rows_d(num_neighbors + 1);
        amgx::thrust::copy(neighbor_rows.begin(), neighbor_rows.end(), neighbor_rows_d.begin());
        cudaCheckError();
        //map column indices of my own matrix (the ones that point outward)
        map_col_indices <<< num_blocks, 512>>>(Ac.row_offsets.raw() + Ac.manager->num_interior_nodes(),
                                               Ac.col_indices.raw(),
                                               Ac.manager->halo_ranges.raw(),
                                               halo_mapping.raw(),
                                               neighbor_rows_d.raw(),
                                               renumbering.raw(),
                                               num_neighbors, c_size, c_size - Ac.manager->num_interior_nodes());
        cudaCheckError();
        IVector temp_row_len(max_num_rows);

        for (int i = 0; i < num_neighbors; i++)
        {
            //map column indices of halo matrices
            int size = halo_rows[i].get_num_rows();
            int num_blocks = std::min(4096, (size + 127) / 128);
            map_col_indices_and_count_rowlen <<< num_blocks, 128, 128 * sizeof(INDEX_TYPE)>>>(
                halo_rows[i].row_offsets.raw(),
                halo_rows[i].col_indices.raw(),
                temp_row_len.raw(),
                global_to_coarse_local.raw(),
                halo_mapping.raw(),
                neighbor_rows_d.raw(),
                Ac.manager->halo_ranges.raw(),
                i,
                Ac.manager->global_id(),
                Ac.manager->base_index(),
                Ac.manager->index_range(),
                num_neighbors,
                size);
                cudaCheckError();

            for (int ring = 0; ring < rings; ring++)
            {
                amgx::thrust::copy(temp_row_len.begin() + halo_btl[i].B2L_rings[0][ring], temp_row_len.begin() + halo_btl[i].B2L_rings[0][ring + 1], Ac.row_offsets.begin() + Ac.manager->halo_offsets[ring * num_neighbors + i]);
            }
        }

        cudaCheckError();
        INDEX_TYPE old_nnz = Ac.row_offsets[Ac.row_offsets.size() - 1];
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(Ac.row_offsets.begin() + c_size, Ac.row_offsets.end(), Ac.row_offsets.begin() + c_size, owned_nnz);
        cudaCheckError();
        //
        // Step 4 - consolidate column indices and values
        //
        int new_nnz = Ac.row_offsets[Ac.row_offsets.size() - 1];

        Ac.col_indices.resize(new_nnz);
        Ac.values.resize((new_nnz + 1 + diag * (Ac.row_offsets.size() - 2)) * A.get_block_size());

        if (diag)
        {
            MVector diags(c_size * Ac.get_block_size());
            amgx::thrust::copy(Ac.values.begin() + old_nnz * Ac.get_block_size(),
                         Ac.values.begin() + old_nnz * Ac.get_block_size() + c_size * Ac.get_block_size(),
                         diags.begin());
            amgx::thrust::copy(diags.begin(), diags.begin() + c_size * Ac.get_block_size(),
                         Ac.values.begin() + Ac.row_offsets[Ac.get_num_rows()]*Ac.get_block_size());
            cudaCheckError();
        }

        int cumulative_num_rows = c_size;

        for (int i = 0; i < num_neighbors; i++)
        {
            for (int ring = 0; ring < rings; ring++)
            {
                int num_rows = halo_btl[i].B2L_rings[0][ring + 1] - halo_btl[i].B2L_rings[0][ring];
                int num_blocks = std::min(4096, (num_rows + 127) / 128);
                reorder_whole_matrix <<< num_blocks, 128>>>(halo_rows[i].row_offsets.raw() + halo_btl[i].B2L_rings[0][ring], halo_rows[i].col_indices.raw(), halo_rows[i].values.raw(), Ac.row_offsets.raw() + Ac.manager->halo_offsets[ring * num_neighbors + i], Ac.col_indices.raw(), Ac.values.raw(), Ac.get_block_size(), num_rows);
                cudaCheckError();

                if (diag)
                {
                    amgx::thrust::copy(halo_rows[i].values.begin() + (halo_rows[i].row_offsets[halo_rows[i].get_num_rows()] + halo_btl[i].B2L_rings[0][ring])*Ac.get_block_size(),
                                 halo_rows[i].values.begin() + (halo_rows[i].row_offsets[halo_rows[i].get_num_rows()] + halo_btl[i].B2L_rings[0][ring + 1])*Ac.get_block_size(),
                                 Ac.values.begin() + (Ac.row_offsets[Ac.get_num_rows()] + cumulative_num_rows)*Ac.get_block_size());
                    cumulative_num_rows += num_rows;
                }
            }
        }

        cudaCheckError();
        Ac.set_num_cols(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1]);
        Ac.set_num_rows(Ac.get_num_cols());
        Ac.set_num_nz(new_nnz);
        Ac.delProps(COO);
        Ac.set_initialized(1);
        Ac.computeDiagonal();
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix_diag(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (A.is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Aggregation AMG Not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        int c_size = Ac.manager->halo_offsets[0];
        int f_size = A.manager->halo_offsets[0];
        int diag = Ac.hasProps(DIAG);
        Ac.manager->inverse_renumbering.resize(c_size);
        //get coarse -> fine renumbering
        int num_blocks = std::min(4096, (c_size + 127) / 128);
        coarse_to_global <<< num_blocks, 128>>>(this->m_aggregates.raw(), this->m_aggregates_fine_idx.raw(), Ac.manager->inverse_renumbering.raw(), f_size, -1 * A.manager->base_index());
        cudaCheckError();
        Ac.manager->set_num_halo_rows(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1] - c_size);

        if (!diag) { Ac.computeDiagonal(); }

        Ac.set_initialized(1);
        std::vector<MVector> diagonals(num_neighbors);

        for (int i = 0; i < num_neighbors; i++)
        {
            int size = Ac.manager->B2L_rings[i][Ac.manager->B2L_rings.size() - 1];
            diagonals[i].resize(Ac.get_block_size()*size);
            int num_blocks = std::min(4096, (size + 127) / 128);
            write_diagonals <<< num_blocks, 128>>>(Ac.values.raw(), Ac.diag.raw(), Ac.manager->B2L_maps[i].raw(), diagonals[i].raw(), Ac.get_block_size(), size);
        }

        cudaCheckError();
        Ac.manager->getComms()->exchange_vectors(diagonals, Ac, this->tag * 100 + 10 + 2);

        for (int i = 0; i < num_neighbors; i++)
        {
            int size = Ac.manager->halo_offsets[i + 1] - Ac.manager->halo_offsets[i];

            if (Ac.hasProps(DIAG)) { amgx::thrust::copy(diagonals[i].begin(), diagonals[i].begin() + Ac.get_block_size()*size, Ac.values.begin() + Ac.get_block_size() * (Ac.diagOffset() + Ac.manager->halo_offsets[i])); }
            else
            {
                int num_blocks = std::min(4096, (size + 127) / 128);
                write_diagonals_back <<< num_blocks, 128>>>(Ac.values.raw(), Ac.diag.raw() + Ac.manager->halo_offsets[i], diagonals[i].raw(), Ac.get_block_size(), size);
                cudaCheckError();
            }
        }

        cudaCheckError();
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix_none(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (A.is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    if (TConfig::memSpace == AMGX_host)
    {
        FatalError("Aggregation AMG Not implemented for host", AMGX_ERR_NOT_IMPLEMENTED);
    }
    else
    {
        int c_size = Ac.manager->halo_offsets[0];
        int f_size = A.manager->halo_offsets[0];
        int diag = Ac.hasProps(DIAG);
        Ac.manager->inverse_renumbering.resize(c_size);
        //get coarse -> fine renumbering
        int num_blocks = std::min(4096, (c_size + 127) / 128);
        coarse_to_global <<< num_blocks, 128>>>(this->m_aggregates.raw(), this->m_aggregates_fine_idx.raw(), Ac.manager->inverse_renumbering.raw(), f_size, 0);
        cudaCheckError();
        Ac.manager->set_num_halo_rows(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1] - c_size);
        Ac.set_initialized(1);

        if (!diag) { Ac.computeDiagonal(); }
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::prepareNextLevelMatrix(const Matrix<TConfig> &A, Matrix<TConfig> &Ac)
{
    if (m_matrix_halo_exchange == 0)
    {
        this->prepareNextLevelMatrix_none(A, Ac);
    }
    else if (m_matrix_halo_exchange == 1)
    {
        this->prepareNextLevelMatrix_diag(A, Ac);
    }
    else if (m_matrix_halo_exchange == 2)
    {
        this->prepareNextLevelMatrix_full(A, Ac);
    }
    else
    {
        FatalError("Invalid Aggregation matrix_halo_exchange parameter", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


__global__ void set_halo_rowlen(INDEX_TYPE *work, INDEX_TYPE *output, INDEX_TYPE  size, INDEX_TYPE diag)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        if (work[idx + 1] - work[idx] > 0)
        {
            output[idx] += work[idx + 1] - work[idx] - (1 - diag);
        }

        idx += blockDim.x * gridDim.x;
    }
}

template <typename T>
__global__ void append_halo_nz(INDEX_TYPE *row_offsets, INDEX_TYPE *new_row_offsets, INDEX_TYPE *col_indices, INDEX_TYPE *new_col_indices, T *values, T *new_values, INDEX_TYPE size, INDEX_TYPE diag, INDEX_TYPE halo_offset, INDEX_TYPE block_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        int add_diag = !diag;

        if (!diag && new_col_indices[new_row_offsets[idx]] != -1) { add_diag = 0; } //if diag or there is already soimething in the row, then don't add diagonal nonzero (inside diag)

        int append_offset = -1;

        for (int i = new_row_offsets[idx]; i < new_row_offsets[idx + 1]; i++)
        {
            if (new_col_indices[i] == -1) {append_offset = i; break;}
        }

        for (int i = row_offsets[idx]; i < row_offsets[idx + 1]; i++)
        {
            if (diag && i == row_offsets[idx])   //if outside diag and this is the first nonzero in a non-empty row, overwrite diagonal value
            {
                for (int j = 0; j < block_size; j++)
                {
                    new_values[(new_row_offsets[size] + halo_offset + idx)*block_size + j] = values[(row_offsets[size] + halo_offset + idx) * block_size + j];
                }
            }

            int col_idx = col_indices[i];

            if (append_offset == -1 && (col_idx != halo_offset + idx)) {printf("ERROR: append offset is -1 but row has nonzeros in it old %d to %d new %d to %d\n", row_offsets[idx], row_offsets[idx + 1], new_row_offsets[idx], new_row_offsets[idx + 1]); append_offset = 0;}

            if (col_idx != halo_offset + idx || add_diag)
            {
                new_col_indices[append_offset] = col_idx;

                for (int j = 0; j < block_size; j++)
                {
                    new_values[append_offset * block_size + j] = values[i * block_size + j];
                }

                append_offset++;
            }
        }

        idx += blockDim.x * gridDim.x;
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::createCoarseB2LMaps(std::vector<IVector> &in_coarse_B2L_maps)
{
    Matrix<TConfig> &A = this->getA();
    m_num_all_aggregates = m_num_aggregates;
    int num_neighbors = A.manager->neighbors.size();
    IndexType max_b2l = 0;

    for (int i = 0; i < num_neighbors; i++ ) { max_b2l = max_b2l > A.manager->B2L_rings[i][1] ? max_b2l : A.manager->B2L_rings[i][1]; }

    IVector B2L_aggregates(max_b2l);
    IVector indices(max_b2l);

    for (int i = 0; i < num_neighbors; i++ )
    {
        int size = A.manager->B2L_rings[i][1];
        thrust_wrapper::fill<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + size, 0);
        thrust_wrapper::sequence<TConfig::memSpace>(indices.begin(), indices.begin() + size);
        //substitute coarse aggregate indices for fine boundary nodes
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin()),
                     amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin() + size),
                     B2L_aggregates.begin());
        //find the unique ones
        amgx::thrust::sort_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin());
        IndexType num_unique = amgx::thrust::unique_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin()).first - B2L_aggregates.begin();
        in_coarse_B2L_maps[i].resize(num_unique);
        //sort it back so we have the original ordering
        amgx::thrust::sort_by_key(indices.begin(), indices.begin() + num_unique, B2L_aggregates.begin());
        amgx::thrust::copy(B2L_aggregates.begin(), B2L_aggregates.begin() + num_unique, in_coarse_B2L_maps[i].begin());
    }

    cudaCheckError();
}


__global__ void populate_coarse_boundary(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE *maps, INDEX_TYPE *output, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        output[flags[maps[indices[idx]]]] = maps[indices[idx]];
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void flag_coarse_boundary(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE *maps, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        flags[maps[indices[idx]]] = 1;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void flag_halo_indices(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE  offset, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        flags[indices[idx] - offset] = 1;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void apply_halo_aggregate_indices(INDEX_TYPE *flags, INDEX_TYPE *indices, INDEX_TYPE *output, INDEX_TYPE offset, INDEX_TYPE aggregates_offset, INDEX_TYPE  size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    while (idx < size)
    {
        output[idx] = flags[indices[idx] - offset] + aggregates_offset;
        idx += blockDim.x * gridDim.x;
    }
}

// renumbering the aggregates/communicationg with neighbors
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::setNeighborAggregates()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    m_num_all_aggregates = m_num_aggregates;

    /* WARNING: the matrix reordering always happens inside createRenumbering routine. There are three ways to get to this routine
       1. matrix_upload_all -> uploadMatrix -> initializeUploadReorderAll -> reorder_matrix -> createRenumbering
       2. read_system_distributed -> renumberMatrixOneRing -> reorder_matrix_owned -> createRenumbering
       3. solver_setup -> ... -> AMG_Level::setup -> createCoarseMatrices -> setNeighborAggregates -> createRenumbering
       If you are reading the renumbering from file you might need to add intercept code in if statement below,
       otherwise this routine will exit before calling createRenumbering routine (in case of single or disjoint partitions).
    */
    if (this->getA().is_matrix_singleGPU()) { return; }

    int num_neighbors = A.manager->neighbors.size();

    //
    // Step 0 - set up coarse matrix metadata
    //
    if (Ac.manager == NULL) { Ac.manager = new DistributedManager<T_Config>(); }

    Ac.manager->resize(A.manager->neighbors.size(), 1);
    Ac.manager->A = &Ac;
    int f_size = A.get_num_rows();
    Ac.manager->setComms(A.manager->getComms());
    Ac.manager->set_global_id(A.manager->global_id());
    Ac.manager->neighbors = A.manager->neighbors;
    Ac.manager->set_base_index(A.manager->base_index());
    Ac.manager->halo_ranges = A.manager->halo_ranges;
    Ac.manager->set_index_range(A.manager->index_range());
    //-------------------------------------- Section 1 - renumbering -----------------------------------------------------------
    //
    // Step 1 - calculate coarse level B2L maps - any aggregate that has a fine boundary node, becomes a coarse boundary node
    //
    m_num_all_aggregates = m_num_aggregates;
    int vec_size = m_num_aggregates + 1; //A.manager->num_boundary_nodes()+1;
    IVector B2L_aggregates(vec_size);

    for (int i = 0; i < A.manager->neighbors.size(); i++)
    {
        thrust_wrapper::fill<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + vec_size, 0);
        int size = A.manager->B2L_rings[i][1];
        int block_size = 128;
        int grid_size = std::min( 4096, ( size + block_size - 1 ) / block_size);
        flag_coarse_boundary <<< grid_size, block_size>>>(B2L_aggregates.raw(), A.manager->B2L_maps[i].raw(), this->m_aggregates.raw(), size);
        cudaCheckError();
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + vec_size, B2L_aggregates.begin());
        (Ac.manager->B2L_maps)[i].resize(B2L_aggregates[vec_size - 1]);
        populate_coarse_boundary <<< grid_size, block_size>>>(B2L_aggregates.raw(), A.manager->B2L_maps[i].raw(), this->m_aggregates.raw(), Ac.manager->B2L_maps[i].raw(), size);
    }

    cudaCheckError();

    for (int i = 0; i < num_neighbors; i++)
    {
        Ac.manager->B2L_rings[i].resize(2);
        Ac.manager->B2L_rings[i][0] = 0;
        Ac.manager->B2L_rings[i][1] = Ac.manager->B2L_maps[i].size();
    }

    DistributedArranger<T_Config> *prep = new DistributedArranger<T_Config>;
    prep->initialize_B2L_maps_offsets(Ac, 1);
    delete prep;
    Ac.set_num_rows(m_num_aggregates);
    IVector renumbering(m_num_aggregates + 1); /* +1 is actually not needed, it will be resized in createRenumbering */
    Ac.manager->createRenumbering(renumbering);
    //
    // Step 2 - renumber aggregates, so boundary nodes will have higher index than interior ones (based on the renumberiong we have been calculating)
    //
    /* WARNING: 1. Thrust scatter and gather routines seem more appropriate here, but they implicitly assume that the input
                and output have certain size correlation, which is not matched by vectors in our case. The only remaining option
                is to use make_permutation as is done below. Example of Thrust scatter and gather calls
                IVector ttt(f_size,-1);
                amgx::thrust::scatter(this->m_aggregates.begin(), this->m_aggregates.begin()+f_size, renumbering.begin(), ttt.begin());
                amgx::thrust::gather(renumbering.begin(), renumbering.end(), this->m_aggregates.begin(), ttt.begin());
                amgx::thrust::copy(ttt.begin(), ttt.end(), this->m_aggregates.begin());

                2. The original thrust composite call is illegal because it uses the same array (m_aggregates) for input and output.
                amgx::thrust::copy(amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin()),
                amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin()+f_size),
                             this->m_aggregates.begin());
                Although it somehow still works, it is much safer to use explicit temporary storage for the intermediate result.
    */
    /* WARNING: must save unreordered aggregates for later use before reordering them. */
    IVector unreordered_aggregates(this->m_aggregates);
    /* WARNING: change Thrust call to explicitly use temporary storage for the intermediate result. The earlier version is illegal, but somehow still works. */
    IVector ttt(f_size, -1);
    amgx::thrust::copy(amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin()),
                 amgx::thrust::make_permutation_iterator(renumbering.begin(), this->m_aggregates.begin() + f_size),
                 ttt.begin());
    amgx::thrust::copy(ttt.begin(), ttt.end(), this->m_aggregates.begin());
    cudaCheckError();

    //we don't need renumbering anymore, it will be identity on the coarse level

    //-------------------------------------- Section 2 - communication -----------------------------------------------------------

    //
    // Step 3 - populate aggregates_fine_idx, which stores for every fine node the original global index of the aggregate (which is lowest global index of nodes aggregated together)
    //

    //
    // These are different when we do /don't do matrix halo exchanges - when we do we need global indices to match nodes,
    // and in this case Ac after computeA will not have the same ordering of halo nodes as after prepareNextLevel_full.
    // However when we do not do matrix halo exchange we are only interested in the ordering of halo nodes on the coarse level,
    // and we can get that by exchanging the (already renumbered) aggregates vector.
    //
    if (m_matrix_halo_exchange == 2)
    {
        //Find original global indices of nodes that have the minimum id in the aggregates.
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(A.manager->inverse_renumbering.begin(), this->m_aggregates_fine_idx.begin()),
                     amgx::thrust::make_permutation_iterator(A.manager->inverse_renumbering.begin(), this->m_aggregates_fine_idx.begin() + f_size),
                     this->m_aggregates_fine_idx.begin());
        thrust_wrapper::transform<TConfig::memSpace>(this->m_aggregates_fine_idx.begin(),
                          this->m_aggregates_fine_idx.begin() + f_size,
                          amgx::thrust::constant_iterator<IndexType>(A.manager->base_index()),
                          this->m_aggregates_fine_idx.begin(),
                          amgx::thrust::plus<IndexType>());
        //communicate
        this->m_aggregates_fine_idx.set_block_dimx(1);
        this->m_aggregates_fine_idx.set_block_dimy(1);
        m_aggregates_fine_idx.dirtybit = 1;
        A.manager->exchange_halo(m_aggregates_fine_idx, this->tag * 100 + 1 * 10 + 0);
    }
    else
    {
        //communicate
        this->m_aggregates.set_block_dimx(1);
        this->m_aggregates.set_block_dimy(1);
        m_aggregates.dirtybit = 1;
        /* WARNING: you should exchange unreordered aggregates, and append them to your own reordered aggregates, to conform to asusmptions done by distributed_mamanger. */
        //A.manager->exchange_halo(m_aggregates, this->tag*100+1*10+0); //wrong
        A.manager->exchange_halo(unreordered_aggregates, this->tag * 100 + 1 * 10 + 0);
        amgx::thrust::copy(unreordered_aggregates.begin() + f_size, unreordered_aggregates.end(), this->m_aggregates.begin() + f_size);
    }

    cudaCheckError();
    //
    // Step 4 - consolidate neighbors' aggregates into own list to be able to perform Galerkin product with the n-ring halo
    //
    IVector &exchanged_aggregates = m_matrix_halo_exchange == 2 ? this->m_aggregates_fine_idx : this->m_aggregates;
    int min_index = amgx::thrust::reduce(exchanged_aggregates.begin() + A.manager->halo_offsets[0], exchanged_aggregates.begin() + A.manager->halo_offsets[num_neighbors], (int)0xFFFFFFF, amgx::thrust::minimum<int>());
    int max_index = amgx::thrust::reduce(exchanged_aggregates.begin() + A.manager->halo_offsets[0], exchanged_aggregates.begin() + A.manager->halo_offsets[num_neighbors], (int)0, amgx::thrust::maximum<int>());
    cudaCheckError();
    int s_size = max_index - min_index + 2;
    IVector scratch(s_size);

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = A.manager->halo_offsets[i + 1] - A.manager->halo_offsets[i];
        //Could also use local minimums to perform the same operation. The results are the same.
        //int min_local = amgx::thrust::reduce(exchanged_aggregates.begin()+A.manager->halo_offsets[i], exchanged_aggregates.begin()+A.manager->halo_offsets[i+1], (int)0xFFFFFFF, amgx::thrust::minimum<int>());
        thrust_wrapper::fill<TConfig::memSpace>(scratch.begin(), scratch.begin() + s_size, 0);
        int block_size = 128;
        int grid_size = std::min( 4096, ( size + block_size - 1 ) / block_size);
        flag_halo_indices <<< grid_size, block_size>>>(scratch.raw(), exchanged_aggregates.raw() + A.manager->halo_offsets[i], min_index /*min_local*/, size);
        cudaCheckError();
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(scratch.begin(), scratch.begin() + s_size, scratch.begin());
        apply_halo_aggregate_indices <<< grid_size, block_size>>>(scratch.raw(), exchanged_aggregates.raw() + A.manager->halo_offsets[i], this->m_aggregates.raw() + A.manager->halo_offsets[i], min_index /*min_local*/, m_num_all_aggregates, size);
        cudaCheckError();
        Ac.manager->halo_offsets[i] = m_num_all_aggregates;
        m_num_all_aggregates += scratch[s_size - 1];
    }

    cudaCheckError();
    Ac.manager->halo_offsets[num_neighbors] = m_num_all_aggregates;
}

//TODO: The consolidate and unconsolidate parts could be made more efficient by only sending the
//      nonzero values
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::consolidateVector(VVector &x)
{
    int my_id = this->getA().manager->global_id();

    if (this->getA().manager->isRootPartition())
    {
        // Here all partitions being consolidated should have same vector size, see TODO above
        INDEX_TYPE num_parts = this->getA().manager->getNumPartsToConsolidate();

        for (int i = 0; i < num_parts; i++)
        {
            int current_part = this->getA().manager->getPartsToConsolidate()[i];

            // Vector has been set to correct size
            if (current_part != my_id)
            {
                //printf("Root partition %d receiving %d -> %d and %d -> %d (total %d)\n", this->getA().manager->global_id(), this->getA().manager->getConsolidationArrayOffsets()[i], this->getA().manager->getConsolidationArrayOffsets()[i+1], this->getA().manager->getConsolidationArrayOffsets()[num_parts+i], this->getA().manager->getConsolidationArrayOffsets()[num_parts+i+1], (int)x.size()/x.get_block_size());
                this->getA().manager->getComms()->recv_vector(x, current_part, 10000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[i + 1] - this->getA().manager->getConsolidationArrayOffsets()[i]));
                this->getA().manager->getComms()->recv_vector(x, current_part, 20000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[num_parts + i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[num_parts + i + 1] - this->getA().manager->getConsolidationArrayOffsets()[num_parts + i]));
            }
        }
    }
    else
    {
        int my_destination_part = this->getA().manager->getMyDestinationPartition();
        int i_off, i_size, b_off, b_size;
        this->getA().manager->getConsolidationOffsets(&i_off, &i_size, &b_off, &b_size);
        // Here all partitions being consolidated should have same vector size, see TODO above
        this->getA().manager->getComms()->send_vector_async(x, my_destination_part, 10000 + my_id, i_off * x.get_block_size(), i_size * x.get_block_size());
        this->getA().manager->getComms()->send_vector_async(x, my_destination_part, 20000 + my_id, b_off * x.get_block_size(), b_size * x.get_block_size());
    }
}

//TODO: The consolidate and unconsolidate parts could be made more efficient by only sending the
//      nonzero values
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::unconsolidateVector(VVector &x)
{
    if (this->getA().manager->isRootPartition())
    {
        INDEX_TYPE num_parts = this->getA().manager->getNumPartsToConsolidate();

        for (int i = 0; i < num_parts; i++)
        {
            int current_part = this->getA().manager->getPartsToConsolidate()[i];

            // Vector has been set to correct size
            if (current_part != this->getA().manager->global_id())
            {
                this->getA().manager->getComms()->send_vector_async(x, current_part, 30000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[i + 1] - this->getA().manager->getConsolidationArrayOffsets()[i]));
                this->getA().manager->getComms()->send_vector_async(x, current_part, 40000 + current_part, x.get_block_size()*this->getA().manager->getConsolidationArrayOffsets()[num_parts + i], x.get_block_size() * (this->getA().manager->getConsolidationArrayOffsets()[num_parts + i + 1] - this->getA().manager->getConsolidationArrayOffsets()[num_parts + i]));
            }
        }
    }
    else
    {
        int my_destination_part = this->getA().manager->getMyDestinationPartition();
        // Vector x is of unknown size
        int i_off, i_size, b_off, b_size;
        this->getA().manager->getConsolidationOffsets(&i_off, &i_size, &b_off, &b_size);
        this->getA().manager->getComms()->recv_vector(x, my_destination_part, 30000 + this->getA().manager->global_id(), i_off * x.get_block_size(), i_size * x.get_block_size());
        this->getA().manager->getComms()->recv_vector(x, my_destination_part, 40000 + this->getA().manager->global_id(), b_off * x.get_block_size(), b_size * x.get_block_size());
    }
}


template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::createCoarseVertices()
{
    //Set the aggregates
    // Pass level index to selector via matrix parameter (used by MIS selector for aggressive_levels)
    this->getA().template setParameter<int>("amg_level_index", this->getLevelIndex());

    fprintf(stderr, "[SA-DBG] createCoarseVertices: num_rows=%d, block_dimy=%d\n",
            (int)this->getA().get_num_rows(), (int)this->getA().get_block_dimy());

    // ------------------------------------------------------------------
    // Aggregate override: if a file was specified AND this is level 0,
    // read node→aggregate mapping from the file instead of running MIS.
    // File format (PETSc GAMG export):
    //   # Level N: P is M x K  (bs_row=3 bs_col=6)
    //   # fine_node  aggregate_id
    //   <fine_node>  <aggregate_id>
    //   ...
    // ------------------------------------------------------------------
    bool used_override = false;
    if (!s_agg_override_file.empty() && this->getLevelIndex() == 0)
    {
        FILE *fp = fopen(s_agg_override_file.c_str(), "r");
        if (!fp)
        {
            fprintf(stderr, "[SA-AGG-OVERRIDE] ERROR: cannot open '%s'\n",
                    s_agg_override_file.c_str());
        }
        else
        {
            int num_rows = this->getA().get_num_rows();
            std::vector<int> h_agg(num_rows, -1);
            int max_agg = -1;
            char line[256];
            int parsed = 0;
            while (fgets(line, sizeof(line), fp))
            {
                if (line[0] == '#') continue;  // skip comment lines
                int fn, ai;
                if (sscanf(line, "%d %d", &fn, &ai) == 2)
                {
                    if (fn >= 0 && fn < num_rows)
                    {
                        h_agg[fn] = ai;
                        if (ai > max_agg) max_agg = ai;
                        ++parsed;
                    }
                    else
                    {
                        fprintf(stderr, "[SA-AGG-OVERRIDE] WARNING: fine_node %d out of range [0,%d)\n",
                                fn, num_rows);
                    }
                }
            }
            fclose(fp);

            if (parsed != num_rows)
            {
                fprintf(stderr, "[SA-AGG-OVERRIDE] WARNING: parsed %d entries but num_rows=%d\n",
                        parsed, num_rows);
            }

            this->m_num_aggregates = max_agg + 1;
            // Upload to device
            this->m_aggregates.resize(num_rows);
            cudaMemcpy(this->m_aggregates.raw(), h_agg.data(),
                       num_rows * sizeof(int), cudaMemcpyHostToDevice);
            // m_aggregates_fine_idx is not used by the SA path; leave empty.
            this->m_aggregates_fine_idx.resize(0);

            fprintf(stderr, "[SA-AGG-OVERRIDE] Loaded %d aggregates from '%s' "
                    "(num_rows=%d, num_aggs=%d)\n",
                    parsed, s_agg_override_file.c_str(),
                    num_rows, this->m_num_aggregates);
            used_override = true;
        }
    }

    if (!used_override)
    {
        // Compute aggregates via selector
        this->m_selector->setAggregates(this->getA(), this->m_aggregates, this->m_aggregates_fine_idx, this->m_num_aggregates);
    }

    fprintf(stderr, "[SA-DBG] createCoarseVertices: m_num_aggregates=%d (coarsening ratio=%.2f)\n",
            this->m_num_aggregates,
            (this->m_num_aggregates > 0) ? (float)this->getA().get_num_rows() / (float)this->m_num_aggregates : 0.0f);

    // ------------------------------------------------------------------
    // Singleton handling: reassign size-1 aggregates to their strongest
    // non-singleton neighbor's aggregate.
    //
    // Singletons (aggregates of size 1) arise at BC nodes where MIS
    // propagation stalls.  If left as singletons with agg=-1, they get
    // zero rows in P_tent and zero rows in P_smooth, making the
    // preconditioner non-SPD (PCG diverges).
    //
    // Fix: merge each singleton into the aggregate of its strongest
    // off-diagonal neighbor (by |A[i,j]| entry).  This is what PETSc
    // GAMG does.  It may slightly enlarge some aggregates but preserves
    // full coverage of the fine grid in the prolongator.
    // ------------------------------------------------------------------
    {
        typedef ValueTypeA MatValueType;
        typedef typename types::PODTypes<MatValueType>::type PodA;

        int num_rows = this->getA().get_num_rows();
        int A_block_nnz = this->getA().get_num_nz();
        int bs = this->getA().get_block_dimy();

        // Download aggregates and A sparsity to host
        std::vector<int> h_agg(num_rows);
        cudaMemcpy(h_agg.data(), this->m_aggregates.raw(),
                   num_rows * sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<int> h_row(num_rows + 1);
        std::vector<int> h_col(A_block_nnz);
        cudaMemcpy(h_row.data(), this->getA().row_offsets.raw(),
                   (num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col.data(), this->getA().col_indices.raw(),
                   A_block_nnz * sizeof(int), cudaMemcpyDeviceToHost);

        // Download A values to find strongest neighbor
        int A_scalar_nnz = A_block_nnz * bs * bs;
        std::vector<MatValueType> h_val(A_scalar_nnz);
        cudaMemcpy(h_val.data(), this->getA().values.raw(),
                   A_scalar_nnz * sizeof(MatValueType), cudaMemcpyDeviceToHost);

        // Count aggregate sizes
        std::vector<int> agg_size(this->m_num_aggregates, 0);
        for (int i = 0; i < num_rows; i++) {
            if (h_agg[i] >= 0 && h_agg[i] < this->m_num_aggregates)
                agg_size[h_agg[i]]++;
        }

        // Iteratively reassign singletons to strongest non-singleton neighbor.
        // Iterate because reassigning one singleton may make another non-singleton.
        int n_singletons_total = 0;
        bool changed = true;
        while (changed) {
            changed = false;
            for (int v = 0; v < num_rows; v++) {
                int my_agg = h_agg[v];
                if (my_agg < 0 || agg_size[my_agg] != 1) continue;

                // v is a singleton — find strongest non-singleton neighbor
                double best_weight = -1.0;
                int best_agg = -1;
                for (int bnz = h_row[v]; bnz < h_row[v + 1]; ++bnz) {
                    int j = h_col[bnz];
                    if (j == v) continue;  // skip self
                    int j_agg = h_agg[j];
                    if (j_agg < 0 || agg_size[j_agg] < 1) continue;
                    // Use Frobenius norm of the bs×bs block as weight
                    double w = 0.0;
                    for (int li = 0; li < bs; ++li)
                        for (int lj = 0; lj < bs; ++lj) {
                            double a = (double)static_cast<PodA>(
                                types::util<MatValueType>::abs(
                                    h_val[bnz * bs * bs + li * bs + lj]));
                            w += a * a;
                        }
                    if (w > best_weight) {
                        best_weight = w;
                        best_agg = j_agg;
                    }
                }

                if (best_agg >= 0) {
                    // Reassign singleton v to best_agg
                    agg_size[my_agg]--;   // now 0
                    agg_size[best_agg]++;
                    h_agg[v] = best_agg;
                    n_singletons_total++;
                    changed = true;
                }
                // If no non-singleton neighbor found, leave as singleton for now
                // (will be compacted out below as agg=-1)
            }
        }

        // Any remaining size-1 aggregates (isolated nodes with no non-singleton
        // neighbors) are marked agg=-1 and excluded from the coarse grid.
        int n_isolated = 0;
        for (int v = 0; v < num_rows; v++) {
            int my_agg = h_agg[v];
            if (my_agg < 0) continue;
            if (agg_size[my_agg] == 1) {
                agg_size[my_agg] = 0;
                h_agg[v] = -1;
                n_isolated++;
            }
        }

        // Renumber aggregates: compact out empty aggregates.
        std::vector<int> agg_map(this->m_num_aggregates, -1);
        int new_num_aggs = 0;
        for (int a = 0; a < this->m_num_aggregates; a++) {
            if (agg_size[a] > 0) {
                agg_map[a] = new_num_aggs++;
            }
        }

        // Apply renumbering to h_agg
        for (int i = 0; i < num_rows; i++) {
            if (h_agg[i] >= 0) {
                h_agg[i] = agg_map[h_agg[i]];
            }
            // h_agg[i] == -1 stays -1 (truly isolated nodes)
        }

        this->m_num_aggregates = new_num_aggs;

        // Upload renumbered aggregates to device
        cudaMemcpy(this->m_aggregates.raw(), h_agg.data(),
                   num_rows * sizeof(int), cudaMemcpyHostToDevice);

        if (n_singletons_total > 0 || n_isolated > 0) {
            fprintf(stderr, "[SA-SINGLETON] %d singletons merged into neighbors, "
                    "%d isolated (no non-singleton neighbor, excluded). "
                    "Final: %d aggregates\n",
                    n_singletons_total, n_isolated, this->m_num_aggregates);
        }

        this->m_singleton_agg_flags.resize(0);
    }

    if ( this->m_print_aggregation_info )
    {
        this->m_selector->printAggregationInfo( this->m_aggregates, this->m_aggregates_fine_idx, this->m_num_aggregates );
    }

    this->getA().template setParameter< int > ("aggregates_num", this->m_num_aggregates); // ptr to aaggregates
}

// Forward declarations of SA kernels used in createCoarseMatrices.
// The full definitions appear later in this file (before smoothProlongator).
template <typename IndexType>
__global__ void build_SA_smoother_block_rowlen_kernel(
    int num_block_rows, int block_size,
    const IndexType * __restrict__ A_row_offsets,
    IndexType * __restrict__ S_row_offsets);

template <typename IndexType, typename ValueType>
__global__ void expand_block_to_scalar_kernel(
    int num_block_rows, int block_size,
    const IndexType * __restrict__ A_row_offsets,
    const IndexType * __restrict__ A_col_indices,
    const ValueType * __restrict__ A_values,
    const IndexType * __restrict__ As_row_offsets,
    IndexType       * __restrict__ As_col_indices,
    ValueType       * __restrict__ As_values);

template <typename IndexType, typename ValueType>
void launch_expand_block_to_scalar(
    int num_block_rows, int block_size, int num_dofs,
    const IndexType *A_row_offsets, const IndexType *A_col_indices,
    const ValueType *A_values,
    const IndexType *As_row_offsets,
    IndexType *As_col_indices, ValueType *As_values);

//  Creating the next level
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::createCoarseMatrices()
{
    typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
    typedef Vector<i64vec_value_type> I64Vector;
    typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
    typedef Vector<i64vec_value_type_h> I64Vector_h;

    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();

    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0).
                Notice that in aggregation path, prolongation P is implicit,
                and is used through the aggregates array. */

    bool const consolidation_level = !A.is_matrix_singleGPU() && this->isConsolidationLevel();

    // bookkeeping for the coarse grid: renumber aggregates,
    // if consolidation compute consolidated halo-offsets, etc
    if (!this->isReuseLevel())
    {
        if (consolidation_level)
        {
            // Consolidation-path steps 1-9
            this->consolidationBookKeeping();
        }
        else
        {
            this->setNeighborAggregates();
        }
    }

    this->getA().setView(ALL);

    // Compute restriction operator
    // TODO: computing the restriction operator could be merged with the selector to save some work
    // If we reuse the level we keep the previous restriction operator
    if (this->isReuseLevel() == false)
    {
        computeRestrictionOperator();
    }

    // --- SA path: check if near-null space is available ---
    // On the finest level, pull near-null space from the AMG object
    fprintf(stderr, "[SA-DBG] aggregation_amg_level::setup: m_null_dim=%d, amg=%p, hasSANearNullSpace=%d\n",
            m_null_dim, (void*)this->amg,
            (this->amg != nullptr) ? (int)this->amg->hasSANearNullSpace() : -1);
    if (m_null_dim == 0 && this->amg != nullptr && this->amg->hasSANearNullSpace())
    {
        const std::vector<double> &nns = this->amg->getSANearNullSpace();
        int nd = this->amg->getSANullDim();
        int nr = this->amg->getSANullRows();
        fprintf(stderr, "[SA-DBG] Pulling near-null space from AMG: null_dim=%d, null_rows=%d, data_size=%zu\n",
                nd, nr, nns.size());
        setNearNullSpace(nd, nr, nns.data());
    }
    fprintf(stderr, "[SA-DBG] After pull: m_null_dim=%d\n", m_null_dim);

    // Flag for distributed SA path — declared outside SA block so it's visible
    // in the RAP and prepareNextLevelMatrix sections below.
    bool sa_distributed = false;

    // Variables for distributed RAP — declared at function scope so they are
    // visible both inside the SA setup block and in the later RAP section.
    int num_owned_fine_pts  = 0;
    int num_owned_coarse_pts = 0;
    IVector_h P_neighbors_h;
    I64Vector_h P_halo_ranges_h;
    I64Vector P_halo_ranges;
    IVector_h P_halo_offsets_h;

    // SA is only supported for real-valued types (float/double), not complex.
    // Use sizeof check: for real types, sizeof(ValueTypeB) == sizeof(PODType);
    // for complex types, sizeof(ValueTypeB) == 2 * sizeof(PODType).
    fprintf(stderr, "[SA-DBG] SA condition: m_null_dim=%d, sizeof(ValueTypeB)=%zu, sizeof(PODType)=%zu\n",
            m_null_dim, sizeof(ValueTypeB), sizeof(typename types::PODTypes<ValueTypeB>::type));
    if (m_null_dim > 0 && sizeof(ValueTypeB) == sizeof(typename types::PODTypes<ValueTypeB>::type))
    {
        // SA path: build P_tent via QR, smooth it, then use for RAP
        // === Compact aggregate IDs to remove gaps ===
        // The selector may produce non-contiguous aggregate IDs (e.g., 0, 2, 5, 7
        // with gaps at 1, 3, 4, 6). P_tent needs contiguous column indices.
        {
            int num_fine = m_aggregates.size();
            // 1. Copy aggregates, sort, unique to find the set of used IDs
            IVector sorted_aggs(m_aggregates);
            thrust_wrapper::sort<TConfig::memSpace>(sorted_aggs.begin(), sorted_aggs.begin() + num_fine);
            auto new_end = amgx::thrust::unique(sorted_aggs.begin(), sorted_aggs.begin() + num_fine);
            int num_unique = (int)(new_end - sorted_aggs.begin());

            if (num_unique < m_num_aggregates)
            {

                // 2. Create mapping: for each fine node, find its new contiguous ID
                //    new_id = lower_bound position in sorted_unique array
                IVector new_aggs(num_fine);
                amgx::thrust::lower_bound(sorted_aggs.begin(), sorted_aggs.begin() + num_unique,
                                           m_aggregates.begin(), m_aggregates.begin() + num_fine,
                                           new_aggs.begin());

                // 3. Copy back and update count
                thrust_wrapper::copy<TConfig::memSpace>(new_aggs.begin(), new_aggs.begin() + num_fine,
                                                         m_aggregates.begin());
                m_num_aggregates = num_unique;
            }
        }
        buildTentativeProlongator();

        // --- Distributed SA: exchange halo rows of P_tent before smoothing ---
        // In the distributed case (multi-GPU), P_tent was built for owned fine rows only.
        // We need halo rows of P_tent (from neighbors) so that:
        //   (a) smoothProlongator can compute S*P_tent correctly (S connects owned to halo)
        //   (b) the Galerkin product P^T*A*P includes off-process contributions
        // This follows the classical AMG distributed RAP pattern.
        sa_distributed = !A.is_matrix_singleGPU();
        num_owned_fine_pts  = 0;
        num_owned_coarse_pts = m_num_aggregates * m_null_dim;  // coarse DOF count

        if (sa_distributed)
        {
            int block_size = A.get_block_dimy();
            // num_owned_fine_pts is DOF-level (P_tent rows = nodes * block_size)
            num_owned_fine_pts = A.manager->halo_offsets[0] * block_size;

            // 1. Set up m_P_tent.manager using initialize_manager
            //    This does an allgather to compute part_offsets for the coarse level.
            DistributedArranger<TConfig> *prep = new DistributedArranger<TConfig>;
            prep->initialize_manager(A, m_P_tent, num_owned_coarse_pts);

            // 2. Initialize Ac.manager from P_tent's manager info
            //    (override what setNeighborAggregates partially set up)
            IndexType num_parts = A.manager->get_num_partitions();
            IndexType my_rank = A.manager->global_id();

            if (Ac.manager == NULL)
            {
                Ac.manager = new DistributedManager<TConfig>();
            }
            Ac.manager->A = &Ac;
            Ac.manager->setComms(A.manager->getComms());
            Ac.manager->set_global_id(my_rank);
            Ac.manager->set_num_partitions(num_parts);
            Ac.manager->part_offsets_h = m_P_tent.manager->part_offsets_h;
            Ac.manager->part_offsets = m_P_tent.manager->part_offsets;
            Ac.manager->set_base_index(Ac.manager->part_offsets_h[my_rank]);
            Ac.manager->set_index_range(num_owned_coarse_pts);
            Ac.manager->num_rows_global = Ac.manager->part_offsets_h[num_parts];

            // 3. P_tent has no halo columns initially (all columns are owned coarse DOFs).
            //    local_to_global_map starts empty.
            Ac.manager->local_to_global_map.resize(0);

            // 4. Copy P manager info for exchange_halo_rows_P
            P_neighbors_h = m_P_tent.manager->neighbors;  // empty initially
            P_halo_ranges_h = m_P_tent.manager->halo_ranges_h;
            P_halo_ranges = m_P_tent.manager->halo_ranges;
            P_halo_offsets_h = m_P_tent.manager->halo_offsets;

            // 5. Exchange halo rows of P_tent from neighbors.
            //    This sends owned boundary rows of P_tent to neighbors and receives
            //    halo rows from neighbors, appending them to m_P_tent.
            //
            //    pack_halo_rows_P uses A.manager->B2L_maps[i] as row indices into P
            //    and A.manager->B2L_rings[i][ring] as the count of boundary rows.
            //    These are node-level indices. When block_size > 1, P_tent is a
            //    DOF-level scalar matrix (rows = nodes * block_size), so we must
            //    temporarily expand B2L maps to DOF-level indices.
            //
            //    For block_size == 1, node indices == DOF indices, so no change needed.

            // Save original B2L maps/rings and expand for block_size > 1
            int num_neighbors = A.manager->num_neighbors();
            std::vector<IVector> orig_B2L_maps;
            std::vector<std::vector<VecInt_t>> orig_B2L_rings;

            if (block_size > 1)
            {
                orig_B2L_maps.resize(num_neighbors);
                orig_B2L_rings.resize(num_neighbors);
                for (int i = 0; i < num_neighbors; i++)
                {
                    // Save originals
                    orig_B2L_maps[i] = A.manager->B2L_maps[i];
                    orig_B2L_rings[i].assign(A.manager->B2L_rings[i].begin(),
                                              A.manager->B2L_rings[i].end());

                    // Expand node-level B2L map to DOF-level:
                    // Each node n maps to DOFs [n*bs, n*bs+1, ..., n*bs+(bs-1)]
                    int node_count = (int)A.manager->B2L_maps[i].size();
                    IVector_h node_B2L_h(A.manager->B2L_maps[i]);
                    IVector_h dof_B2L_h(node_count * block_size);
                    for (int j = 0; j < node_count; j++)
                    {
                        int node = node_B2L_h[j];
                        for (int k = 0; k < block_size; k++)
                        {
                            dof_B2L_h[j * block_size + k] = node * block_size + k;
                        }
                    }
                    A.manager->B2L_maps[i] = dof_B2L_h;

                    // Scale B2L_rings counts by block_size
                    for (size_t r = 0; r < A.manager->B2L_rings[i].size(); r++)
                    {
                        A.manager->B2L_rings[i][r] *= block_size;
                    }
                }
            }

            prep->exchange_halo_rows_P(A, m_P_tent,
                                       Ac.manager->local_to_global_map,
                                       P_neighbors_h, P_halo_ranges_h, P_halo_ranges,
                                       P_halo_offsets_h,
                                       Ac.manager->part_offsets_h, Ac.manager->part_offsets,
                                       num_owned_coarse_pts,
                                       Ac.manager->part_offsets_h[my_rank]);
            cudaCheckError();

            // Restore original B2L maps/rings after exchange
            if (block_size > 1)
            {
                for (int i = 0; i < num_neighbors; i++)
                {
                    A.manager->B2L_maps[i] = orig_B2L_maps[i];
                    A.manager->B2L_rings[i].assign(orig_B2L_rings[i].begin(),
                                                    orig_B2L_rings[i].end());
                }
            }

            delete prep;
        }

        smoothProlongator();

        // Pass the SA-computed rho(D^{-1}A) to the Chebyshev smoother so that
        // lambda_mode=4 can reuse it instead of running a separate power iteration.
        // Must be done before setup_smoother() is called (which calls solver_setup()).
        if (m_sa_rho > 0.0)
        {
            Solver<TConfig> *sm = this->getSmoother();
            if (sm != nullptr)
            {
                Chebyshev_Solver<TConfig> *cheb =
                    dynamic_cast<Chebyshev_Solver<TConfig>*>(sm);
                if (cheb != nullptr)
                    cheb->setSAEigenvalue(m_sa_rho);
            }
        }

        // For the distributed case, set P_tent to show only owned fine rows
        // for the transpose (R = P^T uses owned rows only, like classical path).
        if (sa_distributed)
        {
            m_P_tent.set_initialized(0);
            m_P_tent.set_num_rows(num_owned_fine_pts);
            m_P_tent.addProps(CSR);
            m_P_tent.set_initialized(1);
        }

        // Compute and store P^T for use in restrictResidual()
        transpose(m_P_tent, m_P_tent_T);

        // Near-null space propagation is deferred to after Ac is fully set up
        // (including halo structure for distributed case). See below.
    }
    // --- End SA path ---

    Ac.set_initialized(0);
    Ac.copyAuxData(&A);
    // SA path: use the real-valued smoothed P and P^T for the Galerkin triple product.
    // The standard coarse-A generators use an implicit boolean P (aggregate injection),
    // which is incorrect for SA.  CSR_Multiply::csr_galerkin_product handles real-valued P.
    if (m_null_dim > 0 && m_P_tent.get_num_rows() > 0)
    {
        // ---------------------------------------------------------------
        // Block-size expansion: csr_galerkin_product requires block_size==1
        // for all three matrices.  P_tent is already scalar (DOF-level,
        // 1x1 blocks).  When A has block_size > 1, expand it to scalar CSR
        // A_scalar before the Galerkin product.
        // ---------------------------------------------------------------
        int sa_block_size = A.get_block_dimy();
        Matrix<TConfig> A_scalar_storage;  // only used when block_size > 1
        Matrix<TConfig> *A_for_rap = &A;   // points to A or A_scalar_storage

        if (sa_block_size > 1)
        {
            int num_block_rows = A.get_num_rows();
            int num_dofs       = num_block_rows * sa_block_size;
            int A_block_nnz    = A.get_num_nz();
            int As_nnz         = A_block_nnz * sa_block_size * sa_block_size;

            A_scalar_storage.set_initialized(0);
            A_scalar_storage.addProps(CSR);
            A_scalar_storage.delProps(COO);
            A_scalar_storage.delProps(DIAG);
            A_scalar_storage.setColsReorderedByColor(false);
            A_scalar_storage.resize(num_dofs, num_dofs, As_nnz, 1, 1, 1);

            // Step 1: compute per-scalar-row lengths via existing kernel,
            //         then exclusive scan -> As_row_offsets
            {
                const int threads = 256;
                const int blocks  = std::min(4096, (num_dofs + threads - 1) / threads);
                thrust_wrapper::fill<TConfig::memSpace>(
                    A_scalar_storage.row_offsets.begin(),
                    A_scalar_storage.row_offsets.end(),
                    (IndexType)0);
                build_SA_smoother_block_rowlen_kernel<IndexType><<<blocks, threads>>>(
                    num_block_rows, sa_block_size,
                    A.row_offsets.raw(),
                    A_scalar_storage.row_offsets.raw());
                cudaCheckError();
            }
            thrust_wrapper::exclusive_scan<TConfig::memSpace>(
                A_scalar_storage.row_offsets.begin(),
                A_scalar_storage.row_offsets.end(),
                A_scalar_storage.row_offsets.begin());
            cudaCheckError();

            // Step 2: fill col_indices and values
            // Use wrapper to avoid CUDA <<<>>> template-argument parse issue.
            {
                typedef ValueTypeA MatValueType;
                launch_expand_block_to_scalar<IndexType, MatValueType>(
                    num_block_rows, sa_block_size, num_dofs,
                    A.row_offsets.raw(),
                    A.col_indices.raw(),
                    A.values.raw(),
                    A_scalar_storage.row_offsets.raw(),
                    A_scalar_storage.col_indices.raw(),
                    A_scalar_storage.values.raw());
                cudaCheckError();
            }

            A_scalar_storage.set_initialized(1);
            A_for_rap = &A_scalar_storage;
        }

        if (!sa_distributed)
        {
            // Single-GPU path: local-only triple product
            CSR_Multiply<TConfig>::csr_galerkin_product(
                m_P_tent_T, *A_for_rap, m_P_tent, Ac,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

            // DEBUG diagnostic: check Ac after RAP
            {
                typedef ValueTypeA MVal;
                typedef typename types::PODTypes<MVal>::type PodM;
                int Ac_nnz  = Ac.get_num_nz();
                int Ac_rows = Ac.get_num_rows();
                int Ac_cols = Ac.get_num_cols();
                if (Ac_nnz > 0)
                {
                    std::vector<MVal> Ac_vals_h(Ac_nnz);
                    cudaMemcpy(Ac_vals_h.data(), Ac.values.raw(),
                               Ac_nnz * sizeof(MVal), cudaMemcpyDeviceToHost);
                    std::vector<int> Ac_row_h(Ac_rows + 1);
                    cudaMemcpy(Ac_row_h.data(), Ac.row_offsets.raw(),
                               (Ac_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
                    std::vector<int> Ac_col_h(Ac_nnz);
                    cudaMemcpy(Ac_col_h.data(), Ac.col_indices.raw(),
                               Ac_nnz * sizeof(int), cudaMemcpyDeviceToHost);
                    int nan_cnt = 0, inf_cnt = 0;
                    double max_val = 0.0, min_diag = 1e300, max_diag = 0.0;
                    // find diagonal entries
                    for (int row = 0; row < Ac_rows; ++row)
                    {
                        for (int j = Ac_row_h[row]; j < Ac_row_h[row+1]; ++j)
                        {
                            double v = (double)static_cast<PodM>(types::util<MVal>::abs(Ac_vals_h[j]));
                            if (std::isnan(v)) nan_cnt++;
                            if (std::isinf(v)) inf_cnt++;
                            if (v > max_val) max_val = v;
                            if (Ac_col_h[j] == row)
                            {
                                if (v < min_diag) min_diag = v;
                                if (v > max_diag) max_diag = v;
                            }
                        }
                    }
                    fprintf(stderr, "[DEBUG RAP] Ac: rows=%d cols=%d nnz=%d NaN=%d Inf=%d "
                            "max_val=%.6e min_diag=%.6e max_diag=%.6e cond_diag=%.3e\n",
                            Ac_rows, Ac_cols, Ac_nnz, nan_cnt, inf_cnt,
                            max_val, min_diag, max_diag,
                            (min_diag > 0.0 ? max_diag / min_diag : -1.0));

                    // Dump Ac (scalar, before createBlockGraph) in Matrix Market format.
                    // One file per level: amgx_Ac_level<L>.mtx
                    // This allows offline eigenvalue analysis to check SPD-ness.
                    {
                        char ac_fname[256];
                        snprintf(ac_fname, sizeof(ac_fname),
                                 "/pscratch/sd/m/madams/amgx_Ac_level%d.mtx",
                                 this->getLevelIndex());
                        FILE *fp = fopen(ac_fname, "w");
                        if (fp)
                        {
                            fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
                            fprintf(fp, "%d %d %d\n", Ac_rows, Ac_cols, Ac_nnz);
                            for (int row = 0; row < Ac_rows; ++row)
                            {
                                for (int j = Ac_row_h[row]; j < Ac_row_h[row+1]; ++j)
                                {
                                    // Extract signed scalar value via memcpy
                                    PodM pod_v;
                                    memcpy(&pod_v, &Ac_vals_h[j], sizeof(PodM));
                                    fprintf(fp, "%d %d %.17e\n",
                                            row + 1, Ac_col_h[j] + 1, (double)pod_v);
                                }
                            }
                            fclose(fp);
                            fprintf(stderr, "[DEBUG RAP] Ac (level %d, scalar %d×%d) dumped to %s\n",
                                    this->getLevelIndex(), Ac_rows, Ac_cols, ac_fname);
                        }
                        else
                        {
                            fprintf(stderr, "[DEBUG RAP] WARNING: could not open %s for writing\n",
                                    ac_fname);
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // Regularize Ac:
            // 1. Set zero diagonal entries to 1.0 (degenerate coarse DOFs).
            // 2. Shift ALL diagonals by avg_diag * 1e-6 to handle
            //    semi-definite coarse matrices (rigid body modes in the
            //    null space of Ac).  This ensures DENSE_LU_SOLVER can
            //    factor the matrix.
            // ---------------------------------------------------------------
            {
                typedef ValueTypeA MVal;
                typedef typename types::PODTypes<MVal>::type PodM;
                int Ac_rows = Ac.get_num_rows();
                int Ac_nnz  = Ac.get_num_nz();
                if (Ac_nnz > 0 && Ac_rows > 0)
                {
                    std::vector<MVal> vals_h(Ac_nnz);
                    std::vector<int>  row_h(Ac_rows + 1), col_h(Ac_nnz);
                    cudaMemcpy(vals_h.data(), Ac.values.raw(), Ac_nnz * sizeof(MVal), cudaMemcpyDeviceToHost);
                    cudaMemcpy(row_h.data(), Ac.row_offsets.raw(), (Ac_rows+1) * sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(col_h.data(), Ac.col_indices.raw(), Ac_nnz * sizeof(int), cudaMemcpyDeviceToHost);

                    // Find diagonal positions and compute average
                    std::vector<int> diag_pos(Ac_rows, -1);
                    double diag_sum = 0.0;
                    int n_diag = 0, n_zero = 0;
                    for (int row = 0; row < Ac_rows; ++row)
                    {
                        for (int j = row_h[row]; j < row_h[row+1]; ++j)
                        {
                            if (col_h[j] == row)
                            {
                                diag_pos[row] = j;
                                PodM dval = static_cast<PodM>(types::util<MVal>::abs(vals_h[j]));
                                if (dval < 1e-14)
                                {
                                    PodM one_pod = static_cast<PodM>(1.0);
                                    memcpy(&vals_h[j], &one_pod, sizeof(PodM));
                                    n_zero++;
                                    diag_sum += 1.0;
                                }
                                else
                                {
                                    diag_sum += (double)dval;
                                }
                                n_diag++;
                                break;
                            }
                        }
                    }

                    // Shift all diagonals by avg_diag * 1e-6
                    double avg_diag = (n_diag > 0) ? diag_sum / n_diag : 1.0;
                    PodM shift = static_cast<PodM>(avg_diag * 1e-6);
                    for (int row = 0; row < Ac_rows; ++row)
                    {
                        if (diag_pos[row] >= 0)
                        {
                            PodM d;
                            memcpy(&d, &vals_h[diag_pos[row]], sizeof(PodM));
                            d += shift;
                            memcpy(&vals_h[diag_pos[row]], &d, sizeof(PodM));
                        }
                    }

                    cudaMemcpy(Ac.values.raw(), vals_h.data(), Ac_nnz * sizeof(MVal), cudaMemcpyHostToDevice);
                    if (n_zero > 0 || true)
                    {
                        fprintf(stderr, "[SA-FIX] Ac (%d×%d): %d zero diags→1.0, "
                                "shift=%.2e (avg_diag=%.2e)\n",
                                Ac_rows, Ac_rows, n_zero,
                                avg_diag * 1e-6, avg_diag);
                    }
                }
            }

            // ---------------------------------------------------------------
            // Block-compress scalar Ac → block-null_dim Ac so that the next
            // MIS coarsening operates on aggregate-level block nodes (like
            // PETSc GAMG which annotates Ac with block_size=null_dim).
            //
            // The coarse Ac block size is the column block size of P, which
            // equals m_null_dim (the number of near-null vectors).
            // ---------------------------------------------------------------
            if (m_null_dim > 1)
            {
                this->createBlockGraph(Ac, m_null_dim);
            }
        }
        else
        {
            // ---------------------------------------------------------------
            // Distributed SA path: follow the classical AMG RAP pattern.
            // P_tent has halo rows appended (from exchange_halo_rows_P above),
            // but num_rows is set to owned only. The CSR arrays still contain
            // halo data, so the galerkin product can access halo rows of P
            // via column indices of A that reference halo fine nodes.
            // ---------------------------------------------------------------

            // Initialize CSR workspace (csr_workspace_create only exists on device)
            typedef TemplateConfig<AMGX_device, vecPrec, matPrec, indPrec> TConfig_d_rap;
            void *wk = AMG_Level<TConfig>::amg->getCsrWorkspace();
            if (wk == NULL)
            {
                wk = CSR_Multiply<TConfig_d_rap>::csr_workspace_create(
                    *(AMG_Level<TConfig>::amg->m_cfg),
                    AMG_Level<TConfig>::amg->m_cfg_scope);
                AMG_Level<TConfig>::amg->setCsrWorkspace(wk);
            }

            // For distributed: A_scalar_storage needs the manager from A
            // so that halo columns are correctly referenced.
            if (sa_block_size > 1)
            {
                A_scalar_storage.manager = A.manager;
            }

            // Compute RAP_full = R * A * P (local computation including halo contributions)
            // RAP_full will have num_owned_coarse_pts rows for owned coarse nodes,
            // plus extra rows for halo coarse nodes (from halo rows of P_tent_T).
            Matrix<TConfig> RAP_full;
            RAP_full.set_initialized(0);
            A.setView(OWNED);
            if (sa_block_size > 1)
                A_scalar_storage.setView(OWNED);
            CSR_Multiply<TConfig>::csr_galerkin_product(
                m_P_tent_T, *A_for_rap, m_P_tent, RAP_full,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, wk);
            RAP_full.set_initialized(1);

            // Update m_P_tent.manager with info modified by exchange_halo_rows_P.
            // exchange_RAP_ext -> pack_halo_rows_RAP uses P.manager->neighbors,
            // P.manager->halo_offsets, P.manager->base_index(), etc.
            m_P_tent.manager->neighbors = P_neighbors_h;
            m_P_tent.manager->halo_offsets = P_halo_offsets_h;
            m_P_tent.manager->halo_ranges_h = P_halo_ranges_h;
            m_P_tent.manager->halo_ranges = P_halo_ranges;

            // Exchange RAP rows: send extra rows (halo coarse) to neighbors,
            // receive contributions from neighbors, and assemble final Ac.
            DistributedArranger<TConfig> *prep2 = new DistributedArranger<TConfig>;
            prep2->exchange_RAP_ext(Ac, RAP_full, A, m_P_tent,
                                    P_halo_offsets_h,
                                    Ac.manager->local_to_global_map,
                                    P_neighbors_h, P_halo_ranges_h, P_halo_ranges,
                                    Ac.manager->part_offsets_h, Ac.manager->part_offsets,
                                    num_owned_coarse_pts,
                                    Ac.manager->part_offsets_h[A.manager->global_id()],
                                    wk);
            delete prep2;

            // Column compression: remove unused halo columns from Ac.
            // Some halo columns in local_to_global_map may not appear in
            // the owned rows of Ac after assembly. This mirrors the classical
            // path's column compression (classical_amg_level.cu).
            IndexType nrow = Ac.get_num_rows();
            IndexType ncol = Ac.get_num_cols();
            IndexType nl2g = ncol - nrow;

            if (nl2g > 0)
            {
                IVector l2g_p(nl2g + 1, 0);  // +1 for exclusive_scan
                I64Vector l2g_t(nl2g, 0);
                IndexType nblocks = std::min(4096, (int)((nrow + 127) / 128));

                // Step 1: Flag which halo columns are referenced
                if (nblocks > 0)
                {
                    sa_flag_halo_columns_kernel<IndexType><<<nblocks, 128>>>(
                        nrow, Ac.row_offsets.raw(), Ac.col_indices.raw(), l2g_p.raw());
                }
                cudaCheckError();

                // Step 2: Exclusive scan to get new positions
                thrust_wrapper::exclusive_scan<TConfig::memSpace>(
                    l2g_p.begin(), l2g_p.end(), l2g_p.begin());
                int new_nl2g = l2g_p[nl2g];

                // Step 3: Compress column indices
                if (nblocks > 0)
                {
                    sa_compress_halo_columns_kernel<IndexType><<<nblocks, 128>>>(
                        nrow, Ac.row_offsets.raw(), Ac.col_indices.raw(), l2g_p.raw());
                }
                cudaCheckError();

                // Step 4: Adjust matrix size
                Ac.set_initialized(0);
                Ac.set_num_cols(nrow + new_nl2g);
                Ac.set_initialized(1);

                // Step 5: Compress local_to_global_map
                nblocks = std::min(4096, (int)((nl2g + 127) / 128));
                if (nblocks > 0)
                {
                    sa_compress_l2g_kernel<IndexType, int64_t><<<nblocks, 128>>>(
                        nl2g, Ac.manager->local_to_global_map.raw(),
                        l2g_t.raw(), l2g_p.raw());
                }
                cudaCheckError();
                amgx::thrust::copy(l2g_t.begin(), l2g_t.begin() + new_nl2g,
                                   Ac.manager->local_to_global_map.begin());
                cudaCheckError();
                Ac.manager->local_to_global_map.resize(new_nl2g);
            }

            // Finalize the distributed coarse matrix:
            // renumberMatrixOneRing creates B2L maps and sets up halo structure
            Ac.set_initialized(0);
            Ac.manager->renumberMatrixOneRing(this->isReuseLevel() ? 1 : 0);
            Ac.manager->createOneRingHaloRows();
            Ac.manager->getComms()->set_neighbors(Ac.manager->num_neighbors());
            Ac.setView(OWNED);
            Ac.set_initialized(1);

            // Restore A to ALL view for consistency
            A.setView(ALL);
        }
    }
    else
    {
        this->m_coarseAGenerator->computeAOperator(A, Ac, this->m_aggregates, this->m_R_row_offsets, this->m_R_column_indices, this->m_num_all_aggregates);
    }
    Ac.setColsReorderedByColor(false);
    Ac.setView(FULL);

    if (consolidation_level)
    {
        // Consolidation-path Steps 11-12, send matrices to root, consolidate, final bookkeeping
        this->consolidateCoarseGridMatrix();
    }
    else
    {
        // For the distributed SA path, the coarse matrix is already finalized
        // (renumberMatrixOneRing + createOneRingHaloRows were called above).
        // Only call prepareNextLevelMatrix for non-SA or single-GPU paths.
        if (!(m_null_dim > 0 && m_P_tent.get_num_rows() > 0 && sa_distributed))
        {
            this->prepareNextLevelMatrix(A, Ac);
        }
    }

    A.setView(OWNED);
    Ac.setView(OWNED);

    if (sa_distributed && m_null_dim > 0)
    {
        // For distributed SA, m_next_level_size uses the FULL view of Ac
        int size, offset;
        Ac.getOffsetAndSizeForView(FULL, &offset, &size);
        this->m_next_level_size = size * Ac.get_block_dimy();
    }
    else if (m_null_dim > 0 && A.get_block_dimy() > 1)
    {
        // Block SA path: after block-compression, Ac has num_rows = num_aggs
        // and block_dimy = null_dim.  The next level size is num_aggs * null_dim.
        this->m_next_level_size = Ac.get_num_rows() * Ac.get_block_dimy();
    }
    else
    {
        this->m_next_level_size = this->m_num_all_aggregates * Ac.get_block_dimy();
    }

    // ---------------------------------------------------------------
    // Propagate coarse near-null space to the next level.
    // This is done after Ac is fully set up so that in the distributed
    // case we can exchange halo aggregate near-null space values.
    // ---------------------------------------------------------------
    fprintf(stderr, "[SA-PROP] level=%d  m_null_dim=%d  m_coarse_nns_size=%zu  "
            "m_num_aggregates=%d\n",
            this->getLevelIndex(), m_null_dim,
            (size_t)m_coarse_near_null_space.size(), m_num_aggregates);
    if (m_null_dim > 0 && m_coarse_near_null_space.size() > 0)
    {
        AMG_Level<TConfig> *next = this->getNextLevel(MemorySpace());
        Aggregation_AMG_Level_Base<TConfig> *next_agg =
            dynamic_cast<Aggregation_AMG_Level_Base<TConfig>*>(next);
        fprintf(stderr, "[SA-PROP] level=%d  next=%p  next_agg=%p\n",
                this->getLevelIndex(), (void*)next, (void*)next_agg);
        if (next_agg != nullptr)
        {
            int num_owned_aggs = m_num_aggregates;
            int coarse_dofs_owned = num_owned_aggs * m_null_dim;

            if (sa_distributed && Ac.manager != nullptr &&
                Ac.manager->num_neighbors() > 0)
            {
                // ---------------------------------------------------
                // Distributed case (all null_dim values):
                // Exchange halo near-null space values so the next
                // level's buildTentativeProlongator() can look up
                // near-null space for halo aggregates.
                //
                // The coarse near-null space from QR has layout:
                //   [vec_0(agg_0..agg_{N-1}), vec_1(agg_0..agg_{N-1}), ...]
                // where N = num_owned_aggs.  We need to expand each
                // vector to include halo aggregate values, producing:
                //   [vec_0(agg_0..agg_{T-1}), vec_1(agg_0..agg_{T-1}), ...]
                // where T = total_coarse_nodes (owned + halo).
                //
                // We exchange each near-null vector separately using
                // exchange_halo(), which handles one scalar per node.
                // ---------------------------------------------------

                // Total coarse nodes including halos
                int num_halo_offsets = Ac.manager->num_halo_offsets();
                int total_coarse_nodes = (num_halo_offsets > 0)
                    ? Ac.manager->halo_offset(num_halo_offsets - 1)
                    : num_owned_aggs;

                // Output: null_dim blocks of total_coarse_nodes each
                int total_nns_size = m_null_dim * total_coarse_nodes;
                std::vector<double> h_coarse_nns_double(total_nns_size);

                // Temporary device vector for a single near-null vector
                VVector nns_with_halo(total_coarse_nodes);

                for (int v = 0; v < m_null_dim; ++v)
                {
                    // Copy owned values for vector v into the owned portion.
                    // Source: m_coarse_near_null_space at offset v * num_owned_aggs
                    cudaMemcpy(nns_with_halo.raw(),
                               m_coarse_near_null_space.raw() + v * num_owned_aggs,
                               num_owned_aggs * sizeof(ValueTypeB),
                               cudaMemcpyDeviceToDevice);
                    cudaCheckError();

                    // Mark dirty so exchange_halo sends data
                    nns_with_halo.dirtybit = 1;
                    nns_with_halo.set_block_dimx(1);
                    nns_with_halo.set_block_dimy(1);

                    // Exchange halo values for this near-null vector.
                    // Use distinct tags per vector to avoid message confusion.
                    Ac.manager->exchange_halo(nns_with_halo, 7799 + v);

                    // Copy full vector (owned + halo) to host and convert
                    // to double for setNearNullSpace.
                    std::vector<ValueTypeB> h_vtmp(total_coarse_nodes);
                    cudaMemcpy(h_vtmp.data(), nns_with_halo.raw(),
                               total_coarse_nodes * sizeof(ValueTypeB),
                               cudaMemcpyDeviceToHost);
                    cudaCheckError();

                    int dst_offset = v * total_coarse_nodes;
                    for (int i = 0; i < total_coarse_nodes; ++i)
                    {
                        h_coarse_nns_double[dst_offset + i] = static_cast<double>(
                            *reinterpret_cast<typename types::PODTypes<ValueTypeB>::type*>(&h_vtmp[i]));
                    }
                }

                next_agg->setNearNullSpace(m_null_dim, total_coarse_nodes, h_coarse_nns_double.data());
            }
            else
            {
                // ---------------------------------------------------
                // Single-GPU case (no halo neighbors):
                // Propagate owned near-null space only.
                // ---------------------------------------------------
                int total = coarse_dofs_owned * m_null_dim;

                std::vector<ValueTypeB> h_vtmp(total);
                cudaMemcpy(h_vtmp.data(), m_coarse_near_null_space.raw(),
                           total * sizeof(ValueTypeB), cudaMemcpyDeviceToHost);
                cudaCheckError();

                std::vector<double> h_coarse_nns_double(total);
                for (int i = 0; i < total; ++i)
                {
                    h_coarse_nns_double[i] = static_cast<double>(
                        *reinterpret_cast<typename types::PODTypes<ValueTypeB>::type*>(&h_vtmp[i]));
                }

                fprintf(stderr, "[SA-PROP] level=%d  single-GPU: calling setNearNullSpace("
                        "null_dim=%d, num_rows=%d, total_data=%d)\n",
                        this->getLevelIndex(), m_null_dim, coarse_dofs_owned,
                        (int)h_coarse_nns_double.size());
                next_agg->setNearNullSpace(m_null_dim, coarse_dofs_owned, h_coarse_nns_double.data());
                fprintf(stderr, "[SA-PROP] level=%d  setNearNullSpace done; "
                        "next_agg->m_null_dim=%d\n",
                        this->getLevelIndex(), next_agg->getNullDim());
            }
        }
    }

    if (this->m_print_aggregation_info)
    {
        MatrixAnalysis<TConfig> ana(&Ac);
        ana.aggregatesQuality2(this->m_aggregates, this->m_num_aggregates, A);
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::consolidationBookKeeping()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();


    int num_parts, num_fine_neighbors, my_id;

    if (!A.is_matrix_singleGPU())
    {
        num_parts = A.manager->getComms()->get_num_partitions();
        num_fine_neighbors = A.manager->neighbors.size();
        my_id = A.manager->global_id();
    }
    else
    {
        num_parts = 1;
        num_fine_neighbors = 0;
        my_id = 0;
    }

    // ----------------------------------------------------
    // Consolidate multiple fine matrices into one coarse matrix
    // ----------------------------------------------------
    // ----------------
    // Step 1
    // Decide which partitions should be merged together, store in destination_partitions vector
    // ---------------
    IVector_h &destination_part = A.manager->getDestinationPartitions();
    int my_destination_part = A.manager->getMyDestinationPartition();

    if (my_destination_part >= num_parts)
    {
        FatalError("During consolidation, sending data to partition that doesn't exist", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // Create mapping from coarse partition indices (ranks on the coarse consolidated level) to partition indices on the fine level (ranks on the fine level)
    IVector_h coarse_part_to_fine_part = destination_part;
    amgx::thrust::sort(coarse_part_to_fine_part.begin(), coarse_part_to_fine_part.end());
    cudaCheckError();
    coarse_part_to_fine_part.erase(thrust::unique(coarse_part_to_fine_part.begin(), coarse_part_to_fine_part.end()), coarse_part_to_fine_part.end());
    cudaCheckError();
    //Then, the number of coarse partitions is simply the size of this vector
    int num_coarse_partitions = coarse_part_to_fine_part.size();
    // Create mapping from fine partition indices to coarse partition indices, with fine partitions that are merging together having the same coarse indices
    IVector_h fine_part_to_coarse_part(num_parts);
    amgx::thrust::lower_bound(coarse_part_to_fine_part.begin(), coarse_part_to_fine_part.end(), destination_part.begin(), destination_part.end(), fine_part_to_coarse_part.begin());
    cudaCheckError();
    // Create mapping from this specific partition's neighbors to consolidated coarse neighbors, but using their fine index (aka. destination partition indices for my neighbors)
    IVector_h fine_neigh_to_fine_part;
    A.manager->createNeighToDestPartMap(fine_neigh_to_fine_part, A.manager->neighbors, destination_part, num_fine_neighbors);
    // Create mapping from consolidated coarse neighbors to fine partition indices (even if the current partition is not going to be a root)
    IVector_h coarse_neigh_to_fine_part;
    int num_coarse_neighbors;
    A.manager->createConsolidatedNeighToPartMap(coarse_neigh_to_fine_part, fine_neigh_to_fine_part, my_destination_part, destination_part, num_coarse_neighbors);
    // Create mapping from fine neighbors to coarse neighbors, with fine neighbors this partition is merging with labeled with -1
    IVector_h fine_neigh_to_coarse_neigh;
    A.manager->createNeighToConsNeigh(fine_neigh_to_coarse_neigh, coarse_neigh_to_fine_part, fine_neigh_to_fine_part, my_destination_part, num_fine_neighbors);
    /*
        EXAMPLE
        Take the following partition graph (that describes connections between partitions, vertices are the partitions themselves), this is the same graph that is used in the setup example
        number of partitions num_parts=12
        CSR row_offsets [0 4 8 13 21 25 32 36 41 46 50 57 61]
        CSR col_indices [0 1 3 8
                    0 1 2 3
                    1 2 3 4 5
                    0 1 2 3 4 5 8 10
                    2 4 5 6
                    2 3 4 5 6 7 10
                    4 5 6 7
                    5 6 7 9 10
                    0 3 8 10 11
                    7 9 10 11
                    3 5 7 8 9 10 11
                    8 9 10 11]
        destination_part = [0 0 0 0 4 4 4 4 8 8 8 8]
        coarse_part_to_fine_part = [0 4 8] num_coarse_partitions = 3
        fine_part_to_coarse_part = [0 0 0 0 1 1 1 1 2 2 2 2]
        original neighbor lists correspond to the rows of the matrix, minus the diagonal elements: (part 0)[1 3 8] (part 3)[0 1 2 4 5 8 10] (part 10)[3 5 7 8 9 11]
        fine_neigh_to_fine_part (part 0)[0 0 2] (part 3)[0 0 0 0 1 2 2] (part 10)[0 1 1 2 2 2]
        coarse_neigh_to_fine_part (part 0)[8] (part 3)[4 8] (part 10)[0 4]
        fine_neigh_to_coarse_neigh (part 0)[-1 -1 0] (part 3)[-1 -1 -1 0 0 1 1] (part 10)[0 1 1 -1 -1 -1]
        */
    // --------------------------
    // Step 2
    // Create coarse B2L_maps, by mapping fine B2L maps to coarse indices using this->m_aggregates and eliminating duplicates
    // --------------------------
    std::vector<IVector> coarse_B2L_maps(num_fine_neighbors);
    m_num_all_aggregates = m_num_aggregates;
    int num_neighbors_temp = A.manager->neighbors.size();
    int num_rings = A.manager->B2L_rings[0].size() - 1;

    if (num_rings != 1)
    {
        FatalError("num_rings > 1 not supported in consolidation\n", AMGX_ERR_NOT_IMPLEMENTED);
    }

    IndexType max_b2l = 0;


    for (int i = 0; i < num_neighbors_temp; i++ ) { max_b2l = max_b2l > A.manager->B2L_rings[i][1] ? max_b2l : A.manager->B2L_rings[i][1]; }

    IVector B2L_aggregates(max_b2l);
    IVector indices(max_b2l);

    //TODO: use the algorithm from setNeighborAggregates()
    for (int i = 0; i < num_neighbors_temp; i++ )
    {
        int size = A.manager->B2L_rings[i][1];
        thrust_wrapper::fill<TConfig::memSpace>(B2L_aggregates.begin(), B2L_aggregates.begin() + size, 0);
        thrust_wrapper::sequence<TConfig::memSpace>(indices.begin(), indices.begin() + size);
        //substitute coarse aggregate indices for fine boundary nodes
        amgx::thrust::copy(amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin()),
                        amgx::thrust::make_permutation_iterator(this->m_aggregates.begin(), A.manager->B2L_maps[i].begin() + size),
                        B2L_aggregates.begin());
        //find the unique ones
        amgx::thrust::sort_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin());
        IndexType num_unique = amgx::thrust::unique_by_key(B2L_aggregates.begin(), B2L_aggregates.begin() + size, indices.begin()).first - B2L_aggregates.begin();
        coarse_B2L_maps[i].resize(num_unique);
        //sort it back so we have the original ordering
        amgx::thrust::sort_by_key(indices.begin(), indices.begin() + num_unique, B2L_aggregates.begin());
        amgx::thrust::copy(B2L_aggregates.begin(), B2L_aggregates.begin() + num_unique, coarse_B2L_maps[i].begin());
    }

    cudaCheckError();
    /*
        * EXAMPLE
        say, partition 3 has the following coarse B2L_maps:
        neighbors [0 1 2 4 5 8 10]
        B2L_maps[0(=0)] = [6 7 8]
        B2L_maps[1(=1)] = [8 9 10]
        B2L_maps[2(=2)] = [10 11 12 13]
        B2L_maps[3(=4)] = [13 14 15]
        B2L_maps[4(=5)] = [15 16 17]
        B2L_maps[5(=8)] = [6 18 19]
        B2L_maps[6(=10)] = [17 20 19]
        */
    // ---------------------------------------------------
    // Step 3
    // create new B2L maps for each merged destination neighbor and drop B2L maps to neighbors we are merging with
    // ---------------------------------------------------
    std::vector<IVector> dest_coarse_B2L_maps;
    A.manager->consolidateB2Lmaps(dest_coarse_B2L_maps, coarse_B2L_maps, fine_neigh_to_coarse_neigh, num_coarse_neighbors, num_fine_neighbors);
    /*
        * EXAMPLE
        Then, merging the coarse B2L maps on partition 3, we get:
        coarse_neigh_to_fine_part [4 8]
        dest_coarse_B2L_maps[0(=4)] = [13 14 15 16 17]
        dest_coarse_B2L_maps[1(=8)] = [6 17 18 19 20]
        */
    // -----------------------
    // Step 4
    // Create interior-boundary renumbering of aggregates according to dest_coarse_B2L_maps
    // -----------------------
    // Now renumber the aggregates with all interior aggregates first, boundary aggregates second
    int num_interior_aggregates; //returned by createAggregatesRenumbering
    int num_boundary_aggregates; //returned by createAggregatesRenumbering
    IVector renumbering; //returned by createAggregatesRenumbering
    // Following calls create renumbering array and modifies B2L_maps
    A.manager->createAggregatesRenumbering(renumbering, dest_coarse_B2L_maps, this->m_num_aggregates, num_coarse_neighbors, num_interior_aggregates, num_boundary_aggregates, num_rings);
    /*
        * EXAMPLE
        Partition 3 will get a renumbering vector of size 21, for the 21 owned agggregates:
        [0 1 2 3 4 5 17 6 7 8 9 10 11 12 13 14 15 16 18 19 20]
        num_interior_aggregates = 12
        num_boundary_aggregates = 9
        */
    // -------------------------------------------------
    // Step 5
    // Determine whether root partition, make list of partitions merged into one
    // ------------------------------------------------
    // Check if I'm root partition and how fine partitions (including myself) are merging into me
    // bool is_root_partition = false;
    bool &is_root_partition = this->m_is_root_partition;
    is_root_partition = false; 
    int num_fine_parts_to_consolidate = 0;
    // IVector_h fine_parts_to_consolidate;
    IVector_h &fine_parts_to_consolidate = this->m_fine_parts_to_consolidate;

    for (int i = 0; i < num_parts; i++)
    {
        if (destination_part[i] == my_id)
        {
            is_root_partition = true;
            num_fine_parts_to_consolidate++;
        }
    }

    fine_parts_to_consolidate.resize(num_fine_parts_to_consolidate);
    int count = 0;

    for (int i = 0; i < num_parts; i++)
    {
        if (destination_part[i] == my_id)
        {
            fine_parts_to_consolidate[count] = i;
            count++;
        }
    }

    //save this information as state, as this will also be required during solve for restriction/prolongation
    A.manager->setIsRootPartition(is_root_partition);
    A.manager->setNumPartsToConsolidate(num_fine_parts_to_consolidate);
    A.manager->setPartsToConsolidate(fine_parts_to_consolidate);

    // Create a new distributed communicator for coarse levels that only contains active partitions
    if (Ac.manager == NULL)
    {
        Ac.manager = new DistributedManager<TConfig>();
    }

    Ac.manager->setComms(A.manager->getComms()->Clone());
    Ac.manager->getComms()->createSubComm(coarse_part_to_fine_part, is_root_partition);


    /*
        * EXAMPLE
        isRootPartition is true for partitions 0,4,8 false for others
        num_fine_parts_to_consolidate = 4 for partitions 0,4,8
        fine_parts_to_consolidate (part 0)[0 1 2 3] (part 4)[4 5 6 7] (part 8)[8 9 10 11]
        */
    // ----------------------
    // Step 6
    // Compute number of interior, boundary and total nodes in the consolidated coarse matrix. Create offsets so that partitions being merged together will have their aggregate indices ordered like this:
    // [num_interior(fine_parts_to_consolidate[0]] num_interior(fine_parts_to_consolidate[1]] ... num_interior(fine_parts_to_consolidate[num_fine_parts_to_consolidate]
    //        num_boundary(fine_parts_to_consolidate[0]] num_boundary(fine_parts_to_consolidate[1]] ... num_boundary(fine_parts_to_consolidate[num_fine_parts_to_consolidate] ]
    // ----------------------
    // Gather to get number of interior/boundary aggregates of neighbors I will merge with
    // std::vector<IVector_h> vertex_counts;
    std::vector<IVector_h> &vertex_counts = this->m_vertex_counts;
    // int interior_offset, boundary_offset, total_interior_rows_in_merged, total_boundary_rows_in_merged;
    int interior_offset, boundary_offset;
    int &total_interior_rows_in_merged = this->m_total_interior_rows_in_merged;
    int &total_boundary_rows_in_merged = this->m_total_boundary_rows_in_merged;
    int total_rows_in_merged;
    //Computes these offsets on the root, sends them back
    A.manager->computeConsolidatedOffsets(my_id, my_destination_part, is_root_partition, num_interior_aggregates, num_boundary_aggregates, vertex_counts, fine_parts_to_consolidate, num_fine_parts_to_consolidate, interior_offset, boundary_offset, total_interior_rows_in_merged, total_boundary_rows_in_merged, total_rows_in_merged, A.manager->getComms());
    //Partitions save these offsets, as it will be required during solve restriction/prolongation
    A.manager->setConsolidationOffsets(interior_offset, num_interior_aggregates, boundary_offset + num_interior_aggregates, num_boundary_aggregates);
    /*
        * EXAMPLE
        For root partition 0, say we have the following interior/boundary counts (note that partition 1 has 0 boundary, as it is only connected to partitions it is merging with)
        part 0 - interior: 10 boundary 3
        part 1 - interior: 18
        part 2 - interior: 10 boundary 16
        part 3 - interior: 12 boundary 9
        interior_offset for partitions 0,1,2,3: 0 10 28 38 (total_interior_rows_in_merged 50)
        boundary_offset for partitions 0,1,2,3: 0 3 3 19 (total_boundary_rows_in_merged 28)
        */
    // ----------------------
    // Step 7
    // Each partition renumbers its aggregates and dest_coarse_B2L_maps using offsets computed in Step 6 and permutation in Step 4
    // ----------------------
    // Kernel to renumber the aggregates
    int block_size = 128;
    int grid_size = std::min( 4096, ( A.manager->halo_offsets[0] + block_size - 1 ) / block_size);
    renumberAggregatesKernel <<< grid_size, block_size >>>(renumbering.raw(), interior_offset, boundary_offset, this->m_aggregates.raw(), A.manager->halo_offsets[0], num_interior_aggregates, renumbering.size());
    cudaCheckError();

    for (int i = 0; i < num_coarse_neighbors; i++)
    {
        thrust_wrapper::transform<TConfig::memSpace>(dest_coarse_B2L_maps[i].begin(),
                            dest_coarse_B2L_maps[i].end(),
                            amgx::thrust::constant_iterator<IndexType>(boundary_offset),
                            dest_coarse_B2L_maps[i].begin(),
                            amgx::thrust::plus<IndexType>());
    }

    cudaCheckError();
    /*
        * EXAMPLE
        Partition 3 had a renumbering vector:
        [0 1 2 3 4 5 17 6 7 8 9 10 11 12 13 14 15 16 18 19 20]
        which is now adjusted to account for the consolidated coarse matrices' indices:
        [38 39 40 41 42 43 74 44 45 46 47 48 49 69 70 71 72 73 75 76 77]
        And the dest_coarse_B2L_maps, which looked like:
        dest_coarse_B2L_maps[0(=4)] = [13 14 15 16 17]
        dest_coarse_B2L_maps[1(=8)] = [6 17 18 19 20]
        is now:
        dest_coarse_B2L_maps[0(=4)] = [69 70 71 72 73]
        dest_coarse_B2L_maps[1(=8)] = [74 73 75 76 77]
        */
    // -------------------------------------------------
    // Step 8
    // Send dest_coarse_B2L_maps to root partitions
    // ------------------------------------------------
    // Each fine partition sends to its root the number of coarse neighbors it has, their ids, and the number of boundary nodes for each coarse neighbor
    IVector_h num_bdy_per_coarse_neigh(num_coarse_neighbors);

    for (int i = 0; i < num_coarse_neighbors; i++)
    {
        num_bdy_per_coarse_neigh[i] = dest_coarse_B2L_maps[i].size();
    }

    IVector_h consolidated_coarse_neigh_to_fine_part; //consolidated list of coarse neighbors for the root partition, using fine partition indices
    int num_consolidated_neighbors = 0;
    // std::vector<IVector> consolidated_B2L_maps; //concatenates dest_coarse_B2L_maps received from partitions that are merging into the same root and pointing to the same destination coarse neighbor
    std::vector<IVector> &consolidated_B2L_maps = this->m_consolidated_B2L_maps;
    A.manager->consolidateB2LmapsOnRoot(num_consolidated_neighbors, consolidated_B2L_maps, consolidated_coarse_neigh_to_fine_part, dest_coarse_B2L_maps, coarse_neigh_to_fine_part, num_bdy_per_coarse_neigh, fine_parts_to_consolidate, num_fine_parts_to_consolidate, my_id, my_destination_part, is_root_partition, num_coarse_neighbors, A.manager->getComms());
    //
    // Step 9 - figuring out halo aggregate IDs
    //
    //Now we need to update halo aggregate IDs - this is just a halo exchange on this->m_aggregates between partitions
    //that are being merged together, but we need to send other halos to the root to come up with the halo renumbering
    //TODO: separate transactions, send "real halo" to the root nodes (coarse neighbors) immediately
    //Step 9.1: takes care of synchronizing the aggregate IDs between partitions we are merging together and got consistent halo aggregate IDs for neighbor we are not merging with (which are going to be sent to the root in 9.2)
    A.manager->exchange_halo(this->m_aggregates, 6666);
    /*
        * EXAMPLE 2
        This example is independent from the previous ones.
        Say partition 0 and 1 are merging (into 0) partition 0 is neighbors with 1,2,3 and partition 1 is neighbors with 0,3,4
        Partitions 3 and 4 are merging (into partition 3) and partition 2 is not merging with anyone.
        This example details the renumbering of halo indices on partition 0 and partition 1.
        After the exchange halo, we have:
        this->m_aggregates on partition 0:
        [(fine interior nodes) (fine boundary nodes) (fine halo from part 1) (fine halo from part 2) (fine halo from part 3)]
        [(fine interior nodes) (fine boundary nodes) (13 13 15) (12 15 17) (14 16 18)]
        aggregates on partition 1:
        [(fine interior nodes) (fine boundary nodes) (fine halo from part 0) (fine halo from part 3) (fine halo from part 4)]
        [(fine interior nodes) (fine boundary nodes) (14 16 17) (18 19 19) (15 15 17)]
        indices in  (fine halo from part 0) and (fine halo from part 1) actually contain interior aggregate indices (if they are not connected to partitions 2,3 or 4), because the boundary is disappearing there.
        Indices in halo regions contain remote-local indices.

        This example is used throughout consolidateAndRenumberHalos
        */
    //Step 9.2 - 9.5
    // IVector_h halo_offsets(num_consolidated_neighbors + 1, 0);
    IVector_h &halo_offsets = this->m_consolidated_halo_offsets;
    halo_offsets = IVector_h(num_consolidated_neighbors + 1, 0);
    A.manager->consolidateAndRenumberHalos(this->m_aggregates, A.manager->halo_offsets, halo_offsets, A.manager->neighbors, num_fine_neighbors, consolidated_coarse_neigh_to_fine_part, num_consolidated_neighbors, destination_part, my_destination_part, is_root_partition, fine_parts_to_consolidate, num_fine_parts_to_consolidate, num_parts, my_id, total_rows_in_merged, this->m_num_all_aggregates, A.manager->getComms());

    if (is_root_partition)
    {
        for (int i = 0; i < consolidated_B2L_maps.size(); i++)
        {
            amgx::thrust::sort(consolidated_B2L_maps[i].begin(), consolidated_B2L_maps[i].end());
        }

        this->m_consolidated_neighbors.resize(num_consolidated_neighbors);
        for (int i = 0; i < num_consolidated_neighbors; i++)
        {
            this->m_consolidated_neighbors[i] = fine_part_to_coarse_part[consolidated_coarse_neigh_to_fine_part[i]];
        }
            
        cudaCheckError();
    }
}

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::consolidateCoarseGridMatrix()
{
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();

    int my_id = A.manager->global_id();
        
    IVector_h &destination_part = A.manager->getDestinationPartitions();
    int my_destination_part = A.manager->getMyDestinationPartition();

    // bookkeeping stored in AMG_Level_Base
    std::vector<IVector_h> &vertex_counts = this->m_vertex_counts;
    IVector_h &fine_parts_to_consolidate  = this->m_fine_parts_to_consolidate;

    // bookkeeping stored in either AMG_Level_Base or Acs' DistributedManager
    IVector_h &halo_offsets                     = this->isReuseLevel() ? Ac.manager->getHaloOffsets() : this->m_consolidated_halo_offsets;
    std::vector<IVector> &consolidated_B2L_maps = this->isReuseLevel() ? Ac.manager->getB2Lmaps()     : this->m_consolidated_B2L_maps;

    int num_consolidated_neighbors = this->isRootPartition() ? this->m_consolidated_neighbors.size() : 0;

    if (!this->isRootPartition())
    {
        A.manager->getComms()->send_vector_async(Ac.row_offsets, my_destination_part, 1111);
        A.manager->getComms()->send_vector_async(Ac.col_indices, my_destination_part, 1112);
        A.manager->getComms()->send_vector_async(Ac.values, my_destination_part, 1113);
    }
    else
    {
        int num_fine_parts_to_consolidate = fine_parts_to_consolidate.size();

        int total_num_rows = this->m_num_all_aggregates;
        IVector new_row_offsets(total_num_rows + 1, 0);

        //if diags are inside then we won't be counting those twice when computing halo row length
        if (!Ac.hasProps(DIAG))
        {
            thrust_wrapper::fill<TConfig::memSpace>(new_row_offsets.begin() + halo_offsets[0], new_row_offsets.begin() + halo_offsets[num_consolidated_neighbors], 1);
            cudaCheckError();
        }

        std::vector<IVector> recv_row_offsets(num_fine_parts_to_consolidate);
        std::vector<VecInt_t> num_nz(num_fine_parts_to_consolidate);
        IVector *work_row_offsets;
        std::vector<VecInt_t> index_offset_array(2 * num_fine_parts_to_consolidate + 1);
        int interior_offset = 0;
        int boundary_offset = 0;

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            boundary_offset += vertex_counts[i][0];
        }

        int max_num_nz = 0;

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            int current_part = fine_parts_to_consolidate[i];

            //receive row offsets
            if (current_part != my_id)
            {
                recv_row_offsets[i].resize(total_num_rows + 1);
                A.manager->getComms()->recv_vector(recv_row_offsets[i], current_part, 1111);
                work_row_offsets = &(recv_row_offsets[i]);
                num_nz[i] = (*work_row_offsets)[work_row_offsets->size() - 1];
                max_num_nz = max_num_nz > num_nz[i] ? max_num_nz : num_nz[i];
            }
            else
            {
                work_row_offsets = &(Ac.row_offsets);
                num_nz[i] = Ac.get_num_nz();
            }

            //Get interior row length
            thrust_wrapper::transform<TConfig::memSpace>(work_row_offsets->begin() + interior_offset + 1,
                                work_row_offsets->begin() + interior_offset + vertex_counts[i][0] + 1,
                                work_row_offsets->begin() + interior_offset,
                                new_row_offsets.begin() + interior_offset,
                                amgx::thrust::minus<IndexType>());
            cudaCheckError();
            //Get boundary row length
            thrust_wrapper::transform<TConfig::memSpace>(work_row_offsets->begin() + boundary_offset + 1,
                                work_row_offsets->begin() + boundary_offset + vertex_counts[i][1] + 1,
                                work_row_offsets->begin() + boundary_offset,
                                new_row_offsets.begin() + boundary_offset,
                                amgx::thrust::minus<IndexType>());
            cudaCheckError();
            //Increment halo row length by one for every nonzero that is an edge from the halo into this partition
            int size = halo_offsets[num_consolidated_neighbors] - halo_offsets[0];
            const int block_size = 128;
            const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (size - 1) / block_size + 1);
            set_halo_rowlen <<< num_blocks, block_size>>>(work_row_offsets->raw() + halo_offsets[0], new_row_offsets.raw() + halo_offsets[0], size, Ac.hasProps(DIAG));
            cudaCheckError();
            index_offset_array[i] = interior_offset;
            index_offset_array[num_fine_parts_to_consolidate + i] = boundary_offset;
            interior_offset += vertex_counts[i][0];
            boundary_offset += vertex_counts[i][1];
            index_offset_array[i + 1] = interior_offset;
            index_offset_array[num_fine_parts_to_consolidate + i + 1] = boundary_offset;
        }

        A.manager->setConsolidationArrayOffsets(index_offset_array);
        //Exclusive scan row length array to get row offsets
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(new_row_offsets.begin(), new_row_offsets.end(), new_row_offsets.begin());
        cudaCheckError();
        //Prepare to receive column indices and values
        int num_nz_consolidated = new_row_offsets[new_row_offsets.size() - 1];
        IVector recv_col_indices(max_num_nz);
        IVector new_col_indices(num_nz_consolidated);
        MVector recv_values((max_num_nz + 1 + Ac.hasProps(DIAG) * (halo_offsets[num_consolidated_neighbors] - 1))*Ac.get_block_size());
        MVector new_values((num_nz_consolidated + 1 + Ac.hasProps(DIAG) * (halo_offsets[num_consolidated_neighbors] - 1))*Ac.get_block_size());
        thrust_wrapper::fill<TConfig::memSpace>(new_col_indices.begin() + new_row_offsets[halo_offsets[0]], new_col_indices.end(), -1); //Set all the halo col indices to -1


        if (!Ac.hasProps(DIAG)) { thrust_wrapper::fill<TConfig::memSpace>(new_values.begin() + num_nz_consolidated * Ac.get_block_size(), new_values.end(), types::util<ValueTypeA>::get_zero()); }

        cudaCheckError();
        IVector *work_col_indices;
        MVector *work_values;
        interior_offset = 0;
        boundary_offset = 0;

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            int current_part = fine_parts_to_consolidate[i];
            boundary_offset += vertex_counts[i][0];
        }

        for (int i = 0; i < num_fine_parts_to_consolidate; i++)
        {
            int current_part = fine_parts_to_consolidate[i];

            if (current_part != my_id)
            {
                A.manager->getComms()->recv_vector(recv_col_indices, current_part, 1112, 0, num_nz[i]);
                A.manager->getComms()->recv_vector(recv_values, current_part, 1113, 0, (num_nz[i] + 1 + Ac.hasProps(DIAG) * (halo_offsets[num_consolidated_neighbors] - 1))*Ac.get_block_size());
                work_col_indices = &(recv_col_indices);
                work_row_offsets = &(recv_row_offsets[i]);
                work_values = &(recv_values);
            }
            else
            {
                work_row_offsets = &(Ac.row_offsets);
                work_col_indices = &(Ac.col_indices);
                work_values = &(Ac.values);
            }

            //Put interior rows in place
            amgx::thrust::copy(work_col_indices->begin() + (*work_row_offsets)[interior_offset],
                            work_col_indices->begin() + (*work_row_offsets)[interior_offset + vertex_counts[i][0]],
                            new_col_indices.begin() + new_row_offsets[interior_offset]);
            cudaCheckError();
            amgx::thrust::copy(work_values->begin() + (*work_row_offsets)[interior_offset]*Ac.get_block_size(),
                            work_values->begin() + ((*work_row_offsets)[interior_offset + vertex_counts[i][0]])*Ac.get_block_size(),
                            new_values.begin() + new_row_offsets[interior_offset]*Ac.get_block_size());
            cudaCheckError();
            //Put boundary rows in place
            amgx::thrust::copy(work_col_indices->begin() + (*work_row_offsets)[boundary_offset],
                            work_col_indices->begin() + (*work_row_offsets)[boundary_offset + vertex_counts[i][1]],
                            new_col_indices.begin() + new_row_offsets[boundary_offset]);
            cudaCheckError();
            amgx::thrust::copy(work_values->begin() + (*work_row_offsets)[boundary_offset]*Ac.get_block_size(),
                            work_values->begin() + ((*work_row_offsets)[boundary_offset + vertex_counts[i][1]])*Ac.get_block_size(),
                            new_values.begin() + new_row_offsets[boundary_offset]*Ac.get_block_size());
            cudaCheckError();
            //Process halo rows (merge)
            int size = halo_offsets[num_consolidated_neighbors] - halo_offsets[0];
            const int block_size = 128;
            const int num_blocks = std::min( AMGX_GRID_MAX_SIZE, (size - 1) / block_size + 1);
            //TODO: vectorise this kernel, will be inefficient for larger block sizes
            append_halo_nz <<< num_blocks, block_size>>>(work_row_offsets->raw() + halo_offsets[0],
                    new_row_offsets.raw() + halo_offsets[0],
                    work_col_indices->raw(),
                    new_col_indices.raw(),
                    work_values->raw(),
                    new_values.raw(),
                    size, Ac.hasProps(DIAG), halo_offsets[0], Ac.get_block_size());
            cudaCheckError();

            // Diagonals
            if (Ac.hasProps(DIAG))
            {
                // Diagonal corresponding to interior rows
                amgx::thrust::copy(work_values->begin() + (num_nz[i] + interior_offset)*Ac.get_block_size(),
                                work_values->begin() + (num_nz[i] + interior_offset + vertex_counts[i][0])*Ac.get_block_size(),
                                new_values.begin() + (new_row_offsets[halo_offsets[halo_offsets.size() - 1]] + interior_offset)*Ac.get_block_size());
                // Diagonal corresponding to boundary rows
                amgx::thrust::copy(work_values->begin() + (num_nz[i] + boundary_offset)*Ac.get_block_size(),
                                work_values->begin() + (num_nz[i] + boundary_offset + vertex_counts[i][1])*Ac.get_block_size(),
                                new_values.begin() + (new_row_offsets[halo_offsets[halo_offsets.size() - 1]] + boundary_offset)*Ac.get_block_size());
                cudaCheckError();
            }

            interior_offset += vertex_counts[i][0];
            boundary_offset += vertex_counts[i][1];
        }

        Ac.set_initialized(0);
        Ac.row_offsets = new_row_offsets;
        Ac.col_indices = new_col_indices;
        Ac.values = new_values;
    }

    // A new distributed communicator for coarse levels that only contains active partitions
    // has already been created in consolidatedBookKeeping!

    //
    // Step 12 - finalizing, bookkeping
    //
    if (this->isRootPartition())
    {
        // int my_consolidated_id = fine_part_to_coarse_part[my_id];
        int my_consolidated_id = Ac.manager->getComms()->get_global_id();

        if (!this->isReuseLevel())
        {
             Ac.manager->initializeAfterConsolidation(
                 my_consolidated_id,
                 Ac,
                this->m_consolidated_neighbors,
                this->m_total_interior_rows_in_merged,
                this->m_total_boundary_rows_in_merged,
                this->m_num_all_aggregates,
                this->m_consolidated_halo_offsets,
                this->m_consolidated_B2L_maps,
                1,
                true);

            // this is now stored in Acs DistributedManager
            this->m_consolidated_neighbors.resize(0);
            this->m_consolidated_halo_offsets.resize(0);
            this->m_consolidated_B2L_maps.resize(0);

            Ac.manager->B2L_rings.resize(num_consolidated_neighbors + 1);

            for (int i = 0; i < num_consolidated_neighbors; i++)
            {
                Ac.manager->B2L_rings[i].resize(2);
                Ac.manager->B2L_rings[i][0] = 0;
                Ac.manager->B2L_rings[i][1] = consolidated_B2L_maps[i].size();
            }
        }

        Ac.manager->set_initialized(Ac.row_offsets);
        Ac.manager->getComms()->set_neighbors(num_consolidated_neighbors);
        int new_nnz = Ac.row_offsets[Ac.row_offsets.size() - 1];

        Ac.set_num_nz(new_nnz);
        Ac.set_num_cols(Ac.manager->halo_offsets[Ac.manager->halo_offsets.size() - 1]);
        Ac.set_num_rows(Ac.get_num_cols());

        if (A.hasProps(DIAG)) { Ac.addProps(DIAG); }

        Ac.computeDiagonal();
        Ac.set_initialized(1);
    }
    else
    {
        this->getA().manager->getComms()->send_vector_wait_all(Ac.row_offsets);
        this->getA().manager->getComms()->send_vector_wait_all(Ac.col_indices);
        this->getA().manager->getComms()->send_vector_wait_all(Ac.values);

        Ac.set_initialized(0);
        // set size of Ac to be zero
        Ac.resize(0, 0, 0, 1);
        Ac.set_initialized(1);
    }
}

// -------------------------------------------------------------
// Near-null space helpers
// -------------------------------------------------------------

// Device-side scalar conversion: double -> ValueType.
// For real types (float, double) this is a plain cast.
// For complex types (cuComplex, cuDoubleComplex) the imaginary part is zero.
template <typename ValueType>
static __device__ __inline__ ValueType double_to_valuetype(double v)
{
    return static_cast<ValueType>(v);   // works for float and double
}

template <>
__device__ __inline__ cuComplex double_to_valuetype<cuComplex>(double v)
{
    return make_cuComplex(static_cast<float>(v), 0.f);
}

template <>
__device__ __inline__ cuDoubleComplex double_to_valuetype<cuDoubleComplex>(double v)
{
    return make_cuDoubleComplex(v, 0.);
}

// Kernel: copy-convert double host data to device ValueType vector.
// Each thread handles one element.
template <typename ValueType>
__global__
void copy_double_to_valuetype_kernel(const double * __restrict__ src,
                                     ValueType * __restrict__ dst,
                                     int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        dst[tid] = double_to_valuetype<ValueType>(src[tid]);
}

// setNearNullSpace: upload host double data to device VVector.
// Works for both float and double solver precision.
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::setNearNullSpace(
    int null_dim, int num_rows, const double *data)
{
    m_null_dim = null_dim;
    int total = null_dim * num_rows;

    // Allocate a temporary host double vector and copy to device
    thrust::device_vector<double> d_tmp(data, data + total);

    // Resize the device near-null space vector
    m_near_null_space.resize(total);

    // Launch kernel to convert double -> ValueTypeB
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    copy_double_to_valuetype_kernel<ValueTypeB><<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_tmp.data()),
        m_near_null_space.raw(),
        total);
    cudaCheckError();
}

// ---------------------------------------------------------------
// Kernels for buildTentativeProlongator
// ---------------------------------------------------------------

// Expand node-level aggregates to DOF-level aggregates.
// For DOF row d, dof_aggregates[d] = aggregates[d / block_size].
template <typename IndexType>
__global__
void expand_aggregates_kernel(const IndexType * __restrict__ aggregates,
                              IndexType * __restrict__ dof_aggregates,
                              int num_fine_rows,
                              int block_size)
{
    for (int d = blockDim.x * blockIdx.x + threadIdx.x;
         d < num_fine_rows;
         d += gridDim.x * blockDim.x)
    {
        dof_aggregates[d] = aggregates[d / block_size];
    }
}

// Fill col_indices and values for P_tent (uniform-nnz version).
// Every DOF row i gets null_dim entries: col = agg*null_dim + k, value from P_tent_vals.
// Excluded rows (aggregates[node]==-1, Approach A) get col=0..null_dim-1 and zero values.
template <typename IndexType, typename ValueTypeIn, typename ValueTypeOut>
__global__
void fill_P_tent_csr_kernel(
    int num_fine_rows,
    int null_dim,
    int block_size,
    const IndexType * __restrict__ aggregates,
    const ValueTypeIn * __restrict__ P_tent_vals,
    IndexType * __restrict__ col_indices,
    ValueTypeOut * __restrict__ values)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < num_fine_rows;
         i += gridDim.x * blockDim.x)
    {
        int node = i / block_size;
        int agg  = aggregates[node];
        bool is_excluded = (agg < 0);

        for (int k = 0; k < null_dim; ++k)
        {
            if (is_excluded)
            {
                // Excluded node (Approach A): dummy column index, zero value.
                col_indices[i * null_dim + k] = k;  // valid col to avoid OOB
                values[i * null_dim + k] = ValueTypeOut{};
            }
            else
            {
                col_indices[i * null_dim + k] = agg * null_dim + k;
                ValueTypeIn tmp = P_tent_vals[i * null_dim + k];
                ValueTypeOut *dst = &values[i * null_dim + k];
                if (sizeof(ValueTypeIn) == sizeof(ValueTypeOut))
                    *dst = *reinterpret_cast<ValueTypeOut*>(&tmp);
                else
                {
                    typename types::PODTypes<ValueTypeIn>::type pod_in =
                        *reinterpret_cast<typename types::PODTypes<ValueTypeIn>::type*>(&tmp);
                    typename types::PODTypes<ValueTypeOut>::type pod_out =
                        static_cast<typename types::PODTypes<ValueTypeOut>::type>(pod_in);
                    *dst = *reinterpret_cast<ValueTypeOut*>(&pod_out);
                }
            }
        }
    }
}

// ---------------------------------------------------------------
// buildTentativeProlongator
// ---------------------------------------------------------------

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::buildTentativeProlongator()
{
    fprintf(stderr, "[SA-TENT] level=%d  m_null_dim=%d  m_near_null_space.size=%zu  "
            "m_num_aggregates=%d  A.num_rows=%d  A.block_dimy=%d\n",
            this->getLevelIndex(), m_null_dim, (size_t)m_near_null_space.size(),
            m_num_aggregates, this->getA().get_num_rows(), this->getA().get_block_dimy());
    if (m_null_dim <= 0)
    {
        FatalError("buildTentativeProlongator: null_dim not set", AMGX_ERR_BAD_PARAMETERS);
    }

    if (m_near_null_space.size() == 0)
    {
        FatalError("buildTentativeProlongator: near-null space not set", AMGX_ERR_BAD_PARAMETERS);
    }

    Matrix<TConfig> &A = this->getA();
    int block_size = A.get_block_dimy();
    // In the distributed case, only build P_tent for owned fine rows.
    // The near-null space is only available for owned DOFs, and halo rows
    // of P_tent will be obtained via exchange_halo_rows_P.
    int num_nodes  = (!A.is_matrix_singleGPU() && A.manager != NULL)
                     ? A.manager->halo_offsets[0]
                     : A.get_num_rows();       // block-rows (nodes)
    int num_fine_rows = num_nodes * block_size;  // total DOF rows
    int num_coarse_cols = m_num_aggregates * m_null_dim;
    int nnz_uniform = num_fine_rows * m_null_dim;

    // ----------------------------------------------------------
    // 1. Expand node-level aggregates to DOF-level aggregates.
    // ----------------------------------------------------------
    IVector dof_aggregates(num_fine_rows);
    {
        const int threads = 256;
        const int blocks = std::min(4096, (num_fine_rows + threads - 1) / threads);
        expand_aggregates_kernel<IndexType><<<blocks, threads>>>(
            m_aggregates.raw(),
            dof_aggregates.raw(),
            num_fine_rows,
            block_size);
        cudaCheckError();
    }

    // ----------------------------------------------------------
    // 2. Build aggregate row lists at DOF level (all rows included,
    //    singletons included — they contribute to QR but their
    //    P_tent values will be zeroed after QR).
    // ----------------------------------------------------------
    IVector agg_row_offsets(m_num_aggregates + 1);
    IVector agg_rows(num_fine_rows);

    build_agg_row_lists(num_fine_rows,
                        m_num_aggregates,
                        dof_aggregates.raw(),
                        agg_row_offsets.raw(),
                        agg_rows.raw());

    // ----------------------------------------------------------
    // 3. Run batched QR on the near-null space.
    //    P_tent_vals uses the uniform layout (nnz_uniform entries).
    // ----------------------------------------------------------
    VVector P_tent_vals(nnz_uniform);
    VVector R_out(m_num_aggregates * m_null_dim * m_null_dim);

    batched_qr<ValueTypeB>(m_num_aggregates,
                           m_null_dim,
                           num_fine_rows,
                           agg_row_offsets.raw(),
                           agg_rows.raw(),
                           m_near_null_space.raw(),
                           P_tent_vals.raw(),
                           R_out.raw());

    // ----------------------------------------------------------
    // 4. Build P_tent CSR matrix with uniform nnz (null_dim per row).
    //    Excluded rows (aggregates[node]==-1, Approach A) get
    //    zero values with dummy column indices.
    // ----------------------------------------------------------
    m_P_tent.set_initialized(0);
    m_P_tent.addProps(CSR);
    m_P_tent.resize(num_fine_rows, num_coarse_cols, nnz_uniform, 1, 1, 1);

    // Fill uniform row_offsets: row_offsets[i] = i * null_dim.
    {
        amgx::thrust::transform(
            amgx::thrust::counting_iterator<int>(0),
            amgx::thrust::counting_iterator<int>(num_fine_rows + 1),
            m_P_tent.row_offsets.begin(),
            amgx::thrust::placeholders::_1 * m_null_dim);
        cudaCheckError();
    }

    {
        const int threads = 256;
        const int blocks = std::min(4096, (num_fine_rows + threads - 1) / threads);
        fill_P_tent_csr_kernel<<<blocks, threads>>>(
            num_fine_rows,
            m_null_dim,
            block_size,
            m_aggregates.raw(),
            P_tent_vals.raw(),
            m_P_tent.col_indices.raw(),
            m_P_tent.values.raw());
        cudaCheckError();
    }

    m_P_tent.set_initialized(1);

    // Count excluded rows (agg == -1)
    int n_excluded = 0;
    {
        std::vector<int> h_agg(num_nodes);
        cudaMemcpy(h_agg.data(), m_aggregates.raw(),
                   num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < num_nodes; i++)
            if (h_agg[i] < 0) n_excluded++;
    }

    fprintf(stderr, "[SA-PTENT] num_fine_rows=%d nnz=%d (uniform, %d excluded nodes with zero P_tent rows)\n",
            num_fine_rows, nnz_uniform, n_excluded);

    // ----------------------------------------------------------
    // 5. Store coarse near-null space (R factors from QR)
    //    Reshape from per-aggregate column-major R to global
    //    column-major B_c layout for the next coarser level.
    // ----------------------------------------------------------
    int total_R = m_num_aggregates * m_null_dim * m_null_dim;
    m_coarse_near_null_space.resize(total_R);
    {
        const int threads = 256;
        const int blocks = std::min(4096, (total_R + threads - 1) / threads);
        reshape_R_to_coarse_B_kernel<ValueTypeB><<<blocks, threads>>>(
            m_num_aggregates,
            m_null_dim,
            R_out.raw(),
            m_coarse_near_null_space.raw());
        cudaCheckError();
    }
}

// ---------------------------------------------------------------
// Kernel for estimateSADampingFactor: apply point D^{-1} to a vector
// ---------------------------------------------------------------

template <typename IndexType, typename MatValueType, typename VecValueType>
__global__
void apply_point_dinv_kernel(
    int num_dofs,
    int block_size,
    const IndexType * __restrict__ diag_offsets,  // A.diag, length num_rows
    const MatValueType * __restrict__ A_values,    // A.values (MatPrec)
    VecValueType * __restrict__ w)                 // vector to scale by D^{-1} (VecPrec)
{
    for (int dof = blockDim.x * blockIdx.x + threadIdx.x; dof < num_dofs; dof += gridDim.x * blockDim.x)
    {
        int node = dof / block_size;
        int local = dof % block_size;
        IndexType diag_nz = diag_offsets[node];
        MatValueType diag_val = A_values[diag_nz * block_size * block_size + local * block_size + local];
        if (!types::util<MatValueType>::is_zero(diag_val))
        {
            w[dof] = w[dof] / diag_val;
        }
    }
}

// ---------------------------------------------------------------
// estimateSADampingFactor
// ---------------------------------------------------------------
//
// Returns the SA damping factor omega = (4/3) / rho(D^{-1}A).
// Uses power iteration to estimate rho(D^{-1}A): the spectral radius
// of the Jacobi-preconditioned operator.  Starting from a vector of
// ones, each iteration applies A then D^{-1} and tracks the Rayleigh
// quotient.  Convergence is typically reached in 10-20 iterations.
//
// Note: PETSc GAMG uses omega = 4/(3*rho) by default (same formula).
// The constant 4/3 ≈ 1.333 is the optimal damping for SA prolongator
// smoothing on model problems (Baker et al., 2011).

template <class T_Config>
typename T_Config::VecPrec
Aggregation_AMG_Level_Base<T_Config>::estimateSADampingFactor(int max_iter)
{
    Matrix<TConfig> &A = this->getA();
    int block_size    = A.get_block_dimy();
    int num_rows      = A.get_num_rows();
    int num_dofs      = num_rows * block_size;

    // Fallback: if matrix is empty or diagonal array not allocated, return 0.7 (= 1.4/2).
    if (num_dofs == 0 || A.diag.size() == 0)
    {
        ValueTypeB omega_fb = types::util<ValueTypeB>::get_one();
        typedef typename types::PODTypes<ValueTypeB>::type PodB_fb;
        omega_fb = omega_fb * static_cast<PodB_fb>(0.7);
        return omega_fb;
    }

    // Clamp iteration count.
    if (max_iter <= 0) max_iter = 20;

    // Allocate work vectors (DOF-level, block_dimy set for SpMV).
    VVector w(num_dofs, types::util<ValueTypeB>::get_one());
    VVector v(num_dofs, types::util<ValueTypeB>::get_zero());
    w.set_block_dimy(block_size);
    w.set_block_dimx(1);
    v.set_block_dimy(block_size);
    v.set_block_dimx(1);

    const int threads = 256;
    const int blocks  = std::min(4096, (num_dofs + threads - 1) / threads);

    typedef typename types::PODTypes<ValueTypeB>::type PodB;
    ValueTypeB lambda = types::util<ValueTypeB>::get_one();

    for (int iter = 0; iter < max_iter; ++iter)
    {
        // v = A * w  (SpMV, owned rows only)
        thrust_wrapper::fill<TConfig::memSpace>(v.begin(), v.end(),
                                                types::util<ValueTypeB>::get_zero());
        multiply(A, w, v, OWNED);

        // v = D^{-1} * v  (point diagonal scaling)
        apply_point_dinv_kernel<<<blocks, threads>>>(
            num_dofs, block_size,
            A.diag.raw(),
            A.values.raw(),
            v.raw());
        cudaCheckError();

        // Rayleigh quotient: lambda = <w, v> / <w, w>
        ValueTypeB wv = amgx::thrust::inner_product(
            w.begin(), w.begin() + num_dofs,
            v.begin(), types::util<ValueTypeB>::get_zero());
        ValueTypeB ww = amgx::thrust::inner_product(
            w.begin(), w.begin() + num_dofs,
            w.begin(), types::util<ValueTypeB>::get_zero());
        cudaCheckError();

        if (!types::util<ValueTypeB>::is_zero(ww))
            lambda = wv / ww;

        // Normalize: w = v / ||v||
        ValueTypeB vv = amgx::thrust::inner_product(
            v.begin(), v.begin() + num_dofs,
            v.begin(), types::util<ValueTypeB>::get_zero());
        cudaCheckError();
        PodB norm_v = std::sqrt(static_cast<PodB>(types::util<ValueTypeB>::abs(vv)));
        if (norm_v > (PodB)0.0)
        {
            ValueTypeB inv_norm = types::util<ValueTypeB>::get_one();
            // Scale inv_norm by 1/norm_v using POD arithmetic.
            inv_norm = inv_norm * static_cast<PodB>(1.0 / norm_v);
            amgx::thrust::transform(v.begin(), v.begin() + num_dofs,
                                    w.begin(),
                                    amgx::thrust::placeholders::_1 * inv_norm);
            cudaCheckError();
        }
    }

    // omega = (4/3) / rho(D^{-1}A)  -- classic SA prolongator smoothing factor
    // (Baker et al., 2011).  PETSc GAMG uses the same formula with the same
    // constant, so both solvers produce identical omega given the same rho.
    PodB lambda_pod = static_cast<PodB>(types::util<ValueTypeB>::abs(lambda));
    ValueTypeB omega = types::util<ValueTypeB>::get_one();
    static const PodB omega_num = static_cast<PodB>(4.0 / 3.0);
    if (lambda_pod > (PodB)0.0)
    {
        PodB omega_pod = omega_num / lambda_pod;
        omega = omega * omega_pod;
    }

    return omega;
}

// ---------------------------------------------------------------
// Kernel: expand block-CSR A (block_size > 1) to scalar CSR A_scalar.
//
// Each block-row i of A expands to block_size scalar rows.
// Each block-nonzero (block column node_j) expands to block_size scalar
// columns.  Scalar row (i*bs + li) has (row_offsets[i+1]-row_offsets[i])*bs
// entries.  A_scalar_row_offsets must already be the exclusive-scan result
// (computed by build_SA_smoother_block_rowlen_kernel + exclusive_scan).
//
// Scalar value at position (i*bs+li, node_j*bs+lj) is
//   A_values[bnz * bs * bs + li * bs + lj]
// ---------------------------------------------------------------
template <typename IndexType, typename ValueType>
__global__
void expand_block_to_scalar_kernel(
    int num_block_rows,
    int block_size,
    const IndexType * __restrict__ A_row_offsets,
    const IndexType * __restrict__ A_col_indices,
    const ValueType * __restrict__ A_values,
    const IndexType * __restrict__ As_row_offsets,   // exclusive-scan result
    IndexType       * __restrict__ As_col_indices,
    ValueType       * __restrict__ As_values)
{
    int num_dofs = num_block_rows * block_size;
    for (int dof = blockDim.x * blockIdx.x + threadIdx.x;
         dof < num_dofs;
         dof += gridDim.x * blockDim.x)
    {
        int node = dof / block_size;
        int li   = dof % block_size;

        int s_pos   = As_row_offsets[dof];
        int a_start = A_row_offsets[node];
        int a_end   = A_row_offsets[node + 1];

        for (int bnz = a_start; bnz < a_end; ++bnz)
        {
            int node_j = A_col_indices[bnz];
            for (int lj = 0; lj < block_size; ++lj)
            {
                As_col_indices[s_pos] = node_j * block_size + lj;
                As_values[s_pos]      = A_values[bnz * block_size * block_size
                                                  + li * block_size + lj];
                ++s_pos;
            }
        }
    }
}

// Wrapper to avoid CUDA template-argument-in-<<<>>> parse issue.
// CUDA's <<<>>> syntax cannot handle comma-separated template args directly.
template <typename IndexType, typename ValueType>
void launch_expand_block_to_scalar(
    int num_block_rows, int block_size, int num_dofs,
    const IndexType *A_row_offsets, const IndexType *A_col_indices,
    const ValueType *A_values,
    const IndexType *As_row_offsets,
    IndexType *As_col_indices, ValueType *As_values)
{
    typedef void (*KernPtr)(int, int,
                            const IndexType*, const IndexType*, const ValueType*,
                            const IndexType*, IndexType*, ValueType*);
    KernPtr kptr = expand_block_to_scalar_kernel<IndexType, ValueType>;
    const int threads = 256;
    const int blocks  = std::min(4096, (num_dofs + threads - 1) / threads);
    kptr<<<blocks, threads>>>(
        num_block_rows, block_size,
        A_row_offsets, A_col_indices, A_values,
        As_row_offsets, As_col_indices, As_values);
}

// ---------------------------------------------------------------
// Kernel: build S = I - omega * D^{-1} * A  (scalar CSR, block_size==1)
// ---------------------------------------------------------------
template <typename IndexType, typename ValueType, typename PodType>
__global__
void build_SA_smoother_kernel(
    int num_rows,
    PodType omega,
    const IndexType * __restrict__ row_offsets,
    const IndexType * __restrict__ col_indices,
    const ValueType * __restrict__ A_values,
    const IndexType * __restrict__ A_diag,
    ValueType * __restrict__ S_values)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < num_rows;
         i += gridDim.x * blockDim.x)
    {
        // Get diagonal value
        IndexType diag_nz = A_diag[i];
        ValueType d_ii = A_values[diag_nz];
        ValueType inv_d = types::util<ValueType>::is_zero(d_ii)
                          ? types::util<ValueType>::get_zero()
                          : types::util<ValueType>::get_one() / d_ii;

        int row_start = row_offsets[i];
        int row_end   = row_offsets[i + 1];

        // Convert omega (PodType = real scalar) to ValueType for mixed arithmetic
        ValueType omega_v = types::util<ValueType>::get_one();
        omega_v = omega_v * omega;  // works for both real and complex

        for (int nz = row_start; nz < row_end; ++nz)
        {
            int j = col_indices[nz];
            ValueType a_ij = A_values[nz];
            // S[i,j] = delta(i,j) - omega * a_ij / d_ii
            ValueType s_ij = types::util<ValueType>::get_zero() - omega_v * inv_d * a_ij;
            if (j == i)
                s_ij = s_ij + types::util<ValueType>::get_one();
            S_values[nz] = s_ij;
        }
    }
}

// ---------------------------------------------------------------
// Kernel: build scalar S row offsets from block-CSR A (block_size > 1)
//
// Each block-row i of A (with block_size x block_size blocks) expands
// to block_size scalar rows.  Each block-nonzero expands to block_size
// scalar columns.  So scalar row (i*bs + li) has
//   (row_offsets[i+1] - row_offsets[i]) * block_size  entries.
//
// This kernel fills S_row_offsets[0..num_dofs] (exclusive scan input):
//   S_row_offsets[i*bs + li] = (row_offsets[i+1]-row_offsets[i]) * bs
// Then an exclusive scan converts it to actual offsets.
// ---------------------------------------------------------------
template <typename IndexType>
__global__
void build_SA_smoother_block_rowlen_kernel(
    int num_block_rows,
    int block_size,
    const IndexType * __restrict__ A_row_offsets,
    IndexType * __restrict__ S_row_offsets)   // length num_dofs+1, pre-zeroed
{
    int num_dofs = num_block_rows * block_size;
    for (int dof = blockDim.x * blockIdx.x + threadIdx.x;
         dof < num_dofs;
         dof += gridDim.x * blockDim.x)
    {
        int node = dof / block_size;
        int row_len = A_row_offsets[node + 1] - A_row_offsets[node];
        S_row_offsets[dof] = row_len * block_size;
    }
    // Sentinel: last entry set to 0 (will be filled by exclusive_scan)
    if (blockDim.x * blockIdx.x + threadIdx.x == 0)
        S_row_offsets[num_dofs] = 0;
}

// ---------------------------------------------------------------
// Kernel: fill scalar S col_indices and values from block-CSR A
//
// For scalar row dof = node*bs + li:
//   The block-nonzeros of A for block-row `node` are at
//   A_row_offsets[node] .. A_row_offsets[node+1]-1.
//   For each block-nz at position `bnz` (block column `node_j = A_col_indices[bnz]`):
//     scalar columns: node_j*bs + lj  for lj in [0, bs)
//     scalar value:   A_values[bnz*bs*bs + li*bs + lj]
//     S value:        delta(dof, node_j*bs+lj) - omega * inv_d_li * a_ij
//   where inv_d_li = 1 / A_values[A_diag[node]*bs*bs + li*bs + li]
//
// S_row_offsets must already be the exclusive-scan result.
// ---------------------------------------------------------------
template <typename IndexType, typename ValueType, typename PodType>
__global__
void build_SA_smoother_block_kernel(
    int num_block_rows,
    int block_size,
    PodType omega,
    const IndexType * __restrict__ A_row_offsets,
    const IndexType * __restrict__ A_col_indices,
    const ValueType * __restrict__ A_values,
    const IndexType * __restrict__ A_diag,
    const IndexType * __restrict__ S_row_offsets,
    IndexType * __restrict__ S_col_indices,
    ValueType * __restrict__ S_values)
{
    int num_dofs = num_block_rows * block_size;
    ValueType omega_v = types::util<ValueType>::get_one();
    omega_v = omega_v * omega;

    for (int dof = blockDim.x * blockIdx.x + threadIdx.x;
         dof < num_dofs;
         dof += gridDim.x * blockDim.x)
    {
        int node = dof / block_size;
        int li   = dof % block_size;

        // Point diagonal: A[node,node][li,li]
        IndexType diag_bnz = A_diag[node];
        ValueType d_li = A_values[diag_bnz * block_size * block_size + li * block_size + li];
        ValueType inv_d = types::util<ValueType>::is_zero(d_li)
                          ? types::util<ValueType>::get_zero()
                          : types::util<ValueType>::get_one() / d_li;

        int s_pos = S_row_offsets[dof];  // write position in S arrays

        int a_start = A_row_offsets[node];
        int a_end   = A_row_offsets[node + 1];

        for (int bnz = a_start; bnz < a_end; ++bnz)
        {
            int node_j = A_col_indices[bnz];
            for (int lj = 0; lj < block_size; ++lj)
            {
                int col_dof = node_j * block_size + lj;
                ValueType a_ij = A_values[bnz * block_size * block_size + li * block_size + lj];
                ValueType s_ij = types::util<ValueType>::get_zero() - omega_v * inv_d * a_ij;
                if (col_dof == dof)
                    s_ij = s_ij + types::util<ValueType>::get_one();
                S_col_indices[s_pos] = col_dof;
                S_values[s_pos]      = s_ij;
                ++s_pos;
            }
        }
    }
}

// ---------------------------------------------------------------
// smoothProlongator
// ---------------------------------------------------------------
//
// Smooths the tentative prolongator m_P_tent using full SpGEMM:
//   P_smooth = S * P_tent,  where S = I - omega * D^{-1} * A
//
// For block_size == 1: S has the same sparsity as A (scalar CSR).
// For block_size > 1:  S is the scalar expansion of the block-CSR A.
//   Each block-row i expands to block_size scalar rows; each block-nz
//   expands to block_size scalar columns.
//
// The SpGEMM produces P_smooth with a LARGER sparsity pattern than
// P_tent (fine-node neighbours of each aggregate can contribute new
// column entries).  After this call, m_P_tent is replaced wholesale
// by P_smooth — including its new, larger row_offsets and col_indices.

template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::smoothProlongator()
{
    if (m_null_dim <= 0 || m_P_tent.get_num_rows() == 0)
        FatalError("smoothProlongator: P_tent not built", AMGX_ERR_BAD_PARAMETERS);

    Matrix<TConfig> &A = this->getA();
    int block_size     = A.get_block_dimy();
    int num_rows       = A.get_num_rows();   // node-level rows
    int num_dofs       = num_rows * block_size;

    // Estimate damping factor omega = (4/3) / rho(D^{-1}A)
    ValueTypeB omega_b = estimateSADampingFactor();
    typedef typename types::PODTypes<ValueTypeB>::type PodB;
    PodB omega_pod = types::util<ValueTypeB>::abs(omega_b);

    // Store rho(D^{-1}A) for use by the Chebyshev smoother (lambda_mode=4).
    // estimateSADampingFactor returns omega = (4/3) / rho, so rho = (4/3) / omega.
    static const double omega_num_d = 4.0 / 3.0;
    if (omega_pod > (PodB)0.0)
        m_sa_rho = omega_num_d / static_cast<double>(omega_pod);
    else
        m_sa_rho = 0.0;

    // Per-level SA eigen estimate summary (one line per level, always printed).
    fprintf(stderr, "[SA-EIGEN] level=%d  rho(D^{-1}A)=%.6e  omega=(4/3)/rho=%.6e"
            "  dofs=%d  block=%d\n",
            this->getLevelIndex(), m_sa_rho, (double)omega_pod, num_dofs, block_size);

    // ---------------------------------------------------------------
    // Build S = I - omega * D^{-1} * A  as an explicit scalar CSR matrix.
    // ---------------------------------------------------------------
    Matrix<TConfig> S;
    S.set_initialized(0);
    S.addProps(CSR);

    if (block_size == 1)
    {
        // S has the same sparsity as A (scalar CSR, 1x1 blocks)
        int S_nnz = A.get_num_nz();
        S.resize(num_rows, num_rows, S_nnz, 1, 1, 1);

        // Copy A's sparsity pattern to S
        thrust_wrapper::copy<TConfig::memSpace>(
            A.row_offsets.begin(), A.row_offsets.begin() + num_rows + 1,
            S.row_offsets.begin());
        thrust_wrapper::copy<TConfig::memSpace>(
            A.col_indices.begin(), A.col_indices.begin() + S_nnz,
            S.col_indices.begin());
        cudaCheckError();

        // Compute S values: S[i,j] = delta(i,j) - omega * A[i,j] / A[i,i]
        {
            const int threads = 256;
            const int blocks  = std::min(4096, (num_rows + threads - 1) / threads);
            build_SA_smoother_kernel<<<blocks, threads>>>(
                num_rows,
                omega_pod,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                A.diag.raw(),
                S.values.raw());
            cudaCheckError();
        }
    }
    else
    {
        // block_size > 1: expand block-CSR A to scalar CSR S.
        // S is num_dofs x num_dofs with S_nnz = A_block_nnz * block_size^2.
        int A_block_nnz = A.get_num_nz();
        long long S_nnz_ll = (long long)A_block_nnz * block_size * block_size;
        fprintf(stderr, "[SA-SNNZ] level=%d  A_block_nnz=%d  block_size=%d  "
                "S_nnz=%lld  num_dofs=%d\n",
                this->getLevelIndex(), A_block_nnz, block_size,
                S_nnz_ll, num_dofs);
        fflush(stderr);
        if (S_nnz_ll > (long long)INT_MAX)
        {
            fprintf(stderr, "[SA-SNNZ] ERROR: S_nnz overflows int32! "
                    "S_nnz=%lld > INT_MAX=%d\n", S_nnz_ll, INT_MAX);
            fflush(stderr);
            FatalError("smoothProlongator: S_nnz overflows int32", AMGX_ERR_BAD_PARAMETERS);
        }
        int S_nnz = (int)S_nnz_ll;
        fprintf(stderr, "[SA-KERN] level=%d  before S.resize(%d,%d,%d,1,1,1)\n",
                this->getLevelIndex(), num_dofs, num_dofs, S_nnz);
        fflush(stderr);
        S.resize(num_dofs, num_dofs, S_nnz, 1, 1, 1);
        fprintf(stderr, "[SA-KERN] level=%d  after S.resize  S.row_offsets.size=%d\n",
                this->getLevelIndex(), (int)S.row_offsets.size());
        fflush(stderr);

        // Step 1: compute per-scalar-row lengths, then exclusive scan -> S_row_offsets
        {
            const int threads = 256;
            const int blocks  = std::min(4096, (num_dofs + threads - 1) / threads);
            thrust_wrapper::fill<TConfig::memSpace>(
                S.row_offsets.begin(), S.row_offsets.end(),
                (IndexType)0);
            fprintf(stderr, "[SA-KERN] level=%d  after thrust fill  launching rowlen_kernel blocks=%d threads=%d\n",
                    this->getLevelIndex(), blocks, threads);
            fflush(stderr);
            build_SA_smoother_block_rowlen_kernel<IndexType><<<blocks, threads>>>(
                num_rows, block_size,
                A.row_offsets.raw(),
                S.row_offsets.raw());
            cudaError_t err1 = cudaDeviceSynchronize();
            fprintf(stderr, "[SA-KERN] level=%d  rowlen_kernel done: err=%d (%s)\n",
                    this->getLevelIndex(), (int)err1, cudaGetErrorString(err1));
            if (err1 != cudaSuccess)
                FatalError("build_SA_smoother_block_rowlen_kernel failed", AMGX_ERR_CUDA_FAILURE);
        }
        thrust_wrapper::exclusive_scan<TConfig::memSpace>(
            S.row_offsets.begin(), S.row_offsets.end(),
            S.row_offsets.begin());
        {
            cudaError_t err2 = cudaDeviceSynchronize();
            fprintf(stderr, "[SA-KERN] level=%d  exclusive_scan done: err=%d (%s)\n",
                    this->getLevelIndex(), (int)err2, cudaGetErrorString(err2));
            if (err2 != cudaSuccess)
                FatalError("exclusive_scan failed", AMGX_ERR_CUDA_FAILURE);
        }

        // Step 2: fill col_indices and values
        // A.values and S.values are ValueTypeA (MatPrec), so use that type.
        {
            typedef ValueTypeA MatValueType;
            typedef typename types::PODTypes<MatValueType>::type PodA;
            PodA omega_pod_a = static_cast<PodA>(omega_pod);
            const int threads = 256;
            const int blocks  = std::min(4096, (num_dofs + threads - 1) / threads);
            build_SA_smoother_block_kernel<IndexType, MatValueType, PodA><<<blocks, threads>>>(
                num_rows, block_size,
                omega_pod_a,
                A.row_offsets.raw(),
                A.col_indices.raw(),
                A.values.raw(),
                A.diag.raw(),
                S.row_offsets.raw(),
                S.col_indices.raw(),
                S.values.raw());
            cudaError_t err3 = cudaDeviceSynchronize();
            fprintf(stderr, "[SA-KERN] level=%d  block_kernel done: err=%d (%s)\n",
                    this->getLevelIndex(), (int)err3, cudaGetErrorString(err3));
            if (err3 != cudaSuccess)
                FatalError("build_SA_smoother_block_kernel failed", AMGX_ERR_CUDA_FAILURE);
        }

        // Diagnostic: min/max of point diagonal d_li = A[node,node][li,li]
        // used for damping in build_SA_smoother_block_kernel.
        // Near-zero d_li → inv_d blows up → S has huge entries → P_smooth ill-conditioned.
        {
            typedef ValueTypeA MatValueType;
            typedef typename types::PODTypes<MatValueType>::type PodA;
            int A_block_nnz2 = A.get_num_nz();
            // A.values stores block_size*block_size scalars per block-NNZ entry
            int A_scalar_nnz = A_block_nnz2 * block_size * block_size;
            std::vector<MatValueType> A_vals_h(A_scalar_nnz);
            std::vector<int> A_diag_h(num_rows);
            cudaMemcpy(A_vals_h.data(), A.values.raw(),
                       A_scalar_nnz * sizeof(MatValueType), cudaMemcpyDeviceToHost);
            cudaMemcpy(A_diag_h.data(), A.diag.raw(),
                       num_rows * sizeof(int), cudaMemcpyDeviceToHost);
            double min_d = 1e300, max_d = 0.0;
            int n_zero_d = 0, n_neg_d = 0;
            for (int node = 0; node < num_rows; ++node)
            {
                int diag_bnz = A_diag_h[node];
                for (int li = 0; li < block_size; ++li)
                {
                    int idx = diag_bnz * block_size * block_size + li * block_size + li;
                    MatValueType d_raw = A_vals_h[idx];
                    PodA d_signed;
                    memcpy(&d_signed, &d_raw, sizeof(PodA));
                    double d = std::abs((double)d_signed);
                    if ((double)d_signed < 0.0) n_neg_d++;
                    if (d < 1e-14) n_zero_d++;
                    if (d < min_d) min_d = d;
                    if (d > max_d) max_d = d;
                }
            }
            fprintf(stderr, "[SA-DIAG] level=%d  block=%d  num_dofs=%d  "
                    "min_point_diag=%.4e  max_point_diag=%.4e  "
                    "n_near_zero(|d|<1e-14)=%d  n_negative=%d\n",
                    this->getLevelIndex(), block_size, num_dofs,
                    min_d, max_d, n_zero_d, n_neg_d);
            fflush(stderr);
        }
    }

    S.set_initialized(1);

    // ---------------------------------------------------------------
    // Compute P_smooth = S * P_tent via SpGEMM
    // ---------------------------------------------------------------
    Matrix<TConfig> P_smooth;
    CSR_Multiply<TConfig>::csr_multiply(S, m_P_tent, P_smooth, NULL);

    // DEBUG NaN diagnostic: check P_smooth, S, P_tent for NaN/Inf after SpGEMM
    {
        typedef ValueTypeA MVal;
        typedef typename types::PODTypes<MVal>::type PodM;
        auto check_nan_inf = [](const MVal *d_ptr, int count, const char *label) {
            if (count <= 0) return;
            std::vector<MVal> vals_h(count);
            cudaMemcpy(vals_h.data(), d_ptr, count * sizeof(MVal), cudaMemcpyDeviceToHost);
            int nan_cnt = 0, inf_cnt = 0;
            for (int i = 0; i < count; ++i)
            {
                PodM v = static_cast<PodM>(types::util<MVal>::abs(vals_h[i]));
                if (std::isnan((double)v)) nan_cnt++;
                if (std::isinf((double)v)) inf_cnt++;
            }
            fprintf(stderr, "[DEBUG smoothProlongator] %s: count=%d NaN=%d Inf=%d\n",
                    label, count, nan_cnt, inf_cnt);
        };
        check_nan_inf(P_smooth.values.raw(), P_smooth.get_num_nz(), "P_smooth");
        fprintf(stderr, "[DEBUG smoothProlongator] P_smooth: rows=%d cols=%d nnz=%d\n",
                P_smooth.get_num_rows(), P_smooth.get_num_cols(), P_smooth.get_num_nz());
        // Column norms of P_smooth
        {
            int nrows = P_smooth.get_num_rows();
            int ncols = P_smooth.get_num_cols();
            int nnz   = P_smooth.get_num_nz();
            std::vector<MVal> pv(nnz);
            std::vector<int>  pr(nrows + 1), pc(nnz);
            cudaMemcpy(pv.data(), P_smooth.values.raw(),      nnz   * sizeof(MVal), cudaMemcpyDeviceToHost);
            cudaMemcpy(pr.data(), P_smooth.row_offsets.raw(), (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(pc.data(), P_smooth.col_indices.raw(), nnz   * sizeof(int),  cudaMemcpyDeviceToHost);
            std::vector<double> col_norm2(ncols, 0.0);
            double min_val = 1e300, max_val = 0.0;
            for (int i = 0; i < nnz; ++i)
            {
                double v = (double)static_cast<PodM>(types::util<MVal>::abs(pv[i]));
                col_norm2[pc[i]] += v * v;
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
            double min_cn = 1e300, max_cn = 0.0;
            int ps_zero_cols = 0;
            for (int c = 0; c < ncols; ++c)
            {
                double cn = sqrt(col_norm2[c]);
                if (cn < min_cn) min_cn = cn;
                if (cn > max_cn) max_cn = cn;
                if (cn < 1e-14) ps_zero_cols++;
            }
            fprintf(stderr, "[DEBUG smoothProlongator] P_smooth: min_entry=%.4e max_entry=%.4e "
                    "min_col_norm=%.4e max_col_norm=%.4e zero_cols=%d\n",
                    min_val, max_val, min_cn, max_cn, ps_zero_cols);
            if (ps_zero_cols > 0 && ps_zero_cols <= 20)
            {
                for (int c = 0; c < ncols; ++c)
                    if (sqrt(col_norm2[c]) < 1e-14)
                        fprintf(stderr, "[DEBUG smoothProlongator]   P_smooth zero-col %d (agg=%d, k=%d)\n",
                                c, c / m_null_dim, c % m_null_dim);
            }
        }
        // Column norms of P_tent (before smoothing, stored in m_P_tent)
        {
            int nrows = m_P_tent.get_num_rows();
            int ncols = m_P_tent.get_num_cols();
            int nnz   = m_P_tent.get_num_nz();
            std::vector<MVal> pv(nnz);
            std::vector<int>  pt_r(nrows + 1), pt_c(nnz);
            cudaMemcpy(pv.data(), m_P_tent.values.raw(), nnz * sizeof(MVal), cudaMemcpyDeviceToHost);
            cudaMemcpy(pt_r.data(), m_P_tent.row_offsets.raw(), (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(pt_c.data(), m_P_tent.col_indices.raw(), nnz * sizeof(int), cudaMemcpyDeviceToHost);
            double min_val = 1e300, max_val = 0.0;
            std::vector<double> pt_col_norm2(ncols, 0.0);
            for (int row = 0; row < nrows; ++row)
            {
                for (int j = pt_r[row]; j < pt_r[row+1]; ++j)
                {
                    double v = (double)static_cast<PodM>(types::util<MVal>::abs(pv[j]));
                    if (v < min_val) min_val = v;
                    if (v > max_val) max_val = v;
                    pt_col_norm2[pt_c[j]] += v * v;
                }
            }
            double pt_min_cn = 1e300, pt_max_cn = 0.0;
            int pt_zero_cols = 0;
            for (int c = 0; c < ncols; ++c)
            {
                double cn = sqrt(pt_col_norm2[c]);
                if (cn < pt_min_cn) pt_min_cn = cn;
                if (cn > pt_max_cn) pt_max_cn = cn;
                if (cn < 1e-14) pt_zero_cols++;
            }
            fprintf(stderr, "[DEBUG smoothProlongator] P_tent: rows=%d cols=%d nnz=%d "
                    "min_entry=%.4e max_entry=%.4e min_col_norm=%.4e max_col_norm=%.4e zero_cols=%d\n",
                    nrows, ncols, nnz, min_val, max_val, pt_min_cn, pt_max_cn, pt_zero_cols);
            if (pt_zero_cols > 0 && pt_zero_cols <= 20)
            {
                for (int c = 0; c < ncols; ++c)
                    if (sqrt(pt_col_norm2[c]) < 1e-14)
                        fprintf(stderr, "[DEBUG smoothProlongator]   P_tent zero-col %d (agg=%d, k=%d)\n",
                                c, c / m_null_dim, c % m_null_dim);
            }
        }
        check_nan_inf(S.values.raw(), S.get_num_nz(), "S");
        check_nan_inf(m_P_tent.values.raw(), m_P_tent.get_num_nz(), "P_tent(input)");

        // Zero-row diagnostics: count zero rows in P_tent and P_smooth.
        // For Approach A (singletons removed from coarse grid), P_tent has
        // zero-nnz rows for singleton DOFs.  After SpGEMM, P_smooth should
        // have NO zero rows (SA smoother fills them from algebraic neighbors).
        {
            // P_tent zero rows (structural zeros = zero nnz)
            int nrows_pt = m_P_tent.get_num_rows();
            std::vector<int> pt_ro(nrows_pt + 1);
            cudaMemcpy(pt_ro.data(), m_P_tent.row_offsets.raw(),
                       (nrows_pt + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            int pt_zero_rows = 0;
            for (int r = 0; r < nrows_pt; ++r)
                if (pt_ro[r+1] == pt_ro[r]) pt_zero_rows++;
            fprintf(stderr, "[DEBUG smoothProlongator] P_tent zero-nnz rows: %d / %d\n",
                    pt_zero_rows, nrows_pt);

            // P_smooth zero rows (structural zeros = zero nnz)
            int nrows_ps = P_smooth.get_num_rows();
            int nnz_ps   = P_smooth.get_num_nz();
            std::vector<int> ps_ro(nrows_ps + 1);
            std::vector<int> ps_ci(nnz_ps);
            std::vector<MVal> ps_v(nnz_ps);
            cudaMemcpy(ps_ro.data(), P_smooth.row_offsets.raw(),
                       (nrows_ps + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            if (nnz_ps > 0) {
                cudaMemcpy(ps_ci.data(), P_smooth.col_indices.raw(),
                           nnz_ps * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(ps_v.data(), P_smooth.values.raw(),
                           nnz_ps * sizeof(MVal), cudaMemcpyDeviceToHost);
            }
            int ps_zero_rows = 0;
            int ps_zero_rows_printed = 0;
            for (int r = 0; r < nrows_ps; ++r)
            {
                bool row_zero = true;
                for (int j = ps_ro[r]; j < ps_ro[r+1]; ++j)
                {
                    double v = (double)static_cast<PodM>(
                        types::util<MVal>::abs(ps_v[j]));
                    if (v > 1e-300) { row_zero = false; break; }
                }
                if (row_zero)
                {
                    ps_zero_rows++;
                    if (ps_zero_rows_printed < 10)
                    {
                        fprintf(stderr,
                            "[DEBUG smoothProlongator]   P_smooth zero row %d "
                            "(nnz=%d)\n", r, ps_ro[r+1] - ps_ro[r]);
                        ps_zero_rows_printed++;
                    }
                }
            }
            fprintf(stderr, "[DEBUG smoothProlongator] P_smooth zero rows: %d / %d\n",
                    ps_zero_rows, nrows_ps);
        }
    }

    // Replace m_P_tent with the smoothed prolongator
    m_P_tent.swap(P_smooth);
}

// -------------------------------------------------------------
// Explicit instantiations
// -------------------------------------------------------------

#define AMGX_CASE_LINE(CASE) template class Aggregation_AMG_Level<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// Explicit instantiations for setNearNullSpace (base class method)
#define AMGX_CASE_LINE(CASE) \
    template void Aggregation_AMG_Level_Base<TemplateMode<CASE>::Type>::setNearNullSpace( \
        int, int, const double *);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// Explicit instantiations for buildTentativeProlongator (base class method)
#define AMGX_CASE_LINE(CASE) \
    template void Aggregation_AMG_Level_Base<TemplateMode<CASE>::Type>::buildTentativeProlongator();
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// Explicit instantiations for estimateSADampingFactor (base class method)
#define AMGX_CASE_LINE(CASE) \
    template typename TemplateMode<CASE>::Type::VecPrec \
    Aggregation_AMG_Level_Base<TemplateMode<CASE>::Type>::estimateSADampingFactor(int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// -----------------------------------------------------------------------
// createBlockGraph
//
// Compress a scalar (block_dimy=1) Galerkin product Ac into a block matrix
// with block_dim x block_dim blocks, analogous to PETSc's PCGAMGCreateGraph_AGG.
//
// The coarse Ac block size equals the column block size of P, which is
// m_null_dim (the number of near-null vectors provided by the user).
//
// Scalar entry (r, c) maps to block entry (r/block_dim, c/block_dim) at
// local position (r%block_dim, c%block_dim).  Block values are stored
// row-major within each block:
//   values[blk_nnz * block_dim*block_dim + local_row * block_dim + local_col]
//
// Precondition:  Ac.get_block_dimy() == 1,
//                Ac.get_num_rows() % block_dim == 0
// Postcondition: Ac.get_block_dimy() == block_dim,
//                Ac.get_num_rows()   == old_num_rows / block_dim
// -----------------------------------------------------------------------
template <class T_Config>
void Aggregation_AMG_Level_Base<T_Config>::createBlockGraph(
    Matrix<TConfig> &Ac, int block_dim)
{
    typedef ValueTypeA MVal;
    const int nd       = block_dim;
    const int nd2      = nd * nd;
    int As_rows        = Ac.get_num_rows();   // = num_aggs * nd
    int As_nnz         = Ac.get_num_nz();
    int num_aggs_local = As_rows / nd;        // = num_aggs

    // Download scalar Ac from device
    std::vector<MVal> As_vals(As_nnz);
    std::vector<int>  As_row(As_rows + 1), As_col(As_nnz);
    cudaMemcpy(As_vals.data(), Ac.values.raw(),
               As_nnz * sizeof(MVal), cudaMemcpyDeviceToHost);
    cudaMemcpy(As_row.data(), Ac.row_offsets.raw(),
               (As_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(As_col.data(), Ac.col_indices.raw(),
               As_nnz * sizeof(int), cudaMemcpyDeviceToHost);

    // Pass 1: collect unique block columns per block row using sorted sets
    std::vector<std::set<int>> block_cols(num_aggs_local);
    for (int sr = 0; sr < As_rows; ++sr)
    {
        int bi = sr / nd;
        for (int j = As_row[sr]; j < As_row[sr + 1]; ++j)
            block_cols[bi].insert(As_col[j] / nd);
    }

    // Build block row offsets
    std::vector<int> Ab_row(num_aggs_local + 1, 0);
    for (int bi = 0; bi < num_aggs_local; ++bi)
        Ab_row[bi + 1] = Ab_row[bi] + (int)block_cols[bi].size();
    int Ab_nnz = Ab_row[num_aggs_local];

    // Build block column indices and zero-initialise block values
    std::vector<int>  Ab_col(Ab_nnz);
    std::vector<MVal> Ab_vals(Ab_nnz * nd2, types::util<MVal>::get_zero());
    for (int bi = 0; bi < num_aggs_local; ++bi)
    {
        int pos = Ab_row[bi];
        for (int bj : block_cols[bi])
            Ab_col[pos++] = bj;
    }

    // Pass 2: scatter scalar values into block values (row-major layout)
    for (int sr = 0; sr < As_rows; ++sr)
    {
        int bi = sr / nd, li = sr % nd;
        for (int j = As_row[sr]; j < As_row[sr + 1]; ++j)
        {
            int bj = As_col[j] / nd, lj = As_col[j] % nd;
            // Binary search for bj in Ab_col[Ab_row[bi]..Ab_row[bi+1])
            int lo = Ab_row[bi], hi = Ab_row[bi + 1] - 1, blk_pos = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) / 2;
                if      (Ab_col[mid] == bj) { blk_pos = mid; break; }
                else if (Ab_col[mid] <  bj)   lo = mid + 1;
                else                           hi = mid - 1;
            }
            Ab_vals[blk_pos * nd2 + li * nd + lj] = As_vals[j];
        }
    }

    // Rebuild Ac as a block matrix on the device
    Ac.set_initialized(0);
    Ac.resize(num_aggs_local, num_aggs_local, Ab_nnz, nd, nd, 0);
    cudaMemcpy(Ac.row_offsets.raw(), Ab_row.data(),
               (num_aggs_local + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Ac.col_indices.raw(), Ab_col.data(),
               Ab_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Ac.values.raw(), Ab_vals.data(),
               Ab_nnz * nd2 * sizeof(MVal), cudaMemcpyHostToDevice);

    // Rebuild diagonal index array
    Ac.diag.resize(num_aggs_local);
    {
        std::vector<int> diag_h(num_aggs_local, -1);
        for (int bi = 0; bi < num_aggs_local; ++bi)
            for (int j = Ab_row[bi]; j < Ab_row[bi + 1]; ++j)
                if (Ab_col[j] == bi) { diag_h[bi] = j; break; }
        cudaMemcpy(Ac.diag.raw(), diag_h.data(),
                   num_aggs_local * sizeof(int), cudaMemcpyHostToDevice);
    }
    Ac.set_initialized(1);

    fprintf(stderr,
            "[SA-BLOCK] createBlockGraph: scalar Ac (%d×%d, nnz=%d) → "
            "block-%d Ac (%d×%d, block_nnz=%d)\n",
            As_rows, As_rows, As_nnz,
            nd, num_aggs_local, num_aggs_local, Ab_nnz);
}

// Explicit instantiations for smoothProlongator (base class method)
#define AMGX_CASE_LINE(CASE) \
    template void Aggregation_AMG_Level_Base<TemplateMode<CASE>::Type>::smoothProlongator();
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

// Explicit instantiations for createBlockGraph (base class method)
#define AMGX_CASE_LINE(CASE) \
    template void Aggregation_AMG_Level_Base<TemplateMode<CASE>::Type>::createBlockGraph( \
        Matrix<TemplateMode<CASE>::Type> &, int);
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
}

}
