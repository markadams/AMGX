// SPDX-FileCopyrightText: 2024 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <aggregation/selectors/agg_selector.h>

namespace amgx
{
namespace aggregation
{
namespace mis_selector
{

template <class T_Config> class MISSelector;

template <class T_Config>
class MISSelectorBase : public Selector<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename T_Config::MemSpace MemorySpace;
        typedef typename Matrix<T_Config>::IVector IVector;

        MISSelectorBase(AMG_Config &cfg, const std::string &cfg_scope);
        void setAggregates(Matrix<T_Config> &A,
                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates);

    protected:
        virtual void setAggregates_common_sqblocks(const Matrix<T_Config> &A,
                IVector &aggregates, IVector &aggregates_global, int &num_aggregates) = 0;

        int m_mis_k;              // MIS distance parameter (1=standard, 2=aggressive)
        int m_max_iterations;     // Max iterations for MIS convergence loop
        int m_merge_singletons;   // Whether to merge isolated nodes into neighbors
        int m_weight_formula;     // Edge weight formula (0 or 1)
        int m_aggregation_edge_weight_component; // Block component for edge weights
};

// Specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MISSelector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
    : public MISSelectorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TConfig> Matrix_h;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_h::IVector IVector;

        MISSelector(AMG_Config &cfg, const std::string &cfg_scope)
            : MISSelectorBase<TConfig>(cfg, cfg_scope) {}

    private:
        void setAggregates_common_sqblocks(const Matrix_h &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

// Specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class MISSelector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
    : public MISSelectorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix<TConfig> Matrix_d;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef typename Matrix_d::IVector IVector;

        MISSelector(AMG_Config &cfg, const std::string &cfg_scope)
            : MISSelectorBase<TConfig>(cfg, cfg_scope) {}

    private:
        void setAggregates_common_sqblocks(const Matrix_d &A,
                                           IVector &aggregates, IVector &aggregates_global, int &num_aggregates);
};

template<class T_Config>
class MISSelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope)
        {
            return new MISSelector<T_Config>(cfg, cfg_scope);
        }
};

} // namespace mis_selector
} // namespace aggregation
} // namespace amgx
