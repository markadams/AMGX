#!/usr/bin/env python3
"""Patch AMGx aggregation_amg_level.cu to add MatrixMarket output of A, P_tent, P_smooth."""

import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else "/global/homes/m/madams/amgx-sa/src/aggregation/aggregation_amg_level.cu"

with open(filepath, "r") as f:
    content = f.read()

# Helper function to write device CSR matrix to MatrixMarket
helper = r"""
// Helper: write a device CSR matrix to MatrixMarket file
template <class TConfig>
static void write_mtx_file(const char *filename, const Matrix<TConfig> &M)
{
    typedef typename TConfig::IndPrec IndexType;
    typedef typename TConfig::MatPrec ValueTypeA;
    int nrows = M.get_num_rows();
    int ncols = M.get_num_cols();
    int nnz   = M.get_num_nz();
    std::vector<IndexType> h_row(nrows+1), h_col(nnz);
    std::vector<ValueTypeA> h_val(nnz);
    cudaMemcpy(h_row.data(), M.row_offsets.raw(), (nrows+1)*sizeof(IndexType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col.data(), M.col_indices.raw(), nnz*sizeof(IndexType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_val.data(), M.values.raw(), nnz*sizeof(ValueTypeA), cudaMemcpyDeviceToHost);
    FILE *fp = fopen(filename, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return; }
    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %d %d\n", nrows, ncols, nnz);
    for (int i = 0; i < nrows; i++)
        for (int j = h_row[i]; j < h_row[i+1]; j++)
            fprintf(fp, "%d %d %.15e\n", i+1, h_col[j]+1, (double)h_val[j]);
    fclose(fp);
    printf("[SA-DIAG] Wrote %s (%d x %d, nnz=%d)\n", filename, nrows, ncols, nnz);
    fflush(stdout);
}

"""

# Insert helper before the smoothProlongator function
marker = "void Aggregation_AMG_Level_Base<T_Config>::smoothProlongator()"
idx = content.find(marker)
if idx < 0:
    print("ERROR: could not find smoothProlongator")
    sys.exit(1)

# Find the template line before it
templ_marker = "template <class T_Config>"
templ_idx = content.rfind(templ_marker, 0, idx)
content = content[:templ_idx] + helper + content[templ_idx:]
print("Inserted write_mtx_file helper")

# Add A + P_tent dump after the P_tent fflush
ptent_marker = '[SA-DIAG] P_tent BEFORE smooth: ||P_tent||_F'
idx2 = content.find(ptent_marker)
if idx2 < 0:
    print("ERROR: could not find P_tent BEFORE smooth")
    sys.exit(1)

# Find the fflush after it
fflush_idx = content.find("fflush(stdout);", idx2)
end_line = content.find("\n", fflush_idx) + 1
# Find closing brace of the block
close_brace = content.find("}", end_line)
end_block = content.find("\n", close_brace) + 1

dump_a_ptent = """    write_mtx_file<TConfig>("amgx_A.mtx", A);
    write_mtx_file<TConfig>("amgx_P_tent.mtx", m_P_tent);
"""
content = content[:end_block] + dump_a_ptent + content[end_block:]
print("Inserted A + P_tent dump calls")

# Add P_smooth dump after SpGEMM fflush
spgemm_marker = '[SA-DIAG] SpGEMM:'
idx3 = content.find(spgemm_marker)
if idx3 < 0:
    print("ERROR: could not find SpGEMM print")
    sys.exit(1)

fflush_idx2 = content.find("fflush(stdout);", idx3)
end_line2 = content.find("\n", fflush_idx2) + 1

dump_psmooth = """    write_mtx_file<TConfig>("amgx_P_smooth.mtx", P_smooth);
"""
content = content[:end_line2] + dump_psmooth + content[end_line2:]
print("Inserted P_smooth dump call")

with open(filepath, "w") as f:
    f.write(content)

print("Done patching AMGx")
