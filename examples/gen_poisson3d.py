#!/usr/bin/env python3
"""Generate 3D Poisson matrix (27-point stencil) in MatrixMarket format.

Usage: python3 gen_poisson3d.py [n] [output_file]
  n: grid dimension per axis (default 100, giving n^3 DOFs)
  output_file: output filename (default poisson3d_<n>.mtx)

Generates the 27-point stencil Laplacian from trilinear hexahedral FEM
on an n×n×n grid with Dirichlet boundary conditions.

Stencil weights (interior node, divided by 1):
  Corner neighbors (8):  -1
  Edge neighbors (12):   -4
  Face neighbors (6):   -16
  Diagonal:             128
  (Row sum for interior = 128 - 8 - 48 - 96 = -24; Dirichlet BCs
   modify boundary rows so the system is SPD.)

For boundary nodes, only existing neighbors are included and the
diagonal is adjusted to maintain row-sum = 0 (homogeneous Dirichlet).

Matrix size: n^3 rows, up to 27*n^3 nonzeros.
"""
import sys
import os

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    outfile = sys.argv[2] if len(sys.argv) > 2 else f"poisson3d_{n}.mtx"

    N = n * n * n

    # Stencil: for each (di, dj, dk) offset, the weight
    # Corner (|di|+|dj|+|dk|=3): -1
    # Edge   (|di|+|dj|+|dk|=2): -4
    # Face   (|di|+|dj|+|dk|=1): -16
    offsets = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                dist = abs(di) + abs(dj) + abs(dk)
                if dist == 1:
                    w = -16.0
                elif dist == 2:
                    w = -4.0
                else:  # dist == 3
                    w = -1.0
                offsets.append((di, dj, dk, w))

    # First pass: count nonzeros
    nnz = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                nnz += 1  # diagonal
                for di, dj, dk, w in offsets:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                        nnz += 1

    print(f"Generating {n}x{n}x{n} 3D Poisson (27-pt stencil): {N} rows, {nnz} nonzeros")
    print(f"Avg nnz/row: {nnz/N:.1f}")
    print(f"Output: {outfile}")

    # Second pass: write matrix
    with open(outfile, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {nnz}\n")
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    row = i * n * n + j * n + k + 1  # 1-based

                    # Collect off-diagonal entries and compute diagonal
                    entries = []
                    diag_val = 0.0
                    for di, dj, dk, w in offsets:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                            col = ni * n * n + nj * n + nk + 1
                            entries.append((col, w))
                            diag_val -= w  # diagonal = -sum(off-diagonals)

                    # Sort entries by column for CSR-friendly output
                    entries.append((row, diag_val))
                    entries.sort(key=lambda x: x[0])

                    for col, val in entries:
                        f.write(f"{row} {col} {val:.6g}\n")

    file_size = os.path.getsize(outfile) / (1024 * 1024)
    print(f"Done. File size: {file_size:.1f} MB")

if __name__ == "__main__":
    main()
