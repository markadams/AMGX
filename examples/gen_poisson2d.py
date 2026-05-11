#!/usr/bin/env python3
"""Generate 2D Poisson matrix in MatrixMarket format.

Usage: python3 gen_poisson2d.py [m] [n] [output_file]
  m, n: grid dimensions (default 400 400)
  output_file: output filename (default poisson2d_400.mtx)

Generates the standard 5-point stencil Laplacian on an m×n grid.
Matrix size: m*n rows, approximately 5*m*n nonzeros.
"""
import sys

def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    outfile = sys.argv[3] if len(sys.argv) > 3 else f"poisson2d_{m}.mtx"

    N = m * n
    # Count nonzeros
    nnz = 0
    for i in range(m):
        for j in range(n):
            nnz += 1  # diagonal
            if i > 0: nnz += 1
            if i < m-1: nnz += 1
            if j > 0: nnz += 1
            if j < n-1: nnz += 1

    print(f"Generating {m}x{n} Poisson matrix: {N} rows, {nnz} nonzeros")
    print(f"Output: {outfile}")

    with open(outfile, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {nnz}\n")
        for i in range(m):
            for j in range(n):
                row = i * n + j + 1  # 1-based
                # down neighbor
                if i > 0:
                    f.write(f"{row} {row-n} -1.0\n")
                # left neighbor
                if j > 0:
                    f.write(f"{row} {row-1} -1.0\n")
                # diagonal
                f.write(f"{row} {row} 4.0\n")
                # right neighbor
                if j < n-1:
                    f.write(f"{row} {row+1} -1.0\n")
                # up neighbor
                if i < m-1:
                    f.write(f"{row} {row+n} -1.0\n")

    print("Done.")

if __name__ == "__main__":
    main()
