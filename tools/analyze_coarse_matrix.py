#!/usr/bin/env python3
"""
analyze_coarse_matrix.py — Eigenvalue analysis of AMGX SA-AMG coarse matrices.

Usage:
    python3 tools/analyze_coarse_matrix.py amgx_Ac_level0.mtx [amgx_Ac_level1.mtx ...]

For each Matrix Market file, this script:
  1. Loads the sparse matrix
  2. Checks symmetry (||A - A^T||_F / ||A||_F)
  3. Computes all eigenvalues (dense eig for small matrices, ARPACK for large)
  4. Reports: min/max eigenvalue, condition number, number of negative/near-zero eigenvalues
  5. Plots eigenvalue spectrum (saved as PNG)

The goal is to determine whether each coarse Ac is:
  - SPD (all eigenvalues > 0)
  - Semi-definite (some eigenvalues ≈ 0, expected with Dirichlet BCs)
  - Indefinite (some eigenvalues < 0, which breaks PCG)

Dependencies: numpy, scipy, matplotlib
    pip install numpy scipy matplotlib
"""

import sys
import os
import numpy as np
import scipy.io
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for Perlmutter
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available, skipping plots")


def analyze_matrix(fname):
    """Load and analyze a single Matrix Market file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {fname}")
    print(f"{'='*70}")

    if not os.path.exists(fname):
        print(f"ERROR: file not found: {fname}")
        return

    # Load sparse matrix
    A = scipy.io.mmread(fname)
    A = scipy.sparse.csr_matrix(A)
    n = A.shape[0]
    nnz = A.nnz
    print(f"  Size: {n} x {n},  nnz = {nnz},  avg nnz/row = {nnz/n:.1f}")

    # --- Symmetry check ---
    AT = A.T
    diff = A - AT
    norm_A = scipy.sparse.linalg.norm(A, 'fro')
    norm_diff = scipy.sparse.linalg.norm(diff, 'fro')
    sym_err = norm_diff / norm_A if norm_A > 0 else 0.0
    print(f"  Symmetry: ||A - A^T||_F / ||A||_F = {sym_err:.4e}  "
          f"({'SYMMETRIC' if sym_err < 1e-10 else 'ASYMMETRIC (expected for SA scalar Ac)'})")

    # --- Diagonal analysis ---
    diag = A.diagonal()
    n_neg_diag = np.sum(diag < 0)
    n_zero_diag = np.sum(np.abs(diag) < 1e-14)
    print(f"  Diagonal: min={diag.min():.4e}  max={diag.max():.4e}  "
          f"n_negative={n_neg_diag}  n_near_zero(|d|<1e-14)={n_zero_diag}")

    # --- Eigenvalue analysis ---
    # Use symmetrized matrix for eigenvalue analysis (A + A^T)/2
    A_sym = (A + AT) * 0.5
    A_dense = A_sym.toarray()

    if n <= 2000:
        # Dense eigenvalue decomposition (exact)
        print(f"  Computing all {n} eigenvalues (dense eig)...")
        eigvals = scipy.linalg.eigvalsh(A_dense)
        eigvals_sorted = np.sort(eigvals)

        n_neg = np.sum(eigvals < -1e-10)
        n_near_zero = np.sum(np.abs(eigvals) < 1e-10)
        n_pos = np.sum(eigvals > 1e-10)
        lam_min = eigvals_sorted[0]
        lam_max = eigvals_sorted[-1]
        cond = lam_max / lam_min if lam_min > 0 else float('inf')

        print(f"  Eigenvalues: min={lam_min:.6e}  max={lam_max:.6e}")
        print(f"  Condition number (lam_max/lam_min): {cond:.4e}")
        print(f"  n_negative (< -1e-10): {n_neg}")
        print(f"  n_near_zero (|λ| < 1e-10): {n_near_zero}")
        print(f"  n_positive (> 1e-10): {n_pos}")

        # SPD verdict
        if n_neg > 0:
            print(f"  *** VERDICT: INDEFINITE — {n_neg} negative eigenvalue(s) ***")
            print(f"  *** This BREAKS PCG symmetry requirement ***")
        elif n_near_zero > 0:
            print(f"  VERDICT: SEMI-DEFINITE — {n_near_zero} near-zero eigenvalue(s)")
            print(f"  (Expected with Dirichlet BCs: null space of rigid body modes)")
        else:
            print(f"  VERDICT: SPD — all eigenvalues positive")

        # Print smallest 10 and largest 5 eigenvalues
        print(f"\n  Smallest 10 eigenvalues:")
        for i, lam in enumerate(eigvals_sorted[:10]):
            print(f"    [{i:3d}]  {lam:+.8e}")
        if n > 10:
            print(f"  Largest 5 eigenvalues:")
            for i, lam in enumerate(eigvals_sorted[-5:]):
                print(f"    [{n-5+i:3d}]  {lam:+.8e}")

        # Plot eigenvalue spectrum
        if HAS_MATPLOTLIB:
            base = os.path.splitext(fname)[0]
            plot_fname = base + '_eigenvalues.png'
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Full spectrum
            ax = axes[0]
            ax.plot(eigvals_sorted, 'b.', markersize=3)
            ax.axhline(0, color='r', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Index')
            ax.set_ylabel('Eigenvalue')
            ax.set_title(f'Full spectrum: {os.path.basename(fname)}\n'
                         f'n={n}, min={lam_min:.3e}, max={lam_max:.3e}')
            ax.grid(True, alpha=0.3)

            # Zoom on small eigenvalues (bottom 20% of range)
            ax2 = axes[1]
            n_show = max(20, n // 5)
            ax2.plot(eigvals_sorted[:n_show], 'r.', markersize=4)
            ax2.axhline(0, color='k', linewidth=1.0, linestyle='--')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Eigenvalue')
            ax2.set_title(f'Smallest {n_show} eigenvalues\n'
                          f'n_neg={n_neg}, n_zero={n_near_zero}')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(plot_fname, dpi=120)
            plt.close()
            print(f"\n  Plot saved: {plot_fname}")

    else:
        # Large matrix: compute only a few extreme eigenvalues via ARPACK
        k = min(20, n - 2)
        print(f"  Matrix too large for dense eig (n={n}). "
              f"Computing {k} extreme eigenvalues via ARPACK...")
        try:
            # Smallest algebraic eigenvalues
            eigvals_small, _ = scipy.sparse.linalg.eigsh(
                A_sym, k=k, which='SA', tol=1e-8, maxiter=10000)
            # Largest algebraic eigenvalues
            eigvals_large, _ = scipy.sparse.linalg.eigsh(
                A_sym, k=k, which='LA', tol=1e-8, maxiter=10000)
            lam_min = eigvals_small.min()
            lam_max = eigvals_large.max()
            n_neg = np.sum(eigvals_small < -1e-10)
            n_near_zero = np.sum(np.abs(eigvals_small) < 1e-10)
            cond = lam_max / lam_min if lam_min > 0 else float('inf')
            print(f"  Eigenvalues (approx): min={lam_min:.6e}  max={lam_max:.6e}")
            print(f"  Condition number (approx): {cond:.4e}")
            print(f"  n_negative (< -1e-10) in smallest {k}: {n_neg}")
            print(f"  n_near_zero (|λ| < 1e-10) in smallest {k}: {n_near_zero}")
            if n_neg > 0:
                print(f"  *** VERDICT: INDEFINITE — {n_neg} negative eigenvalue(s) ***")
            elif n_near_zero > 0:
                print(f"  VERDICT: SEMI-DEFINITE — {n_near_zero} near-zero eigenvalue(s)")
            else:
                print(f"  VERDICT: SPD (based on {k} smallest eigenvalues)")
            print(f"\n  Smallest {k} eigenvalues:")
            for i, lam in enumerate(sorted(eigvals_small)):
                print(f"    [{i:3d}]  {lam:+.8e}")
        except Exception as e:
            print(f"  ARPACK failed: {e}")


def main():
    if len(sys.argv) < 2:
        # Default: look for amgx_Ac_level*.mtx in current directory
        import glob
        files = sorted(glob.glob("amgx_Ac_level*.mtx"))
        if not files:
            print("Usage: python3 analyze_coarse_matrix.py <file1.mtx> [file2.mtx ...]")
            print("  or run in directory containing amgx_Ac_level*.mtx files")
            sys.exit(1)
    else:
        files = sys.argv[1:]

    print(f"AMGX SA-AMG Coarse Matrix Eigenvalue Analysis")
    print(f"Analyzing {len(files)} matrix file(s)...")

    for fname in files:
        analyze_matrix(fname)

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == '__main__':
    main()
