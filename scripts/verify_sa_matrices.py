#!/usr/bin/env python3
"""
Verify SA smoothing matrices from PETSc and AMGx.
Reads 6 MatrixMarket files, checks A and P_tent match,
computes reference P_smooth, and identifies which solver is correct.
"""
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import norm as spnorm
import sys, os

def load(name):
    """Load a MatrixMarket file and return as CSR matrix."""
    m = mmread(name)
    if hasattr(m, 'tocsr'):
        return m.tocsr()
    return csr_matrix(m)

def rel_diff(A, B, label=""):
    """Compute relative Frobenius norm difference."""
    nA = spnorm(A, 'fro')
    nB = spnorm(B, 'fro')
    diff = spnorm(A - B, 'fro')
    rel = diff / max(nA, nB) if max(nA, nB) > 0 else 0
    print(f"  {label}: ||A||_F={nA:.10e}, ||B||_F={nB:.10e}, ||A-B||_F={diff:.10e}, rel={rel:.4e}")
    return rel

# Directory with .mtx files
d = sys.argv[1] if len(sys.argv) > 1 else "."

print("=" * 70)
print("  SA Smoothing Matrix Verification")
print("=" * 70)

# Load matrices
print("\nLoading matrices...")
A_petsc = load(os.path.join(d, "petsc_A.mtx"))
A_amgx = load(os.path.join(d, "amgx_A.mtx"))
Pt_petsc = load(os.path.join(d, "petsc_P_tent.mtx"))
Pt_amgx = load(os.path.join(d, "amgx_P_tent.mtx"))
Ps_petsc = load(os.path.join(d, "petsc_P_smooth.mtx"))
Ps_amgx = load(os.path.join(d, "amgx_P_smooth.mtx"))

print(f"  A_petsc:  {A_petsc.shape}, nnz={A_petsc.nnz}")
print(f"  A_amgx:   {A_amgx.shape}, nnz={A_amgx.nnz}")
print(f"  Pt_petsc: {Pt_petsc.shape}, nnz={Pt_petsc.nnz}")
print(f"  Pt_amgx:  {Pt_amgx.shape}, nnz={Pt_amgx.nnz}")
print(f"  Ps_petsc: {Ps_petsc.shape}, nnz={Ps_petsc.nnz}")
print(f"  Ps_amgx:  {Ps_amgx.shape}, nnz={Ps_amgx.nnz}")

# Check A matrices match
print("\n--- Check 1: A matrices ---")
rel_a = rel_diff(A_petsc, A_amgx, "A_petsc vs A_amgx")
assert rel_a < 1e-12, f"A matrices differ! rel={rel_a}"
print("  PASS: A matrices match")

# Check P_tent matrices match
print("\n--- Check 2: P_tent matrices ---")
rel_pt = rel_diff(Pt_petsc, Pt_amgx, "Pt_petsc vs Pt_amgx")
if rel_pt > 1e-12:
    print(f"  WARNING: P_tent differs (rel={rel_pt:.4e})")
    print("  This means aggregates differ!")
else:
    print("  PASS: P_tent matrices match")

# Check P_smooth difference
print("\n--- Check 3: P_smooth matrices ---")
rel_ps = rel_diff(Ps_petsc, Ps_amgx, "Ps_petsc vs Ps_amgx")
print(f"  P_smooth nnz: PETSc={Ps_petsc.nnz}, AMGx={Ps_amgx.nnz}, diff={Ps_petsc.nnz - Ps_amgx.nnz}")

# Compute reference P_smooth from scratch
print("\n--- Check 4: Reference P_smooth computation ---")
# Use A_petsc (same as A_amgx) and Pt_petsc (same as Pt_amgx)
A = A_petsc
Pt = Pt_petsc

# Get diagonal
D_vec = np.array(A.diagonal()).flatten()
D_inv = diags(1.0 / D_vec)

# AMGx omega (from 100 power iterations on D^{-1}A)
# PETSc omega: 4/3 / emax where emax=1.969974058297559
omega_petsc = 4.0 / 3.0 / 1.969974058297559
print(f"  omega_petsc = {omega_petsc:.15e}")

# Compute S = I - omega * D^{-1} * A
n = A.shape[0]
I_n = eye(n, format='csr')
S_ref = I_n - omega_petsc * D_inv @ A
print(f"  S_ref: {S_ref.shape}, nnz={S_ref.nnz}")

# Compute P_smooth_ref = S_ref * P_tent
Ps_ref = S_ref @ Pt
print(f"  P_smooth_ref: {Ps_ref.shape}, nnz={Ps_ref.nnz}, ||P_ref||_F={spnorm(Ps_ref, 'fro'):.10e}")

# Compare reference with both implementations
print("\n--- Check 5: Compare implementations against reference ---")
rel_petsc_ref = rel_diff(Ps_petsc, Ps_ref, "Ps_petsc vs Ps_ref (Python)")
rel_amgx_ref = rel_diff(Ps_amgx, Ps_ref, "Ps_amgx vs Ps_ref (Python)")

# Now use AMGx's omega (slightly different due to power iteration)
# AMGx rho ≈ 1.97, omega = 4/3/1.97 ≈ 0.6769
omega_amgx = 4.0 / 3.0 / 1.97  # approximate
S_amgx = I_n - omega_amgx * D_inv @ A
Ps_amgx_ref = S_amgx @ Pt_amgx
print(f"\n  Using omega_amgx={omega_amgx:.15e}:")
rel_amgx_ref2 = rel_diff(Ps_amgx, Ps_amgx_ref, "Ps_amgx vs Ps_ref(omega_amgx)")

# Compute Ac for each
print("\n--- Check 6: Galerkin coarse operators ---")
Ac_petsc = Ps_petsc.T @ A @ Ps_petsc
Ac_amgx = Ps_amgx.T @ A @ Ps_amgx
Ac_ref = Ps_ref.T @ A @ Ps_ref
print(f"  Ac_petsc: {Ac_petsc.shape}, nnz={Ac_petsc.nnz}, ||Ac||_F={spnorm(Ac_petsc, 'fro'):.10e}")
print(f"  Ac_amgx:  {Ac_amgx.shape}, nnz={Ac_amgx.nnz}, ||Ac||_F={spnorm(Ac_amgx, 'fro'):.10e}")
print(f"  Ac_ref:   {Ac_ref.shape}, nnz={Ac_ref.nnz}, ||Ac||_F={spnorm(Ac_ref, 'fro'):.10e}")

rel_diff(Ac_petsc, Ac_ref, "Ac_petsc vs Ac_ref")
rel_diff(Ac_amgx, Ac_ref, "Ac_amgx vs Ac_ref")
rel_diff(Ac_petsc, Ac_amgx, "Ac_petsc vs Ac_amgx")

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  A match:      rel_diff = {rel_a:.4e} {'PASS' if rel_a < 1e-12 else 'FAIL'}")
print(f"  P_tent match: rel_diff = {rel_pt:.4e} {'PASS' if rel_pt < 1e-12 else 'FAIL'}")
print(f"  P_smooth gap: rel_diff = {rel_ps:.4e}")
print(f"  PETSc vs ref: rel_diff = {rel_petsc_ref:.4e} {'PASS' if rel_petsc_ref < 1e-10 else 'CHECK'}")
print(f"  AMGx vs ref:  rel_diff = {rel_amgx_ref:.4e} {'PASS' if rel_amgx_ref < 1e-10 else 'CHECK'}")
if rel_petsc_ref < rel_amgx_ref:
    print("\n  --> PETSc P_smooth is CLOSER to reference (Python scipy)")
else:
    print("\n  --> AMGx P_smooth is CLOSER to reference (Python scipy)")
print("=" * 70)
