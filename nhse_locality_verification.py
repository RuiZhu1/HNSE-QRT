"""
nhse_locality_verification.py

Numerical verification of the LOCALITY claims that are actually well-defined
for the native single-particle (direct-sum) Hilbert space of the NHSE channel
QRT paper -- Lemma 1 (quasi-locality of K, U_H, and the Julia defect operators)
and Proposition 23 (the transition-amplitude lower bound on A_diamond).

WHY THIS SCRIPT EXISTS / WHAT IT DELIBERATELY DOES NOT DO
-----------------------------------------------------------------------------
The previous script's "windowed convergence" figure was mislabeled: it only
computed the channel asymmetry of an ISOLATED chain of length w, never a true
partial trace of a larger bulk chain restricted to a window. That is a
legitimate finite-size scaling check but it is NOT a test of the paper's
buffered local-channel reduction (Definitions 45-48, Assumption 3, Theorem 15
in Sec. XVII).

A genuine implementation of Definitions 45-48 requires an actual partial
trace Tr_{Vbar}[rho_V (x) sigma_Vbar], which in turn requires a TENSOR-PRODUCT
decomposition H = H_V (x) H_Vbar of the full Hilbert space. The paper is
explicit (Sec. XVII, Definition 45, Remark 23) that the native single-particle
Hilbert space used everywhere else in the paper is a DIRECT SUM over sites,
H_L = (+)_n C|n>, NOT a tensor product -- so "trace out the sites outside V"
has no canonical meaning here. A tensor-product realization would have to be
supplied independently, e.g. by second-quantizing the single particle into a
many-mode Fock space (H_V (x) H_Vbar with dim H_V = 2^ell for hard-core
bosons/fermions on ell sites) and lifting the single-particle Julia channel to
a Gaussian channel on that Fock space. That is a materially bigger undertaking
(fermionic/bosonic Gaussian-channel formalism) than a same-day numpy patch,
and it involves a real modeling choice that isn't dictated by the paper.

So: rather than silently bolt on an arbitrary tensor-product convention and
call it "the" buffered reduction (which would repeat the original mislabeling
problem in a different place), this script instead tests the two locality
statements that ARE fully well-defined on the native direct-sum space:

  (1) Lemma 1: the Kraus operator K_{H,t}, the propagator U_H(t), and the
      Julia defect operators D_K, D_{K^dagger} are exponentially quasi-local,
      i.e. |P_n A P_m| <= C exp(-mu |n-m|) for fixed t, uniformly in system
      size L.

  (2) Proposition 23: the transition-amplitude lower bound
        A_diamond(N_{H,t}) >= (1/2) max_m sum_n | |K(n,m)|^2 - |K(m,n)|^2 |
      is evaluated directly (no partial trace needed) and its L-dependence is
      tracked to check thermodynamic persistence (Corollary 7 / Prop. 16
      territory) for the Hatano-Nelson family.

If you want the actual Fock-space buffered-reduction calculation, that is a
separate, larger piece of work with a real modeling choice to make first --
flag it and we can scope it properly rather than fake it here.

Requires: numpy, scipy, matplotlib.
Run with: python3 nhse_locality_verification.py
"""

import numpy as np
from scipy.linalg import expm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Basic building blocks (same conventions as the original script)
# ----------------------------------------------------------------------

def sqrtm_psd(A):
    """Positive-semidefinite matrix square root via eigendecomposition."""
    w, v = np.linalg.eigh((A + A.conj().T) / 2)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.conj().T


def U_from_H(H, t):
    """Non-Hermitian propagator U_H(t) = exp(-i H t)."""
    return expm(-1j * H * t)


def hn_chain(L, g, t0=1.0):
    """Open-boundary Hatano-Nelson chain Hamiltonian of length L."""
    H = np.zeros((L, L), dtype=complex)
    for n in range(L - 1):
        H[n, n + 1] = t0 * np.exp(g)
        H[n + 1, n] = t0 * np.exp(-g)
    return H


def julia_pieces(H, t, kappa):
    """
    Return K, D_K, D_{K^dagger} for the sub-normalized contraction
    K = U_H(t) / sqrt(kappa), per Definition 3 / Proposition 1.
    """
    n = H.shape[0]
    K = U_from_H(H, t) / np.sqrt(kappa)
    Kd = K.conj().T
    DK = sqrtm_psd(np.eye(n) - Kd @ K)
    DKd = sqrtm_psd(np.eye(n) - K @ Kd)
    return K, DK, DKd


# ----------------------------------------------------------------------
# 2. Lemma 1: exponential quasi-locality of K, U_H(t), D_K, D_{K^dagger}
# ----------------------------------------------------------------------

def offdiag_decay_profile(M):
    """
    For a matrix M on a 1D chain, return (r, envelope(r)) where
    envelope(r) = max_n |M[n, n+r]| combined with max_n |M[n+r, n]|,
    i.e. the largest matrix element at spatial separation r, for
    r = 0 .. L-1. This is the direct numerical analogue of the bound
    ||P_n A P_m|| <= C exp(-mu |n-m|) in Lemma 1.
    """
    L = M.shape[0]
    profile = np.zeros(L)
    for r in range(L):
        vals = [abs(M[n, n + r]) for n in range(L - r)] + \
               [abs(M[n + r, n]) for n in range(L - r)]
        profile[r] = max(vals) if vals else 0.0
    return np.arange(L), profile


def fit_decay_rate(r, profile, rmin=1, rmax=None):
    """Fit log(profile) ~ -mu * r + const over r in [rmin, rmax]."""
    if rmax is None:
        rmax = len(r) - 1
    mask = (r >= rmin) & (r <= rmax) & (profile > 1e-14)
    if mask.sum() < 2:
        return float("nan")
    A = np.vstack([r[mask], np.ones(mask.sum())]).T
    slope, _ = np.linalg.lstsq(A, np.log(profile[mask]), rcond=None)[0]
    return -slope


def run_lemma1_check(g=0.5, t_op=0.35, t0=1.0, L=60):
    print("=" * 70)
    print("Lemma 1 check: exponential quasi-locality of K, U_H(t), D_K, D_K^dag")
    print("=" * 70)

    H = hn_chain(L, g, t0)
    kappa = (1.05 * np.linalg.norm(U_from_H(H, t_op), 2)) ** 2
    print(f"L = {L}, kappa = {kappa:.6f}\n")

    U = U_from_H(H, t_op)
    K, DK, DKd = julia_pieces(H, t_op, kappa)

    results = {}
    print(f"{'operator':>18} {'fitted mu':>12}")
    for name, M in [("U_H(t)", U), ("K = U/sqrt(kappa)", K),
                    ("D_K", DK), ("D_K^dagger", DKd)]:
        r, profile = offdiag_decay_profile(M)
        mu = fit_decay_rate(r, profile, rmin=2, rmax=min(25, L // 2))
        mu_long = fit_decay_rate(r, profile, rmin=2, rmax=48)
        results[name] = (r, profile, mu)
        print(f"{name:>18} {mu:12.4f}   (rmax=48: {mu_long:12.4f})")
    print()
    return results


def plot_lemma1(results, outbase="fig_lemma1_locality"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, (r, profile, mu) in results.items():
        mask = profile > 1e-14
        ax.semilogy(r[mask], profile[mask], 'o-', ms=3,
                    label=fr"{name} ($\mu\approx{mu:.2f}$)")
    ax.set_xlabel(r"site distance $|n-m|$")
    ax.set_ylabel(r"$\max |M_{n,n\pm r}|$")
    ax.set_title("Lemma 1: quasi-locality of K, U, D_K", fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outbase + ".pdf")
    plt.savefig(outbase + ".png", dpi=150)
    print(f"Saved {outbase}.pdf/.png\n")


# ----------------------------------------------------------------------
# 3. Proposition 23 lower bound and its L-dependence
#    (Corollary 7 / Prop. 16 thermodynamic-persistence territory)
# ----------------------------------------------------------------------

def prop23_lower_bound(K):
    """
    A_diamond(N_H,t) >= (1/2) max_m sum_n | |K(n,m)|^2 - |K(m,n)|^2 |
    -- Proposition 23 of the paper. K is the sub-normalized propagator,
    with K[n, m] interpreted as the amplitude from site m to site n.
    """
    P = np.abs(K) ** 2          # P[n, m] = |K(n,m)|^2
    asym = np.abs(P - P.T)      # asym[n, m] = | |K(n,m)|^2 - |K(m,n)|^2 |
    return 0.5 * np.max(asym.sum(axis=0))   # sum over n, then max over m


def run_prop23_scaling(g=0.5, t_op=0.35, t0=1.0,
                        lengths=(4, 8, 12, 20, 30, 45, 60, 90, 120)):
    print("=" * 70)
    print("Proposition 23 lower bound: thermodynamic persistence with L")
    print("=" * 70)
    # One L-independent normalization for the fixed-kappa comparison.
    # ||U_L(t)|| increases with L and saturates, so L_ref >= max(lengths)
    # guarantees K stays a strict contraction for every tested L.
    L_ref = max(max(lengths), 200)
    kappa_fixed = (1.05 * np.linalg.norm(
        U_from_H(hn_chain(L_ref, g, t0), t_op), 2)) ** 2
    print(f"fixed kappa (from L_ref={L_ref}) = {kappa_fixed:.6f}")
    print(f"{'L':>5} {'kappa_L':>10} {'bound(tight)':>14} "
          f"{'kappa_L*bound':>14} {'bound(fixed)':>14}")

    bounds, bounds_fixed = [], []
    for L in lengths:
        H = hn_chain(L, g, t0)
        kappa = (1.05 * np.linalg.norm(U_from_H(H, t_op), 2)) ** 2
        K, _, _ = julia_pieces(H, t_op, kappa)
        bound = prop23_lower_bound(K)
        K_fix, _, _ = julia_pieces(H, t_op, kappa_fixed)
        bound_fix = prop23_lower_bound(K_fix)
        bounds.append(bound)
        bounds_fixed.append(bound_fix)
        print(f"{L:5d} {kappa:10.4f} {bound:14.6f} "
              f"{kappa * bound:14.6f} {bound_fix:14.6f}")
    print()
    print("Note: kappa_L*bound is the un-normalized numerator. If it is constant")
    print("in L while bound(tight) drifts, the drift is a normalization effect.")
    print()
    return np.array(lengths), np.array(bounds), np.array(bounds_fixed)


def plot_prop23(lengths, bounds, bounds_fixed, outbase="fig_prop23_persistence"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(lengths, bounds, 'o-', color='#1f4e79',
            label=r'tight $\kappa_L$ (varies with $L$)')
    ax.plot(lengths, bounds_fixed, 's--', color='#c0504d',
            label=r'fixed $\kappa$ ($L$-independent)')
    ax.set_xlabel("chain length L (sites)")
    ax.set_ylabel(r"Prop. 23 bound on $A_\diamond$")
    ax.set_title("Prop. 23 bound vs L", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outbase + ".pdf")
    plt.savefig(outbase + ".png", dpi=150)
    print(f"Saved {outbase}.pdf/.png\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    results = run_lemma1_check()
    plot_lemma1(results)

    lengths, bounds, bounds_fixed = run_prop23_scaling()
    plot_prop23(lengths, bounds, bounds_fixed)

    print("=" * 70)
    print("NOTE: this script deliberately does NOT attempt Definitions 45-48 /")
    print("Assumption 3 / Theorem 15 (buffered local-channel reduction), because")
    print("that requires a tensor-product Hilbert-space realization that is not")
    print("canonically available for the native single-particle direct-sum")
    print("space used here. See the module docstring for details.")
    print("=" * 70)
