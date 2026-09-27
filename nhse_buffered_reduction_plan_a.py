"""
nhse_buffered_reduction_plan_a.py

"Plan A" numerical test of the buffered local-channel reduction idea from
Sec. XVII of the paper (Definitions 45-48, Assumption 3, Theorem 15).

MODELING CHOICE AND WHY IT'S NEEDED
-----------------------------------------------------------------------------
Definition 46 requires a genuine partial trace Tr_{Vbar}[rho_V (x) sigma_Vbar],
which needs a tensor-product Hilbert space H = H_V (x) H_Vbar. The paper's
native single-particle space is a DIRECT SUM over sites, not a tensor product
(Remark 23), so this is not automatically available.

We resolve this the way free-fermion physics normally resolves it: promote
"one particle in V with single-particle density matrix rho_V, independently
of one particle in Vbar with density matrix sigma_Vbar" to a genuine 2-particle
free-fermion state. Because rho_V and sigma_Vbar have disjoint spatial support,
this 2-particle state has a block-diagonal single-particle correlation matrix

    C(0) = rho_V (embedded)  (+)  sigma_Vbar (embedded)      (direct sum)

Evolution under the (sub-normalized, strictly-contractive) propagator K is,
for free/non-interacting fermions, exactly

    C(t) = K C(0) K^dagger,

and the reduced state on window V is simply the V-V block of C(t). This is
the standard Peschel/Klich correlation-matrix construction used throughout
free-fermion entanglement-entropy calculations, and it needs only poly(L)
linear algebra, not an exponentially large Fock space.

We deliberately work in the "success branch" (sub-normalized, not re-Julia-
dilated) sense, exactly analogous to how the paper itself treats E_succ before
introducing the Julia completion in Sec. I. This is "Plan A" as opposed to
the heavier "Plan B" (re-dilating the reduced map with a second, paper-
unspecified auxiliary environment).

WHAT THIS BUYS US -- AND THE ONE HONEST CAVEAT
-----------------------------------------------------------------------------
Because C(0) is linear in rho_V ONLY UP TO the additive sigma_Vbar term (which
is a FIXED matrix once sigma_Vbar is fixed), the map rho_V -> C_V(t) is
AFFINE, not linear:

    Phi(rho_V) = L(rho_V) + b,

    L(rho_V) = K_sub rho_V K_sub^dagger,      K_sub  = K[idx_V, idx_V]
    b        = (1/dVbar) K_cross K_cross^dagger,  K_cross = K[idx_V, idx_Vbar]

This affine splitting is exact and is not an approximation; it just means the
paper's A_diamond machinery (built for literal linear CPTP maps) has to be
applied separately to the two pieces rather than to one single quantity:

  * The LINEAR part L is itself a single-Kraus-operator sub-normalized map.
    K_sub is a strict contraction (submatrix of a strict contraction), so we
    can Julia-dilate it exactly as in the original two-site calibration and
    reuse the validated exact Acin discrimination formula to get an honest
    A_diamond-type number for L.

  * The OFFSET b captures exactly the "background leakage from the rest of
    the system" that Assumption 3 (Eq. 358-359) is actually about. We compare
    b computed from a full bulk chain of length L_bulk (tracing out
    EVERYTHING outside V) against b computed from an isolated buffered window
    of width ell + 2*b_buf (tracing out only the buffer), using the SAME
    kappa in both cases per Definition 47, and check whether the difference
    shrinks exponentially with buffer thickness. We do the same for the
    "reversed" (transpose-K) offset, mirroring Eq. 359.

Both pieces are reported and plotted separately below. Neither claims to BE
the paper's single scalar A_diamond(M_V) -- that quantity is not literally
defined for an affine map -- but together they give an honest, well-defined,
and directly falsifiable numerical test of the buffer-convergence idea that
underlies Assumption 3 / Theorem 15.

Requires: numpy, scipy, matplotlib.
Run with: python3 nhse_buffered_reduction_plan_a.py
"""

import numpy as np
from scipy.linalg import expm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Basic building blocks (same conventions as the earlier scripts)
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


def julia_from_K(K):
    """
    Build the Julia-dilated unitary for an ALREADY-normalized strict
    contraction K (no extra kappa needed -- K is used as-is, exactly as
    Proposition 1 requires for any contraction).
    """
    d = K.shape[0]
    Kd = K.conj().T
    DK = sqrtm_psd(np.eye(d) - Kd @ K)
    DKd = sqrtm_psd(np.eye(d) - K @ Kd)
    top = np.hstack([K, DKd])
    bot = np.hstack([DK, -Kd])
    return np.vstack([top, bot])


def half_diamond_A(Utilde):
    """
    Exact A_diamond = (1/2)||N - N^rev||_diamond for the unitary conjugation
    channel N(X) = Utilde X Utilde^dagger, via the Acin discrimination
    formula (Acin, PRL 87, 177901 (2001)) -- same formula validated against
    Theorem 7's closed form in the earlier calibration script.
    """
    W = Utilde.conj().T @ Utilde.T
    ev = np.linalg.eigvals(W)
    ang = np.sort(np.angle(ev) % (2 * np.pi))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    maxgap = np.max(gaps)
    enclosing_arc = 2 * np.pi - maxgap
    if enclosing_arc > np.pi:
        return 1.0
    phi = enclosing_arc / 2.0
    return np.sin(phi)


# ----------------------------------------------------------------------
# 2. Window bookkeeping (no environment doubling needed at this level --
#    we work directly with the un-dilated K, per the free-fermion embedding)
# ----------------------------------------------------------------------

def window_indices(n_sites, v_start, ell):
    idx_V = list(range(v_start, v_start + ell))
    idx_Vbar = [i for i in range(n_sites) if i not in idx_V]
    return idx_V, idx_Vbar


# ----------------------------------------------------------------------
# 3. The two pieces of the affine reduced map: linear part L, offset b
# ----------------------------------------------------------------------

def linear_part_Adiamond(K, idx_V):
    """A_diamond of L(rho_V) = K_sub rho_V K_sub^dagger, K_sub = K[V, V]."""
    K_sub = K[np.ix_(idx_V, idx_V)]
    Ut = julia_from_K(K_sub)
    return half_diamond_A(Ut)


def leakage_offset(K, idx_V, idx_Vbar):
    """
    b = (1/dVbar) K_cross K_cross^dagger, K_cross = K[V, Vbar].
    Undefined (no complement to leak from) when dVbar == 0 (buffer = 0,
    isolated system exactly equal to the window); returns None in that case.
    """
    dVbar = len(idx_Vbar)
    if dVbar == 0:
        return None
    K_cross = K[np.ix_(idx_V, idx_Vbar)]
    return (K_cross @ K_cross.conj().T) / dVbar


def leakage_offset_rev(K, idx_V, idx_Vbar):
    """
    Same quantity but for the reciprocal map (K -> K^T, per Eq. 11 of the
    paper: K_{H^T,t} = K_{H,t}^T). K_cross_rev = K^T[V, Vbar] = K[Vbar, V]^T.
    Returns None when dVbar == 0, for the same reason as leakage_offset.
    """
    dVbar = len(idx_Vbar)
    if dVbar == 0:
        return None
    K_cross_rev = K[np.ix_(idx_Vbar, idx_V)].T
    return (K_cross_rev @ K_cross_rev.conj().T) / dVbar


# ----------------------------------------------------------------------
# 4. Main experiment
# ----------------------------------------------------------------------

def run_buffer_convergence(g=0.5, t_op=0.35, t0=1.0, ell=4, L_bulk=60,
                            buffers=(0, 1, 2, 3, 4, 5, 6, 8, 10)):
    print("=" * 70)
    print("Plan A: buffered local-channel reduction via free-fermion")
    print("correlation-matrix embedding")
    print("=" * 70)

    v_start = L_bulk // 2 - ell // 2
    H_bulk = hn_chain(L_bulk, g, t0)
    U_bulk_norm = np.linalg.norm(U_from_H(H_bulk, t_op), 2)
    kappa = (1.05 * U_bulk_norm) ** 2
    print(f"window width ell = {ell}, bulk length L_bulk = {L_bulk}")
    print(f"common kappa = {kappa:.6f} (from ||U_bulk(t)||_op = {U_bulk_norm:.6f})\n")

    K_bulk = U_from_H(H_bulk, t_op) / np.sqrt(kappa)
    idx_V_bulk, idx_Vbar_bulk = window_indices(L_bulk, v_start, ell)

    # These do NOT depend on the buffer size -- computed once.
    A_full = linear_part_Adiamond(K_bulk, idx_V_bulk)
    b_full = leakage_offset(K_bulk, idx_V_bulk, idx_Vbar_bulk)
    b_full_rev = leakage_offset_rev(K_bulk, idx_V_bulk, idx_Vbar_bulk)
    print(f"A_diamond of the LINEAR part, full bulk chain: {A_full:.6f}")
    print(f"||offset_full||_op = {np.linalg.norm(b_full, 2):.6f}")
    print(f"||offset_full_rev||_op = {np.linalg.norm(b_full_rev, 2):.6f}\n")

    print(f"{'b':>4} {'||U_W(t)||_op':>14} {'A_window':>10} {'|A_full-A_W|':>14} "
          f"{'||b_full-b_W||':>16} {'||b_full_rev-b_W_rev||':>22}")
    results = []
    for buf in buffers:
        w_width = ell + 2 * buf
        H_w = hn_chain(w_width, g, t0)
        U_w_norm = np.linalg.norm(U_from_H(H_w, t_op), 2)
        if U_w_norm >= np.sqrt(kappa):
            print(f"  [WARNING] buffer={buf}: ||U_w||={U_w_norm:.4f} >= "
                  f"sqrt(kappa)={np.sqrt(kappa):.4f}; K_w not a strict "
                  f"contraction, results unreliable.")
        K_w = U_from_H(H_w, t_op) / np.sqrt(kappa)  # SAME kappa, Definition 47
        idx_V_w, idx_Vbar_w = window_indices(w_width, buf, ell)

        A_w = linear_part_Adiamond(K_w, idx_V_w)
        b_w = leakage_offset(K_w, idx_V_w, idx_Vbar_w)
        b_w_rev = leakage_offset_rev(K_w, idx_V_w, idx_Vbar_w)

        dA = abs(A_full - A_w)
        if b_w is None:
            # buffer = 0: no complement to leak from, offset undefined.
            db, db_rev = float("nan"), float("nan")
            db_str, db_rev_str = "  n/a (buf=0)", "     n/a (buf=0)"
        else:
            db = np.linalg.norm(b_full - b_w, 2)
            db_rev = np.linalg.norm(b_full_rev - b_w_rev, 2)
            db_str, db_rev_str = f"{db:16.6f}", f"{db_rev:22.6f}"

        results.append((buf, U_w_norm, A_w, dA, db, db_rev))
        print(f"{buf:4d} {U_w_norm:14.6f} {A_w:10.6f} {dA:14.6f} "
              f"{db_str} {db_rev_str}")
    print()
    return A_full, results


def fit_rate(b, y):
    mask = y > 1e-12
    if mask.sum() < 2:
        return float("nan")
    A = np.vstack([b[mask], np.ones(mask.sum())]).T
    slope, _ = np.linalg.lstsq(A, np.log(y[mask]), rcond=None)[0]
    return -slope


def plot_results(results, outbase="fig_buffered_reduction_plan_a"):
    b = np.array([r[0] for r in results])
    dA = np.array([r[3] for r in results])
    db = np.array([r[4] for r in results])
    db_rev = np.array([r[5] for r in results])

    mu_A = fit_rate(b, dA)
    mu_b = fit_rate(b, db)
    mu_brev = fit_rate(b, db_rev)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogy(b, dA, 'o-', label=fr'$|A_{{full}}-A_{{window}}|$ ($\mu\approx{mu_A:.2f}$)')
    ax.semilogy(b, db, 's-', label=fr'$\|b_{{full}}-b_{{window}}\|$ ($\mu\approx{mu_b:.2f}$)')
    ax.semilogy(b, db_rev, '^-', label=fr'$\|b^{{rev}}_{{full}}-b^{{rev}}_{{window}}\|$ ($\mu\approx{mu_brev:.2f}$)')
    ax.set_xlabel('buffer thickness b (sites)')
    ax.set_ylabel('deviation from bulk value')
    ax.set_title('Buffered local-channel reduction (Plan A, free-fermion embedding)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outbase + '.pdf')
    plt.savefig(outbase + '.png', dpi=150)
    print(f"Saved {outbase}.pdf/.png")
    print(f"Fitted decay rates: mu_A={mu_A:.4f}, mu_offset={mu_b:.4f}, "
          f"mu_offset_rev={mu_brev:.4f}")


if __name__ == "__main__":
    A_full, results = run_buffer_convergence()
    plot_results(results)

    print()
    print("=" * 70)
    print("Reminder: the LINEAR part (A_full vs A_window) and the OFFSET")
    print("terms (leakage from the traced-out complement) are reported")
    print("separately because the reduced map here is affine, not linear --")
    print("see the module docstring for why. Exponential decay of ALL THREE")
    print("curves with buffer thickness is the honest numerical analogue of")
    print("Assumption 3 / Theorem 15 for this free-fermion realization.")
    print("=" * 70)