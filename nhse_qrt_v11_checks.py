#!/usr/bin/env python3
"""V11 independent checks: position-resolved skin budget Phi_X, nonnormality budget Phi_Q.

Every statement in the V11 manuscript that is a theorem is tested here on explicit data.
Large-L Hatano-Nelson values use the exact closed form b_j = g(L-1)+log(2 L_j^2/(L+1)),
which involves no ill-conditioned eigenproblem.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eig

rng = np.random.default_rng(7)


# ----------------------------------------------------------------- models
def hn(L, g, pbc=False):
    H = np.zeros((L, L), complex)
    for n in range(L - 1):
        H[n + 1, n] += np.exp(g)
        H[n, n + 1] += np.exp(-g)
    if pbc:
        H[0, L - 1] += np.exp(g)
        H[L - 1, 0] += np.exp(-g)
    return H


def atlas(H):
    w, vl, vr = eig(H, left=True, right=True)
    vl = vl / np.linalg.norm(vl, axis=0)
    vr = vr / np.linalg.norm(vr, axis=0)
    return w, vl, vr


def cell_density(v, q):
    return (np.abs(v.reshape(-1, q)) ** 2).sum(axis=1)


def budgets(H, q=1):
    """return Phi_Q, Phi_X and per-mode arrays b, beta, F_pinched"""
    w, vl, vr = atlas(H)
    N = H.shape[0]
    b, beta, fpin, dens = [], [], [], []
    for j in range(N):
        r, l = vr[:, j], vl[:, j]
        b.append(-np.log(abs(np.vdot(l, r))))
        p, qd = cell_density(r, q), cell_density(l, q)
        bc = np.sum(np.sqrt(p * qd))
        beta.append(-np.log(bc))
        # pinched (cell-block) fidelity: sum_x |Pi_x r||Pi_x l| |<l_x|r_x>|
        F = 0.0
        for x in range(N // q):
            rx, lx = r[x * q:(x + 1) * q], l[x * q:(x + 1) * q]
            F += abs(np.vdot(lx, rx))
        fpin.append(F)
        dens.append((p, qd))
    return np.mean(b), np.mean(beta), np.array(b), np.array(beta), np.array(fpin), dens


def hn_phi_exact(L, g, chunk=400):
    """Phi_Q(HN, OBC) from R_j = e^{g(L-1)} L_j (proved identity)."""
    n = np.arange(1, L + 1)
    wgt = np.exp(-2 * g * (n - 1))
    k = np.pi * np.arange(1, L + 1) / (L + 1)
    Lj2 = np.empty(L)
    for s in range(0, L, chunk):
        Lj2[s:s + chunk] = (wgt[None, :] * np.sin(np.outer(k[s:s + chunk], n)) ** 2).sum(axis=1)
    return g * (L - 1) + np.mean(np.log(2 * Lj2 / (L + 1)))


def hn_asym(L, g):
    u = np.exp(-2 * g)
    return (g * (L - 1) - np.log(L + 1) + np.log(1 / np.tanh(g) / 2)
            + 2 / L * (np.log(L + 1) + np.log(1 - u)) - 2 / L * np.log(1 - u ** (L + 1)))


def main():
    print("=== C1  HN: Phi_X = Phi_Q, direct eigen-solver vs exact vs asymptotic (g=0.25)")
    g = 0.25
    for L in [6, 10, 16, 24, 32, 40]:
        pq, px, *_ = budgets(hn(L, g))
        print(f"L={L:3d}  Phi_Q={pq:.9f}  Phi_X={px:.9f}  exact={hn_phi_exact(L, g):.9f}  asym={hn_asym(L, g):.9f}")

    print("\n=== C2  HN identity R_j = e^{g(L-1)} L_j, and Chernoff optimality at s=1/2 (L=12)")
    L, g = 12, 0.4
    n = np.arange(1, L + 1)
    maxdev, chern = 0.0, 0.0
    for j in range(1, L + 1):
        s2 = np.sin(np.pi * j * n / (L + 1)) ** 2
        R = np.sqrt((np.exp(2 * g * (n - 1)) * s2).sum())
        Lj = np.sqrt((np.exp(-2 * g * (n - 1)) * s2).sum())
        maxdev = max(maxdev, abs(R / (np.exp(g * (L - 1)) * Lj) - 1))
        p, qd = np.exp(2 * g * (n - 1)) * s2 / R ** 2, np.exp(-2 * g * (n - 1)) * s2 / Lj ** 2
        ss = np.linspace(0, 1, 2001)
        vals = np.array([np.sum(p ** s * qd ** (1 - s)) for s in ss])
        chern = max(chern, abs(ss[np.argmin(vals)] - 0.5))
    print(f"max |R_j/(e^(g(L-1))L_j)-1| = {maxdev:.2e};  max |argmin_s - 1/2| = {chern:.2e}")

    print("\n=== C3  Large L (exact closed form): residual vs log-term, g=0.25 and 0.7")
    for g in [0.25, 0.7]:
        for L in [100, 400, 1600, 6400]:
            ex = hn_phi_exact(L, g)
            print(f"g={g} L={L:5d}  Phi-[g(L-1)-log(L+1)] = {ex - (g*(L-1)-np.log(L+1)):.8f}"
                  f"   log(coth g/2)={np.log(1/np.tanh(g)/2):.8f}   |Phi-asym|={abs(ex-hn_asym(L,g)):.2e}")

    print("\n=== C4  Sandwich |<l|r>| <= F_pinched <= BC on random banded q=2 non-Hermitian matrices")
    worst = 0
    for trial in range(20):
        Lc, q = 7, 2
        H = np.zeros((Lc * q, Lc * q), complex)
        for x in range(Lc):
            for y in range(max(0, x - 1), min(Lc, x + 2)):
                H[x*q:(x+1)*q, y*q:(y+1)*q] = rng.normal(size=(q, q)) + 1j * rng.normal(size=(q, q))
        _, _, b, beta, fp, _ = budgets(H, q)
        ov = np.exp(-b)
        ok = np.all(ov <= fp + 1e-12) and np.all(fp <= np.exp(-beta) + 1e-12)
        worst = max(worst, 0 if ok else 1)
    print("sandwich violated in any trial:", bool(worst))

    print("\n=== C5  Reciprocity: complex-symmetric banded Toeplitz (nonnormal, no skin)")
    L = 14
    a1, a2 = 0.7 + 0.3j, 0.4 - 0.5j
    H = np.zeros((L, L), complex)
    for d, a in [(1, a1), (2, a2)]:
        for n in range(L - d):
            H[n + d, n] = a
            H[n, n + d] = a
    H += 0.1j * np.eye(L)
    pq, px, _, _, _, dens = budgets(H)
    print(f"symmetric?{np.allclose(H, H.T)}  Phi_Q={pq:.4f} (>0)  Phi_X={px:.2e}  max|p-q|={max(np.abs(p-qq).max() for p,qq in dens):.2e}")

    print("\n=== C6  Translation-invariant PBC, q=2, random blocks: Phi_X=0 while Phi_Q>0")
    Lc, q = 9, 2
    A, B, C = [rng.normal(size=(q, q)) + 1j * rng.normal(size=(q, q)) for _ in range(3)]
    HP = np.zeros((Lc * q, Lc * q), complex)
    for x in range(Lc):
        HP[x*q:(x+1)*q, x*q:(x+1)*q] = C
        y = (x + 1) % Lc
        HP[y*q:(y+1)*q, x*q:(x+1)*q] += A
        HP[x*q:(x+1)*q, y*q:(y+1)*q] += B
    pq, px, *_ = budgets(HP, q)
    print(f"Phi_Q(PBC)={pq:.4f}  Phi_X(PBC)={px:.2e}")

    print("\n=== C7  Non-covariant unitary: 0 <= Phi_X(UHU^H) <= Phi_Q(H) (L=16, HN g=0.25)")
    H = hn(16, 0.25)
    Z = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    U, _ = np.linalg.qr(Z)
    pq0, px0, *_ = budgets(H)
    pq1, px1, *_ = budgets(U @ H @ U.conj().T)
    print(f"Phi_Q: {pq0:.6f} -> {pq1:.6f}   Phi_X: {px0:.6f} -> {px1:.6f}")
    P = np.eye(16)[rng.permutation(16)]
    pq2, px2, *_ = budgets(P @ H @ P.T)
    print(f"cell permutation (free):  Phi_X: {px0:.6f} -> {px2:.6f}")

    print("\n=== C8  Tilt filter on HN (L=20, g1=0.35 -> g2=0.15)")
    L, g1, g2 = 20, 0.35, 0.15
    d = g1 - g2
    s = np.exp(-d * np.arange(L))
    S = np.diag(s)
    H1, H2 = hn(L, g1), hn(L, g2)
    print(f"||S H1 S^-1 - H2|| = {np.linalg.norm(S @ H1 @ np.linalg.inv(S) - H2, 2):.2e}")
    w1, vl1, vr1 = atlas(H1)
    w2, vl2, vr2 = atlas(H2)
    i1, i2 = np.argsort(w1.real), np.argsort(w2.real)
    lip_ok, id_err, betas, succ = True, 0.0, [], []
    for a, c in zip(i1, i2):
        p, qd = np.abs(vr1[:, a]) ** 2, np.abs(vl1[:, a]) ** 2
        Zp, Zq = np.sum(s ** 2 * p), np.sum(s ** -2 * qd)
        beta0 = -np.log(np.sum(np.sqrt(p * qd)))
        beta1 = beta0 + 0.5 * np.log(Zp * Zq)
        p2, q2 = np.abs(vr2[:, c]) ** 2, np.abs(vl2[:, c]) ** 2
        id_err = max(id_err, abs(beta1 - (-np.log(np.sum(np.sqrt(p2 * q2))))))
        lip_ok &= abs(beta1 - beta0) <= (L - 1) * d + 1e-12
        succ.append((np.log(Zp), np.log(Zq * s.min() ** 2)))  # K=S/smax (smax=1), K'=smin S^-1
    lp = np.array(succ)
    print(f"beta' identity error = {id_err:.2e};  Lipschitz |dbeta|<=log cond S holds: {lip_ok};  "
          f"log cond S = {(L-1)*d:.3f}")
    print(f"log p_succ (right) min/max = {lp[:,0].min():.2f}/{lp[:,0].max():.2f};  (left) {lp[:,1].min():.2f}/{lp[:,1].max():.2f};  "
          f"-2 d (L-1) = {-2*d*(L-1):.2f}")

    print("\n=== C9  Counterexample: reference-sufficiency is necessary (classical dichotomies)")
    def phi(p, qd):
        return -np.log(np.sum(np.sqrt(np.array(p) * np.array(qd))))
    pO, qO = [.75, .25, 0, 0], [.25, .75, 0, 0]
    pP, qP = [0, 0, .9, .1], [0, 0, .1, .9]
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]], float)  # column-stochastic: 4 -> 3
    R0 = max(phi(pO, qO) - phi(pP, qP), 0)
    R1 = max(phi(T @ pO, T @ qO) - phi(T @ pP, T @ qP), 0)
    print(f"Phi_X(O)={phi(pO,qO):.4f}  Phi_X(P)={phi(pP,qP):.4f}  R before={R0:.4f}  R after T={R1:.4f}")

    print("\n=== C10  First-order sensitivity: sup_{|dH|<=eps} |d lambda| = eps e^{b}  (random 8x8)")
    M = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    w, vl, vr = atlas(M)
    eps = 1e-7
    j = 0
    dH = eps * np.outer(vl[:, j], vr[:, j].conj())
    w_new = np.linalg.eigvals(M + dH)
    shift = np.min(np.abs(w_new - w[j]))
    print(f"measured shift={shift:.4e}   eps*e^b={eps/abs(np.vdot(vl[:,j], vr[:,j])):.4e}")


if __name__ == "__main__":
    main()
