#!/usr/bin/env python3
"""Independent numerical checks for the V30 manuscript (NHSE_QRT_V30.tex).

Every check targets a load-bearing statement and prints the quantity it tests.
Numerical tolerances are stated explicitly; nothing is forced to pass.
Run:  python nhse_qrt_v30_checks.py   (numpy, scipy)
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import eig, eigh, norm, svd
from scipy.linalg import block_diag
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

rng = np.random.default_rng(30001)
HALF_LOG2 = 0.5 * np.log(2.0)


# ------------------------------------------------------------------ helpers
def collar(N, sigma, w):
    """cells in the width-w collar: sigma=+1 right end, sigma=-1 left end"""
    return list(range(N - w, N)) if sigma > 0 else list(range(w))


def leak_pair(p, q, N, sigma, w):
    """(lambda^R, lambda^L): R tested at sigma, L at the dual boundary -sigma.
    Complements are summed directly (no 1-x cancellation)."""
    cR, cL = collar(N, sigma, w), collar(N, -sigma, w)
    mR = np.ones(N, bool); mR[cR] = False
    mL = np.ones(N, bool); mL[cL] = False
    return p[mR].sum(), q[mL].sum()


def s_score(p, q, N, sigma, w):
    lR, lL = leak_pair(p, q, N, sigma, w)
    return max(0.0, -0.5 * np.log(max(lR + lL, 1e-300)))


def sqrtm_h(M):
    ew, U = eigh((M + M.conj().T) / 2)
    return (U * np.sqrt(np.clip(ew, 0, None))) @ U.conj().T


def root_fid(A, B):
    return float(np.sum(svd(sqrtm_h(A) @ sqrtm_h(B), compute_uv=False)))


def rand_state(d, rank=None):
    r = rank or d
    Z = rng.normal(size=(d, r)) + 1j * rng.normal(size=(d, r))
    rho = Z @ Z.conj().T
    return rho / np.trace(rho).real


def cell_probs(v, N, d):
    return (np.abs(v) ** 2).reshape(N, d).sum(1) / norm(v) ** 2


def biorth_pairs(G):
    """right eigvecs of G and matched left eigvecs (G^dagger L = conj(lambda) L), unit-normalised"""
    ev, R = eig(G)
    evl, Lv = eig(G.conj().T)
    out = []
    for m in range(len(ev)):
        r = R[:, m] / norm(R[:, m])
        l = Lv[:, int(np.argmin(np.abs(evl - np.conj(ev[m]))))]
        out.append((ev[m], r, l / norm(l)))
    return out


def hn_law(L, g):
    """exact HN eigen-response cell laws (similarity to the symmetric chain), rows = modes j=1..L"""
    n = np.arange(1, L + 1); P = []; Q = []
    for j in range(1, L + 1):
        k = np.pi * j / (L + 1); s2 = np.sin(k * n) ** 2
        p = np.exp(2 * g * (n - L)) * s2; q = np.exp(-2 * g * (n - 1)) * s2
        P.append(p / p.sum()); Q.append(q / q.sum())
    return np.array(P), np.array(Q)


def allowed_T(N, W):
    """classical collar-non-expanding maps: A[y,x]=True iff x->y never enters a collar x was not in"""
    def depth(x): return (min(x + 1, W + 1), min(N - x, W + 1))
    A = np.zeros((N, N), bool)
    for x in range(N):
        for y in range(N):
            A[y, x] = all(depth(y)[k] >= depth(x)[k] for k in (0, 1))
    return A


def rand_T(A):
    N = A.shape[0]; T = np.zeros((N, N))
    for x in range(N):
        idx = np.where(A[:, x])[0]
        T[idx, x] = rng.dirichlet(np.ones(len(idx)) * rng.uniform(.2, 2))
    return T


# ------------------------------------------------------------------ checks
def c01_reciprocity():
    """Lemma 1 / Thm 1: collar-reciprocal => s=0 exactly; G^T=UGU^dag (U cell-local, U^T=U) => collar-reciprocal;
    U^T=-U => no simple eigenvalue."""
    s_max = 0.0
    for _ in range(3000):                       # (a) random collar-reciprocal data (identical cell laws)
        N = int(rng.integers(4, 15)); w = int(rng.integers(1, N // 2 + 1))
        p = rng.dirichlet(np.ones(N) * rng.uniform(.05, 1))
        if rng.random() < .5:                   # push mass into the two collars: the hardest case
            c = np.zeros(N); c[collar(N, 1, w)] = rng.random(); c[collar(N, -1, w)] = rng.random()
            p = .001 * p + .999 * c / c.sum()
        for sg in (1, -1):
            s_max = max(s_max, s_score(p, p, N, sg, w))
    diff = 0.0; s_gen = 0.0; gap = 0.0
    for _ in range(150):                        # (b),(c) reciprocal generators
        N = int(rng.integers(4, 10)); d = 2
        for sgn in (1, -1):
            blocks = []
            for _c in range(N):
                if sgn > 0:
                    S = rng.normal(size=(d, d)); S = S + S.T; ww, V = eigh(S); blocks.append((V * np.exp(1j * ww)) @ V.T)
                else:
                    Q, _ = np.linalg.qr(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)))
                    blocks.append(Q @ np.array([[0, 1], [-1, 0]]) @ Q.T)
            U = block_diag(*blocks)
            B = rng.normal(size=(N * d, N * d)) + 1j * rng.normal(size=(N * d, N * d))
            G = B + U.conj().T @ B.T @ U        # satisfies G^T = U G U^dagger
            assert np.allclose(G.T, U @ G @ U.conj().T)
            if sgn > 0:
                for lam, r, l in biorth_pairs(G):
                    pr, pl = cell_probs(r, N, d), cell_probs(l, N, d)
                    diff = max(diff, np.abs(pr - pl).max())
                    for w in range(1, N // 2 + 1):
                        s_gen = max(s_gen, s_score(pr, pl, N, 1, w), s_score(pr, pl, N, -1, w))
            else:
                ev = eig(G)[0]; scale = np.abs(ev).max()
                gap = max(gap, max(min(abs(ev[i] - ev[j]) for j in range(len(ev)) if j != i) for i in range(len(ev))) / scale)
    ok = s_max < 1e-12 and diff < 1e-10 and s_gen < 1e-12 and gap < 1e-10   # exact zeros up to floating point
    print(f"V30-01 reciprocity: collar-reciprocal max s={s_max:.1e}; U^T=U: max |cell law R-L|={diff:.1e}, max s={s_gen:.1e}; "
          f"U^T=-U: max rel. distance to nearest other eigenvalue={gap:.1e} (all degenerate); passed={ok}")
    return ok


def c02_spectral_sensitivity():
    """Thm 2: F <= sqrt(lR)+sqrt(lL) (quantum root fidelity);
    generator level: kappa_m >= e^s/sqrt2, residue identity, forward residue >= sqrt((e^{2s}-1)/2)."""
    worst_F = np.inf
    for _ in range(3000):
        N = int(rng.integers(4, 12)); w = int(rng.integers(1, N // 2 + 1))
        eps = 10 ** rng.uniform(-6, -0.2)
        def conc(sg):
            u = np.zeros(N, complex); idx = collar(N, sg, w)
            u[idx] = rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx)); u /= norm(u)
            v = rng.normal(size=N) + 1j * rng.normal(size=N); v *= np.sqrt(eps) / norm(v)
            x = u * np.sqrt(1 - eps) + v; return x / norm(x)
        sg = int(rng.choice([1, -1]))
        r, l = conc(sg), conc(-sg)
        if rng.random() < .4:
            r2, l2 = conc(sg), conc(-sg)
            rho = .7 * np.outer(r, r.conj()) + .3 * np.outer(r2, r2.conj())
            sig = .7 * np.outer(l, l.conj()) + .3 * np.outer(l2, l2.conj())
        else:
            rho, sig = np.outer(r, r.conj()), np.outer(l, l.conj())
        lR, lL = leak_pair(np.real(np.diag(rho)), np.real(np.diag(sig)), N, sg, w)
        worst_F = min(worst_F, np.sqrt(lR) + np.sqrt(lL) - root_fid(rho, sig))
    worst_k = np.inf; worst_fwd = np.inf; ident = []; n_pos = 0
    for _ in range(400):
        N = int(rng.integers(4, 12)); d = int(rng.integers(1, 3))
        G = np.zeros((N * d, N * d), complex)
        for n in range(N - 1):
            G[(n + 1) * d:(n + 2) * d, n * d:(n + 1) * d] = rng.normal(size=(d, d)) * np.exp(rng.uniform(0, 1.2))
            G[n * d:(n + 1) * d, (n + 1) * d:(n + 2) * d] = rng.normal(size=(d, d)) * np.exp(-rng.uniform(0, 1.2))
        G += np.diag(rng.normal(size=N * d)) + .01 * rng.normal(size=G.shape)
        for lam, r, l in biorth_pairs(G):
            kap = 1 / abs(np.vdot(l, r))
            if kap > 1e6: continue                       # keep eig in its accurate regime
            P = np.outer(r, l.conj()) / np.vdot(l, r)
            pr, pl = cell_probs(r, N, d), cell_probs(l, N, d)
            for w in range(1, N // 2 + 1):
                s = s_score(pr, pl, N, 1, w)
                lR, lL = leak_pair(pr, pl, N, 1, w)
                Pp = np.zeros(N * d); Pp[(N - w) * d:] = 1; Pm = np.zeros(N * d); Pm[:w * d] = 1
                fwd = norm(np.diag(Pp) @ P @ np.diag(Pm), 2)
                ident.append(abs(fwd - kap * np.sqrt((1 - lR) * (1 - lL))) / kap)   # error relative to the scale kappa of P_m
                worst_k = min(worst_k, np.log(kap) - (s - HALF_LOG2))
                worst_fwd = min(worst_fwd, fwd - np.sqrt((np.exp(2 * s) - 1) / 2))
                n_pos += s > 1
    ident = np.array(ident)
    ok = worst_F >= -1e-12 and worst_k >= -1e-9 and worst_fwd >= -1e-9 and ident.max() < 1e-6 and n_pos > 100
    print(f"V30-02 spectral sensitivity: min[sqrt(lR)+sqrt(lL)-F]={worst_F:.3f}; min[log kappa-(s-log2/2)]={worst_k:.3f}; "
          f"min[fwd residue - sqrt((e^2s-1)/2)]={worst_fwd:.3f}; residue identity err/kappa max={ident.max():.1e} "
          f"(median {np.median(ident):.1e}); cases with s>1: {n_pos}; passed={ok}")
    return ok


PSIS = [lambda lr, ll: np.maximum(0, -0.5 * np.log(np.maximum(lr + ll, 1e-300))),   # skin score s
        lambda lr, ll: -0.5 * np.log(np.maximum(lr + ll, 1e-300)),
        lambda lr, ll: np.sqrt(1 / np.maximum(lr + ll, 1e-300)),
        lambda lr, ll: np.exp(-3 * lr) + np.exp(-ll), lambda lr, ll: 2 - lr - ll]
for _t in (.2, .6, 1.0, 1.5):
    PSIS.append((lambda t: lambda lr, ll: np.maximum(t - lr - ll, 0))(_t))


def profile(P, Q, N, W):
    """rows: responses; columns: (sigma,w); entries: (lR, lL) pairs"""
    return np.array([[leak_pair(p, q, N, s, w) for s in (1, -1) for w in range(1, W + 1)] for p, q in zip(P, Q)])


def c03_monotones_and_golden_rule():
    """Thm 3: Psi_psi non-increasing; faithful monotone; free (skinless) set closed under free morphisms."""
    N, W = 16, 4; A = allowed_T(N, W); bad = 0; tests = 0; ok_maps = 0; free_bad = 0; free_tests = 0
    for it in range(600):
        MX = int(rng.integers(2, 8)); MY = int(rng.integers(1, 8))
        P = rng.dirichlet(np.ones(N) * rng.uniform(.05, 1), size=MX); Qd = rng.dirichlet(np.ones(N) * rng.uniform(.05, 1), size=MX)
        for m in range(MX):
            if rng.random() < .7:
                a = rng.uniform(.3, 3); n = np.arange(N)
                P[m] = np.exp(a * n) * P[m]; P[m] /= P[m].sum(); Qd[m] = np.exp(-a * n) * Qd[m]; Qd[m] /= Qd[m].sum()
        pi = rng.dirichlet(np.ones(MX)); kap = rng.dirichlet(np.ones(MY), size=MX).T
        pip = kap @ pi
        T = rand_T(A)
        # collar non-expansion at operator level, every sigma,w
        okT = all(T[np.ix_(collar(N, s, w), [x for x in range(N) if x not in collar(N, s, w)])].sum() < 1e-14
                  for s in (1, -1) for w in range(1, W + 1))
        ok_maps += okT
        Pn = np.array([T @ ((kap[mp] * pi) @ P) / pip[mp] for mp in range(MY)])
        Qn = np.array([T @ ((kap[mp] * pi) @ Qd) / pip[mp] for mp in range(MY)])
        lx = profile(P, Qd, N, W); ly = profile(Pn, Qn, N, W)
        for psi in PSIS:
            vx = pi @ psi(lx[..., 0], lx[..., 1]).sum(1); vy = pip @ psi(ly[..., 0], ly[..., 1]).sum(1)
            tests += 1; bad += int(vy > vx + 1e-9 * (1 + abs(vx)))
        sx = PSIS[0](lx[..., 0], lx[..., 1]).max(1); sy = PSIS[0](ly[..., 0], ly[..., 1]).max(1)   # faithful monotone
        tests += 1; bad += int(pip @ sy > pi @ sx + 1e-9 * (1 + pi @ sx))
        # golden rule: skinless inputs (identical cell laws => s=0) stay skinless
        R0 = rng.dirichlet(np.ones(N), size=MX)
        R0n = np.array([T @ ((kap[mp] * pi) @ R0) / pip[mp] for mp in range(MY)])
        l0 = profile(R0n, R0n, N, W); free_tests += 1
        free_bad += int(PSIS[0](l0[..., 0], l0[..., 1]).max() > 0)
    ok = bad == 0 and ok_maps == 600 and free_bad == 0
    print(f"V30-03 leakage monotones: tests={tests}, violations={bad}; maps collar-non-expanding={ok_maps}/600; "
          f"skinless->skinless failures={free_bad}/{free_tests}; passed={ok}")
    return ok


def c04_fidelity_monotone():
    """Thm 3(iii): Phi_Q monotone under common CPTP + response mixing; additive on tensor products."""
    bad = 0; worst_add = 0.0
    for _ in range(250):
        d = 4; MX = int(rng.integers(2, 5)); MY = int(rng.integers(1, 5))
        R = [rand_state(d) for _ in range(MX)]; Lm = [rand_state(d) for _ in range(MX)]
        pi = rng.dirichlet(np.ones(MX)); kap = rng.dirichlet(np.ones(MY), size=MX).T; pip = kap @ pi
        K = [(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) for _ in range(3)]
        Sm = np.linalg.inv(sqrtm_h(sum(k.conj().T @ k for k in K))); K = [k @ Sm for k in K]
        Lam = lambda r: sum(k @ r @ k.conj().T for k in K)
        Phi = lambda pp, RR, LL: -np.log(sum(pp[m] * root_fid(RR[m], LL[m]) for m in range(len(pp))))
        Rn = [Lam(sum(kap[mp, m] * pi[m] * R[m] for m in range(MX)) / pip[mp]) for mp in range(MY)]
        Ln = [Lam(sum(kap[mp, m] * pi[m] * Lm[m] for m in range(MX)) / pip[mp]) for mp in range(MY)]
        bad += int(Phi(pip, Rn, Ln) > Phi(pi, R, Lm) + 1e-9)
        d2 = 3; R2 = [rand_state(d2) for _ in range(2)]; L2 = [rand_state(d2) for _ in range(2)]; pi2 = rng.dirichlet(np.ones(2))
        RR = [np.kron(R[a], R2[b]) for a in range(MX) for b in range(2)]; LL = [np.kron(Lm[a], L2[b]) for a in range(MX) for b in range(2)]
        pp = np.array([pi[a] * pi2[b] for a in range(MX) for b in range(2)])
        worst_add = max(worst_add, abs(Phi(pp, RR, LL) - Phi(pi, R, Lm) - Phi(pi2, R2, L2)))
    ok = bad == 0 and worst_add < 1e-8
    print(f"V30-04 Phi_Q monotone (violations={bad}) and additive (err={worst_add:.1e}); passed={ok}")
    return ok


def c05_calibration():
    """Thm 7 (gauge bound, disordered / quasiperiodic tridiagonal chains, via eig) and Cor 2 (HN, two-sided)."""
    worst = np.inf; nt = 0; n_big = 0
    for _ in range(300):
        L = int(rng.integers(6, 22))
        tp = np.exp(rng.uniform(-.3, 1.2, size=L - 1)); tm = np.exp(rng.uniform(-1.2, .3, size=L - 1))
        V = rng.normal(size=L) * rng.uniform(0, 2)
        if rng.random() < .3: V = 2 * np.cos(2 * np.pi * 0.6180339887 * np.arange(L))      # quasiperiodic (AAH) potential
        G = np.diag(V).astype(complex)
        for n in range(L - 1): G[n + 1, n] = tp[n]; G[n, n + 1] = tm[n]
        Gam = np.concatenate([[0], np.cumsum(0.5 * np.log(tp / tm))])
        h = np.sqrt(tp * tm); e0, phi = eigh(np.diag(V) + np.diag(h, 1) + np.diag(h, -1))
        for lam, r, l in biorth_pairs(G):
            j = int(np.argmin(np.abs(e0 - lam.real)))
            p, q = np.abs(r) ** 2, np.abs(l) ** 2
            a = min(phi[0, j] ** 2, phi[-1, j] ** 2)
            for w in range(1, L // 2 + 1):
                Gp = min(Gam[L - 1] - Gam[n] for n in range(0, L - w)); Gm = min(Gam[n] - Gam[0] for n in range(w, L))
                bound = min(Gp, Gm) - 0.5 * np.log(2 / a)
                s = s_score(p, q, L, 1, w)
                worst = min(worst, s - bound); nt += 1; n_big += bound > 1
    # HN two-sided bound, analytic law (eig is unreliable at large L: pseudospectral sensitivity ~ e^{gL})
    dev = 0.0
    for L, g in [(14, .3), (20, .45)]:
        G = np.zeros((L, L));
        for k in range(L - 1): G[k + 1, k] = np.exp(g); G[k, k + 1] = np.exp(-g)
        P1, Q1 = hn_law(L, g)
        for lam, r, l in biorth_pairs(G):
            p, q = np.abs(r) ** 2, np.abs(l) ** 2
            dev = max(dev, min(max(np.abs(p - P1[b]).max(), np.abs(q - Q1[b]).max()) for b in range(L)))
    lo = np.inf; hi = -np.inf; wrong = 0.0
    for L, g in [(10, .3), (40, .4), (100, .25), (200, .6), (400, .15), (400, 1.5)]:
        P, Q = hn_law(L, g); C = 1.5 * np.log(L + 1) + g + 0.5 * np.log(1 / (1 - np.exp(-2 * g)))
        for w in range(1, L // 2 + 1):
            s = np.array([s_score(P[j], Q[j], L, 1, w) for j in range(L)])
            lo = min(lo, np.min(s - (g * w - C))); hi = max(hi, np.max(s - (g * w + C)))
            wrong = max(wrong, max(s_score(P[j], Q[j], L, -1, w) for j in range(L)))
    ok = worst >= -1e-9 and n_big > 50 and dev < 1e-6 and lo >= 0 and hi <= 0 and wrong == 0.0
    print(f"V30-05 calibration: gauge bound min slack={worst:.3f} over {nt} (mode,w) pairs ({n_big} with bound>1); "
          f"HN analytic-vs-eig dev={dev:.1e}; |s-gw|<=C_L slack lo={lo:.3f}, hi={-hi:.3f}; HN wrong-orientation max s={wrong:.1e}; passed={ok}")
    return ok


def c06_controls():
    """Hermitian chain: s=0; periodic HN: s=0; Hermitian chain + non-reciprocal bulk bond: Phi_Q>0 at every L,
    kappa bounded uniformly in L, hence s <= log kappa + log2/2 bounded (Thm 2) -- all widths w <= L/2."""
    L = 60
    H = np.zeros((L, L))
    for k in range(L - 1): H[k + 1, k] = H[k, k + 1] = 1.0
    allw = lambda N: [(s, w) for s in (1, -1) for w in range(1, N // 2 + 1)]
    sh = max(s_score(np.abs(r) ** 2, np.abs(l) ** 2, L, s, w) for _, r, l in biorth_pairs(H) for s, w in allw(L))
    g = .4; Hp = np.zeros((L, L), complex)
    for k in range(L): Hp[(k + 1) % L, k] += np.exp(g); Hp[k, (k + 1) % L] += np.exp(-g)
    sp = max(s_score(np.abs(r) ** 2, np.abs(l) ** 2, L, s, w) for _, r, l in biorth_pairs(Hp) for s, w in allw(L))
    rows = []; thm2 = 0
    for L2 in (40, 60, 100, 160):
        H2 = np.zeros((L2, L2), complex)
        for k in range(L2 - 1): H2[k + 1, k] = H2[k, k + 1] = 1.0
        H2[L2 // 2, L2 // 2 + 1] = 3.0; H2[L2 // 2 + 1, L2 // 2] = 0.2
        pairs = biorth_pairs(H2)
        kaps = [1 / abs(np.vdot(l, r)) for _, r, l in pairs]
        sd = [max(s_score(np.abs(r) ** 2, np.abs(l) ** 2, L2, s, w) for s, w in allw(L2)) for _, r, l in pairs]
        thm2 += sum(sd[m] > np.log(kaps[m]) + HALF_LOG2 + 1e-9 for m in range(L2))
        rows.append((L2, np.mean(np.log(kaps)), max(kaps), max(sd)))
    ok = sh < 1e-12 and sp < 1e-12 and thm2 == 0 and all(r[1] > 0.5 and r[2] < 2.1 and r[3] < 1.1 for r in rows)
    print(f"V30-06 controls: Hermitian max s={sh:.1e}; periodic HN max s={sp:.1e}; bulk bond (L, mean log kappa, max kappa, max_w s): "
          + ", ".join(f"({a},{b:.3f},{c:.3f},{d:.3f})" for a, b, c, d in rows) + f"; Thm2 violations={thm2}; passed={ok}")
    return ok


def c07_leakage_completeness():
    """Thm 4(a): LP feasibility of the leakage preorder <=> no max-affine witness; the Farkas certificate separates."""
    agree = 0; n = 0; feas_count = 0; sep_ok = 0; sep_n = 0
    for _ in range(300):
        d = int(rng.integers(2, 7)); M = int(rng.integers(2, 6)); Mp = int(rng.integers(1, 6))
        LX = rng.uniform(0, 1, size=(M, d)); pi = rng.dirichlet(np.ones(M))
        if rng.random() < .5:           # build a dominated target
            J = rng.dirichlet(np.ones(Mp), size=M) * pi[:, None]
            pip = J.sum(0); LY = (J.T @ LX) / pip[:, None]
            LY = np.minimum(1, LY + rng.uniform(-.03, .15, size=LY.shape))
        else:
            pip = rng.dirichlet(np.ones(Mp)); LY = rng.uniform(0, 1, size=(Mp, d))
        # primal: J>=0, row sums pi, column sums pip, sum_m J_mm' LX_m <= pip_m' LY_m'
        nv = M * Mp; Aeq = []; beq = []
        for m in range(M):
            r = np.zeros(nv); r[m * Mp:(m + 1) * Mp] = 1; Aeq.append(r); beq.append(pi[m])
        for mp in range(Mp - 1):
            r = np.zeros(nv); r[mp::Mp] = 1; Aeq.append(r); beq.append(pip[mp])
        Aub = []; bub = []
        for mp in range(Mp):
            for k in range(d):
                r = np.zeros(nv); r[mp::Mp] = LX[:, k]; Aub.append(r); bub.append(pip[mp] * LY[mp, k])
        prim = linprog(np.zeros(nv), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=[(0, None)] * nv, method='highs')
        feas = prim.status == 0
        # dual certificate: a (M), b (Mp), c (Mp x d, >=0); a_m + b_m' + c_m'.LX_m >= 0; minimise pi.a + pip.b + sum pip c.LY (boxed)
        na = M; nb = Mp; nc = Mp * d; nv2 = na + nb + nc
        A2 = []; b2 = []
        for m in range(M):
            for mp in range(Mp):
                r = np.zeros(nv2); r[m] = -1; r[na + mp] = -1; r[na + nb + mp * d: na + nb + (mp + 1) * d] = -LX[m]
                A2.append(r); b2.append(0)
        cobj = np.concatenate([pi, pip, (pip[:, None] * LY).ravel()])
        bnds = [(-1, 1)] * (na + nb) + [(0, 1)] * nc
        dual = linprog(cobj, A_ub=A2, b_ub=b2, bounds=bnds, method='highs')
        cert = dual.fun < -1e-9
        agree += int(feas != cert); n += 1; feas_count += feas
        if cert:
            a = dual.x[:na]; b = dual.x[na:na + nb]; c = dual.x[na + nb:].reshape(Mp, d)
            psi = lambda lam: np.max(-b - c @ lam)
            PsiX = sum(pi[m] * psi(LX[m]) for m in range(M)); PsiY = sum(pip[mp] * psi(LY[mp]) for mp in range(Mp))
            sep_n += 1; sep_ok += int(PsiX < PsiY - 1e-12)
    ok = agree == n and 30 < feas_count < n - 30 and sep_ok == sep_n
    print(f"V30-07 leakage-preorder completeness: primal-feasible XOR certificate in {agree}/{n} "
          f"(feasible={feas_count}); certificate psi separates in {sep_ok}/{sep_n}; passed={ok}")
    return ok


def shadow_lp(X, Y, N, W, u=None):
    """Commuting (cell-diagonal) sector. X=(pi,PR,PL), Y=(pi',PR',PL') on the same cells/collars.
    Variables K(m',y|m,x)>=0, kappa(m'|m). Constraints: sum_y K = kappa (M2), classical (M3), column-stochastic kappa.
    If u is None: minimise l1 distance of the image to Y (returns distance, duals of the image constraints).
    Else: maximise <u, image> (value function V_u)."""
    pi, PR, PL = X; M = len(pi); Mp = Y[1].shape[0] if Y is not None else u.shape[1]
    A = allowed_T(N, W)
    nK = Mp * N * M * N; nk = Mp * M
    iK = lambda mp, y, m, x: ((mp * N + y) * M + m) * N + x
    ik = lambda mp, m: nK + mp * M + m
    nimg = 2 * Mp * N
    use_err = u is None
    nv = nK + nk + (2 * nimg if use_err else 0)
    rows = []; beq = []
    Aeq = lil_matrix((M * N * Mp + M + (nimg if use_err else 0), nv)); r = 0
    for mp in range(Mp):
        for m in range(M):
            for x in range(N):
                for y in range(N): Aeq[r, iK(mp, y, m, x)] = 1
                Aeq[r, ik(mp, m)] = -1; beq.append(0); r += 1
    for m in range(M):
        for mp in range(Mp): Aeq[r, ik(mp, m)] = 1
        beq.append(1); r += 1
    img_rows = []
    if use_err:
        piY, PRY, PLY = Y
        for h, (P, PY) in enumerate(((PR, PRY), (PL, PLY))):
            for mp in range(Mp):
                for y in range(N):
                    for m in range(M):
                        for x in range(N): Aeq[r, iK(mp, y, m, x)] = pi[m] * P[m, x]
                    e = (h * Mp + mp) * N + y
                    Aeq[r, nK + nk + e] = -1; Aeq[r, nK + nk + nimg + e] = 1
                    beq.append(piY[mp] * PY[mp, y]); img_rows.append(r); r += 1
    bounds = [(0, None) if A[y, x] else (0, 0) for mp in range(Mp) for y in range(N) for m in range(M) for x in range(N)]
    bounds += [(0, 1)] * nk + ([(0, None)] * (2 * nimg) if use_err else [])
    c = np.zeros(nv)
    if use_err:
        c[nK + nk:] = 1
    else:
        for h, P in enumerate((PR, PL)):
            for mp in range(Mp):
                for y in range(N):
                    for m in range(M):
                        for x in range(N): c[iK(mp, y, m, x)] -= u[h, mp, y] * pi[m] * P[m, x]
    res = linprog(c, A_eq=Aeq.tocsr(), b_eq=np.array(beq), bounds=bounds, method='highs')
    if use_err:
        duals = res.eqlin.marginals[img_rows].reshape(2, Mp, N)
        return res.fun, duals
    return -res.fun


def rand_shadow(M, N, skin):
    pi = rng.dirichlet(np.ones(M)); PR = rng.dirichlet(np.ones(N) * .5, size=M); PL = rng.dirichlet(np.ones(N) * .5, size=M)
    if skin:
        n = np.arange(N)
        for m in range(M):
            a = rng.uniform(.5, 2)
            PR[m] *= np.exp(a * n); PR[m] /= PR[m].sum(); PL[m] *= np.exp(-a * n); PL[m] /= PL[m].sum()
    return pi, PR, PL


def c08_full_completeness_commuting():
    """Thm 4(b) in the commuting sector: X->Y (LP) <=> V_u(X) >= V_u(Y) for all u; the separating u is a violated monotone."""
    N, W = 6, 2; n = 0; sep = 0; infeas = 0; mono_viol = 0; feas = 0
    for _ in range(60):
        M = int(rng.integers(1, 3)); Mp = int(rng.integers(1, 3))
        X = rand_shadow(M, N, skin=True)
        if rng.random() < .5:            # image of X under a random free processing (reachable by construction)
            A = allowed_T(N, W); kap = rng.dirichlet(np.ones(Mp), size=M).T; pip = kap @ X[0]
            Ts = [[rand_T(A) for m in range(M)] for mp in range(Mp)]
            PRY = np.array([sum(kap[mp, m] * X[0][m] * Ts[mp][m] @ X[1][m] for m in range(M)) / pip[mp] for mp in range(Mp)])
            PLY = np.array([sum(kap[mp, m] * X[0][m] * Ts[mp][m] @ X[2][m] for m in range(M)) / pip[mp] for mp in range(Mp)])
            Y = (pip, PRY, PLY)
        else:
            Y = rand_shadow(Mp, N, skin=rng.random() < .5)
        dist, duals = shadow_lp(X, Y, N, W); n += 1
        if dist > 1e-7:
            infeas += 1
            u = duals                    # separating payoff = gradient of the distance w.r.t. the target (LP dual)
            VX = shadow_lp(X, None, N, W, u=u); VY = shadow_lp(Y, None, N, W, u=u)
            sep += int(VX < VY - 1e-9)
        else:
            feas += 1
            for _k in range(3):
                u = rng.normal(size=(2, Y[1].shape[0], N))
                VX = shadow_lp(X, None, N, W, u=u); VY = shadow_lp(Y, None, N, W, u=u)
                mono_viol += int(VX < VY - 1e-7)
    ok = sep == infeas and mono_viol == 0 and feas > 10 and infeas > 10
    print(f"V30-08 complete monotones (commuting sector): reachable={feas}, unreachable={infeas}; "
          f"separating V_u found in {sep}/{infeas}; monotone violations on reachable pairs={mono_viol}; passed={ok}")
    return ok


def c09_conversions():
    """Prop 4 (dilution band), Thm 6 (HN incomparability, exact), Prop 5 (approximate-conversion lower bound) + LP attainment."""
    L = 24; W = 6
    band = 0.0
    for g in (.3, .6):
        P, Q = hn_law(L, g)
        for j in (0, 7, 19):
            for t in (.5, 1e-1, 1e-3, 1e-6, 1e-10):
                bulk = np.zeros(L); bulk[L // 2] = 1.0
                p2 = (1 - t) * P[j] + t * bulk; q2 = (1 - t) * Q[j] + t * bulk
                for w in range(1, W + 1):
                    s0 = s_score(P[j], Q[j], L, 1, w); s1 = s_score(p2, q2, L, 1, w)
                    band = max(band, abs(s1 - min(s0, 0.5 * np.log(1 / (2 * t)))) - HALF_LOG2)
    # exact incomparability: mode-averaged collar masses strictly ordered in one orientation for every g != g'
    strict_fail = 0; pairs = 0
    for g, gp in [(.6, .3), (.5, .1), (.8, .5), (.4, 0.0), (-.3, .2)]:
        for L2 in (8, 16, 24):
            P, Q = hn_law(L2, g); Pp, Qp = hn_law(L2, gp)
            for w in range(1, L2 // 2 + 1):
                cR = lambda PP, sg: np.mean([PP[j][collar(L2, sg, w)].sum() for j in range(L2)])
                # X=HN(g)->Y=HN(g') needs c_Y <= c_X in every orientation; find a strictly violated one
                viol = (cR(Pp, -1) > cR(P, -1) * (1 + 1e-12)) or (cR(Pp, 1) > cR(P, 1) * (1 + 1e-12))
                viol_rev = (cR(P, -1) > cR(Pp, -1) * (1 + 1e-12)) or (cR(P, 1) > cR(Pp, 1) * (1 + 1e-12))
                pairs += 1; strict_fail += int(not (viol and viol_rev))
    # approximate conversion HN(g)->HN(g') (single mode, commuting sector): LP minimum vs proved lower bound
    ratios = []; errs = []; gaps = []
    A = None
    for L2 in (12, 16, 20, 24):
        W2 = L2 // 4; A = allowed_T(L2, W2)
        P, Q = hn_law(L2, .8); Pp, Qp = hn_law(L2, .4); j = 0
        p, q, pp, qp = P[j], Q[j], Pp[j], Qp[j]
        nT = L2 * L2; ne = 4 * L2; nv = nT + ne
        Aeq = lil_matrix((3 * L2, nv)); beq = np.zeros(3 * L2)
        for x in range(L2):
            for y in range(L2): Aeq[x, y * L2 + x] = 1
            beq[x] = 1
        for h, (src, tgt) in enumerate(((p, pp), (q, qp))):
            for y in range(L2):
                rr = L2 + h * L2 + y
                for x in range(L2): Aeq[rr, y * L2 + x] = src[x]
                Aeq[rr, nT + h * 2 * L2 + y] = -1; Aeq[rr, nT + h * 2 * L2 + L2 + y] = 1; beq[rr] = tgt[y]
        c = np.zeros(nv); c[nT:] = 1
        bnds = [(0, None) if A[y, x] else (0, 0) for y in range(L2) for x in range(L2)] + [(0, None)] * ne
        res = linprog(c, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bnds, method='highs')
        lb = sum(2 * max(max(tg[collar(L2, sg, w)].sum() - sr[collar(L2, sg, w)].sum() for sg in (1, -1) for w in range(1, W2 + 1)), 0)
                 for sr, tg in ((p, pp), (q, qp)))
        errs.append(res.fun); ratios.append(res.fun / lb); gaps.append(res.fun - lb)
    decay = all(errs[i + 1] < errs[i] / 5 for i in range(len(errs) - 1))
    ok = band <= 1e-9 and strict_fail == 0 and min(gaps) >= -1e-8 and decay   # HiGHS tolerance ~1e-9 absolute
    print(f"V30-09 conversions: dilution band excess={band:.1e}; HN(g),HN(g') mutually non-convertible in {pairs - strict_fail}/{pairs} "
          f"(L,w,g,g') cases; approx. HN(.8)->HN(.4) min l1 error {', '.join(f'{e:.1e}' for e in errs)} (L=12..24), "
          f"ratio to proved lower bound {', '.join(f'{r:.6f}' for r in ratios)}; passed={ok}")
    return ok


def main():
    checks = [c01_reciprocity, c02_spectral_sensitivity, c03_monotones_and_golden_rule, c04_fidelity_monotone,
              c05_calibration, c06_controls, c07_leakage_completeness, c08_full_completeness_commuting, c09_conversions]
    passed = 0
    for f in checks:
        try:
            passed += int(bool(f()))
        except Exception as exc:
            print(f"ERROR in {f.__name__}: {exc!r}")
    print(f"SUMMARY: {passed}/{len(checks)} V30 checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
