#!/usr/bin/env python3
"""Independent numerical checks for the V32 manuscript (NHSE_QRT_V32.tex).

Every check targets a load-bearing statement and prints the quantity it tests.
Numerical tolerances are stated explicitly; nothing is forced to pass.
Objects have uniform weights. Free processings (Definition 7) are convex mixtures of label-blind processings with
collar non-expansion (C) and density non-increase (D). Two resource layers: direction (paired score s) and
accumulation (collar Bessel bound B).
Run:  python nhse_qrt_v32_checks.py   (numpy, scipy)
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.linalg import eig, eigh, inv, norm, svd, eigvalsh
from scipy.linalg import block_diag, expm
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
    print(f"V32-01 reciprocity: collar-reciprocal max s={s_max:.1e}; U^T=U: max |cell law R-L|={diff:.1e}, max s={s_gen:.1e}; "
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
    print(f"V32-02 spectral sensitivity: min[sqrt(lR)+sqrt(lL)-F]={worst_F:.3f}; min[log kappa-(s-log2/2)]={worst_k:.3f}; "
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
    print(f"V32-03 leakage monotones: tests={tests}, violations={bad}; maps collar-non-expanding={ok_maps}/600; "
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
    print(f"V32-04 Phi_Q monotone (violations={bad}) and additive (err={worst_add:.1e}); passed={ok}")
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
    print(f"V32-05 calibration: gauge bound min slack={worst:.3f} over {nt} (mode,w) pairs ({n_big} with bound>1); "
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
    print(f"V32-06 controls: Hermitian max s={sh:.1e}; periodic HN max s={sp:.1e}; bulk bond (L, mean log kappa, max kappa, max_w s): "
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
    print(f"V32-07 leakage-preorder completeness: primal-feasible XOR certificate in {agree}/{n} "
          f"(feasible={feas_count}); certificate psi separates in {sep_ok}/{sep_n}; passed={ok}")
    return ok


def acc_lp(X, Y, N, W, Mp=None, u=None, use_D=True, rand_obj=None):
    """Commuting (cell-diagonal) sector, uniform weights (Definition 1). Free processings (Definition 7):
    Theta = sum_f K_f (x) A_f over deterministic relabellings f:[M]->[M']; A_f >= 0 with the collar-non-expanding
    pattern (C), column sums t_f, sum_f t_f = 1, uniform output weights, and (D): (M'/M) sum_f rowsum_y(A_f) <= 1
    on every collar cell y.  Y given: minimise the l1 distance of the image to Y (returns distance, duals of the
    image rows).  u given: maximise <u, image>.  rand_obj given: image of a random vertex (a reachable target)."""
    PR, PL = X; M = PR.shape[0]
    if Mp is None: Mp = Y[0].shape[0] if Y is not None else (u.shape[1] if u is not None else rand_obj.shape[1])
    A = allowed_T(N, W)
    collar_cells = [c for c in range(N) if c < W or c >= N - W]
    fs = list(itertools.product(range(Mp), repeat=M)); F = len(fs)
    nA = F * N * N; iA = lambda f, y, x: (f * N + y) * N + x
    nimg = 2 * Mp * N; use_err = Y is not None
    nv = nA + F + (2 * nimg if use_err else 0)
    Aeq = lil_matrix((F * N + 1 + (Mp - 1) + (nimg if use_err else 0), nv)); beq = []; r = 0
    for f in range(F):
        for x in range(N):
            for y in range(N): Aeq[r, iA(f, y, x)] = 1
            Aeq[r, nA + f] = -1; beq.append(0); r += 1
    for f in range(F): Aeq[r, nA + f] = 1
    beq.append(1); r += 1
    for mp in range(Mp - 1):                                     # uniform output weights
        for f, fm in enumerate(fs): Aeq[r, nA + f] = sum(1 for m in range(M) if fm[m] == mp)
        beq.append(M / Mp); r += 1
    def coeffs(h, mp, y):
        P = PR if h == 0 else PL; out = {}
        for f, fm in enumerate(fs):
            src = [m for m in range(M) if fm[m] == mp]
            if src:
                wx = sum(P[m] for m in src) / M
                for x in range(N): out[iA(f, y, x)] = wx[x]
        return out
    img_rows = []
    if use_err:
        QR_, QL_ = Y
        for h, Q in enumerate((QR_, QL_)):
            for mp in range(Mp):
                for y in range(N):
                    for k, v in coeffs(h, mp, y).items(): Aeq[r, k] = v
                    e = (h * Mp + mp) * N + y
                    Aeq[r, nA + F + e] = -1; Aeq[r, nA + F + nimg + e] = 1
                    beq.append(Q[mp, y] / Mp); img_rows.append(r); r += 1
    bounds = [(0, None) if A[y, x] else (0, 0) for f in range(F) for y in range(N) for x in range(N)]
    bounds += [(0, 1)] * F + ([(0, None)] * (2 * nimg) if use_err else [])
    c = np.zeros(nv)
    if use_err:
        c[nA + F:] = 1
    else:
        uu = u if u is not None else rand_obj
        for h in (0, 1):
            for mp in range(Mp):
                for y in range(N):
                    for k, v in coeffs(h, mp, y).items(): c[k] -= uu[h, mp, y] * v
    kw = dict(A_eq=Aeq.tocsr(), b_eq=np.array(beq), bounds=bounds, method='highs')
    if use_D:                                                    # axiom (D)
        Aub = lil_matrix((len(collar_cells), nv))
        for i, y in enumerate(collar_cells):
            for f in range(F):
                for x in range(N): Aub[i, iA(f, y, x)] = Mp / M
        kw.update(A_ub=Aub.tocsr(), b_ub=np.ones(len(collar_cells)))
    res = linprog(c, **kw)
    if use_err:
        return res.fun, res.eqlin.marginals[img_rows].reshape(2, Mp, N)
    if rand_obj is not None:
        z = res.x; img = np.zeros((2, Mp, N))
        for h in (0, 1):
            for mp in range(Mp):
                for y in range(N):
                    img[h, mp, y] = sum(v * z[k] for k, v in coeffs(h, mp, y).items()) * Mp
        img = np.clip(img, 0, None); img /= img.sum(2, keepdims=True)
        return img[0], img[1]
    return -res.fun


def c08_full_completeness_commuting():
    """Thm 9(b) in the commuting sector with uniform weights and axiom (D): X->Y (LP) <=> V_u(X) >= V_u(Y) for all u;
    the separating u is a violated monotone; no monotone violation on reachable pairs; the collar Bessel bound B never
    increases on reachable pairs; images reachable without (D) are partly blocked by (D)."""
    N, W = 6, 2; sep = 0; infeas = 0; feas = 0; mono = 0; Bviol = 0; blocked = 0; nD = 0
    collar_cells = [c for c in range(N) if c < W or c >= N - W]
    Bc = lambda PR_: PR_.sum(0)[collar_cells].max()             # commuting sector: B = max collar-cell occupation
    for it in range(60):
        M = int(rng.integers(2, 4)); Mp = int(rng.integers(1, 4))
        PR = rng.dirichlet(np.ones(N) * .5, size=M); PL = rng.dirichlet(np.ones(N) * .5, size=M); n = np.arange(N)
        for m in range(M):
            a = rng.uniform(.3, 1.5); PR[m] *= np.exp(a * n); PR[m] /= PR[m].sum(); PL[m] *= np.exp(-a * n); PL[m] /= PL[m].sum()
        X = (PR, PL); kind = it % 3
        if kind == 0:
            Y = acc_lp(X, None, N, W, Mp=Mp, rand_obj=rng.normal(size=(2, Mp, N)))
        elif kind == 1:
            Y = acc_lp(X, None, N, W, Mp=Mp, use_D=False, rand_obj=rng.normal(size=(2, Mp, N))); nD += 1
        else:
            Y = (rng.dirichlet(np.ones(N) * .5, size=Mp), rng.dirichlet(np.ones(N) * .5, size=Mp))
        dist, duals = acc_lp(X, Y, N, W)
        if dist > 1e-7:
            infeas += 1; blocked += int(kind == 1)
            VX = acc_lp(X, None, N, W, u=duals); VY = acc_lp(Y, None, N, W, u=duals)
            sep += int(VX < VY - 1e-9)
        else:
            feas += 1; Bviol += int(Bc(Y[0]) > Bc(PR) + 1e-7)
            for _ in range(3):
                uu = rng.normal(size=(2, Y[0].shape[0], N))
                mono += int(acc_lp(X, None, N, W, u=uu) < acc_lp(Y, None, N, W, u=uu) - 1e-7)
    ok = sep == infeas and mono == 0 and Bviol == 0 and feas > 10 and infeas > 10
    print(f"V32-08 complete monotones (commuting sector, uniform weights, axiom (D)): reachable={feas}, unreachable={infeas}; "
          f"separating V_u found in {sep}/{infeas}; monotone violations on reachable pairs={mono}; B increased on reachable pairs={Bviol}; "
          f"images reachable without (D) that (D) blocks: {blocked}/{nD}; passed={ok}")
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
    print(f"V32-09 conversions: dilution band excess={band:.1e}; HN(g),HN(g') mutually non-convertible in {pairs - strict_fail}/{pairs} "
          f"(L,w,g,g') cases; approx. HN(.8)->HN(.4) min l1 error {', '.join(f'{e:.1e}' for e in errs)} (L=12..24), "
          f"ratio to proved lower bound {', '.join(f'{r:.6f}' for r in ratios)}; passed={ok}")
    return ok


# ------------------------------------------------------------------ generator level (Section 5)
def cplx(*shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def cell_proj(N, d, sigma, w):
    """projector onto the w cells nearest boundary sigma, d levels per cell"""
    P = np.zeros((N * d, N * d))
    for c in collar(N, sigma, w):
        P[c * d:(c + 1) * d, c * d:(c + 1) * d] = np.eye(d)
    return P


def biorth_mat(G):
    """eigenvalues, unit right eigenvectors (columns), matched unit left eigenvectors G^dag L = conj(lam) L"""
    lam, R = eig(G)
    Lm = inv(R).conj().T
    return lam, R / norm(R, axis=0), Lm / norm(Lm, axis=0)


def gen_chain(N, d, g):
    """generic complex cell chain with bias g (non-normal, simple spectrum); (G+G^+)/2 is a random Hermitian chain"""
    G = np.zeros((N * d, N * d), complex)
    for n in range(N):
        G[n * d:(n + 1) * d, n * d:(n + 1) * d] = cplx(d, d)
        if n + 1 < N:
            G[(n + 1) * d:(n + 2) * d, n * d:(n + 1) * d] = np.exp(g) * cplx(d, d)
            G[n * d:(n + 1) * d, (n + 1) * d:(n + 2) * d] = np.exp(-g) * cplx(d, d)
    return G


def hn_mat(L, g):
    return np.diag(np.full(L - 1, np.exp(g)), -1) + np.diag(np.full(L - 1, np.exp(-g)), 1)


def unitary(d):
    Q, R = np.linalg.qr(cplx(d, d))
    return Q * (np.diag(R) / abs(np.diag(R)))


def embedding(N, d, shiftL, extraR, d2):
    """isometry: cell-local unitary, internal enlargement d->d2, boundary moved outward (shiftL cells left, extraR right)"""
    Np = N + shiftL + extraR; J = np.eye(d2, d)
    V = np.zeros((Np * d2, N * d), complex)
    for n in range(N):
        V[(n + shiftL) * d2:(n + shiftL + 1) * d2, n * d:(n + 1) * d] = J @ unitary(d)
    return V, Np


def nullity(Mat, tol=1e-9):
    s = svd(Mat, compute_uv=False)
    return int(Mat.shape[1] - np.sum(s > tol * max(1.0, s[0])))


def hom_dim(G, Gp):
    """dim{T : Gp T = T G and Gp^dag T = T G^dag} (column-major vec)"""
    d, dp = G.shape[0], Gp.shape[0]
    A = np.kron(np.eye(d), Gp) - np.kron(G.T, np.eye(dp))
    B = np.kron(np.eye(d), Gp.conj().T) - np.kron(G.conj(), np.eye(dp))
    return nullity(np.vstack([A, B]))


def response_dim(R, L, Rp, Lp):
    """dim{T : T R_m in span R'_m and T L_m in span L'_m for all listed m}: the Kraus condition of label-blind (M1)"""
    dp = Rp.shape[0]; rows = []
    for m in range(R.shape[1]):
        for x, y in ((R[:, m], Rp[:, m]), (L[:, m], Lp[:, m])):
            rows.append(np.kron(x.reshape(1, -1), np.eye(dp) - np.outer(y, y.conj())))
    return nullity(np.vstack(rows))


def window(Gp, targets):
    """right/left unit eigenvectors of Gp at the eigenvalues closest to `targets` (in that order)"""
    lam, R, Lm = biorth_mat(Gp)
    idx = [int(np.argmin(abs(lam - t))) for t in targets]
    return R[:, idx], Lm[:, idx]


def c10_generator_realization():
    """Thm 5: (a) embeddings G'=V h(G) V^+ + QKQ are free; (b) for full-spectrum data the Kraus operators of a label-blind
    free morphism are joint intertwiners, a window that is not reducing admits none, and the solution space is spanned by
    one isometry per irreducible block (Schur); functional calculus leaves the data unchanged."""
    out = {}
    # (a) embedding: eigen-data, collar non-expansion, leakage domination, condition-number invariance
    N, d, d2 = 5, 2, 3
    G = gen_chain(N, d, .25); V, Np = embedding(N, d, 2, 1, d2)
    Q = np.eye(Np * d2) - V @ V.conj().T
    Gp = V @ G @ V.conj().T + Q @ cplx(Np * d2, Np * d2) @ Q
    lam, R, Lm = biorth_mat(G); Rp, Lp = window(Gp, lam)
    out['eig'] = max(1 - abs(np.vdot(Rp[:, m], V @ R[:, m])) for m in range(len(lam))) + \
                 max(1 - abs(np.vdot(Lp[:, m], V @ Lm[:, m])) for m in range(len(lam)))
    out['C'] = min(eigvalsh(cell_proj(N, d, s, w) - V.conj().T @ cell_proj(Np, d2, s, w) @ V).min()
                   for s in (1, -1) for w in (1, 2))
    marg = np.inf; kap = 0.0
    for m in range(len(lam)):
        kap = max(kap, abs(abs(np.vdot(Lp[:, m], Rp[:, m])) / abs(np.vdot(Lm[:, m], R[:, m])) - 1))
        for s in (1, -1):
            for w in (1, 2):
                for x, xp, sg in ((R[:, m], Rp[:, m], s), (Lm[:, m], Lp[:, m], -s)):
                    marg = min(marg, norm(cell_proj(N, d, sg, w) @ x) ** 2 - norm(cell_proj(Np, d2, sg, w) @ xp) ** 2)
    out['dom'] = marg; out['kap'] = kap
    # (b) rigidity: commutant of {G,G^+} and intertwiner space vs response-constraint space
    L = 8
    out['comm_HN'] = hom_dim(hn_mat(L, .3), hn_mat(L, .3)); out['comm_herm'] = hom_dim(hn_mat(L, 0), hn_mat(L, 0))
    G1, G2 = gen_chain(2, 2, .2), gen_chain(2, 2, .2) + 9 * np.eye(4)
    Gb = block_diag(G1, G2); out['comm_block'] = hom_dim(Gb, Gb)
    rows = []
    for name, G0 in (("generic", gen_chain(4, 2, .2)), ("two blocks", Gb), ("Hermitian", (lambda A: A + A.conj().T)(cplx(8, 8)))):
        V, Np = embedding(4, 2, 1, 1, 3); Q = np.eye(Np * 3) - V @ V.conj().T
        K = cplx(Np * 3, Np * 3); K = K + K.conj().T if name == "Hermitian" else K
        Gp = V @ G0 @ V.conj().T + Q @ K @ Q
        lam, R, Lm = biorth_mat(G0); Rp, Lp = window(Gp, lam)
        rows.append((name, response_dim(R, Lm, Rp, Lp), hom_dim(G0, Gp)))
    out['rows'] = rows
    # a window that is invariant but not reducing (one-way coupling either way) admits no Kraus operator at all
    G0 = gen_chain(4, 2, .2); lam, R, Lm = biorth_mat(G0); D = 8
    nr = []
    for up in (True, False):
        B = cplx(D, 4) if up else cplx(4, D)
        K = cplx(4, 4) + 12 * np.eye(4)
        Gp = np.block([[G0, B], [np.zeros((4, D)), K]]) if up else np.block([[G0, np.zeros((D, 4))], [B, K]])
        Rp, Lp = window(Gp, lam)
        nr.append(response_dim(R, Lm, Rp, Lp))
    out['nonreducing'] = nr
    # functional calculus: same eigen-data, same score
    G = hn_mat(10, .2); lam, R, Lm = biorth_mat(G); dev = 0.0
    for f, fs in ((lambda M: expm(-.7j * M), lambda x: np.exp(-.7j * x)),
                  (lambda M: inv((2.3 + .4j) * np.eye(10) - M), lambda x: 1 / (2.3 + .4j - x)),
                  (lambda M: M @ M + 3 * M, lambda x: x * x + 3 * x)):
        Rf, Lf = window(f(G), fs(lam))
        dev = max(dev, max(1 - abs(np.vdot(Rf[:, m], R[:, m])) for m in range(10)),
                  max(1 - abs(np.vdot(Lf[:, m], Lm[:, m])) for m in range(10)))
    out['fc'] = dev
    ok = (out['eig'] < 1e-9 and out['C'] > -1e-12 and out['dom'] > -1e-12 and out['kap'] < 1e-9
          and out['comm_HN'] == 1 and out['comm_herm'] == L and out['comm_block'] == 2
          and all(a == b for _, a, b in rows) and [r[1] for r in rows] == [1, 2, 8]
          and nr == [0, 0] and dev < 1e-9)
    print(f"V32-10 generator level: embedding eigen-data err={out['eig']:.1e}, min eig(Pi-V^+Pi'V)={out['C']:.1e}, "
          f"min collar-weight loss={out['dom']:.1e}, |kappa'/kappa-1|={out['kap']:.1e}; dim commutant{{G,G^+}}: "
          f"HN={out['comm_HN']}, Hermitian chain={out['comm_herm']}, two blocks={out['comm_block']}; "
          f"dim(label-blind Kraus space) vs dim(joint intertwiners): " + ", ".join(f"{a} {b}/{c}" for a, b, c in rows) +
          f"; invariant non-reducing windows: {nr}; functional calculus err={dev:.1e}; passed={ok}")
    return ok


def c11_not_free():
    """Remark 6: one-way coupling raises kappa_m by exactly sqrt(1+|x_m|^2); the imaginary gauge field raises kappa from 1;
    the reciprocal perturbation G -> G - delta*A maps HN(g) to c*HN(g_delta), g_delta > g, and raises the faithful monotone;
    mode tagging is excluded (pairwise non-orthogonal R_m, unequal kappa_m)."""
    N, d = 5, 2; G = gen_chain(N, d, .2); D = N * d
    lam, R, Lm = biorth_mat(G)
    K = cplx(4, 4) + 12 * np.eye(4); C = cplx(4, D)
    Gc = np.block([[G, np.zeros((D, 4))], [C, K]])
    Rc, Lc = window(Gc, lam)
    fac = 0.0; left = 0.0; grow = np.inf
    for m in range(D):
        x = np.linalg.solve(lam[m] * np.eye(4) - K, C @ R[:, m])
        left = max(left, 1 - abs(np.vdot(Lc[:, m], np.concatenate([Lm[:, m], np.zeros(4)]))))
        k0 = 1 / abs(np.vdot(Lm[:, m], R[:, m])); k1 = 1 / abs(np.vdot(Lc[:, m], Rc[:, m]))
        fac = max(fac, abs(k1 / k0 - np.sqrt(1 + norm(x) ** 2)))
        grow = min(grow, k1 / k0 - 1)
    # imaginary gauge field on the Hermitian chain
    Lh = 12; S = np.diag(np.exp(.25 * np.arange(Lh)))
    lg, Rg, Lg = biorth_mat(S @ hn_mat(Lh, 0) @ inv(S))
    kap_g = min(1 / abs(np.vdot(Lg[:, j], Rg[:, j])) for j in range(Lh))
    # reciprocal perturbation: eigen-data identity at small L, monotone increase from the exact laws
    g, dl = .3, .4; gd = .5 * np.log((np.exp(g) - dl) / (np.exp(-g) - dl))
    A = hn_mat(10, 0); l1, R1, L1 = biorth_mat(hn_mat(10, g) - dl * A); l2, R2, L2 = biorth_mat(hn_mat(10, gd))
    o1 = np.argsort(l1.real); o2 = np.argsort(l2.real)
    same = max(np.abs(np.abs(R1[:, o1]) ** 2 - np.abs(R2[:, o2]) ** 2).max(), np.abs(np.abs(L1[:, o1]) ** 2 - np.abs(L2[:, o2]) ** 2).max())
    Sfa = lambda P, Q, Lc_, W: np.mean([max(s_score(P[j], Q[j], Lc_, sg, w) for sg in (1, -1) for w in range(1, W + 1))
                                        for j in range(Lc_)])
    rows = []
    for Lc_ in (20, 40, 100):
        W = Lc_ // 2; P0, Q0 = hn_law(Lc_, g); P1, Q1 = hn_law(Lc_, gd)
        rows.append((Lc_, Sfa(P0, Q0, Lc_, W), Sfa(P1, Q1, Lc_, W)))
    # tagging: excluded when right eigenvectors are pairwise non-orthogonal and kappa_m are not all equal
    lt, Rt, Ltt = biorth_mat(hn_mat(8, .3))
    ov = min(abs(np.vdot(Rt[:, a], Rt[:, b])) for a in range(8) for b in range(8) if a != b)
    ks = [1 / abs(np.vdot(Ltt[:, m], Rt[:, m])) for m in range(8)]
    ok = (left < 1e-9 and fac < 1e-8 and grow > 0 and kap_g > 1.5 and gd > g and same < 1e-8
          and all(b > a for _, a, b in rows) and ov > 1e-3 and max(ks) / min(ks) > 1.01)
    print(f"V32-11 not free: one-way coupling L'=(L,0) err={left:.1e}, kappa'/kappa=sqrt(1+|x|^2) err={fac:.1e}, "
          f"min increase {grow:.2e}; imaginary gauge g=.25 on Hermitian chain: min kappa 1 -> {kap_g:.2f}; "
          f"reciprocal G-{dl}A: HN({g}) -> HN({gd:.3f}) cell laws err={same:.1e}, faithful monotone S (L, before, after): "
          + ", ".join(f"({a},{b:.2f},{c:.2f})" for a, b, c in rows) +
          f"; tagging: min |<R_a|R_b>|={ov:.3f}, kappa_max/kappa_min={max(ks)/min(ks):.2f} (excluded); passed={ok}")
    return ok


# ------------------------------------------------------------------ accumulation layer (Sections 3-4, 7)
def collars_1d(N, d, W):
    """{(sigma,w): projector} for a chain of N cells with d levels per cell"""
    out = {}
    for s in (1, -1):
        for w in range(1, W + 1):
            out[(s, w)] = cell_proj(N, d, s, w)
    return out


def gen_states(G, window=None):
    """uniform-weight generator datum with simple spectrum: list of (rho^R, rho^L)"""
    lam, R, Lm = biorth_mat(G)
    idx = range(len(lam)) if window is None else window
    return [(np.outer(R[:, m], R[:, m].conj()), np.outer(Lm[:, m], Lm[:, m].conj())) for m in idx]


def eigenspace_states(G, tol=1e-6):
    """semisimple spectrum (Definition 2): k responses Q^R/k, Q^L/k per eigenvalue of multiplicity k"""
    lam, R = eig(G); Linv = inv(R); D = len(lam); used = np.zeros(D, bool); out = []; mults = []
    scale = max(1.0, np.abs(lam).max())
    for i in range(D):
        if used[i]: continue
        cl = [j for j in range(D) if not used[j] and abs(lam[j] - lam[i]) < tol * scale]
        used[cl] = True; k = len(cl); mults.append(k)
        QR_, _ = np.linalg.qr(R[:, cl]); QL_, _ = np.linalg.qr(Linv[cl, :].conj().T)
        out += [(QR_ @ QR_.conj().T / k, QL_ @ QL_.conj().T / k)] * k
    return out, mults


def B_of(states, col):
    """collar Bessel bound B_sigma(w) = || Pi S Pi ||, S = sum_m rho^R_m"""
    S = sum(a for a, _ in states)
    return {k: eigvalsh(P @ S @ P).max() for k, P in col.items()}


def scores_of(states, col):
    out = []
    for a, b in states:
        row = {}
        for (s, w), P in col.items():
            lR = 1 - np.trace(P @ a).real; lL = 1 - np.trace(col[(-s, w)] @ b).real
            row[(s, w)] = max(0.0, -0.5 * np.log(max(lR + lL, 1e-300)))
        out.append(row)
    return out


def hn_exact_states(L, g):
    n = np.arange(1, L + 1); out = []
    for j in range(1, L + 1):
        phi = np.sin(np.pi * j * n / (L + 1)); r = np.exp(g * n) * phi; l = np.exp(-g * n) * phi
        r /= norm(r); l /= norm(l); out.append((np.outer(r, r), np.outer(l, l)))
    return out


def gdse_triangle(N, gamma, dis):
    """reciprocal 2D lattice H(k) = 2 cos kx + 2i gamma cos ky on x,y >= 0, x+y <= N-1, weak real disorder"""
    sites = [(x, y) for x in range(N) for y in range(N) if x + y <= N - 1]
    idx = {s: i for i, s in enumerate(sites)}
    H = np.zeros((len(sites), len(sites)), complex)
    for (x, y), i in idx.items():
        for dx, dy, t in ((1, 0, 1.0), (0, 1, 1j * gamma)):
            j = idx.get((x + dx, y + dy))
            if j is not None:
                H[i, j] = H[j, i] = t
    H += np.diag(np.random.default_rng(7).uniform(-dis, dis, len(sites)))
    return H, sites


def z2_skin(N, g, delta):
    """HN(g) and HN(g)^T coupled antisymmetrically; cell-major basis 2n+s; G^T = U G U^+ with U = 1 (x) i sigma_y"""
    A = hn_mat(N, g); J = np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    Hs = np.block([[A, delta * J], [delta * J, A.T]])
    perm = np.array([s * N + n for n in range(N) for s in range(2)])
    H = Hs[np.ix_(perm, perm)]
    U = np.kron(np.eye(N), np.array([[0, 1], [-1, 0]]))
    assert np.allclose(H.T, U @ H @ U.conj().T) and np.allclose(U.T, -U)
    return H


def bulk_state(N, d, W):
    v = np.zeros(N * d, complex)
    for c in range(W, N - W):
        v[c * d:(c + 1) * d] = cplx(d)
    v /= norm(v)
    return np.outer(v, v.conj())


def c12_accumulation_values():
    """Accumulation definition, Corollary HN and the reciprocal-accumulation corollary: free side B <= 1; HN, Z2 and GDSE accumulate (B grows with size) while
    reciprocal ones carry no direction; non-normal chains without skin keep B bounded; HN and Z2 obey the proved bounds."""
    out = {}
    Nh, dh = 8, 2; G = gen_chain(Nh, dh, 0); G = (G + G.conj().T) / 2
    out['herm'] = max(B_of(gen_states(G), collars_1d(Nh, dh, 4)).values())
    Ls = 24; Hs = np.zeros((Ls, Ls))
    for i in range(Ls - 1): Hs[i, i + 1] = Hs[i + 1, i] = 0.3 if i % 2 == 0 else 1.0
    ev = eigvalsh(Hs); edge = list(np.argsort(abs(np.linalg.eigvals(Hs)))[:2])
    out['edge'] = max(B_of(gen_states(Hs, window=edge), collars_1d(Ls, 1, 3)).values())
    hnrows = []; bound_ok = True; nb = 0
    for L in (20, 40, 80):
        st = hn_exact_states(L, 0.3); B = B_of(st, collars_1d(L, 1, L // 2))
        hnrows.append((L, B[(1, 2)], max(B[(-1, w)] for w in range(1, L // 2 + 1))))
        for g in (0.3, 0.5):
            stg = hn_exact_states(L, g); Bg = B_of(stg, collars_1d(L, 1, L // 2))
            for w in range(1, L // 2 + 1):
                eps = (L + 1) ** 3 * np.exp(-2 * g * w) / 8
                cR = np.mean([np.trace(cell_proj(L, 1, 1, w) @ a).real for a, _ in stg])
                nb += 1; bound_ok &= (cR >= 1 - eps - 1e-12) and (Bg[(1, w)] >= (L / w) * (1 - eps) - 1e-9)
    z0, m0 = eigenspace_states(z2_skin(14, 0.5, 0.0)); colz = collars_1d(14, 2, 7)
    Bz0 = B_of(z0, colz); sz0 = max(max(r.values()) for r in scores_of(z0, colz))
    z_ok = all(Bz0[(s, w)] >= (14 / w) * (1 - 15 ** 3 * np.exp(-1.0 * w) / 8) - 1e-9 for s in (1, -1) for w in range(1, 8)) and set(m0) == {2}
    zrows = []
    for Nz in (8, 14, 20):
        st, ml = eigenspace_states(z2_skin(Nz, 0.5, 0.3)); col = collars_1d(Nz, 2, 2)
        B = B_of(st, col); zrows.append((Nz, B[(1, 2)], B[(-1, 2)], max(max(r.values()) for r in scores_of(st, col)), sorted(set(ml))))
    grows = []
    for Nt in (12, 18, 24):
        res = []
        for gam in (0.6, 0.0):
            H, sites = gdse_triangle(Nt, gam, 0.05); st = gen_states(H)
            Pd = np.diag([1.0 if x + y > Nt - 3 else 0.0 for (x, y) in sites])
            S = sum(a for a, _ in st); res.append(eigvalsh(Pd @ S @ Pd).max())
            if gam > 0: prof = max(np.abs(np.diag(a) - np.diag(b)).max() for a, b in st)
        grows.append((Nt, res[0], res[1], prof))
    bb = []; rc = []
    for Lb in (40, 80, 160):
        Hb = hn_mat(Lb, 0).astype(complex); Hb[Lb // 2, Lb // 2 + 1] = 3.0; Hb[Lb // 2 + 1, Lb // 2] = 0.2
        bb.append(max(B_of(gen_states(Hb), collars_1d(Lb, 1, 4)).values()))
        rg = np.random.default_rng(Lb)
        Hr = hn_mat(Lb, 0).astype(complex) + np.diag(rg.uniform(-1, 1, Lb) + 1j * rg.uniform(-1, 1, Lb))
        rc.append(max(B_of(gen_states(Hr), collars_1d(Lb, 1, 4)).values()))
    ok = (abs(out['herm'] - 1) < 1e-9 and out['edge'] < 1 + 1e-9
          and hnrows[1][1] > 1.8 * hnrows[0][1] and hnrows[2][1] > 1.8 * hnrows[1][1] and max(r[2] for r in hnrows) < 0.1 and hnrows[-1][2] < 1e-6 and bound_ok
          and z_ok and sz0 < 1e-6 and all(r[3] < 1e-6 and r[4] == [2] for r in zrows) and zrows[-1][1] > 1.5 * zrows[0][1]
          and all(r[3] < 1e-10 and r[2] < 1 + 1e-6 for r in grows) and grows[-1][1] > 1.5 * grows[0][1]
          and bb[-1] < 1.2 * bb[0] and rc[-1] < 1.5 * max(rc[:2]))
    print(f"V32-12 accumulation: free side B = {out['herm']:.3f} (Hermitian, complete), {out['edge']:.3f} (SSH edge-state window); "
          f"HN g=0.3 (L, B(+,2), max B(-,w)): " + ", ".join(f"({a},{b:.1f},{c:.0e})" for a, b, c in hnrows) +
          f"; HN accumulation bounds (Corollary HN) on c^R and B hold in {nb}/{nb} (L,g,w) cases: {bound_ok}; Z2 decoupled (N=14): reciprocal-accumulation corollary bound {z_ok}, max s={sz0:.0e}; "
          "Z2 coupled delta=0.3 (N, B(+,2), B(-,2), max s, mult.): " + ", ".join(f"({a},{b:.2f},{c:.2f},{d:.0e},{e})" for a, b, c, d, e in zrows) +
          "; GDSE triangle (N, B diag edge gamma=0.6, same at gamma=0, |R|^2-|L|^2): " + ", ".join(f"({a},{b:.1f},{c:.2f},{d:.0e})" for a, b, c, d in grows) +
          "; no skin, max B for L=40,80,160: bulk bond " + ", ".join(f"{x:.2f}" for x in bb) + ", 1D reciprocal random potential "
          + ", ".join(f"{x:.2f}" for x in rc) + f"; passed={ok}")
    return ok


def c13_accumulation_monotone():
    """Accumulation cannot grow (lemma) and axiom (D): B never increases under free processings (mixtures over relabellings, appending free
    objects, uniform merges, dilution, embeddings, erasure); coarse-graining and label splitting violate (D) and raise B."""
    viol = 0; tests = 0; worst = np.inf
    N, d, W = 5, 2, 2
    inputs = [gen_states(gen_chain(N, d, g)) for g in (0.0, 0.4, 0.9)] + [eigenspace_states(z2_skin(N, 0.5, 0.3))[0]]
    for trial in range(90):
        cur = inputs[trial % len(inputs)]; cn, cd = N, d; col = collars_1d(cn, cd, W); B0 = B_of(cur, col)
        for step in range(int(rng.integers(2, 5))):
            M = len(cur); op = int(rng.integers(0, 6))
            if op == 0:      # mixture over label permutations and cell-local unitaries ((D) with equality)
                K = 3; ps = rng.dirichlet(np.ones(K)); perms = [rng.permutation(M) for _ in range(K)]
                Us = [np.block([[unitary(cd) if i == j else np.zeros((cd, cd)) for j in range(cn)] for i in range(cn)]) for _ in range(K)]
                cur = [tuple(sum(pk * U @ cur[int(np.where(pr == mp)[0][0])][h] @ U.conj().T for pk, pr, U in zip(ps, perms, Us)) for h in (0, 1))
                       for mp in range(M)]
            elif op == 1:    # append a free object: internal factor with r orthonormal states (M' = rM)
                dF = 2; r = int(rng.integers(1, dF + 1)); E = np.eye(dF)
                cur = [(np.kron(a, np.outer(E[n], E[n])), np.kron(b, np.outer(E[n], E[n]))) for a, b in cur for n in range(r)]
                cd = cd * dF; col = collars_1d(cn, cd, W)
            elif op == 2 and M % 2 == 0:   # uniform merge of random pairs (M' = M/2)
                pr = rng.permutation(M)
                cur = [tuple((cur[pr[2 * i]][h] + cur[pr[2 * i + 1]][h]) / 2 for h in (0, 1)) for i in range(M // 2)]
            elif op == 3:    # dilution with a bulk state
                t = rng.uniform(0, .5); om = bulk_state(cn, cd, W); cur = [((1 - t) * a + t * om, (1 - t) * b + t * om) for a, b in cur]
            elif op == 4:    # boundary-compatible embedding
                sl, er = int(rng.integers(0, 2)), int(rng.integers(0, 2)); d2 = cd + int(rng.integers(0, 2))
                V, Np = embedding(cn, cd, sl, er, d2); ncol = collars_1d(Np, d2, W)
                for k in col:
                    worst = min(worst, eigvalsh(col[k] - V.conj().T @ ncol[k] @ V).min(), eigvalsh(ncol[k] - ncol[k] @ V @ V.conj().T @ ncol[k]).min())
                cur = [(V @ a @ V.conj().T, V @ b @ V.conj().T) for a, b in cur]; cn, cd, col = Np, d2, ncol
            else:            # erasure to a bulk state (M' = 1)
                om = bulk_state(cn, cd, W); cur = [(om, om)]
            B1 = B_of(cur, col)
            for key in B1:
                tests += 1; viol += int(B1[key] > B0[key] + 1e-9)
            B0 = B1
    # necessity: Hermitian complete data (B = 1) under (C)-channels that violate (D)
    G = gen_chain(N, d, 0); G = (G + G.conj().T) / 2; st = gen_states(G); col = collars_1d(N, d, 2)
    pt = lambda X: np.einsum('aibi->ab', X.reshape(N, d, N, d))
    Bpt = max(B_of([(pt(a), pt(b)) for a, b in st], collars_1d(N, 1, 2)).values())       # coarse-graining: (C) holds, Xi(1) = 2*1
    Bsp = max(B_of(st + st, col).values())                                                # splitting into 2M labels, Xibar = id
    ok = viol == 0 and worst > -1e-12 and Bpt > 1.9 and Bsp > 1.9
    print(f"V32-13 accumulation monotone: 90 random sequences (mixtures over relabellings, appending free objects, uniform merges, "
          f"dilution, embeddings, erasure): {tests} comparisons of B, violations={viol} ((C),(D) slack of embeddings {worst:.0e}); "
          f"(D) violated: coarse-graining an internal level B 1 -> {Bpt:.2f}, label splitting B 1 -> {Bsp:.2f}; passed={ok}")
    return ok


def c14_accumulation_relations():
    """Reciprocity theorem on eigenspaces (Z2); direction => accumulation and accumulation => non-normality."""
    st, ml = eigenspace_states(z2_skin(10, 0.5, 0.3)); col = collars_1d(10, 2, 5)
    recip = max(abs(np.trace(P @ a).real - np.trace(P @ b).real) for a, b in st for P in col.values())
    worst_a = np.inf; na = 0
    sets = [(hn_exact_states(L, g), collars_1d(L, 1, L // 4)) for L, g in ((16, .3), (30, .5), (40, .2))]
    sets += [(gen_states(gen_chain(7, 2, g)), collars_1d(7, 2, 3)) for g in (0.2, 0.6)]
    for data, colx in sets:
        B = B_of(data, colx); sc = scores_of(data, colx)
        for key, P in colx.items():
            lb = sum(1 - np.exp(-2 * s[key]) for s in sc) / np.trace(P).real
            worst_a = min(worst_a, B[key] - lb); na += 1
    rows = []; ok_b = True
    H, sites = gdse_triangle(14, 0.6, 0.05)
    for name, G, P in (("HN L=16", hn_mat(16, 0.3), cell_proj(16, 1, 1, 4)), ("random chain", gen_chain(8, 2, 0.4), cell_proj(8, 2, 1, 2)),
                       ("GDSE", H, np.diag([1.0 if x + y > 11 else 0.0 for (x, y) in sites]))):
        lam, R, Lm = biorth_mat(G); M = len(lam); S1 = R @ R.conj().T; r = int(round(np.trace(P).real))
        E = np.trace(P @ S1).real - r; k2 = (1 / np.abs(np.einsum('im,im->m', Lm.conj(), R)) ** 2).sum()
        bound = (M - r) / (M - r - E); rows.append((name, E, k2, bound)); ok_b &= k2 >= bound * (1 - 1e-9)
    ok = recip < 1e-8 and set(ml) == {2} and worst_a > -1e-9 and ok_b
    print(f"V32-14 relations: Z2 eigenspaces |Tr[Pi Q^R]-Tr[Pi Q^L]| <= {recip:.0e}, multiplicities {sorted(set(ml))}; "
          f"B - sum_m(1-e^(-2s_m))/rank Pi >= {worst_a:.3f} over {na} (data, collar) pairs; "
          + "; ".join(f"{n}: E={E:.2f}, sum kappa^2={k:.3g} >= {b:.3g}" for n, E, k, b in rows) + f"; passed={ok}")
    return ok


def main():
    checks = [c01_reciprocity, c02_spectral_sensitivity, c03_monotones_and_golden_rule, c04_fidelity_monotone,
              c05_calibration, c06_controls, c07_leakage_completeness, c08_full_completeness_commuting, c09_conversions,
              c10_generator_realization, c11_not_free, c12_accumulation_values, c13_accumulation_monotone,
              c14_accumulation_relations]
    passed = 0
    for f in checks:
        try:
            passed += int(bool(f()))
        except Exception as exc:
            print(f"ERROR in {f.__name__}: {exc!r}")
    print(f"SUMMARY: {passed}/{len(checks)} V32 checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == '__main__':
    raise SystemExit(main())
