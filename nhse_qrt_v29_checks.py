#!/usr/bin/env python3
"""Independent V29 checks. Every check targets a load-bearing statement of the V29 manuscript.
Where a statement could NOT be confirmed, the script reports it as such (no forced pass)."""
from __future__ import annotations
import numpy as np
from numpy.linalg import eig, eigh, svd
from scipy.optimize import linprog

rng = np.random.default_rng(29001)
LOG2 = np.log(2.0)

# ---------------------------------------------------------------- helpers
def collar(N, sigma, w):
    """cells in the width-w collar at boundary sigma (0=left,1=right)"""
    return list(range(w)) if sigma == 0 else list(range(N - w, N))

def leak_R(p, N, s, w): return 1 - p[collar(N, s, w)].sum()
def leak_L(q, N, s, w): return 1 - q[collar(N, 1 - s, w)].sum()      # dual collar iota(sigma)
def lam_sum(p, q, N, s, w): return 0.5 * (leak_R(p, N, s, w) + leak_L(q, N, s, w))
def g_sum(p, q, N, s, w): return -0.5 * np.log(max(lam_sum(p, q, N, s, w), 1e-300))

def sqrtm_h(M):
    ew, U = eigh((M + M.conj().T) / 2)
    return (U * np.sqrt(np.clip(ew, 0, None))) @ U.conj().T

def root_fid(A, B):
    return float(np.sum(svd(sqrtm_h(A) @ sqrtm_h(B), compute_uv=False)))

def rand_state(d, rank=None, pure=False):
    r = 1 if pure else (rank or d)
    Z = rng.normal(size=(d, r)) + 1j * rng.normal(size=(d, r))
    rho = Z @ Z.conj().T
    return rho / np.trace(rho).real

# ---------------------------------------------------------------- checks
def c01_reciprocal_no_go():
    """Thm: objects with R=L and orthogonal dual collars have g^Sigma <= (1/2)log 2 for every w, sigma."""
    worst = -np.inf
    for _ in range(4000):
        N = int(rng.integers(4, 15)); w = int(rng.integers(1, N // 2 + 1))
        # random mixed state on C^N, cells = basis; R = L
        rho = rand_state(N, rank=int(rng.integers(1, N + 1)))
        # push mass to a boundary at random to make the test hard
        if rng.random() < .5:
            s = int(rng.integers(0, 2)); P = np.zeros(N); P[collar(N, s, w)] = 1
            rho = (1 - .999) * rho + .999 * np.diag(P / P.sum())
        p = np.real(np.diag(rho))
        for s in (0, 1):
            worst = max(worst, g_sum(p, p, N, s, w))
    ok = worst <= 0.5 * LOG2 + 1e-12
    print(f"V29-01 reciprocal no-go: max g^Sigma={worst:.6f} <= {0.5*LOG2:.6f}; passed={ok}")
    return ok

def c02_skin_implies_nonnormal():
    """Thm: Phi_Q,m >= g^Sigma_m - log 3   (quantum root fidelity, projective cell collars)."""
    worst = np.inf
    for _ in range(3000):
        N = int(rng.integers(4, 12)); w = int(rng.integers(1, N // 2 + 1)); d = N
        eps = 10 ** rng.uniform(-6, -0.3)
        def conc(s):
            v = rng.normal(size=d) + 1j * rng.normal(size=d)
            v = v / np.linalg.norm(v) * np.sqrt(eps)
            u = np.zeros(d, complex); idx = collar(N, s, w)
            u[idx] = rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))
            u = u / np.linalg.norm(u)
            x = u * np.sqrt(1 - eps) + v; return x / np.linalg.norm(x)
        s = int(rng.integers(0, 2))
        r = conc(s); l = conc(1 - s)
        if rng.random() < .4:  # mixed responses
            r2, l2 = conc(s), conc(1 - s)
            rho = .7 * np.outer(r, r.conj()) + .3 * np.outer(r2, r2.conj())
            sig = .7 * np.outer(l, l.conj()) + .3 * np.outer(l2, l2.conj())
        else:
            rho, sig = np.outer(r, r.conj()), np.outer(l, l.conj())
        p = np.real(np.diag(rho)); q = np.real(np.diag(sig))
        F = root_fid(rho, sig)
        Phi = -np.log(max(F, 1e-300))
        worst = min(worst, Phi - (g_sum(p, q, N, s, w) - np.log(3)))
    ok = worst >= -1e-9
    print(f"V29-02 skin => non-normal: min slack={worst:.4f}; passed={ok}")
    return ok

def allowed_T(N, W):
    """classical shadow maps that never bring mass closer (truncated depth) to any boundary"""
    def depth(x): return (min(x + 1, W + 1), min(N - x, W + 1))
    A = np.zeros((N, N), bool)                     # A[y,x]
    for x in range(N):
        for y in range(N):
            A[y, x] = all(depth(y)[k] >= depth(x)[k] for k in (0, 1))
    return A

def rand_T(A):
    N = A.shape[0]; T = np.zeros((N, N))
    for x in range(N):
        idx = np.where(A[:, x])[0]; T[idx, x] = rng.dirichlet(np.ones(len(idx)) * rng.uniform(.2, 2))
    return T

def check_collar_nonexpanding(T, N, W):
    """(c) at operator level: for every sigma,w: x outside collar_sigma,w => T(.|x) outside collar_sigma,w"""
    for s in (0, 1):
        for w in range(1, W + 1):
            B = collar(N, s, w); notA = [x for x in range(N) if x not in B]
            if T[np.ix_(B, notA)].sum() > 1e-14: return False
    return True

PSIS = [lambda l: -0.5 * np.log(np.maximum(l, 1e-300)),
        lambda l: np.sqrt(1 / np.maximum(l, 1e-300)),      # convex nonincreasing
        lambda l: np.exp(-3 * l), lambda l: 1 - l]
for t in (.1, .3, .6, .9): PSIS.append((lambda t: lambda l: np.maximum(t - l, 0))(t))

def profile(P, Q, N, W):
    return np.array([[lam_sum(p, q, N, s, w) for s in (0, 1) for w in range(1, W + 1)] for p, q in zip(P, Q)])

def c03_monotones():
    """All Psi_psi (psi convex nonincreasing, coordinatewise) are nonincreasing under morphisms = (allowed T) o (response mixing kappa)."""
    N, W = 16, 4; A = allowed_T(N, W); bad = 0; tests = 0; ok_maps = 0
    for _ in range(600):
        MX = int(rng.integers(2, 8)); MY = int(rng.integers(1, 8))
        P = rng.dirichlet(np.ones(N) * rng.uniform(.05, 1), size=MX); Qd = rng.dirichlet(np.ones(N) * rng.uniform(.05, 1), size=MX)
        # make skin-like responses
        for m in range(MX):
            if rng.random() < .7:
                a = rng.uniform(.3, 3); n = np.arange(N)
                P[m] = np.exp(a * n) * P[m]; P[m] /= P[m].sum(); Qd[m] = np.exp(-a * n) * Qd[m]; Qd[m] /= Qd[m].sum()
        pi = rng.dirichlet(np.ones(MX)); kap = rng.dirichlet(np.ones(MY), size=MX).T   # MY x MX column-stochastic
        pip = kap @ pi
        T = rand_T(A); ok_maps += int(check_collar_nonexpanding(T, N, W))
        Pn = np.array([T @ ((kap[mp] * pi) @ P) / pip[mp] for mp in range(MY)])
        Qn = np.array([T @ ((kap[mp] * pi) @ Qd) / pip[mp] for mp in range(MY)])
        lx = profile(P, Qd, N, W); ly = profile(Pn, Qn, N, W)
        for psi in PSIS:
            vx = pi @ psi(lx); vy = pip @ psi(ly)
            tests += 1; bad += int(np.any(vy > vx + 1e-9 * (1 + abs(vx))))
        # max-type (convex, coordinatewise nonincreasing)
        vx = pi @ np.max(-0.5 * np.log(np.maximum(lx, 1e-300)), axis=1); vy = pip @ np.max(-0.5 * np.log(np.maximum(ly, 1e-300)), axis=1)
        tests += 1; bad += int(vy > vx + 1e-9 * (1 + abs(vx)))
    ok = bad == 0 and ok_maps == 600
    print(f"V29-03 convex-decreasing leakage monotones: tests={tests}; violations={bad}; T collar-nonexpanding={ok_maps}/600; passed={ok}")
    return ok

def c04_fidelity_monotone_and_additive():
    """Phi_Q = -log sum_m pi_m F(rho^R_m, rho^L_m): monotone under (common CPTP) o (kappa-mixing), additive on tensor products."""
    bad = 0; worst_add = 0.0
    for _ in range(250):
        d = 4; MX = int(rng.integers(2, 5)); MY = int(rng.integers(1, 5))
        R = [rand_state(d) for _ in range(MX)]; Lm = [rand_state(d) for _ in range(MX)]
        pi = rng.dirichlet(np.ones(MX)); kap = rng.dirichlet(np.ones(MY), size=MX).T; pip = kap @ pi
        K = [(rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) for _ in range(3)]
        S = sum(k.conj().T @ k for k in K); Sm = np.linalg.inv(sqrtm_h(S)); K = [k @ Sm for k in K]
        Lam = lambda r: sum(k @ r @ k.conj().T for k in K)
        Phi = lambda pp, RR, LL: -np.log(sum(pp[m] * root_fid(RR[m], LL[m]) for m in range(len(pp))))
        Rn = [Lam(sum(kap[mp, m] * pi[m] * R[m] for m in range(MX)) / pip[mp]) for mp in range(MY)]
        Ln = [Lam(sum(kap[mp, m] * pi[m] * Lm[m] for m in range(MX)) / pip[mp]) for mp in range(MY)]
        bad += int(Phi(pip, Rn, Ln) > Phi(pi, R, Lm) + 1e-9)
        # tensor additivity (two independent ensembles)
        d2 = 3; M2 = 2; R2 = [rand_state(d2) for _ in range(M2)]; L2 = [rand_state(d2) for _ in range(M2)]; pi2 = rng.dirichlet(np.ones(M2))
        RR = [np.kron(R[a], R2[b]) for a in range(MX) for b in range(M2)]; LL = [np.kron(Lm[a], L2[b]) for a in range(MX) for b in range(M2)]
        pp = np.array([pi[a] * pi2[b] for a in range(MX) for b in range(M2)])
        worst_add = max(worst_add, abs(Phi(pp, RR, LL) - Phi(pi, R, Lm) - Phi(pi2, R2, L2)))
    ok = bad == 0 and worst_add < 1e-8
    print(f"V29-04 Phi_Q monotone(violations={bad}) & additive(err={worst_add:.2e}); passed={ok}")
    return ok

def hn_data(L, g):
    n = np.arange(1, L + 1)
    H = np.zeros((L, L))
    for k in range(L - 1): H[k + 1, k] = np.exp(g); H[k, k + 1] = np.exp(-g)
    wr, vr = eig(H); wl, vl = eig(H.T)
    # match left/right by eigenvalue
    order = [int(np.argmin(np.abs(wl - x))) for x in wr]
    P = np.abs(vr) ** 2; P /= P.sum(0); Q = np.abs(vl[:, order]) ** 2; Q /= Q.sum(0)
    return P.T, Q.T   # rows = responses

def hn_law(L, g):
    n = np.arange(1, L + 1); P = []; Q = []
    for j in range(1, L + 1):
        k = np.pi * j / (L + 1); s2 = np.sin(k * n) ** 2
        p = np.exp(2 * g * (n - 1 - (L - 1))) * s2; q = np.exp(-2 * g * (n - 1)) * s2
        P.append(p / p.sum()); Q.append(q / q.sum())
    return np.array(P), np.array(Q)

def c05_hn_calibration():
    """HN: (a) analytic law = numerical eig at small L (exact similarity H=S H0 S^-1);
       (b) g^Sigma = g w + O(log L) uniformly in mode j, explicit constant, up to L=400 (analytic law; eig is unstable there)."""
    dev = 0.0
    for L, g in [(14, .3), (20, .45)]:
        P0, Q0 = hn_data(L, g); P1, Q1 = hn_law(L, g)
        for a in range(L):   # match by nearest distribution (eig order arbitrary)
            dev = max(dev, min(np.max(np.abs(P0[a] - P1[b])) for b in range(L)), min(np.max(np.abs(Q0[a] - Q1[b])) for b in range(L)))
    worst_lo = np.inf; worst_hi = -np.inf
    for L, g in [(40, .4), (100, .25), (200, .6), (400, .15)]:
        P, Q = hn_law(L, g); C = np.log(L + 1) + 1.0 + 0.5 * np.log(1 / (1 - np.exp(-2 * g)))
        for w in range(1, min(L // 3, 40)):
            gs = np.array([-0.5*np.log(0.5*(P[j][:L-w].sum()+Q[j][w:].sum())) for j in range(L)])  # complement summed directly (no 1-x cancellation)
            worst_lo = min(worst_lo, np.min(gs - (g * w - C))); worst_hi = max(worst_hi, np.max(gs - (g * w + C)))
    ok = dev < 1e-6 and worst_lo >= -1e-9 and worst_hi <= 1e-9
    print(f"V29-05 HN: analytic-vs-eig dev={dev:.1e}; g^Sigma - g w in [-C,+C]: lo slack={worst_lo:.3f}, hi slack={-worst_hi:.3f}; passed={ok}")
    return ok

def c06_controls():
    """Hermitian (g=0) and periodic HN give bounded g^Sigma; bulk non-normal perturbation has Phi_Q>0 with bounded g^Sigma."""
    L = 60; P, Q = hn_data(L, 0.0)
    g0 = max(g_sum(P[j], Q[j], L, s, w) for j in range(L) for s in (0, 1) for w in (1, 5, 15))
    # periodic HN
    g = .4; H = np.zeros((L, L), complex)
    for k in range(L): H[(k + 1) % L, k] += np.exp(g); H[k, (k + 1) % L] += np.exp(-g)
    wr, vr = eig(H); wl, vl = eig(H.T); order = [int(np.argmin(np.abs(wl - x))) for x in wr]
    Pp = np.abs(vr) ** 2; Pp /= Pp.sum(0); Qp = np.abs(vl[:, order]) ** 2; Qp /= Qp.sum(0)
    gp = max(g_sum(Pp[:, j], Qp[:, j], L, s, w) for j in range(L) for s in (0, 1) for w in (5, 15))
    # bulk-localized non-normality: Hermitian chain + strong non-Hermitian defect in the middle
    H2 = np.zeros((L, L), complex)
    for k in range(L - 1): H2[k + 1, k] = H2[k, k + 1] = 1.0
    H2[L // 2, L // 2 + 1] = 3.0; H2[L // 2 + 1, L // 2] = 0.2
    wr, vr = eig(H2); wl, vl = eig(H2.T); order = [int(np.argmin(np.abs(wl - x))) for x in wr]
    phis = []; gmax = 0
    for j in range(L):
        r = vr[:, j] / np.linalg.norm(vr[:, j]); l = vl[:, order[j]] / np.linalg.norm(vl[:, order[j]])
        phis.append(-np.log(max(abs(np.vdot(r, l)), 1e-300)))
        p = np.abs(r) ** 2; q = np.abs(l) ** 2
        gmax = max(gmax, max(g_sum(p, q, L, s, w) for s in (0, 1) for w in (5, 15)))
    ok = g0 <= .5 * LOG2 + 1e-9 and gp < 1.0 and np.mean(phis) > 0.05 and gmax < 1.5
    print(f"V29-06 controls: Hermitian max g={g0:.3f}; periodic HN max g={gp:.3f}; bulk-defect mean Phi_Q={np.mean(phis):.3f}, max g={gmax:.3f}; passed={ok}")
    return ok

def c07_strassen_1d():
    """scalar leakage sector: (all hinge tests pass) <=> (exists kappa-coupling with E[lam|lam'] <= lam')."""
    agree = 0; n = 0; pos = 0
    for _ in range(400):
        k = int(rng.integers(2, 6)); k2 = int(rng.integers(2, 6))
        lam = np.sort(rng.uniform(0, 1, size=k)); nu = rng.dirichlet(np.ones(k))
        if rng.random() < .5:      # construct a truly dominated nu' : average (Jensen) then shift up
            groups = rng.integers(0, k2, size=k)
            lam2 = np.array([lam[groups == a] @ nu[groups == a] / max(nu[groups == a].sum(), 1e-300) if nu[groups == a].sum() > 0 else 1.0 for a in range(k2)])
            lam2 = np.minimum(1, lam2 + rng.uniform(0, .2, size=k2)); nu2 = np.array([nu[groups == a].sum() for a in range(k2)])
            keep = nu2 > 0; lam2, nu2 = lam2[keep], nu2[keep]
        else:
            lam2 = rng.uniform(0, 1, size=k2); nu2 = rng.dirichlet(np.ones(k2))
        ts = np.unique(np.concatenate([lam, lam2, [1.0]]))
        hinge = all((nu2 @ np.maximum(t - lam2, 0)) <= (nu @ np.maximum(t - lam, 0)) + 1e-12 for t in ts)
        # LP
        k2 = len(lam2); nv = k * k2; Aeq = np.zeros((k + k2, nv)); beq = np.concatenate([nu, nu2])
        for i in range(k):
            for j in range(k2): Aeq[i, i * k2 + j] = 1; Aeq[k + j, i * k2 + j] = 1
        Aub = np.zeros((k2, nv))
        for j in range(k2):
            for i in range(k): Aub[j, i * k2 + j] = lam[i] - lam2[j]
        res = linprog(np.zeros(nv), A_ub=Aub, b_ub=np.zeros(k2), A_eq=Aeq[:-1], b_eq=beq[:-1], bounds=[(0, None)] * nv, method='highs')
        feas = res.status == 0
        agree += int(feas == hinge); n += 1; pos += int(hinge)
    ok = agree == n and pos > 40
    print(f"V29-07 Strassen (1D) hinge<=>coupling: agree={agree}/{n}; nontrivial positives={pos}; passed={ok}")
    return ok

def c08_hn_order_and_dilution():
    """(i) active-orientation (sigma=right) leakage of HN(g) <= that of HN(g'), g>g'>=0, same mode (MLR); wrong orientation is REVERSED.
       (ii) EXPLORATORY: exact shadow morphism HN(g)->HN(g') under operator-level free maps (report only).
       (iii) dilution D_t = (1-t) id + t (prepare bulk) is a free morphism; |g^Sigma(D_t X) - min(g, (1/2)log(1/t))| <= (1/2)log 2."""
    L = 24; W = 6; A = allowed_T(L, W)
    def law(g, j):
        n = np.arange(1, L + 1); k = np.pi * j / (L + 1)
        p = np.exp(2 * g * (n - 1)) * np.sin(k * n) ** 2; q = np.exp(-2 * g * (n - 1)) * np.sin(k * n) ** 2
        return p / p.sum(), q / q.sum()
    viol_act = 0; viol_rev = 0; feas = 0; tot = 0
    for g, gp in [(.6, .3), (.5, .1), (.8, .5), (.4, 0.0)]:
        for j in (1, 3, 8, 12, 20, 24):
            p, q = law(g, j); pp, qp = law(gp, j)
            for w in range(1, W + 1):
                viol_act += int(lam_sum(p, q, L, 1, w) > lam_sum(pp, qp, L, 1, w) + 1e-12)
                viol_rev += int(lam_sum(p, q, L, 0, w) < lam_sum(pp, qp, L, 0, w) - 1e-12)
            nv = L * L; idx = lambda y, x: y * L + x
            Aeq = []; beq = []
            for x in range(L):
                r = np.zeros(nv); r[[idx(y, x) for y in range(L)]] = 1; Aeq.append(r); beq.append(1)
            for src, tgt in ((p, pp), (q, qp)):
                for y in range(L):
                    r = np.zeros(nv)
                    for x in range(L): r[idx(y, x)] = src[x]
                    Aeq.append(r); beq.append(tgt[y])
            bounds = [(0, None) if A[y, x] else (0, 0) for y in range(L) for x in range(L)]
            res = linprog(np.zeros(nv), A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=bounds, method='highs')
            feas += int(res.status == 0); tot += 1
    # (iii) dilution
    bulk = np.zeros(L); bulk[L // 2] = 1.0; worst_band = 0.0; bad_c = 0
    for g in (.3, .6):
        for j in (1, 8, 20):
            p, q = law(g, j)
            for t in (1e-1, 1e-3, 1e-6, 1e-10):
                Ti = (1 - t) * np.eye(L) + t * np.outer(bulk, np.ones(L))
                bad_c += int(not check_collar_nonexpanding(Ti, L, W))
                p2, q2 = Ti @ p, Ti @ q
                for w in range(1, W + 1):
                    gd = g_sum(p2, q2, L, 1, w); g0 = g_sum(p, q, L, 1, w)
                    worst_band = max(worst_band, abs(gd - min(g0, .5 * np.log(1 / t))) - .5 * LOG2)
    ok = viol_act == 0 and viol_rev == 0 and bad_c == 0 and worst_band <= 1e-9
    print(f"V29-08 (i) active-orientation order violations={viol_act}, reversed-orientation violations={viol_rev}; "
          f"(ii) exact shadow morphism HN(g)->HN(g') feasible in {feas}/{tot} (exploratory); "
          f"(iii) dilution collar-nonexpanding failures={bad_c}, band excess={worst_band:.2e}; passed={ok}")
    return ok

def main():
    checks = [c01_reciprocal_no_go, c02_skin_implies_nonnormal, c03_monotones, c04_fidelity_monotone_and_additive,
              c05_hn_calibration, c06_controls, c07_strassen_1d, c08_hn_order_and_dilution]
    passed = 0
    for f in checks:
        try: passed += int(bool(f()))
        except Exception as exc: print(f"ERROR in {f.__name__}: {exc!r}")
    print(f"SUMMARY: {passed}/{len(checks)} V29 checks passed")
    return 0 if passed == len(checks) else 1

if __name__ == '__main__': raise SystemExit(main())
