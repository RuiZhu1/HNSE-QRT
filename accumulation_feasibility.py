#!/usr/bin/env python3
"""Feasibility checks for option (ii): an accumulation resource that also sees reciprocal skin effects.

Accumulation = collar Bessel bound of the right responses:
    S = sum_m pi_m rho^R_m / pi_max ,   B_sigma(w) = || Pi_{sigma,w} S Pi_{sigma,w} || ,  K_k = Ky Fan k-norm of Pi S Pi.
Any orthonormal family has S <= 1, hence B <= 1 (Bessel's inequality). Free operations (proposed): deterministic
relabelling f (merging allowed, no splitting) and one channel Xi with (C) Xi^+(Pi') <= Pi and (C*) Pi' Xi(1) Pi' <= Pi'.

 A1  values: Hermitian (complete, edge-state window, thermal weights) B <= 1; HN, GDSE, Z2 skin B >> 1;
     non-normal chains without skin B ~ 1
 A2  monotonicity of B and K_k under random free operations (quantum data and commuting sector)
 A3  necessity: (C) without (C*), label splitting, and rank-normalised occupancy each fail
 A4  accumulation forces non-normality (complete data): sum_m kappa_m^2 >= (M-r)/(M-r-E), E = Tr[Pi S_1] - r
 A5  directional skin forces accumulation: B_sigma(w) >= sum_m (pi_m/pi_max)(1 - e^{-2 s_m}) / rank Pi
Run: python accumulation_feasibility.py   (numpy)
"""
from __future__ import annotations
import itertools
import numpy as np
from numpy.linalg import eig, eigh, eigvalsh, inv, norm, qr

rng = np.random.default_rng(32001)
results = []


def report(tag, ok, text):
    results.append(ok)
    print(f"{tag} {text}; passed={ok}")


# ------------------------------------------------------------------ data model
def cplx(*s):
    return rng.normal(size=s) + 1j * rng.normal(size=s)


def unitary(d):
    Q, R = qr(cplx(d, d))
    return Q * (np.diag(R) / abs(np.diag(R)))


def collars_1d(N, d, W):
    """{(sigma,w): projector} for a chain of N cells with d levels per cell"""
    out = {}
    for s in (1, -1):
        for w in range(1, W + 1):
            cells = range(N - w, N) if s > 0 else range(w)
            P = np.zeros((N * d, N * d))
            for c in cells:
                P[c * d:(c + 1) * d, c * d:(c + 1) * d] = np.eye(d)
            out[(s, w)] = P
    return out


def biorth(G):
    lam, R = eig(G)
    Lm = inv(R).conj().T
    return lam, R / norm(R, axis=0), Lm / norm(Lm, axis=0)


def gen_data(G, weights=None, window=None):
    """generator datum: list of (pi, rhoR, rhoL), pure responses"""
    lam, R, Lm = biorth(G)
    idx = range(len(lam)) if window is None else window
    w = np.ones(len(lam)) if weights is None else weights
    data = [(w[m], np.outer(R[:, m], R[:, m].conj()), np.outer(Lm[:, m], Lm[:, m].conj())) for m in idx]
    tot = sum(p for p, _, _ in data)
    return [(p / tot, a, b) for p, a, b in data]


def frame(data):
    pmax = max(p for p, _, _ in data)
    return sum(p * a for p, a, _ in data) / pmax


def B_all(data, col):
    S = frame(data)
    return {k: eigvalsh(P @ S @ P).max() for k, P in col.items()}


def kyfan(data, col, k):
    S = frame(data)
    return {key: np.sort(eigvalsh(P @ S @ P))[::-1][:k].sum() for key, P in col.items()}


def paired_scores(data, col):
    """V31 skin score s_m(sigma,w) for every response"""
    out = []
    for p, a, b in data:
        row = {}
        for (s, w), P in col.items():
            lR = 1 - np.trace(P @ a).real
            lL = 1 - np.trace(col[(-s, w)] @ b).real
            row[(s, w)] = max(0.0, -0.5 * np.log(max(lR + lL, 1e-300)))
        out.append(row)
    return out


# ------------------------------------------------------------------ models
def hn(L, g):
    return np.diag(np.full(L - 1, np.exp(g)), -1) + np.diag(np.full(L - 1, np.exp(-g)), 1)


def hn_exact_data(L, g):
    """exact HN eigen-data via the similarity S = diag(e^{gn}) (avoids eig at large L)"""
    n = np.arange(1, L + 1); data = []
    for j in range(1, L + 1):
        phi = np.sin(np.pi * j * n / (L + 1))
        r = np.exp(g * n) * phi; l = np.exp(-g * n) * phi
        r /= norm(r); l /= norm(l)
        data.append((1 / L, np.outer(r, r), np.outer(l, l)))
    return data


def gen_chain(N, d, g, herm=False):
    G = np.zeros((N * d, N * d), complex)
    for n in range(N):
        G[n * d:(n + 1) * d, n * d:(n + 1) * d] = cplx(d, d)
        if n + 1 < N:
            G[(n + 1) * d:(n + 2) * d, n * d:(n + 1) * d] = np.exp(g) * cplx(d, d)
            G[n * d:(n + 1) * d, (n + 1) * d:(n + 2) * d] = np.exp(-g) * cplx(d, d)
    return (G + G.conj().T) / 2 if herm else G


def gdse_triangle(N, gamma, dis):
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
    """two HN chains of opposite chirality coupled antisymmetrically; TRS^dagger with T = i sigma_y (TT* = -1).
    Basis is cell-major: index 2n+s."""
    A = hn(N, g); J = np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1)
    Hs = np.block([[A, delta * J], [delta * J, A.T]])
    perm = np.array([s * N + n for n in range(N) for s in range(2)])
    H = Hs[np.ix_(perm, perm)]
    U = np.kron(np.eye(N), np.array([[0, 1], [-1, 0]]))
    assert np.allclose(H.T, U @ H @ U.conj().T)
    return H


def eigenspace_data(G, tol=1e-6):
    """semisimple degenerate spectrum: one response per eigenvalue cluster, rho = projector onto the
    (right / left) eigenspace divided by its dimension, weight = dimension / D"""
    lam, R = eig(G); Linv = inv(R)
    D = len(lam); used = np.zeros(D, bool); data = []
    scale = max(1.0, np.abs(lam).max())
    for i in range(D):
        if used[i]: continue
        cl = [j for j in range(D) if not used[j] and abs(lam[j] - lam[i]) < tol * scale]
        used[cl] = True
        QR_, _ = qr(R[:, cl]); QL_, _ = qr(Linv[cl, :].conj().T)
        k = len(cl)
        data.append((k / D, QR_ @ QR_.conj().T / k, QL_ @ QL_.conj().T / k))
    return data


# ------------------------------------------------------------------ free operations (proposed class)
def apply_channel(data, chan):
    return [(p, chan(a), chan(b)) for p, a, b in data]


def merge(data, f):
    """deterministic relabelling f: label m -> f[m] (merging allowed)"""
    out = {}
    for (p, a, b), t in zip(data, f):
        if t in out:
            q, A, Bm = out[t]; out[t] = (q + p, A + p * a, Bm + p * b)
        else:
            out[t] = (p, p * a, p * b)
    return [(q, A / q, Bm / q) for q, A, Bm in out.values()]


def embed_op(N, d, shiftL, extraR, d2):
    """isometry: cell-local unitaries, internal enlargement d -> d2, boundary moved outward"""
    Np = N + shiftL + extraR; J = np.eye(d2, d)
    V = np.zeros((Np * d2, N * d), complex)
    for n in range(N):
        V[(n + shiftL) * d2:(n + shiftL + 1) * d2, n * d:(n + 1) * d] = J @ unitary(d)
    return V, Np


def cell_mixed_unitary(N, d, K=3):
    ps = rng.dirichlet(np.ones(K))
    Us = [np.kron(np.eye(N), np.eye(d)) if d == 1 else
          np.block([[unitary(d) if i == j else np.zeros((d, d)) for j in range(N)] for i in range(N)]) for _ in range(K)]
    return lambda X: sum(p * U @ X @ U.conj().T for p, U in zip(ps, Us))


def bulk_state(N, d, W):
    v = np.zeros(N * d, complex)
    for c in range(W, N - W):
        v[c * d:(c + 1) * d] = cplx(d)
    v /= norm(v)
    return np.outer(v, v.conj())


def check_C_Cstar(V, colin, colout):
    c1 = min(eigvalsh(colin[k] - V.conj().T @ colout[k] @ V).min() for k in colin)
    c2 = min(eigvalsh(colout[k] - colout[k] @ V @ V.conj().T @ colout[k]).min() for k in colout)
    return min(c1, c2)


# ------------------------------------------------------------------ A1 values
def a1_values():
    W = 4; rows = []
    N, d = 8, 2; col = collars_1d(N, d, W)
    Bh = max(B_all(gen_data(gen_chain(N, d, 0, herm=True)), col).values())
    th = np.exp(-np.linspace(0, 3, N * d)); Bth = max(B_all(gen_data(gen_chain(N, d, 0, herm=True), weights=th), col).values())
    # Hermitian SSH chain in the topological phase: window = the two edge states only
    Ls = 24; v, w_ = 0.3, 1.0
    Hs = np.zeros((Ls, Ls))
    for i in range(Ls - 1):
        Hs[i, i + 1] = Hs[i + 1, i] = v if i % 2 == 0 else w_
    ev, U = eigh(Hs); edge = list(np.argsort(abs(ev))[:2])
    col_s = collars_1d(Ls, 1, 3)
    Bedge = max(B_all(gen_data(Hs, window=edge), col_s).values())
    rows.append(("Hermitian complete", Bh)); rows.append(("Hermitian thermal weights", Bth)); rows.append(("SSH edge-state window", Bedge))
    # HN (exact), growth with L
    hnrows = []
    for L in (20, 40, 80):
        data = hn_exact_data(L, 0.3); colL = collars_1d(L, 1, L // 4)
        B = B_all(data, colL)
        hnrows.append((L, B[(1, 2)], B[(1, L // 4)], max(B[(-1, w)] for w in range(1, L // 4 + 1))))
    # GDSE triangle: reciprocal, paired score exactly zero, accumulation at the diagonal edge; growth with N
    gd = []
    for Nt in (12, 18, 24):
        H, sites = gdse_triangle(Nt, 0.6, 0.05); data = gen_data(H)
        Pd = np.diag([1.0 if x + y > Nt - 3 else 0.0 for (x, y) in sites])
        Pl = np.diag([1.0 if x < 2 else 0.0 for (x, y) in sites])
        prof = max(np.abs(np.diag(a_) - np.diag(b_)).max() for _, a_, b_ in data)
        S = frame(data)
        gd.append((Nt, eigvalsh(Pd @ S @ Pd).max(), eigvalsh(Pl @ S @ Pl).max(), prof))
    # Z2 skin effect (Kramers-degenerate), eigenspace projectors; growth with N
    zz = []
    for Nz in (8, 14, 20):
        dz = eigenspace_data(z2_skin(Nz, 0.5, 0.3)); colz = collars_1d(Nz, 2, 2)
        Bz = B_all(dz, colz); sz = paired_scores(dz, colz)
        deg = sorted(set(round(p * 2 * Nz) for p, _, _ in dz))
        zz.append((Nz, Bz[(1, 2)], Bz[(-1, 2)], max(max(r.values()) for r in sz), deg))
    # non-normal chains WITHOUT skin: B stays bounded as L grows
    bb = []; rc = []
    for Lb in (40, 80, 160):
        Hb = hn(Lb, 0).astype(complex); Hb[Lb // 2, Lb // 2 + 1] = 3.0; Hb[Lb // 2 + 1, Lb // 2] = 0.2
        bb.append((Lb, max(B_all(gen_data(Hb), collars_1d(Lb, 1, 4)).values())))
        rg = np.random.default_rng(Lb)
        Hr = hn(Lb, 0).astype(complex) + np.diag(rg.uniform(-1, 1, Lb) + 1j * rg.uniform(-1, 1, Lb))   # 1D reciprocal, complex potential
        rc.append((Lb, max(B_all(gen_data(Hr), collars_1d(Lb, 1, 4)).values())))
    ok = (Bh < 1 + 1e-9 and Bth < 1 + 1e-9 and Bedge < 1 + 1e-9
          and hnrows[1][1] > 1.8 * hnrows[0][1] and hnrows[2][1] > 1.8 * hnrows[1][1] and max(r[3] for r in hnrows) < 1e-2
          and all(r[3] < 1e-10 for r in gd) and gd[-1][1] > 1.5 * gd[0][1]
          and all(r[4] == [2] and r[3] < 0.05 for r in zz) and zz[-1][1] > 1.5 * zz[0][1] and zz[-1][2] > 1.5 * zz[0][2]
          and bb[-1][1] < 1.2 * bb[0][1] and rc[-1][1] < 1.5 * max(r[1] for r in rc[:2]))
    report("A1 values", ok,
           "free side: " + ", ".join(f"{a} B={b:.3f}" for a, b in rows) +
           "; HN g=0.3 (L, B(+,2), B(+,L/4), max B(-,w)): " + ", ".join(f"({a},{b:.1f},{c:.1f},{e:.0e})" for a, b, c, e in hnrows) +
           "; GDSE triangle (N, B(diag edge,2), B(left edge,2), |R|^2-|L|^2): " + ", ".join(f"({a},{b:.1f},{c:.2f},{e:.0e})" for a, b, c, e in gd) +
           "; Z2 skin (N, B(+,2), B(-,2), max paired s, eigenspace dims): " + ", ".join(f"({a},{b:.2f},{c:.2f},{e:.3f},{f})" for a, b, c, e, f in zz) +
           "; no skin: bulk bond (L, max B): " + ", ".join(f"({a},{b:.2f})" for a, b in bb) +
           ", 1D reciprocal random complex potential (L, max B): " + ", ".join(f"({a},{b:.2f})" for a, b in rc))
    return ok


# ------------------------------------------------------------------ A2 monotonicity
def a2_monotone():
    viol = 0; tests = 0; worstC = np.inf
    inputs = []
    N, d, W = 6, 2, 2
    for g in (0.0, 0.3, 0.8):
        inputs.append((gen_data(gen_chain(N, d, g)), N, d))
    inputs.append((eigenspace_data(z2_skin(N, 0.5, 0.3)), N, 2))
    for trial in range(120):
        data, n, dd = inputs[trial % len(inputs)]
        col = collars_1d(n, dd, W)
        cur, cn, cd, ccol = data, n, dd, col
        B0 = B_all(cur, ccol); K0 = {k: kyfan(cur, ccol, k) for k in (1, 2, 3)}
        for step in range(int(rng.integers(2, 5))):
            op = rng.integers(0, 4)
            if op == 0:                                   # boundary-compatible embedding
                sl, er, d2 = int(rng.integers(0, 2)), int(rng.integers(0, 2)), cd + int(rng.integers(0, 2))
                V, Np = embed_op(cn, cd, sl, er, d2); ncol = collars_1d(Np, d2, W)
                worstC = min(worstC, check_C_Cstar(V, ccol, ncol))
                cur = apply_channel(cur, lambda X, V=V: V @ X @ V.conj().T); cn, cd, ccol = Np, d2, ncol
            elif op == 1:                                 # cell-local mixed unitary (unital, commutes with collars)
                cur = apply_channel(cur, cell_mixed_unitary(cn, cd))
            elif op == 2:                                 # dilution with a bulk state
                t = rng.uniform(0, 0.5); om = bulk_state(cn, cd, W)
                cur = apply_channel(cur, lambda X, t=t, om=om: (1 - t) * X + t * np.trace(X) * om)
            else:                                         # deterministic merging relabelling
                M = len(cur); f = rng.integers(0, max(1, M // 2), size=M)
                cur = merge(cur, f)
            B1 = B_all(cur, ccol); K1 = {k: kyfan(cur, ccol, k) for k in (1, 2, 3)}
            for key in B1:
                tests += 1; viol += int(B1[key] > B0[key] + 1e-9)
                for k in (1, 2, 3):
                    tests += 1; viol += int(K1[k][key] > K0[k][key] + 1e-9)
            B0, K0 = B1, K1
    # commuting sector: arbitrary stochastic maps obeying (C) and (C*), with merges
    def allowed(Nn, Ww):
        depth = lambda x: (min(x + 1, Ww + 1), min(Nn - x, Ww + 1))
        return np.array([[all(depth(y)[q] >= depth(x)[q] for q in (0, 1)) for x in range(Nn)] for y in range(Nn)])
    cviol = 0; ctests = 0
    Nn, Ww = 12, 3; A = allowed(Nn, Ww); colset = [c for c in range(Nn) if c < Ww or c >= Nn - Ww]
    for trial in range(300):
        M = int(rng.integers(3, 9)); P = rng.dirichlet(np.ones(Nn) * 0.3, size=M); pi = rng.dirichlet(np.ones(M))
        n = np.arange(Nn)
        for m in range(M):
            if rng.random() < .6: P[m] *= np.exp(rng.uniform(.3, 2) * n * rng.choice([1, -1])); P[m] /= P[m].sum()
        T = np.zeros((Nn, Nn))
        for x in range(Nn):
            idx = np.where(A[:, x])[0]; T[idx, x] = rng.dirichlet(np.ones(len(idx)))
        bulk = Nn // 2
        for y in colset:                                   # (C*): collar rows sub-stochastic; excess sent to the bulk
            s = T[y].sum()
            if s > 1:
                exc = T[y] * (1 - 1 / s); T[y] -= exc; T[bulk] += exc
        assert np.allclose(T.sum(0), 1) and all(T[y].sum() <= 1 + 1e-12 for y in colset)
        def Bc(PP, ww):
            Sd = (ww[:, None] * PP).sum(0) / ww.max()
            return {(s, w): (Sd[list(range(Nn - w, Nn)) if s > 0 else list(range(w))]).max() for s in (1, -1) for w in range(1, Ww + 1)}
        b0 = Bc(P, pi)
        f = rng.integers(0, max(1, M - 2), size=M); Pn = []; wn = []
        for t in sorted(set(f)):
            ms = np.where(f == t)[0]; q = pi[ms].sum(); wn.append(q); Pn.append(T @ (pi[ms] @ P[ms]) / q)
        b1 = Bc(np.array(Pn), np.array(wn))
        for key in b0:
            ctests += 1; cviol += int(b1[key] > b0[key] + 1e-12)
    ok = viol == 0 and cviol == 0 and worstC > -1e-12
    report("A2 monotonicity", ok, f"quantum data (generic chains g=0,0.3,0.8 and Z2), 120 random sequences of embeddings / "
           f"cell-local mixed unitaries / bulk dilutions / merges: {tests} comparisons of B and Ky Fan k=1,2,3, violations={viol} "
           f"(min eig of (C),(C*) slacks {worstC:.1e}); commuting sector, 300 random (C)+(C*) stochastic maps with merges: "
           f"{ctests} comparisons, violations={cviol}")
    return ok


# ------------------------------------------------------------------ A3 necessity of the restrictions
def a3_necessity():
    N, d = 6, 2; data = gen_data(gen_chain(N, d, 0, herm=True)); col = collars_1d(N, d, 2)
    B0 = max(B_all(data, col).values())
    # (a) partial trace over the internal level: satisfies (C) (Xi^+(Pi') = Pi' (x) 1 = Pi) but not (C*)
    def ptrace(X):
        return np.einsum('aibi->ab', X.reshape(N, d, N, d))
    col1 = collars_1d(N, 1, 2)
    Bpt = max(B_all(apply_channel(data, ptrace), col1).values())
    # (b) splitting through a mixture of two relabellings (id and shift into 2M labels)
    split = [(p / 2, a, b) for p, a, b in data] + [(p / 2, a, b) for p, a, b in data]
    Bsp = max(B_all(split, col).values())
    # (c) rank-normalised trace occupancy is not monotone under (C)+(C*): crowded direction moved into a rank-1 collar
    v = np.zeros(4); v[0] = v[1] = 1 / np.sqrt(2); u = np.zeros(4); u[0], u[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    e2, e3 = np.eye(4)[2], np.eye(4)[3]
    dat = [(0.25, np.outer(x, x), np.outer(x, x)) for x in (v, (v + 0.2 * e2) / norm(v + 0.2 * e2), (v + 0.2 * e3) / norm(v + 0.2 * e3), u)]
    Pin = np.diag([1., 1., 0., 0.])                      # rank-2 collar in H (cells 0,1 of one boundary)
    E4 = np.eye(4)                                       # unitary: v -> e0 (collar of H'), u -> e3, e2 -> e1, e3 -> e2 (bulk)
    V = np.outer(E4[0], v) + np.outer(E4[3], u) + np.outer(E4[1], e2) + np.outer(E4[2], e3)
    Pout = np.diag([1., 0., 0., 0.])                     # rank-1 collar in H'
    cC = eigvalsh(Pin - V.T @ Pout @ V).min(); cS = eigvalsh(Pout - Pout @ V @ V.T @ Pout).min()
    S = frame(dat); Sout = V @ S @ V.T
    occ_in = np.trace(Pin @ S) / 2; occ_out = np.trace(Pout @ Sout) / 1
    Bin = eigvalsh(Pin @ S @ Pin).max(); Bout = eigvalsh(Pout @ Sout @ Pout).max()
    ok = abs(B0 - 1) < 1e-9 and Bpt > 1.9 and Bsp > 1.9 and cC > -1e-12 and cS > -1e-12 and occ_out > occ_in + 0.1 and Bout <= Bin + 1e-12
    report("A3 necessity", ok, f"Hermitian complete data B={B0:.3f}; (C) without (C*) (partial trace over the internal level) -> B={Bpt:.3f}; "
           f"label splitting by a mixture of relabellings -> B={Bsp:.3f}; rank-normalised occupancy under a (C)+(C*) isometry "
           f"{occ_in:.3f} -> {occ_out:.3f} while B {Bin:.3f} -> {Bout:.3f}")
    return ok


# ------------------------------------------------------------------ A4 accumulation forces non-normality
def a4_nonnormality():
    rows = []; ok = True
    cases = [("HN L=16 g=0.3", hn(16, 0.3), collars_1d(16, 1, 4)[(1, 4)]),
             ("random chain g=0.4", gen_chain(8, 2, 0.4), collars_1d(8, 2, 2)[(1, 2)])]
    H, sites = gdse_triangle(14, 0.6, 0.05)
    cases.append(("GDSE triangle", H, np.diag([1.0 if x + y > 11 else 0.0 for (x, y) in sites])))
    for name, G, P in cases:
        lam, R, Lm = biorth(G); M = len(lam)
        S1 = R @ R.conj().T; r = int(round(np.trace(P).real)); E = np.trace(P @ S1).real - r
        kap2 = (1 / np.abs(np.einsum('im,im->m', Lm.conj(), R)) ** 2).sum()
        bound = (M - r) / (M - r - E) if E < M - r else np.inf
        rows.append((name, E, kap2, bound)); ok &= kap2 >= bound * (1 - 1e-9)
    report("A4 accumulation => non-normality", ok, "; ".join(f"{n}: collar excess E={E:.2f}, sum kappa^2={k:.3g} >= bound {b:.3g}"
                                                           for n, E, k, b in rows))
    return ok


# ------------------------------------------------------------------ A5 directional skin forces accumulation
def a5_paired_implies_acc():
    worst = np.inf; n = 0
    sets = [(hn_exact_data(L, g), collars_1d(L, 1, L // 4)) for L, g in ((16, .3), (30, .5), (40, .2))]
    for g in (0.2, 0.6):
        sets.append((gen_data(gen_chain(7, 2, g)), collars_1d(7, 2, 3)))
    for data, col in sets:
        B = B_all(data, col); sc = paired_scores(data, col); pmax = max(p for p, _, _ in data)
        for key, P in col.items():
            lb = sum((p / pmax) * (1 - np.exp(-2 * s[key])) for (p, _, _), s in zip(data, sc)) / np.trace(P).real
            worst = min(worst, B[key] - lb); n += 1
    ok = worst > -1e-9
    report("A5 paired => accumulation", ok, f"B_sigma(w) - sum_m (pi_m/pi_max)(1-e^(-2 s_m))/rank Pi >= {worst:.3f} over {n} (data, collar) pairs")
    return ok



# ------------------------------------------------------------------ uniform-weight objects and the density axiom (D)
# Objects: M responses with uniform weights (each eigenmode counts once). S_X = sum_m rho^R_m.
# Free processing: convex combination of label-blind processings, each channel obeying (C), output weights uniform,
# and (D): (M'/M) Pi' Xibar(1) Pi' <= Pi', Xibar = label-averaged channel. Then S_Y = (M'/M) Xibar(S_X).
def B_uniform(states, col):
    S = sum(a for a, _ in states)
    return {k: eigvalsh(P @ S @ P).max() for k, P in col.items()}


def kyfan_uniform(states, col, k):
    S = sum(a for a, _ in states)
    return {key: np.sort(eigvalsh(P @ S @ P))[::-1][:k].sum() for key, P in col.items()}


def a6_density_axiom():
    """B is non-increasing under the convex class with (C) per component and (D) on the average, including
    mixtures over different relabellings, appending free objects (M' > M), uniform merges, erasure, dilution, embeddings."""
    viol = 0; tests = 0; kviol = 0; ktests = 0; worstD = np.inf; kinc = 0
    N, d, W = 5, 2, 2
    inputs = [[(a, b) for _, a, b in gen_data(gen_chain(N, d, g))] for g in (0.0, 0.4, 0.9)]
    zz = eigenspace_data(z2_skin(N, 0.5, 0.3))
    inputs.append([(a, b) for p, a, b in zz for _ in range(round(p * 2 * N))])   # each 2-dim eigenspace as 2 copies of P/2
    for trial in range(90):
        cur = inputs[trial % len(inputs)]; cn, cd = N, (2 if trial % len(inputs) == 3 else d); col = collars_1d(cn, cd, W)
        B0 = B_uniform(cur, col)
        for step in range(int(rng.integers(2, 5))):
            M = len(cur); op = int(rng.integers(0, 6)); Mprev = M
            if op == 0:      # mixture over random label permutations and cell-local unitaries (unital: (D) with equality)
                K = 3; ps = rng.dirichlet(np.ones(K)); perms = [rng.permutation(M) for _ in range(K)]
                Us = [np.block([[unitary(cd) if i == j else np.zeros((cd, cd)) for j in range(cn)] for i in range(cn)]) for _ in range(K)]
                new = []
                for mp in range(M):
                    A_ = sum(pk * U @ cur[int(np.where(pr == mp)[0][0])][0] @ U.conj().T for pk, pr, U in zip(ps, perms, Us))
                    Bm = sum(pk * U @ cur[int(np.where(pr == mp)[0][0])][1] @ U.conj().T for pk, pr, U in zip(ps, perms, Us))
                    new.append((A_, Bm))
                cur = new; Dslack = 0.0
            elif op == 1:    # append a free object: internal factor C^r with orthonormal states e_n, n < r <= dF (M' = r M)
                dF = 2; r = int(rng.integers(1, dF + 1)); E = np.eye(dF)
                cur = [(np.kron(a, np.outer(E[n], E[n])), np.kron(b, np.outer(E[n], E[n]))) for a, b in cur for n in range(r)]
                cd = cd * dF; ncol = collars_1d(cn, cd, W)
                Xi1 = np.kron(np.eye(cn * cd // dF), np.diag([1.0] * r + [0.0] * (dF - r)) / r)   # Xibar(1)
                Dslack = min(eigvalsh(P - (r * M / M) * P @ Xi1 @ P).min() for P in ncol.values())
                col = ncol; B0 = {k: v for k, v in B0.items()}
            elif op == 2 and M % 2 == 0:   # uniform merge of random pairs (M' = M/2)
                pr = rng.permutation(M); cur = [((cur[pr[2 * i]][0] + cur[pr[2 * i + 1]][0]) / 2, (cur[pr[2 * i]][1] + cur[pr[2 * i + 1]][1]) / 2) for i in range(M // 2)]
                Dslack = 0.5
            elif op == 3:    # dilution with a bulk state
                t = rng.uniform(0, .5); om = bulk_state(cn, cd, W)
                cur = [((1 - t) * a + t * om, (1 - t) * b + t * om) for a, b in cur]; Dslack = t
            elif op == 4:    # boundary-compatible embedding
                sl, er = int(rng.integers(0, 2)), int(rng.integers(0, 2)); d2 = cd + int(rng.integers(0, 2))
                V, Np = embed_op(cn, cd, sl, er, d2); ncol = collars_1d(Np, d2, W)
                worstD = min(worstD, check_C_Cstar(V, col, ncol))
                cur = [(V @ a @ V.conj().T, V @ b @ V.conj().T) for a, b in cur]; cn, cd, col = Np, d2, ncol; Dslack = 0.0
            else:            # erasure to a bulk state (M' = 1)
                om = bulk_state(cn, cd, W); cur = [(om, om)]; Dslack = 1.0
            worstD = min(worstD, Dslack)
            B1 = B_uniform(cur, col)
            for key in B1:
                tests += 1; viol += int(B1[key] > B0[key] + 1e-9)
            B0 = B1
    ok = viol == 0 and worstD > -1e-12
    report("A6 density axiom (D)", ok, f"uniform-weight data (generic chains g=0,0.4,0.9 and Z2), 90 random sequences including mixtures over "
           f"different relabellings and appending free objects: {tests} comparisons of B, violations={viol}; min (C)/(D) slack {worstD:.1e}")
    return ok


def acc_lp(X, Y, N, W, Mp=None, u=None, use_D=True, rand_obj=None):
    """Commuting sector, uniform weights. Free processings = sum_f K_f (x) A_f, A_f >= 0 with allowed pattern (C),
    column sums t_f, sum t_f = 1, uniform output, and (D): (M'/M) sum_f rowsum_y(A_f) <= 1 on collar cells.
    Modes: Y given -> min l1 distance (returns dist, duals); u given -> max <u, image>; rand_obj -> random vertex, returns image."""
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix
    PR, PL = X; M = PR.shape[0]
    if Mp is None: Mp = Y[0].shape[0] if Y is not None else u.shape[1]
    depth = lambda x: (min(x + 1, W + 1), min(N - x, W + 1))
    allowed = np.array([[all(depth(y)[q] >= depth(x)[q] for q in (0, 1)) for x in range(N)] for y in range(N)])
    collar_cells = [c for c in range(N) if c < W or c >= N - W]
    fs = list(itertools.product(range(Mp), repeat=M)); F = len(fs)
    nA = F * N * N; iA = lambda f, y, x: (f * N + y) * N + x
    nimg = 2 * Mp * N; use_err = Y is not None
    nv = nA + F + (2 * nimg if use_err else 0)
    rows_eq = F * N + 1 + (Mp - 1) + (nimg if use_err else 0)
    Aeq = lil_matrix((rows_eq, nv)); beq = []; r = 0
    for f in range(F):
        for x in range(N):
            for y in range(N): Aeq[r, iA(f, y, x)] = 1
            Aeq[r, nA + f] = -1; beq.append(0); r += 1
    for f in range(F): Aeq[r, nA + f] = 1
    beq.append(1); r += 1
    for mp in range(Mp - 1):
        for f, fm in enumerate(fs): Aeq[r, nA + f] = sum(1 for m in range(M) if fm[m] == mp)
        beq.append(M / Mp); r += 1
    img_rows = []; Ub = []; bub = []
    if use_D:
        Aub = lil_matrix((len(collar_cells), nv))
        for i, y in enumerate(collar_cells):
            for f in range(F):
                for x in range(N): Aub[i, iA(f, y, x)] = Mp / M
        Ub = Aub.tocsr(); bub = np.ones(len(collar_cells))
    def coeffs(h, mp, y):
        P = PR if h == 0 else PL; out = {}
        for f, fm in enumerate(fs):
            src = [m for m in range(M) if fm[m] == mp]
            if src:
                wx = sum(P[m] for m in src) / M
                for x in range(N): out[iA(f, y, x)] = wx[x]
        return out
    if use_err:
        QR_, QL_ = Y
        for h, Q in enumerate((QR_, QL_)):
            for mp in range(Mp):
                for y in range(N):
                    for k, v in coeffs(h, mp, y).items(): Aeq[r, k] = v
                    e = (h * Mp + mp) * N + y
                    Aeq[r, nA + F + e] = -1; Aeq[r, nA + F + nimg + e] = 1
                    beq.append(Q[mp, y] / Mp); img_rows.append(r); r += 1
    bounds = [(0, None) if allowed[y, x] else (0, 0) for f in range(F) for y in range(N) for x in range(N)]
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
    if use_D: kw.update(A_ub=Ub, b_ub=bub)
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


def a7_completeness_with_D():
    """Thm 9(b) survives: X->Y (LP, new class) <=> V_u(X) >= V_u(Y) for all u; separating u found for unreachable pairs;
    no monotone violation on reachable pairs; B never increases on reachable pairs; V31-class images that violate (D) are blocked."""
    N, W = 6, 2; sep = 0; infeas = 0; feas = 0; mono = 0; Bviol = 0; blocked = 0; nv31 = 0
    collar_cells = [c for c in range(N) if c < W or c >= N - W]
    Bc = lambda PR_: PR_.sum(0)[collar_cells].max()
    for it in range(60):
        M = int(rng.integers(2, 4)); Mp = int(rng.integers(1, 4))
        PR = rng.dirichlet(np.ones(N) * .5, size=M); PL = rng.dirichlet(np.ones(N) * .5, size=M); n = np.arange(N)
        for m in range(M):
            a = rng.uniform(.3, 1.5); PR[m] *= np.exp(a * n); PR[m] /= PR[m].sum(); PL[m] *= np.exp(-a * n); PL[m] /= PL[m].sum()
        X = (PR, PL); kind = it % 3
        if kind == 0:
            Y = acc_lp(X, None, N, W, Mp=Mp, rand_obj=rng.normal(size=(2, Mp, N)))
        elif kind == 1:
            Y = acc_lp(X, None, N, W, Mp=Mp, use_D=False, rand_obj=rng.normal(size=(2, Mp, N))); nv31 += 1
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
    report("A7 completeness with (D)", ok, f"commuting sector, uniform weights: reachable={feas}, unreachable={infeas}; separating V_u found in "
           f"{sep}/{infeas}; monotone violations on reachable pairs={mono}; B increased on reachable pairs={Bviol}; "
           f"V31-class images (no (D)) blocked: {blocked}/{nv31}")
    return ok

if __name__ == "__main__":
    for f in (a1_values, a2_monotone, a3_necessity, a4_nonnormality, a5_paired_implies_acc, a6_density_axiom, a7_completeness_with_D):
        f()
    print(f"SUMMARY: {sum(results)}/{len(results)} accumulation feasibility checks passed")
