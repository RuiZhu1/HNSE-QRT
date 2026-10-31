#!/usr/bin/env python3
"""Route 2, step 1 -- numerical toys for a resource theory of NON-RECIPROCITY, with the NHSE as its extensive limit.

Setting (commensurate-delay linear networks; classical waves, or mode operators of passive linear quantum optics)
  object       S(z): scattering matrix seen from the external ports, z = e^{i phi}, phi = frequency x unit delay
  free object  reciprocal, S(z)^T = S(z)            (Lorentz reciprocity, photonic convention U = 1)
  free op      insert the object ONCE into a passive reciprocal network: junctions are symmetric contractions,
               internal ports are joined by reciprocal lines with delay >= 1.  Loops, resonances and interference
               are all allowed.
  candidate monotone   N1(S) = mean_phi  1/2 || S_A(e^{i phi}) ||_1 ,   S_A = (S - S^T)/2   ("non-reciprocity budget")

Parts
  A  golden rule: reciprocal networks with loops stay reciprocal
  B  single-frequency figures of merit are NOT monotones (interference, resonance); Jensen's formula
  C  the budget N1: proof-step checks, random and adversarial tests, several devices, dispersive objects
  D  NHSE chains: exact gauge-flux identity, forward gain vs. point-gap winding (needs mpmath)
  E  loops: 2-port cascades keep S21/S12 multiplicative; isolation-bandwidth corollary; pure-phase elements
     isolate only inside a loop (Mach-Zehnder); lattice analogue (|t+|=|t-| chains are reciprocal up to gauge)
  F  gain and quantum noise: collapse with free classical gain; series networks keep S21/S12 even with gain;
     noise price N1 <= sum_i beta_i (1 + nu_i); ring amplifier; HN-type amplifier chain
  G  lattices: robust stability <=> r^2|S12 S21| < 1; multi-channel transfer matrices (det flux, symplectic pairing);
     network winding vanishes for passive / reciprocal / one-channel phase-only cells; HN chain rates; length
     stability threshold at flux rate = gain rate; flux per passive cell
  (A-F take about 1.5 min, G about 1-2 min; run e.g. "python3 route2_nonreciprocity_toys.py G" for one part)
  The note route2_note.tex states and proves the results these checks support.

Proof sketch of the bound tested in C (one or more frequency-independent passive devices D_i, external ports on
reciprocal junctions; see C0 for the numerical check of each step):
  (1) S_ext - S_ext^T = 2 X Delta X^T, with X(z) = S_EI (1 - C S_II)^{-1} C restricted to device ports and
      Delta = (+)_i (D_i - D_i^T)/2.  [resolvent identity; C symmetric; S_EI^T = S_IE]
  (2) For every constant vector s:  mean_phi |X(e^{i phi}) s|^2 <= |s|^2.  X s is the output produced by launching s
      from the device ports at t = 0 into delay>=1 lines of a passive network; Parseval + passivity.
  (3) Youla form Delta = sum_j sigma_j (u_j w_j^T - w_j u_j^T); for a rank-2 skew matrix
      1/2 ||a b^T - b a^T||_1 = sqrt(|a|^2|b|^2 - |a^dag b|^2) <= |a||b|; Cauchy-Schwarz with (2) gives
      N1(S_ext) <= sum_j sigma_j = sum_i 1/2 ||(D_i)_A||_1.
  So N1(S) <= Cost(S) := min over realisations of S of sum_i 1/2||(D_i)_A||_1, and Cost is a monotone by construction
  (free networks add no devices).  N1 itself is NOT a monotone on dispersive objects: C5 (a 4-port junction found by
  random search, stored in c4_counterexample.npz) raises N1 of a narrowband ring object by ~17%, and C6 (the ring
  closed into a loop by one lossless coupler) by a factor ~2.2.  The bound holds only relative to elementary components.

Run:  python3 route2_nonreciprocity_toys.py        (numpy; mpmath for part D2)
"""
from __future__ import annotations
import sys
import time
import numpy as np

rng = np.random.default_rng(20261016)
TWO_PI = 2 * np.pi


def grid(n):
    """phase grid offset by half a step (avoids landing exactly on zeros placed on the unit circle)"""
    phi = (np.arange(n) + 0.5) * TWO_PI / n
    return np.exp(1j * phi)


# ------------------------------------------------------------------ building blocks
def rand_unitary(n):
    Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def sym_contr(n, smin=0.0):
    """random symmetric contraction (Takagi form U diag(s) U^T): a passive reciprocal junction"""
    U = rand_unitary(n)
    return (U * rng.uniform(smin, 1.0, n)) @ U.T


def contr(n, smin=0.0):
    """random contraction: a passive, generally non-reciprocal device"""
    return (rand_unitary(n) * rng.uniform(smin, 1.0, n)) @ rand_unitary(n).conj().T


def half_trace_anti(S):
    """1/2 ||(S - S^T)/2||_1, batched over leading axes"""
    A = (S - np.swapaxes(S, -1, -2)) / 2
    return 0.5 * np.linalg.svd(A, compute_uv=False).sum(-1)


class Net:
    """Junction blocks (symmetric ones are reciprocal; a non-symmetric block is a device) with globally numbered ports.
    ext: external ports.  lines: (p, q, delay, amplitude) reciprocal lines joining internal ports p and q."""

    def __init__(self, blocks, ext, lines):
        n = sum(b.shape[0] for b in blocks)
        SJ = np.zeros((n, n), complex)
        i = 0
        for b in blocks:
            k = b.shape[0]
            SJ[i:i + k, i:i + k] = b
            i += k
        self.SJ = SJ
        self.E = np.array(ext, int)
        self.I = np.array([p for p in range(n) if p not in set(int(e) for e in ext)], int)
        self.lines = list(lines)
        self.pos = {int(p): k for k, p in enumerate(self.I)}

    def _C(self, z):
        n = len(self.I)
        C = np.zeros((len(z), n, n), complex)
        for p, q, d, a in self.lines:
            i, j = self.pos[int(p)], self.pos[int(q)]
            C[:, i, j] = a * z ** d
            C[:, j, i] = a * z ** d
        return C

    def S(self, z):
        z = np.atleast_1d(z)
        C = self._C(z)
        SJ, E, I = self.SJ, self.E, self.I
        A = np.eye(len(I))[None] - SJ[np.ix_(I, I)][None] @ C
        Y = np.linalg.solve(A, np.broadcast_to(SJ[np.ix_(I, E)], (len(z), len(I), len(E))))
        return SJ[np.ix_(E, E)][None] + SJ[np.ix_(E, I)][None] @ C @ Y

    def X(self, z, cols):
        """X(z) = S_EI (1 - C S_II)^{-1} C, columns = given (internal) junction ports"""
        z = np.atleast_1d(z)
        C = self._C(z)
        SJ, E, I = self.SJ, self.E, self.I
        A = np.eye(len(I))[None] - C @ SJ[np.ix_(I, I)][None]
        M = np.linalg.solve(np.swapaxes(A, -1, -2), np.broadcast_to(SJ[np.ix_(E, I)].T, (len(z), len(I), len(E))))
        Xfull = np.swapaxes(M, -1, -2) @ C          # S_EI A^{-1} C
        return Xfull[:, :, [self.pos[c] for c in cols]]


def random_embedding(dev_blocks, n_ext, smin=0.0, n_junc=None):
    """insert the device blocks into a random passive reciprocal network; external ports sit on reciprocal junctions"""
    n_junc = n_junc or int(rng.integers(1, 4))
    blocks = [sym_contr(int(rng.integers(2, 5)), smin) for _ in range(n_junc)]
    nsym = sum(b.shape[0] for b in blocks)
    if nsym < n_ext + 1:
        return random_embedding(dev_blocks, n_ext, smin, n_junc + 1)
    blocks = blocks + list(dev_blocks)
    ntot = sum(b.shape[0] for b in blocks)
    ext = [int(e) for e in rng.choice(nsym, n_ext, replace=False)]
    internal = [p for p in range(ntot) if p not in ext]
    if len(internal) % 2:                      # add a 1-port reciprocal junction (partial reflector / absorber)
        blocks.insert(0, sym_contr(1, smin))
        ext = [e + 1 for e in ext]
        ntot += 1
        internal = [p for p in range(ntot) if p not in ext]
    pairs = rng.permutation(internal).reshape(-1, 2)
    lines = [(int(p), int(q), int(rng.integers(1, 4)), np.exp(-rng.uniform(0, .05) + 2j * np.pi * rng.uniform()))
             for p, q in pairs]
    net = Net(blocks, ext, lines)
    ndev = sum(b.shape[0] for b in dev_blocks)
    net.dev_ports = list(range(ntot - ndev, ntot))
    return net


def compose(obj: Net, n_ext2, smin=0.0):
    """insert a (dispersive) object, itself a Net, into a random reciprocal network through one slot"""
    k = len(obj.E)
    while True:
        blocks2 = [sym_contr(int(rng.integers(2, 5)), smin) for _ in range(int(rng.integers(1, 4)))]
        n2 = sum(b.shape[0] for b in blocks2)
        if n2 >= n_ext2 + k:
            break
    perm = [int(p) for p in rng.permutation(n2)]
    ext2, slot, rest = perm[:n_ext2], perm[n_ext2:n_ext2 + k], perm[n_ext2 + k:]
    if len(rest) % 2:
        blocks2.insert(0, sym_contr(1, smin))
        ext2 = [e + 1 for e in ext2]; slot = [s + 1 for s in slot]; rest = [0] + [r + 1 for r in rest]
        n2 += 1
    lines = [(p, q, int(rng.integers(1, 4)), np.exp(-rng.uniform(0, .05) + 2j * np.pi * rng.uniform()))
             for p, q in np.array(rest, int).reshape(-1, 2)]
    lines += [(p + n2, q + n2, d, a) for p, q, d, a in obj.lines]
    lines += [(slot[i], int(obj.E[i]) + n2, 1, 1.0) for i in range(k)]
    net = Net(blocks2 + [obj.SJ], ext2, lines)
    net.n_outer = n2
    return net


def splitter():
    """reciprocal Y-junction (symmetric contraction, lossless for the symmetric excitation)"""
    a = 1 / np.sqrt(2)
    return np.array([[0, a, a], [a, 0, 0], [a, 0, 0]], complex)


def coupler(r):
    k = np.sqrt(1 - r ** 2)
    return np.array([[0, r, 1j * k, 0], [r, 0, 0, 1j * k], [1j * k, 0, 0, r], [0, 1j * k, r, 0]], complex)


def twoport(tf, tb):
    """2-port with transmission tf from its port 0 to port 1 and tb back; no reflection"""
    return np.array([[0, tb], [tf, 0]], complex)


# ------------------------------------------------------------------ A
def part_A():
    worst = 0.0
    zs = np.exp(1j * rng.uniform(0, TWO_PI, 6)) * rng.uniform(.3, 1, 6)
    for _ in range(300):
        net = random_embedding([], int(rng.integers(2, 5)), smin=float(rng.choice([0, .9])))
        S = net.S(zs)
        worst = max(worst, float((half_trace_anti(S) / np.linalg.norm(S, 2, axis=(-2, -1))).max()))
    print(f"A  golden rule: 300 random reciprocal networks with loops, max N(S_A)/||S|| = {worst:.1e}")
    return worst < 1e-12


# ------------------------------------------------------------------ B
def part_B():
    ok = True
    # B1 cosquare invariance under invertible congruence, mu <-> 1/mu pairing
    S = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    cs = lambda M: np.linalg.solve(M.T, M)
    e1 = np.sort_complex(np.linalg.eigvals(cs(S)))
    e2 = np.sort_complex(np.linalg.eigvals(cs(A.T @ S @ A)))
    inv = np.sort_complex(1 / e1)
    b1 = np.allclose(e1, e2, atol=1e-8) and np.allclose(e1, inv, atol=1e-8)
    D = twoport(0.9, 0.3)
    mu = np.linalg.eigvals(cs(D))
    print(f"B1 cosquare spectrum invariant under S -> A^T S A and closed under mu -> 1/mu: {b1}; "
          f"isolator (0.9 fwd, 0.3 back) cosquare eigenvalues {np.round(np.sort(np.abs(mu)), 4)} = (1/3, 3) = isolation ratio")
    ok &= b1
    # B2 Mach-Zehnder (equal arms) around a pure-phase non-reciprocal element: broadband, unbounded isolation ratio
    dphi = 0.05
    tf, tb = 1.0, np.exp(1j * dphi)
    Z = grid(512)
    print(f"B2 MZI around a pure-phase element (|tf|=|tb|=1, phase difference {dphi}; isolation ratio of the element = 1):")
    for eps in (1e-1, 1e-3, 1e-5):
        w = -tb * (1 - eps)
        net = Net([splitter(), splitter(), twoport(tf, tb), twoport(w, w)], [0, 3],
                  [(1, 6, 1, 1.0), (7, 4, 1, 1.0), (2, 8, 1, 1.0), (9, 5, 1, 1.0)])
        S = net.S(Z)
        fwd, bwd = np.abs(S[:, 1, 0]), np.abs(S[:, 0, 1])
        ratio = fwd / bwd
        n1 = half_trace_anti(S).mean() / half_trace_anti(twoport(tf, tb))
        print(f"     eps={eps:6.0e}: isolation ratio {ratio.min():9.1f} (flat over the band: spread {np.ptp(ratio)/ratio.mean():.0e}), "
              f"forward transmission {fwd.mean():.4f}, N1/budget {n1:.3f}")
        ok &= n1 <= 1 + 1e-9
    # B3 ring resonator around the same kind of element: narrowband perfect isolation, Jensen sum rule
    print("B3 critically coupled ring around a pure-phase element (phase difference 0.01):")
    Zf = grid(2 ** 18)
    d = 0.01
    for r in (0.9, 0.99, 0.999):
        a = r
        t1 = (r - a * Zf) / (1 - r * a * Zf)                        # bus 0->1, excites the CW mode (tf = 1)
        t2 = (r - a * np.exp(1j * d) * Zf) / (1 - r * a * np.exp(1j * d) * Zf)   # bus 1->0, CCW mode (tb = e^{id})
        iso_db = 20 * np.log10(np.abs(t2) / np.abs(t1))
        bw = np.mean(iso_db > 20)
        jensen = np.mean(np.log(np.abs(t2) / np.abs(t1)))
        n1 = np.mean(np.abs(t1 - t2) / 2) / (a * abs(1 - np.exp(1j * d)) / 2)
        k = np.argmax(iso_db)
        print(f"     r={r:6}: peak isolation {iso_db[k]:6.1f} dB (grid-limited; exact: infinite) with pass transmission {abs(t2[k]):.3f}; "
              f">20 dB over {bw:.1e} of the band; mean signed log-isolation {jensen:+.1e} (Jensen: 0); N1/budget {n1:.3f}")
        ok &= abs(jensen) < 1e-3 and n1 <= 1 + 1e-9
    print("   => isolation ratio at one frequency, or its band average, can be raised without bound by free networks;"
          " neither is a monotone.")
    return ok


# ------------------------------------------------------------------ C
def part_C():
    ok = True
    Z = grid(4096)
    # C0 the three proof steps
    err1 = 0.0; e2 = 0.0; e3 = 0.0
    for _ in range(60):
        devs = [contr(int(rng.integers(2, 4)))]
        net = random_embedding(devs, int(rng.integers(2, 4)), smin=float(rng.choice([0, .9])))
        Sx = net.S(Z)
        X = net.X(Z, net.dev_ports)
        Dl = (devs[0] - devs[0].T) / 2
        err1 = max(err1, float(np.abs((Sx - np.swapaxes(Sx, -1, -2)) / 2 - X @ Dl @ np.swapaxes(X, -1, -2)).max()))
        for _k in range(5):
            s = rng.normal(size=len(net.dev_ports)) + 1j * rng.normal(size=len(net.dev_ports)); s /= np.linalg.norm(s)
            e2 = max(e2, float(np.mean(np.linalg.norm(X @ s, axis=-1) ** 2)))
        a = rng.normal(size=4) + 1j * rng.normal(size=4); b = rng.normal(size=4) + 1j * rng.normal(size=4)
        M = np.outer(a, b) - np.outer(b, a)
        lhs = 0.5 * np.linalg.svd(M, compute_uv=False).sum()
        rhs = np.sqrt(np.linalg.norm(a) ** 2 * np.linalg.norm(b) ** 2 - abs(np.vdot(a, b)) ** 2)
        e3 = max(e3, abs(lhs - rhs) / rhs)
    print(f"C0 proof steps: (1) |S_A - X Delta X^T| max {err1:.1e};  (2) max mean|X s|^2 over unit s = {e2:.4f} (<= 1);  "
          f"(3) rank-2 skew trace-norm formula rel. err {e3:.1e}")
    ok &= err1 < 1e-10 and e2 <= 1 + 1e-9 and e3 < 1e-10
    # C1 / C2 random networks, one device and several devices
    w1 = 0.0; w2 = 0.0; viol = 0; n = 0
    for t in range(450):
        k = 1 if t < 300 else int(rng.integers(2, 4))
        devs = [contr(int(rng.integers(2, 4)), smin=float(rng.choice([0, .9]))) for _ in range(k)]
        budget = sum(half_trace_anti(D) for D in devs)
        net = random_embedding(devs, int(rng.integers(2, 4)), smin=float(rng.choice([0, .9, .99])))
        val = float(half_trace_anti(net.S(Z)).mean() / budget)
        n += 1; viol += val > 1 + 1e-9
        if k == 1: w1 = max(w1, val)
        else: w2 = max(w2, val)
    D = contr(2); thru = np.array([[0, 1], [1, 0]], complex)
    direct = Net([thru, thru, D], [0, 2], [(1, 4, 1, 1.0), (3, 5, 1, 1.0)])
    tight = float(half_trace_anti(direct.S(Z)).mean() / half_trace_anti(D))
    print(f"C1 one device, 300 random embeddings: max N1(S_ext)/budget = {w1:.3f}  (direct wiring gives {tight:.6f}: bound is tight)")
    print(f"C2 2-3 devices, 150 random embeddings: max N1(S_ext)/(sum of budgets) = {w2:.3f};  violations {viol}/{n}")
    ok &= viol == 0
    # C3 adversarial hill climb on the reciprocal part (junctions merged into one symmetric contraction) and line phases
    Zc, Zf = grid(1024), grid(16384)
    best = []
    for seed in range(10):
        D = contr(2) if seed % 2 else 0.999 * twoport(1.0, np.exp(0.05j))
        budget = half_trace_anti(D)
        net = random_embedding([D], 2)
        nsym = net.SJ.shape[0] - 2
        cur = half_trace_anti(net.S(Zc)).mean() / budget
        for _ in range(300):
            P = rng.normal(size=(nsym, nsym)) + 1j * rng.normal(size=(nsym, nsym)); P = (P + P.T) * 0.03
            old = (net.SJ, net.lines)
            Ss = net.SJ[:nsym, :nsym] + P
            sv = np.linalg.svd(Ss, compute_uv=False)[0]
            if sv > 1: Ss = Ss / sv
            SJ = net.SJ.copy(); SJ[:nsym, :nsym] = Ss; net.SJ = SJ
            net.lines = [(p, q, d, np.clip(abs(a) * np.exp(rng.normal() * .01), 0, 1) * np.exp(1j * (np.angle(a) + rng.normal() * .2)))
                         for p, q, d, a in net.lines]
            v = half_trace_anti(net.S(Zc)).mean() / budget
            if v > cur: cur = v
            else: net.SJ, net.lines = old
        best.append(float(half_trace_anti(net.S(Zf)).mean() / budget))
    print(f"C3 adversarial hill climb (10 starts x 300 steps): best N1/budget found = {max(best):.3f}")
    ok &= max(best) <= 1 + 1e-6
    # C4 dispersive objects: N1(Theta(S)) <= N1(S)?  (conjecture; not covered by the proof)
    worst = 0.0; viol = 0; n = 0
    for _ in range(150):
        obj = random_embedding([contr(int(rng.integers(2, 4)))], int(rng.integers(2, 4)), smin=float(rng.choice([0, .9])))
        v1 = half_trace_anti(obj.S(Z)).mean()
        v2 = half_trace_anti(compose(obj, int(rng.integers(2, 4)), smin=float(rng.choice([0, .9]))).S(Z)).mean()
        n += 1; viol += v2 > v1 * (1 + 1e-6) + 1e-12
        if v1 > 1e-10: worst = max(worst, v2 / v1)
    # adversarial: narrowband object (ring), optimise the outer network
    ring = Net([coupler(0.99), 0.99 * twoport(1.0, np.exp(0.05j))], [0, 1], [(2, 4, 1, 1.0), (5, 3, 1, 1.0)])
    v1 = half_trace_anti(ring.S(grid(2 ** 15))).mean()
    bestr = 0.0
    for seed in range(6):
        net = compose(ring, 2); no = net.n_outer
        cur = half_trace_anti(net.S(grid(4096))).mean()
        for _ in range(200):
            P = rng.normal(size=(no, no)) + 1j * rng.normal(size=(no, no)); P = (P + P.T) * 0.03
            old = (net.SJ, net.lines)
            Ss = net.SJ[:no, :no] + P
            sv = np.linalg.svd(Ss, compute_uv=False)[0]
            if sv > 1: Ss = Ss / sv
            SJ = net.SJ.copy(); SJ[:no, :no] = Ss; net.SJ = SJ
            net.lines = [(p, q, d, (np.clip(abs(a) * np.exp(rng.normal() * .01), 0, 1) * np.exp(1j * (np.angle(a) + rng.normal() * .2))
                                    if (p < no and q < no) else a)) for p, q, d, a in net.lines]
            v = half_trace_anti(net.S(grid(4096))).mean()
            if v > cur: cur = v
            else: net.SJ, net.lines = old
        bestr = max(bestr, float(half_trace_anti(net.S(grid(2 ** 15))).mean()))
    print(f"C4 is N1 a monotone on dispersive objects?  random: max N1(Theta(S))/N1(S) = {worst:.3f}, violations {viol}/{n}; "
          f"adversarial on a narrowband ring object: best ratio {bestr / v1:.3f}")
    # C5 stored counterexample (found by the C4 search in a full run): one reciprocal 4-port junction around the ring object
    import os
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c4_counterexample.npz")
    if os.path.exists(f):
        dat = np.load(f)
        lines = [(int(p), int(q), int(d), complex(a)) for (p, q, d), a in zip(dat["lines_pq"], dat["lines_amp"])]
        net = Net([dat["SJ"]], list(dat["E"]), lines)
        no = dat["SJ"].shape[0] - ring.SJ.shape[0]
        outer = dat["SJ"][:no, :no]
        same_obj = np.allclose(dat["SJ"][no:, no:], ring.SJ)
        vals = [(half_trace_anti(net.S(grid(n))).mean(), half_trace_anti(ring.S(grid(n))).mean()) for n in (2 ** 15, 2 ** 18)]
        budget = half_trace_anti(0.99 * twoport(1.0, np.exp(0.05j)))
        print(f"C5 stored counterexample: outer network = one {no}-port junction, symmetric to {np.abs(outer - outer.T).max():.0e}, "
              f"largest singular value {np.linalg.svd(outer, compute_uv=False)[0]:.6f}; object = the ring: {same_obj}")
        print(f"     N1(Theta(S))/N1(S) = {vals[0][0] / vals[0][1]:.4f} (2^15 pts) / {vals[1][0] / vals[1][1]:.4f} (2^18 pts);  "
              f"N1(Theta(S)) = {vals[1][0]:.4f} <= elementary budget {budget:.4f}")
        print("   => N1 is NOT a monotone on dispersive objects; it lower-bounds the elementary cost, which is the monotone.")
    # C6 explicit counterexample: close the ring object into a loop with one lossless directional coupler (r = 1/2)
    Zf = grid(2 ** 15)
    loop = Net([coupler(0.5), coupler(0.99), 0.99 * twoport(1.0, np.exp(0.05j))], [0, 2],
               [(1, 4, 1, -1j), (5, 3, 1, 1.0), (6, 8, 1, 1.0), (9, 7, 1, 1.0)])
    r15 = half_trace_anti(loop.S(Zf)).mean() / half_trace_anti(ring.S(Zf)).mean()
    r18 = half_trace_anti(loop.S(grid(2 ** 18))).mean() / half_trace_anti(ring.S(grid(2 ** 18))).mean()
    print(f"C6 explicit loop: ring object between ports 1 and 3 of a lossless coupler (r=1/2, line phase -i): "
          f"N1(Theta(S))/N1(S) = {r15:.4f} (2^15) / {r18:.4f} (2^18)")
    ok &= r18 > 1
    return ok


# ------------------------------------------------------------------ D
def part_D():
    ok = True
    worst = 0.0
    for _ in range(200):
        N = int(rng.integers(3, 30))
        tp = rng.normal(size=N - 1) + 1j * rng.normal(size=N - 1)
        tm = rng.normal(size=N - 1) + 1j * rng.normal(size=N - 1)
        V = rng.normal(size=N) + 1j * rng.normal(size=N)
        H = np.diag(V) + np.diag(tp, -1) + np.diag(tm, 1)
        z = 3 * (rng.normal() + 1j * rng.normal())
        G = np.linalg.inv(z * np.eye(N) - H)
        worst = max(worst, abs((G[-1, 0] / G[0, -1]) / (np.prod(tp) / np.prod(tm)) - 1))
    print(f"D1 any one-channel nearest-neighbour chain (complex disorder, any z): G_N1/G_1N = prod t+/t- , max rel. err {worst:.1e}")
    ok &= worst < 1e-9
    try:
        import mpmath as mp
    except ImportError:
        print("D2 skipped (mpmath not installed)")
        return ok
    mp.mp.dps = 60
    T = {1: .5, -1: .5, 2: -.25, -2: .25}         # H_{m,n} = t_{m-n}; PBC loop h(k)=cos k + (i/2) sin 2k (figure eight)
    h = lambda k: sum(t * np.exp(-1j * j * k) for j, t in T.items())

    def winding(z):
        k = np.linspace(0, TWO_PI, 4001)
        return int(np.round(np.sum(np.diff(np.unwrap(np.angle(h(k) - z)))) / TWO_PI))

    def ends(z, N):
        A = mp.matrix(N, N)
        for m in range(N):
            A[m, m] = mp.mpc(z)
            for j, t in T.items():
                if 0 <= m - j < N: A[m, m - j] -= t
        e1 = mp.matrix(N, 1); e1[0] = 1
        eN = mp.matrix(N, 1); eN[N - 1] = 1
        return float(mp.log(abs(mp.lu_solve(A, e1)[N - 1]))), float(mp.log(abs(mp.lu_solve(A, eN)[0])))

    Ns = [10, 20, 30, 40, 50]
    print("D2 chain with reciprocal nearest-neighbour hopping (t_{+-1}=1/2) and next-nearest hopping t_{+2}=-1/4, t_{-2}=+1/4:\n"
          "   equal magnitudes, non-reciprocal only in phase; the PBC loop is a figure eight with winding -1 and +1 lobes")
    agree = 0; pts = [-0.6 + 0.15j, -0.8 - 0.1j, 0.6 + 0.15j, 0.8 - 0.1j, 0.5j, 1.3, -1.2 + 0.3j]
    for z in pts:
        w = winding(z)
        L = np.array([ends(z, N) for N in Ns])
        sF = np.polyfit(Ns, L[:, 0], 1)[0]; sB = np.polyfit(Ns, L[:, 1], 1)[0]
        pred = {-1: (True, False), 1: (False, True), 0: (False, False)}[w]
        agree += (sF > 0, sB > 0) == pred
        print(f"     z={z!s:>9}: winding {w:+d} | growth rate of log|G_N1| {sF:+.3f}, of log|G_1N| {sB:+.3f} | log-isolation rate {sF - sB:+.3f}")
    print(f"   => exponential gain to the right iff winding=-1, to the left iff winding=+1: {agree}/{len(pts)} points; "
          "where w=0 both directions decay, even when their ratio is exponential (z=1.3): gain, not the ratio, marks the NHSE.")
    ok &= agree == len(pts)
    return ok


def part_E():
    """loops: single-channel geometry cannot turn phase non-reciprocity into directionality; one loop can"""
    ok = True
    # E1 cascade of 2-ports (reflections, delays): S21/S12 is multiplicative at every frequency
    worst = 0.0
    for _ in range(100):
        k = int(rng.integers(2, 6))
        blocks = [contr(2) for _ in range(k)]
        lines = [(2 * i + 1, 2 * (i + 1), int(rng.integers(1, 4)), np.exp(2j * np.pi * rng.uniform()) * rng.uniform(.5, 1))
                 for i in range(k - 1)]
        S = Net(blocks, [0, 2 * k - 1], lines).S(grid(64))
        pred = np.prod([b[1, 0] / b[0, 1] for b in blocks])
        worst = max(worst, float(np.abs(S[:, 1, 0] / S[:, 0, 1] / pred - 1).max()))
    print(f"E1 cascade of 2-ports with reflections: S21/S12 = prod_i S21^(i)/S12^(i) at every frequency, max rel. err {worst:.1e}")
    ok &= worst < 1e-9
    # E2 isolation-bandwidth corollary: mean|S21 - S12| <= sum_i |D21^(i) - D12^(i)|
    worst = 0.0
    for _ in range(200):
        devs = [contr(2) for _ in range(int(rng.integers(1, 3)))]
        S = random_embedding(devs, 2, smin=float(rng.choice([0, .9]))).S(grid(4096))
        worst = max(worst, float(np.abs(S[:, 1, 0] - S[:, 0, 1]).mean() / sum(abs(D[1, 0] - D[0, 1]) for D in devs)))
    print(f"E2 corollary on 200 random 2-port networks: max mean|S21-S12| / sum_i|D21-D12| = {worst:.3f} (<= 1)")
    ok &= worst <= 1 + 1e-9
    # E3 pure-phase elements (|tf| = |tb| = 1, phase difference Delta): cascade vs. Mach-Zehnder loop
    print("E3 N pure-phase elements, N*Delta = pi:")
    for n in (4, 16, 64):
        dl = np.pi / n
        el = twoport(1.0, np.exp(1j * dl))
        casc = Net([el] * n, [0, 2 * n - 1], [(2 * i + 1, 2 * (i + 1), 1, 1.0) for i in range(n - 1)]).S(grid(64))
        lines = [(1, 6, 1, 1.0)] + [(6 + 2 * i + 1, 6 + 2 * (i + 1), 1, 1.0) for i in range(n - 1)] + \
                [(6 + 2 * n - 1, 4, 1, 1.0), (2, 5, n + 1, 1.0)]
        S = Net([splitter(), splitter()] + [el] * n, [0, 3], lines).S(grid(256))
        budget = n * half_trace_anti(el)
        n1 = half_trace_anti(S).mean()
        print(f"     N={n:3d}: cascade max||S21/S12|-1| = {np.abs(np.abs(casc[:, 1, 0] / casc[:, 0, 1]) - 1).max():.0e};  "
              f"MZI min|S21| = {np.abs(S[:, 1, 0]).min():.6f}, max|S12| = {np.abs(S[:, 0, 1]).max():.0e}, "
              f"N1 = {n1:.4f}, budget = {budget:.4f}, efficiency = {n1 / budget:.4f} (1/pi = {1 / np.pi:.4f})")
        ok &= np.abs(np.abs(casc[:, 1, 0] / casc[:, 0, 1]) - 1).max() < 1e-9 and np.abs(S[:, 0, 1]).max() < 1e-9
    # E4 lattice analogue: nearest-neighbour chains with |t+| = |t-| (random phases, complex disorder) are reciprocal up to gauge
    worst = 0.0
    for _ in range(100):
        N = int(rng.integers(3, 25))
        r = rng.uniform(.3, 2, N - 1)
        tp = r * np.exp(2j * np.pi * rng.uniform(size=N - 1)); tm = r * np.exp(2j * np.pi * rng.uniform(size=N - 1))
        H = np.diag(rng.normal(size=N) + 1j * rng.normal(size=N)) + np.diag(tp, -1) + np.diag(tm, 1)
        G = np.linalg.inv(2 * (rng.normal() + 1j * rng.normal()) * np.eye(N) - H)
        worst = max(worst, float(np.abs(np.abs(G) - np.abs(G.T)).max() / np.abs(G).max()))
    T = {1: .5, -1: .5, 2: -.25, -2: .25}
    flux = T[1] ** 2 * T[-2] / (T[-1] ** 2 * T[2])
    print(f"E4 nearest-neighbour chains with |t+|=|t-| (random phases, complex disorder): max | |G_mn| - |G_nm| | / max|G| = {worst:.1e};"
          f"\n     figure-eight chain of D2: gauge-invariant loop product t1^2 t-2 / (t-1^2 t2) = {flux:+.0f} (a pi 'non-reciprocal flux')")
    ok &= worst < 1e-9
    return ok


# ------------------------------------------------------------------ F  (gain and quantum noise)
def pos_part(H):
    w, V = np.linalg.eigh((H + H.conj().T) / 2)
    return (V * np.clip(w, 0, None)) @ V.conj().T


class GNet(Net):
    """Net whose junctions may amplify.  Quantum-limited noise emitted at junction outputs: Q_k = (S_k S_k^dag - I)^+
    (normal-ordered quanta, vacuum inputs); gain-energy form on junction inputs: R_k = (S_k^dag S_k - I)^+."""

    def __init__(self, blocks, ext, lines):
        super().__init__(blocks, ext, lines)
        Q = np.zeros_like(self.SJ); R = np.zeros_like(self.SJ); i = 0
        for b in blocks:
            b = np.asarray(b, complex); k = b.shape[0]
            Q[i:i + k, i:i + k] = pos_part(b @ b.conj().T - np.eye(k))
            R[i:i + k, i:i + k] = pos_part(b.conj().T @ b - np.eye(k))
            i += k
        self.Q, self.R = Q, R

    def stable(self, n=None, chunk=32):
        """det(1 - S_II C(z)) is a polynomial with value 1 at z=0: stable iff no zero in the closed unit disk
        (argument principle on the unit circle; evaluated in chunks to bound memory)"""
        I = self.I; SII = self.SJ[np.ix_(I, I)]
        n = n or max(4096, 4 * len(I) * max([d for _p, _q, d, _a in self.lines] + [1]))
        z = grid(n); ds = []
        for i in range(0, n, chunk):
            sgn, ld = np.linalg.slogdet(np.eye(len(I))[None] - SII[None] @ self._C(z[i:i + chunk]))
            ds.append(sgn * np.exp(ld))
        d = np.concatenate(ds)
        w = int(np.round(np.sum(np.diff(np.unwrap(np.angle(np.append(d, d[0]))))) / TWO_PI))
        return w == 0 and np.abs(d).min() > 1e-9

    def noise_ext(self, z):
        z = np.atleast_1d(z); C = self._C(z); SJ, E, I = self.SJ, self.E, self.I
        A = np.eye(len(I))[None] - SJ[np.ix_(I, I)][None] @ C
        WI = SJ[np.ix_(E, I)][None] @ C @ np.linalg.inv(A)
        W = np.concatenate([WI, np.broadcast_to(np.eye(len(E)), (len(z), len(E), len(E)))], axis=2)
        idx = np.concatenate([I, E])
        return W @ self.Q[np.ix_(idx, idx)][None] @ np.swapaxes(W.conj(), -1, -2)

    def _arrive(self, z):
        C = self._C(z); I = self.I
        return np.linalg.solve(np.eye(len(I))[None] - C @ self.SJ[np.ix_(I, I)][None], C)   # (1 - C S_II)^{-1} C

    def noise_at(self, z, ports):
        """band-averaged quanta matrix arriving at the inputs of the given internal ports"""
        K = self._arrive(np.atleast_1d(z))[:, [self.pos[p] for p in ports], :]
        QI = self.Q[np.ix_(self.I, self.I)]
        return (K @ QI[None] @ np.swapaxes(K.conj(), -1, -2)).mean(0)

    def gain_energy(self, z, ports):
        """E with e(s) = s^dag E s: energy gained in junctions by the waves launched from the given ports at t = 0"""
        Aw = self._arrive(np.atleast_1d(z))[:, :, [self.pos[p] for p in ports]]
        RI = self.R[np.ix_(self.I, self.I)]
        return (np.swapaxes(Aw.conj(), -1, -2) @ RI[None] @ Aw).mean(0)


def amp2(g):
    """reciprocal, reflectionless amplifier 2-port (quantum limited: g^2 - 1 noise quanta per output)"""
    return np.array([[0, g], [g, 0]], complex)


def mirror(rho):
    t = np.sqrt(1 - rho ** 2)
    return np.array([[rho, t], [t, -rho]], complex)


def lam_max(M):
    return float(np.linalg.eigvalsh((M + M.conj().T) / 2).max())


def noise_bound(net, devs, dev_ports, Z):
    return sum(half_trace_anti(D) * (1 + lam_max(net.noise_at(Z, p))) for D, p in zip(devs, dev_ports))


def part_F():
    ok = True
    Z = grid(4096)
    # F1 collapse: classical gain + one loop turns any weak element into a perfect broadband directional amplifier
    D = twoport(1.0, np.exp(0.05j)); G = 100.0
    g = np.sqrt(G) / abs((D[1, 0] - D[0, 1]) / 2)
    net = GNet([splitter(), splitter(), D, twoport(-D[0, 1], -D[0, 1]), amp2(g)], [0, 11],
               [(1, 6, 1, 1.0), (7, 4, 1, 1.0), (2, 8, 1, 1.0), (9, 5, 1, 1.0), (3, 10, 1, 1.0)])
    S = net.S(Z); N = net.noise_ext(Z)
    g21 = np.abs(S[:, 1, 0]) ** 2
    print(f"F1 collapse (MZI cancellation + amplifier {g:.1f}): |S21|^2 = {g21.min():.6f}..{g21.max():.6f}, max|S12| = {np.abs(S[:, 0, 1]).max():.0e}, "
          f"max|S11|,|S22| = {max(np.abs(S[:, 0, 0]).max(), np.abs(S[:, 1, 1]).max()):.0e}, stable {net.stable()}")
    print(f"     N1 = {half_trace_anti(S).mean():.3f} = {half_trace_anti(S).mean() / half_trace_anti(D):.0f} x budget;  "
          f"added noise: port 2 = {N[:, 1, 1].real.mean():.0f} quanta (quantum limit {G - 1:.0f}), port 1 = {N[:, 0, 0].real.mean():.0e}")
    ok &= abs(g21.mean() - G) < 1e-6 and np.abs(S[:, 0, 1]).max() < 1e-9 and net.stable()
    # F2 series networks with gain: S21/S12 is still the product of the elements' ratios (gauge flux conserved)
    worst = 0.0; nst = 0
    for _ in range(200):
        k = int(rng.integers(2, 7))
        blocks = []
        for _i in range(k):
            r = rng.random()
            if r < .3: blocks.append(contr(2))
            elif r < .6: blocks.append(amp2(rng.uniform(1, 1.5) * np.exp(2j * np.pi * rng.uniform())))
            else: blocks.append(sym_contr(2) * rng.uniform(.5, 1.4))
        lines = [(2 * i + 1, 2 * (i + 1), int(rng.integers(1, 4)), np.exp(2j * np.pi * rng.uniform())) for i in range(k - 1)]
        net = GNet(blocks, [0, 2 * k - 1], lines)
        if not net.stable(): continue
        nst += 1
        S = net.S(grid(64))
        pred = np.prod([b[1, 0] / b[0, 1] for b in blocks])
        worst = max(worst, float(np.abs(S[:, 1, 0] / S[:, 0, 1] / pred - 1).max()))
    print(f"F2 {nst} stable series networks with amplifiers, mirrors and devices: S21/S12 = prod_i S21^(i)/S12^(i), max rel. err {worst:.1e}")
    ok &= worst < 1e-9
    # F3 single-frequency completeness of the series theory: T_B T_A^{-1} is reciprocal (det 1) for any A, B with equal ratio
    def T(S): return np.array([[S[0, 1] * S[1, 0] - S[0, 0] * S[1, 1], S[1, 1]], [-S[0, 0], 1]]) / S[0, 1]
    def S_of(T_): return np.array([[-T_[1, 0] / T_[1, 1], 1 / T_[1, 1]], [np.linalg.det(T_) / T_[1, 1], T_[0, 1] / T_[1, 1]]])
    errs = 0.0
    for _ in range(200):
        A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); B = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        B[1, 0] = B[0, 1] * A[1, 0] / A[0, 1]          # same ratio S21/S12
        Rm = S_of(T(B) @ np.linalg.inv(T(A)))
        errs = max(errs, abs(Rm[0, 1] - Rm[1, 0]) / abs(Rm).max())
    print(f"F3 equal S21/S12  =>  B = (reciprocal 2-port) after A at a fixed frequency: max asymmetry of the connector {errs:.1e}")
    ok &= errs < 1e-9
    # F4 noise price: N1(S) <= sum_i beta(D_i) (1 + nu_i), nu_i = quanta of quantum-limited amplifier noise reaching device i
    Zc = grid(2048)
    viol = 0; worst = 0.0; ident = 0.0; n = 0; tries = 0
    while n < 150 and tries < 3000:
        tries += 1
        blocks = []
        for _i in range(int(rng.integers(2, 5))):
            m = int(rng.integers(2, 5)); U = rand_unitary(m)
            s = rng.uniform(0.3, 1.0, m); s[rng.random(m) < .4] *= rng.uniform(1.0, 1.6)
            blocks.append((U * s) @ U.T)
        devs = [contr(int(rng.integers(2, 4))) for _i in range(int(rng.integers(1, 3)))]
        nsym = sum(b.shape[0] for b in blocks); allb = blocks + devs; ntot = sum(b.shape[0] for b in allb)
        ext = [int(e) for e in rng.choice(nsym, 2, replace=False)]
        internal = [p for p in range(ntot) if p not in ext]
        if len(internal) % 2: continue
        lines = [(int(p), int(q), int(rng.integers(1, 4)), np.exp(-rng.uniform(0, .05) + 2j * np.pi * rng.uniform()))
                 for p, q in rng.permutation(internal).reshape(-1, 2)]
        net = GNet(allb, ext, lines)
        if not net.stable(): continue
        n += 1
        dev_ports = []; off = nsym
        for D in devs:
            dev_ports.append(list(range(off, off + D.shape[0]))); off += D.shape[0]
        val = half_trace_anti(net.S(Zc)).mean() / noise_bound(net, devs, dev_ports, Zc)
        viol += val > 1 + 1e-9; worst = max(worst, val)
        netT = GNet([b.T for b in allb], ext, lines)
        for p in dev_ports:
            Eg = net.gain_energy(Zc, p); NT = netT.noise_at(Zc, p)
            ident = max(ident, float(np.abs(NT - Eg.conj()).max() / (1 + np.abs(Eg).max())))
    D = contr(2); g = 3.0
    tight = GNet([amp2(g), amp2(g), D], [0, 3], [(1, 4, 1, 1.0), (5, 2, 1, 1.0)])
    tv = half_trace_anti(tight.S(Z)).mean() / noise_bound(tight, [D], [[4, 5]], Z)
    print(f"F4 noise price on {n} random stable networks with gain: max N1/bound = {worst:.3f}, violations {viol}; "
          f"transposition identity E(N) = conj(nu-matrix of N^T): rel. err {ident:.1e}; amplifiers on both device ports: ratio {tv:.6f} (tight)")
    ok &= viol == 0 and ident < 1e-9 and abs(tv - 1) < 1e-9
    # F5 ring with gain inside: externally quantum limited at the peak, N1 above the passive budget, paid by noise at the device
    Zr = grid(2 ** 16)
    for r, g, dl in [(0.9, 1.1105, 0.02), (0.999, 1.00099, 0.05)]:
        D = twoport(1.0, np.exp(1j * dl))
        net = GNet([coupler(r), amp2(g), D], [0, 1], [(2, 4, 1, 1.0), (5, 6, 1, 1.0), (7, 3, 1, 1.0)])
        S = net.S(Zr); N = net.noise_ext(Zr)
        k = int(np.argmax(np.abs(S[:, 1, 0])))
        ql = np.trace(pos_part(S[k] @ S[k].conj().T - np.eye(2))).real
        nu = lam_max(net.noise_at(Zr, [6, 7])); b = half_trace_anti(D)
        print(f"F5 ring r={r}, gain {g}, element phase {dl}: stable {net.stable()}; at the peak |S21|^2 = {abs(S[k, 1, 0]) ** 2:.0f}, |S12|^2 = {abs(S[k, 0, 1]) ** 2:.2f}, "
              f"excess noise {np.trace(N[k]).real - ql:+.1e} quanta; N1 = {half_trace_anti(S).mean() / b:.1f} x budget <= (1+nu) = {1 + nu:.0f}")
        ok &= half_trace_anti(S).mean() <= b * (1 + nu)
    # F6 HN-type amplifier chain (series network): isolation = exp(2 gamma N) exactly, noise close to the quantum limit
    gam, g, rho = 0.15, 1.15, 0.2
    print(f"F6 HN-type amplifier chain: each bond = element (1, e^-{2 * gam}) + amplifier {g}; partial mirrors (rho={rho}) between bonds")
    for Nb in (2, 8, 16):
        blocks = [mirror(rho)]; lines = []; devs = []; dev_ports = []
        for _j in range(Nb):
            base = sum(bb.shape[0] for bb in blocks)
            D = twoport(1.0, np.exp(-2 * gam)); devs.append(D)
            blocks += [D, amp2(g), mirror(rho)]
            lines += [(base - 1, base, 1, 1.0), (base + 1, base + 2, 1, 1.0), (base + 3, base + 4, 1, 1.0)]
            dev_ports.append([base, base + 1])
        net = GNet(blocks, [0, sum(bb.shape[0] for bb in blocks) - 1], lines)
        S = net.S(Z); N = net.noise_ext(Z)
        ql = np.array([np.trace(pos_part(s @ s.conj().T - np.eye(2))).real for s in S])
        ratio = np.abs(S[:, 1, 0] / S[:, 0, 1])
        n1 = half_trace_anti(S).mean(); bud = sum(half_trace_anti(D) for D in devs)
        print(f"     N={Nb:2d}: stable {net.stable()}; |S21|^2 up to {np.max(np.abs(S[:, 1, 0]) ** 2):6.1f}; |S21/S12| = {ratio.mean():.4f} "
              f"(spread {np.ptp(ratio):.0e}; exp(2 gamma N) = {np.exp(2 * gam * Nb):.4f}); total noise / quantum limit = {np.mean(np.trace(N, axis1=1, axis2=2).real) / ql.mean():.3f}; "
              f"N1 = {n1:.2f} vs passive budget {bud:.2f}, noise-price bound {noise_bound(net, devs, dev_ports, Z):.1f}")
        ok &= abs(ratio.mean() / np.exp(2 * gam * Nb) - 1) < 1e-9 and net.stable()
    return ok


# ------------------------------------------------------------------ G  (lattices: stability, transfer matrices, winding)
def tmat(S, m):
    """transfer matrix (a_L, b_L) -> (b_R, a_R) of a 2m-port, ports ordered (L_1..L_m, R_1..R_m)"""
    S11, S12, S21, S22 = S[:m, :m], S[:m, m:], S[m:, :m], S[m:, m:]
    iS12 = np.linalg.inv(S12)
    return np.block([[S21 - S22 @ iS12 @ S11, S22 @ iS12], [-iS12 @ S11, iS12]])


def net_winding(T, m):
    mu = np.abs(np.linalg.eigvals(T))
    return int(np.sum(mu < 1)) - m, np.sort(mu)


def mirror_chain(Nb, gam, g, rho):
    blocks = [mirror(rho)]; lines = []
    for _j in range(Nb):
        base = sum(bb.shape[0] for bb in blocks)
        blocks += [twoport(1.0, np.exp(-2 * gam)), amp2(g), mirror(rho)]
        lines += [(base - 1, base, 1, 1.0), (base + 1, base + 2, 1, 1.0), (base + 3, base + 4, 1, 1.0)]
    return GNet(blocks, [0, sum(bb.shape[0] for bb in blocks) - 1], lines)


def part_G():
    ok = True
    # G1 robust stability of matched two-ports: stable for every termination |rho| = r iff r^2 max|S12 S21| < 1
    agree = 0; n = 0
    for _ in range(60):
        k = int(rng.integers(1, 4)); r = rng.uniform(.3, 1)
        blocks = [twoport(rng.uniform(.8, 1.6), rng.uniform(.3, 1.2)) for _i in range(k)]
        lines = [(2 * i - 1, 2 * i, int(rng.integers(1, 3)), np.exp(2j * np.pi * rng.uniform())) for i in range(1, k)]
        S = GNet(blocks, [0, 2 * k - 1], lines).S(grid(1024))
        crit = r ** 2 * np.abs(S[:, 0, 1] * S[:, 1, 0]).max() < 1
        all_stable = True
        for th1 in np.linspace(0, TWO_PI, 6, endpoint=False):
            for th2 in np.linspace(0, TWO_PI, 4, endpoint=False):
                net = GNet([np.array([[r * np.exp(1j * th1)]])] + blocks + [np.array([[r * np.exp(1j * th2)]])], [],
                           [(0, 1, 1, 1.0)] + [(p + 1, q + 1, d, a) for p, q, d, a in lines] + [(2 * k, 2 * k + 1, 1, 1.0)])
                all_stable &= net.stable(n=1024)
        n += 1; agree += crit == all_stable
    print(f"G1 matched two-ports closed by reflectors |rho|=r (24 phase pairs each): stable for all phases <=> r^2 max|S12 S21| < 1 in {agree}/{n}")
    ok &= agree == n
    # G2 transfer matrices: det T = det S21 / det S12; reciprocal (any gain) => T^T Omega T = Omega, spectrum closed under mu -> 1/mu
    for m in (1, 2, 3):
        Om = np.block([[np.zeros((m, m)), np.eye(m)], [-np.eye(m), np.zeros((m, m))]])
        es = ep = ed = 0.0
        for _ in range(100):
            A = rng.normal(size=(2 * m, 2 * m)) + 1j * rng.normal(size=(2 * m, 2 * m)); Sr = (A + A.T) * rng.uniform(.2, 1.5)
            T = tmat(Sr, m); mu = np.linalg.eigvals(T)
            es = max(es, np.abs(T.T @ Om @ T - Om).max() / np.abs(T).max() ** 2)
            ep = max(ep, np.abs(np.sort_complex(mu) - np.sort_complex(1 / mu)).max() / np.abs(mu).max())
            B = rng.normal(size=(2 * m, 2 * m)) + 1j * rng.normal(size=(2 * m, 2 * m))
            ed = max(ed, abs(np.linalg.det(tmat(B, m)) / (np.linalg.det(B[m:, :m]) / np.linalg.det(B[:m, m:])) - 1))
        print(f"G2 m={m}: reciprocal => T^T Omega T = Omega (err {es:.0e}), mu -> 1/mu (err {ep:.0e});  det T = det S21/det S12 (err {ed:.0e})")
        ok &= es < 1e-9 and ep < 1e-8 and ed < 1e-9
    # G3 network winding w = #{|mu|<1} - m vanishes for passive cells, for reciprocal cells, and for one-channel cells with |S21/S12| = 1
    def count(make, m, trials=1000):
        c = 0
        for _ in range(trials):
            c += net_winding(tmat(make(), m), m)[0] != 0
        return c
    def passive4():
        return contr(4)
    def recip4():
        A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); return (A + A.T) * rng.uniform(.5, 2.5) / np.linalg.norm(A + A.T, 2)
    def phase_cell(m):
        def f():
            A = rng.normal(size=(2 * m, 2 * m)) + 1j * rng.normal(size=(2 * m, 2 * m)); R = (A + A.T) * rng.uniform(1, 2.5) / np.linalg.norm(A + A.T, 2)
            Lay = np.zeros((2 * m, 2 * m), complex)
            for c in range(m):
                Lay[m + c, c] = 1.0; Lay[c, m + c] = np.exp(0.3j) if c == 0 else 1.0
            return tmat_to_S(tmat(Lay, m) @ tmat(R, m), m)
        return f
    def tmat_to_S(T, m):
        T11, T12, T21, T22 = T[:m, :m], T[:m, m:], T[m:, :m], T[m:, m:]
        iT22 = np.linalg.inv(T22)
        return np.block([[-iT22 @ T21, iT22], [T11 - T12 @ iT22 @ T21, T12 @ iT22]])
    c_pas = count(passive4, 2); c_rec = count(recip4, 2); c_1 = count(phase_cell(1), 1); c_2 = count(phase_cell(2), 2)
    print(f"G3 cells with network winding w != 0 (1000 random each): passive two-channel {c_pas}; reciprocal two-channel with gain {c_rec}; "
          f"one-channel phase-only with gain {c_1}; two-channel phase-only with gain {c_2}")
    ok &= c_pas == 0 and c_rec == 0 and c_1 == 0 and c_2 > 0
    # G4 HN-type chain: where the cell's network winding is -1, the end-to-end forward gain grows at the rate log|mu_1|
    g, rho, gam = 1.15, 0.2, 0.15
    cell = GNet([twoport(1.0, np.exp(-2 * gam)), amp2(g), mirror(rho)], [0, 5], [(1, 2, 1, 1.0), (3, 4, 1, 1.0)])
    z = grid(512); Sc = cell.S(z)
    w = np.zeros(len(z), int); lmu = np.zeros(len(z))
    for k, zz in enumerate(z):
        wk, mu = net_winding(tmat(np.array([[0, zz], [zz, 0]]), 1) @ tmat(Sc[k], 1), 1); w[k] = wk; lmu[k] = np.log(mu[0])
    S16 = mirror_chain(16, gam, g, rho).S(z); S32 = mirror_chain(32, gam, g, rho).S(z)
    rf = (np.log(np.abs(S32[:, 1, 0])) - np.log(np.abs(S16[:, 1, 0]))) / 16
    print(f"G4 HN-type chain (gamma={gam}, g={g}, rho={rho}): w=-1 on {np.mean(w == -1):.2f} of the band; there the forward growth rate (N=16->32) "
          f"is {rf[w == -1].mean():.4f} vs log|mu_1| = {lmu[w == -1].mean():.4f} (max deviation {np.abs(rf - lmu)[w == -1].max():.0e}); "
          f"where w=0 it is <= {rf[w == 0].max():.4f}")
    ok &= np.all(rf[w == -1] > 0) and np.all(rf[w == 0] < 0)
    # G5 stability of long chains: sharp threshold at flux per bond 2 gamma = log-gain per bond 2 log g
    print(f"G5 chain stability, 2 log g = {2 * np.log(g):.3f}:")
    for gm in (0.0, 0.13, 0.145):
        res = [(Nb, mirror_chain(Nb, gm, g, rho).stable()) for Nb in (8, 24, 64)]
        print(f"     flux per bond 2 gamma = {2 * gm:.2f}: " + ", ".join(f"N={a}: {'stable' if b else 'UNSTABLE'}" for a, b in res))
        if gm == 0.145: ok &= all(b for _a, b in res)
        if gm == 0.0: ok &= not res[-1][1]
    # G6 flux per passive cell: lambda <= -log(1 - 2 beta / T) for flat cells; magnitude element saturates it, phase-only MZI cell is quadratic
    for gm in (0.05, 0.2):
        D = twoport(1.0, np.exp(-2 * gm)); b = half_trace_anti(D)
        print(f"G6 magnitude element (gamma={gm}): flux {2 * gm:.4f} = bound {-np.log(1 - 2 * b):.4f}")
    for dl in (0.1, 0.4):
        el = twoport(1.0, np.exp(1j * dl))
        S = Net([splitter(), splitter(), el, twoport(1.0, 1.0)], [0, 3],
                [(1, 6, 1, 1.0), (7, 4, 1, 1.0), (2, 8, 1, 1.0), (9, 5, 1, 1.0)]).S(grid(64))
        lam = np.log(np.abs(S[:, 1, 0] / S[:, 0, 1])).mean(); b = half_trace_anti(el)
        print(f"G6 phase-only element in an MZI cell (phase {dl}): |S21| = {np.abs(S[:, 1, 0]).mean():.4f}, flux {lam:.5f} = -log cos(phase/2) "
              f"= {-np.log(np.cos(dl / 2)):.5f}; budget {b:.4f}; passive bound {-np.log(1 - 2 * b):.4f}")
        ok &= abs(lam + np.log(np.cos(dl / 2))) < 1e-9
    return ok


def main():
    parts = {"A": part_A, "B": part_B, "C": part_C, "D": part_D, "E": part_E, "F": part_F, "G": part_G}
    sel = sys.argv[1:] or list(parts)
    res = {}
    for p in sel:
        t0 = time.time()
        res[p] = bool(parts[p]())
        print(f"   [part {p}: {'consistent with claims' if res[p] else 'CHECK FAILED'}, {time.time() - t0:.0f}s]")
    print("SUMMARY: " + ", ".join(f"{p}={'ok' if v else 'FAIL'}" for p, v in res.items()))
    return 0 if all(res.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
