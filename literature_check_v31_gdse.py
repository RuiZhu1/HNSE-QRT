"""Literature check for V31: the geometry-dependent skin effect (GDSE) of reciprocal 2D lattices is invisible to V31's paired skin score.

Model: H(k) = 2 cos kx + 2i*gamma cos ky, i.e. symmetric couplings 1 (x) and i*gamma (y), plus weak real on-site
disorder that lifts the chiral zero modes (simple spectrum). H^T = H, so H is reciprocal in the sense of V31 Theorem 2.
Square OBC: separable, no skin (control). Triangle OBC (x,y>=0, x+y<=N-1): GDSE at the diagonal edge.
Checks: (1) |R|^2 == |L|^2 pointwise  => by V31 Lemma 1 the paired skin score is 0 for every collar pair;
(2) right-eigenvector occupancy of a collar C: occ(C) = sum_m ||Pi_C R_m||^2 / |C|, which equals 1 for ANY orthonormal basis.
Run: python literature_check_v31_gdse.py   (numpy)"""
import numpy as np
from numpy.linalg import eig, inv, norm

def build(sites, gamma):
    idx = {s: i for i, s in enumerate(sites)}
    H = np.zeros((len(sites), len(sites)), complex)
    for (x, y), i in idx.items():
        for (dx, dy, t) in ((1, 0, 1.0), (0, 1, 1j * gamma)):
            j = idx.get((x + dx, y + dy))
            if j is not None:
                H[i, j] = H[j, i] = t          # symmetric (reciprocal) couplings
    H += np.diag(np.random.default_rng(7).uniform(-0.05, 0.05, len(sites)))   # reciprocal on-site disorder: lifts chiral zero modes
    return H, idx

def analyse(name, sites, gamma, collars):
    H, idx = build(sites, gamma)
    assert np.allclose(H, H.T)
    lam, R = eig(H)
    Lm = inv(R).conj().T
    R = R / norm(R, axis=0); Lm = Lm / norm(Lm, axis=0)
    prof_diff = np.abs(np.abs(R) ** 2 - np.abs(Lm) ** 2).max()
    gap = min(np.sort(np.abs(lam[i] - np.delete(lam, i)))[0] for i in range(len(lam)))
    M = len(sites)
    out = []
    for cname, pred in collars.items():
        C = [idx[s] for s in sites if pred(s)]
        occ = (np.abs(R[C, :]) ** 2).sum() / len(C)
        frac50 = np.mean((np.abs(R[C, :]) ** 2).sum(0) >= 0.5)
        out.append(f"{cname}: |C|={len(C)} ({len(C)/M:.0%} of sites), occupancy={occ:.2f}, modes with >=50% weight there={frac50:.0%}")
    kap = 1 / np.abs(np.einsum('im,im->m', Lm.conj(), R))
    print(f"{name}: min eigenvalue gap={gap:.1e}; M={M}, max_m,site | |R|^2-|L|^2 | = {prof_diff:.1e}  -> paired score s = 0 for every collar pair (Lemma 1)")
    print(f"   median log10 kappa_m = {np.median(np.log10(kap)):.2f}, max = {np.log10(kap).max():.2f}")
    for o in out: print("   " + o)

N, g, w = 30, 0.6, 2
square = [(x, y) for x in range(N) for y in range(N)]
tri = [(x, y) for x in range(N) for y in range(N) if x + y <= N - 1]
analyse("square", square, g, {
    "left edge x<w": lambda s: s[0] < w, "bottom edge y<w": lambda s: s[1] < w,
    "corner x+y<w+1": lambda s: s[0] + s[1] <= w})
analyse("triangle", tri, g, {
    "left edge x<w": lambda s: s[0] < w, "bottom edge y<w": lambda s: s[1] < w,
    "diagonal edge x+y>N-1-w": lambda s: s[0] + s[1] > N - 1 - w,
    "corner (0,N-1) region": lambda s: s[0] < w + 1 and s[1] > N - 2 - w,
    "corner (N-1,0) region": lambda s: s[1] < w + 1 and s[0] > N - 2 - w})
