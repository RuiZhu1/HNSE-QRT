#!/usr/bin/env python3
"""Data and figures for the paper draft (fig2_budget_noise.pdf, fig3_lattice.pdf).
Uses the network code of ../route2_nonreciprocity_toys.py.  Data are cached in figdata.npz;
delete it to recompute (the stability map takes a few minutes)."""
import importlib.util, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("toys", os.path.join(HERE, "..", "route2_nonreciprocity_toys.py"))
T = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]; spec.loader.exec_module(T)
rng = np.random.default_rng(2026)
T.rng = rng
CACHE = os.path.join(HERE, "figdata.npz")


def ring_curves(delta=0.05, rs=(0.9, 0.99, 0.999), n=2 ** 18):
    phi = (np.arange(n) + 0.5) * 2 * np.pi / n - np.pi
    z = np.exp(1j * phi); out = {}
    for r in rs:
        a = r
        tf = (r - a * z) / (1 - r * a * z); tb = (r - a * np.exp(1j * delta) * z) / (1 - r * a * np.exp(1j * delta) * z)
        out[r] = np.abs(tf - tb) / 2 / (a * np.sin(delta / 2))
    mzi = np.full_like(phi, 0.5)          # MZI with the same element: 1/2|S21-S12| = |1-e^{i delta}|/4 = budget/2
    return phi, out, mzi


def random_points(n_pass=200, n_act=200):
    Z = T.grid(2048); P = []; A = []
    while len(P) < n_pass:
        devs = [T.contr(int(rng.integers(2, 4))) for _ in range(int(rng.integers(1, 3)))]
        net = T.random_embedding(devs, int(rng.integers(2, 4)), smin=float(rng.choice([0, .9])))
        P.append((T.half_trace_anti(net.S(Z)).mean(), sum(T.half_trace_anti(D) for D in devs)))
    tries = 0
    while len(A) < n_act and tries < 5000:
        tries += 1
        blocks = []
        for _ in range(int(rng.integers(2, 5))):
            m = int(rng.integers(2, 5)); U = T.rand_unitary(m)
            s = rng.uniform(0.3, 1.0, m); s[rng.random(m) < .5] *= rng.uniform(1.0, 2.0)
            blocks.append((U * s) @ U.T)
        devs = [T.contr(int(rng.integers(2, 4))) for _ in range(int(rng.integers(1, 3)))]
        nsym = sum(b.shape[0] for b in blocks); allb = blocks + devs; ntot = sum(b.shape[0] for b in allb)
        ext = [int(e) for e in rng.choice(nsym, 2, replace=False)]
        internal = [p for p in range(ntot) if p not in ext]
        if len(internal) % 2: continue
        lines = [(int(p), int(q), int(rng.integers(1, 4)), np.exp(-rng.uniform(0, .05) + 2j * np.pi * rng.uniform()))
                 for p, q in rng.permutation(internal).reshape(-1, 2)]
        net = T.GNet(allb, ext, lines)
        if not net.stable(n=2048): continue
        dp = []; off = nsym
        for D in devs:
            dp.append(list(range(off, off + D.shape[0]))); off += D.shape[0]
        A.append((T.half_trace_anti(net.S(Z)).mean(), T.noise_bound(net, devs, dp, Z)))
    return np.array(P), np.array(A)


def examples():
    Z = T.grid(2 ** 16); ex = {}
    D = T.contr(2); g = 3.0
    net = T.GNet([T.amp2(g), T.amp2(g), D], [0, 3], [(1, 4, 1, 1.0), (5, 2, 1, 1.0)])
    ex["tight"] = (T.half_trace_anti(net.S(Z)).mean(), T.noise_bound(net, [D], [[4, 5]], Z))
    D = T.twoport(1.0, np.exp(0.02j))
    net = T.GNet([T.coupler(0.9), T.amp2(1.1105), D], [0, 1], [(2, 4, 1, 1.0), (5, 6, 1, 1.0), (7, 3, 1, 1.0)])
    ex["ring"] = (T.half_trace_anti(net.S(Z)).mean(), T.noise_bound(net, [D], [[6, 7]], Z))
    Dm = T.twoport(1.0, np.exp(0.05j)); Gt = 100.0; gg = np.sqrt(Gt) / abs((Dm[1, 0] - Dm[0, 1]) / 2)
    net = T.GNet([T.splitter(), T.splitter(), Dm, T.twoport(-Dm[0, 1], -Dm[0, 1]), T.amp2(gg)], [0, 11],
                 [(1, 6, 1, 1.0), (7, 4, 1, 1.0), (2, 8, 1, 1.0), (9, 5, 1, 1.0), (3, 10, 1, 1.0)])
    ex["collapse"] = (T.half_trace_anti(net.S(Z)).mean(), T.noise_bound(net, [Dm], [[6, 7]], Z))
    return ex


def stability_map(xs, ys, rho=0.2, Ns=(12, 48)):
    M = np.zeros((len(ys), len(xs)), int)      # 2 = stable at Ns[-1], 1 = stable at Ns[0] only, 0 = unstable at Ns[0]
    for i, f in enumerate(ys):
        for j, x in enumerate(xs):
            g = np.exp(x / 2); gam = f / 2
            s0 = T.mirror_chain(Ns[0], gam, g, rho).stable()
            s1 = s0 and T.mirror_chain(Ns[1], gam, g, rho).stable()
            M[i, j] = 2 if s1 else (1 if s0 else 0)
        print(f"  stability map row {i + 1}/{len(ys)}", flush=True)
    return M


def winding_rates(g=1.15, rho=0.2, gam=0.15, n=512):
    cell = T.GNet([T.twoport(1.0, np.exp(-2 * gam)), T.amp2(g), T.mirror(rho)], [0, 5], [(1, 2, 1, 1.0), (3, 4, 1, 1.0)])
    z = T.grid(n); Sc = cell.S(z); w = np.zeros(n, int); lmu = np.zeros(n)
    for k, zz in enumerate(z):
        wk, mu = T.net_winding(T.tmat(np.array([[0, zz], [zz, 0]]), 1) @ T.tmat(Sc[k], 1), 1); w[k] = wk; lmu[k] = np.log(mu[0])
    S16 = T.mirror_chain(16, gam, g, rho).S(z); S32 = T.mirror_chain(32, gam, g, rho).S(z)
    rf = (np.log(np.abs(S32[:, 1, 0])) - np.log(np.abs(S16[:, 1, 0]))) / 16
    return np.angle(z), w, lmu, rf


if __name__ == "__main__":
    if os.path.exists(CACHE):
        d = dict(np.load(CACHE, allow_pickle=True))
    else:
        print("computing figure data ...", flush=True)
        phi, rc, mzi = ring_curves()
        P, A = random_points()
        ex = examples()
        xs = np.round(np.arange(0.04, 0.41, 0.04), 2); ys = np.round(np.arange(0.0, 0.45, 0.05), 2)
        M = stability_map(xs, ys)
        ph, w, lmu, rf = winding_rates()
        d = dict(phi=phi, r09=rc[0.9], r099=rc[0.99], r0999=rc[0.999], mzi=mzi, P=P, A=A,
                 ex_names=np.array(list(ex)), ex_vals=np.array(list(ex.values())), xs=xs, ys=ys, M=M,
                 ph=ph, w=w, lmu=lmu, rf=rf)
        np.savez_compressed(CACHE, **d)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b"})
    BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
    # ---------------- Fig. 2
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(wspace=0.3))
    a = ax[0]; phi = d["phi"]; sel = np.abs(phi) < 0.09
    for key, col, lab in [("r09", BLUE, r"ring, $r=0.9$"), ("r099", ORANGE, r"ring, $r=0.99$"), ("r0999", AQUA, r"ring, $r=0.999$")]:
        y = d[key]; a.plot(phi[sel], y[sel], color=col, lw=1.1, label=f"{lab}  (mean {y.mean():.2f})")
    a.plot(phi[sel], d["mzi"][sel], color=GRAY, lw=1.0, ls=(0, (4, 2)), label=f"Mach-Zehnder  (mean {d['mzi'].mean():.2f})")
    a.axhline(1.0, color="#0b0b0b", lw=0.6, ls=":")
    a.text(-0.087, 1.12, "element budget", ha="left", va="bottom", fontsize=7, color="#52514e")
    a.set_yscale("log"); a.set_ylim(0.006, 60); a.set_xlim(-0.09, 0.09); a.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
    a.set_xlabel(r"frequency $\varphi$ (rad per unit delay)"); a.set_ylabel(r"$\frac{1}{2}\|S_A(\varphi)\|_1\;/\;\beta(D)$")
    a.legend(frameon=False, loc="lower left", handlelength=1.8, borderaxespad=0.2)
    a.set_title("(a) passive ring and interferometer", fontsize=8, loc="left")
    b = ax[1]; P, A = d["P"], d["A"]
    b.scatter(P[:, 1], P[:, 0], s=9, color=BLUE, alpha=0.75, linewidths=0, label=r"passive: bound $\sum\beta$")
    b.scatter(A[:, 1], A[:, 0], s=9, color=ORANGE, alpha=0.75, linewidths=0, label=r"with gain: bound $\sum\beta(1+\nu)$")
    lo, hi = 3e-3, 3e4
    b.plot([lo, hi], [lo, hi], color="#0b0b0b", lw=0.7, label=r"$\mathcal{N}_1=$ bound")
    labels = {"tight": "amplifiers on\nboth ports", "ring": "ring amplifier", "collapse": "MZI + amplifier"}
    offs = {"tight": (-7, 2), "ring": (7, -3), "collapse": (-7, 5)}
    for nm, (n1, bd) in zip(d["ex_names"], d["ex_vals"]):
        b.scatter([bd], [n1], s=26, facecolor="white", edgecolor="#0b0b0b", linewidths=0.9, zorder=5)
        ha = "right" if offs[nm][0] < 0 else "left"
        b.annotate(labels[nm], (bd, n1), xytext=offs[nm], textcoords="offset points", fontsize=6.5, ha=ha, va="center", color="#0b0b0b")
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(lo, hi); b.set_ylim(lo, hi)
    b.set_xlabel("bound"); b.set_ylabel(r"$\mathcal{N}_1(S)$")
    b.legend(frameon=False, loc="upper left", handlelength=1.4, markerscale=1.4, borderaxespad=0.2)
    b.set_title("(b) random and constructed networks", fontsize=8, loc="left")
    for x in ax:
        x.spines["top"].set_visible(False); x.spines["right"].set_visible(False)
    fig.savefig(os.path.join(HERE, "fig2_budget_noise.pdf"), bbox_inches="tight")
    # ---------------- Fig. 3
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(wspace=0.3))
    a = ax[0]; xs, ys, M = d["xs"], d["ys"], d["M"]
    X, Y = np.meshgrid(xs, ys)
    for val, col, lab, mk in [(2, BLUE, r"stable at $N=48$", "o"), (1, ORANGE, r"stable at $N=12$ only", "s"),
                              (0, GRAY, r"unstable at $N=12$", "x")]:
        m = M == val
        a.scatter(X[m], Y[m], s=16, color=col, marker=mk, linewidths=0.9 if mk == "x" else 0, label=lab)
    t = np.linspace(0, 0.44, 10); a.plot(t, t, color="#0b0b0b", lw=0.7, label="flux rate = gain rate")
    a.set_xlim(0.0, 0.44); a.set_ylim(-0.02, 0.44)
    a.set_xlabel(r"log-gain per bond, $2\log g$"); a.set_ylabel(r"gauge flux per bond, $2\gamma$")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.3, columnspacing=1.0)
    a.set_title("(a) stability of long amplifier chains", fontsize=8, loc="left")
    b = ax[1]; ph, w, lmu, rf = d["ph"], d["w"], d["lmu"], d["rf"]
    o = np.argsort(ph); ph, w, lmu, rf = ph[o], w[o], lmu[o], rf[o]
    edges = np.flatnonzero(np.diff(np.r_[0, (w == -1).astype(int), 0]))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        b.axvspan(ph[s0] / np.pi, ph[min(s1, len(ph) - 1)] / np.pi, color="#e9e8e4", lw=0)
    b.plot(ph / np.pi, lmu, color=ORANGE, lw=1.1, label=r"$\log|\mu_1|$ of the cell")
    b.scatter(ph[::6] / np.pi, rf[::6], s=8, color=BLUE, linewidths=0, zorder=4, label=r"measured, $N=16\to32$")
    b.axhline(0, color="#52514e", lw=0.5)
    b.set_xlim(-1, 1); b.set_xlabel(r"frequency $\varphi/\pi$"); b.set_ylabel("forward growth rate per bond")
    handles, labs = b.get_legend_handles_labels()
    handles.append(Patch(color="#e9e8e4")); labs.append(r"winding $w=-1$")
    b.legend(handles, labs, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.4, columnspacing=1.0)
    b.set_title("(b) network winding and end-to-end gain", fontsize=8, loc="left")
    for x in ax:
        x.spines["top"].set_visible(False); x.spines["right"].set_visible(False)
    fig.savefig(os.path.join(HERE, "fig3_lattice.pdf"), bbox_inches="tight")
    print("wrote fig2_budget_noise.pdf, fig3_lattice.pdf")
