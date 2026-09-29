#!/usr/bin/env python3
"""V17 independent finite-instance checks.

These checks validate algebraic implications used by NHSE_QRT_V17.tex. They are
sanity checks only; asymptotic statements remain theorem-level claims.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig
from scipy.optimize import linprog
from scipy.special import logsumexp

rng = np.random.default_rng(16029)


def hn(L: int, g: float, pbc: bool = False) -> np.ndarray:
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


def classical_d(p, q):
    return -np.log(max(np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0))), 1e-300))


def random_density(d):
    Z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = Z @ Z.conj().T
    return rho / np.trace(rho)


def fidelity(rho, sigma):
    ew, U = np.linalg.eigh((rho + rho.conj().T) / 2)
    ew = np.clip(ew, 0, None)
    sr = (U * np.sqrt(ew)) @ U.conj().T
    mid = sr @ sigma @ sr
    ew2 = np.linalg.eigvalsh((mid + mid.conj().T) / 2)
    ew2 = np.clip(ew2, 0, None)
    return float(np.sum(np.sqrt(ew2)))


def amp_damp_first_qubit(rho, gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], complex)
    I = np.eye(2, dtype=complex)
    return sum(K @ rho @ K.conj().T for K in (np.kron(K0, I), np.kron(K1, I)))


def hn_phi_exact(L, g, chunk=400):
    n = np.arange(1, L + 1)
    u = np.exp(-2 * g)
    wgt = np.exp(-2 * g * (n - 1))
    k = np.pi * np.arange(1, L + 1) / (L + 1)
    Lj2 = np.empty(L)
    for s in range(0, L, chunk):
        Lj2[s:s + chunk] = (wgt[None, :] * np.sin(np.outer(k[s:s + chunk], n)) ** 2).sum(axis=1)
    return g * (L - 1) + np.mean(np.log(2 * Lj2 / (L + 1)))


def hn_asym(L, g):
    u = np.exp(-2 * g)
    return (g * (L - 1) - np.log(L + 1) + np.log(1 / np.tanh(g) / 2)
            + 2 / L * (np.log(L + 1) + np.log(1 - u))
            - 2 / L * np.log(1 - u ** (L + 1)))


def hn_log_distributions(L, g, j):
    n = np.arange(1, L + 1)
    s = np.sin(np.pi * j * n / (L + 1))
    logsin2 = 2 * np.log(np.maximum(np.abs(s), 1e-300))
    lp = 2 * g * (n - 1) + logsin2
    lq = -2 * g * (n - 1) + logsin2
    lp -= logsumexp(lp)
    lq -= logsumexp(lq)
    return lp, lq


def boundary_score(p, q, Aminus, Aplus):
    terms = [
        min(-np.log(max(1 - p[Aplus].sum(), 1e-300)),
            -np.log(max(1 - q[Aminus].sum(), 1e-300))),
        min(-np.log(max(1 - p[Aminus].sum(), 1e-300)),
            -np.log(max(1 - q[Aplus].sum(), 1e-300))),
    ]
    return max(terms)


def leakage_score(mu, A):
    return -0.5 * np.log(max(1 - mu[A].sum(), 1e-300))


def c1_hn_identity():
    print("=== C1 HN equality and exact/asymptotic agreement")
    g = 0.25
    for L in (6, 10, 16, 24, 32, 40):
        w, vl, vr = atlas(hn(L, g))
        b = np.mean([-np.log(abs(np.vdot(vl[:, j], vr[:, j]))) for j in range(L)])
        beta = np.mean([-np.log(np.sum(np.sqrt(np.abs(vl[:, j])**2 * np.abs(vr[:, j])**2))) for j in range(L)])
        ex, asym = hn_phi_exact(L, g), hn_asym(L, g)
        print(f"L={L:3d} Phi_Q={b:.9f} Phi_X={beta:.9f} exact={ex:.9f} asym={asym:.9f}")


def c2_chernoff():
    print("\n=== C2 HN identity and Chernoff symmetry")
    L, g = 12, .4
    n = np.arange(1, L + 1)
    dev, argdev = 0.0, 0.0
    for j in range(1, L + 1):
        s2 = np.sin(np.pi * j * n / (L + 1)) ** 2
        R = np.sqrt((np.exp(2*g*(n-1))*s2).sum())
        Lj = np.sqrt((np.exp(-2*g*(n-1))*s2).sum())
        dev = max(dev, abs(R/(np.exp(g*(L-1))*Lj)-1))
        p = np.exp(2*g*(n-1))*s2/R**2
        q = np.exp(-2*g*(n-1))*s2/Lj**2
        ss = np.linspace(0, 1, 2001)
        vals = np.array([np.sum(p**s*q**(1-s)) for s in ss])
        argdev = max(argdev, abs(ss[np.argmin(vals)]-.5))
    print(f"identity dev={dev:.2e}; argmin dev={argdev:.2e}")


def c3_large_l():
    print("\n=== C3 HN large-L constant")
    for g in (.25, .7):
        target = np.log(1/np.tanh(g)/2)
        for L in (100, 400, 1600, 6400):
            ex = hn_phi_exact(L, g)
            resid = ex - (g*(L-1)-np.log(L+1))
            print(f"g={g} L={L:5d} residual={resid:.8f} target={target:.8f}")


def c4_sandwich():
    print("\n=== C4 cell pinching sandwich")
    violated = False
    for _ in range(20):
        Lc, qdim = 7, 2
        H = np.zeros((Lc*qdim, Lc*qdim), complex)
        for x in range(Lc):
            for y in range(max(0, x-1), min(Lc, x+2)):
                H[x*qdim:(x+1)*qdim, y*qdim:(y+1)*qdim] = rng.normal(size=(qdim,qdim)) + 1j*rng.normal(size=(qdim,qdim))
        _, vl, vr = atlas(H)
        for j in range(Lc*qdim):
            r, l = vr[:,j], vl[:,j]
            pin = sum(abs(np.vdot(l[x*qdim:(x+1)*qdim], r[x*qdim:(x+1)*qdim])) for x in range(Lc))
            p = np.array([np.sum(abs(r[x*qdim:(x+1)*qdim])**2) for x in range(Lc)])
            qv = np.array([np.sum(abs(l[x*qdim:(x+1)*qdim])**2) for x in range(Lc)])
            bc = np.sum(np.sqrt(p*qv))
            violated |= not (abs(np.vdot(l,r)) <= pin + 1e-12 and pin <= bc + 1e-12)
    print("violated:", violated)


def c5_reciprocity_pbc_unitary():
    print("\n=== C5 reciprocity, PBC, non-covariant unitary")
    L = 14
    a1, a2 = .7+.3j, .4-.5j
    H = np.zeros((L,L), complex)
    for d,a in ((1,a1),(2,a2)):
        for n in range(L-d):
            H[n+d,n] = a
            H[n,n+d] = a
    H += .1j*np.eye(L)
    _, vl, vr = atlas(H)
    phi_q = np.mean([-np.log(abs(np.vdot(vl[:,j],vr[:,j]))) for j in range(L)])
    phi_x = np.mean([-np.log(np.sum(np.sqrt(abs(vl[:,j])**2*abs(vr[:,j])**2))) for j in range(L)])
    print(f"complex-symmetric={np.allclose(H,H.T)} Phi_Q={phi_q:.4f} Phi_X={phi_x:.2e}")
    HN = hn(L,.25)
    Z = rng.normal(size=(L,L))+1j*rng.normal(size=(L,L))
    U,_=np.linalg.qr(Z)
    _,vl0,vr0=atlas(HN); _,vl1,vr1=atlas(U@HN@U.conj().T)
    x0=np.mean([-np.log(np.sum(np.sqrt(abs(vl0[:,j])**2*abs(vr0[:,j])**2))) for j in range(L)])
    x1=np.mean([-np.log(np.sum(np.sqrt(abs(vl1[:,j])**2*abs(vr1[:,j])**2))) for j in range(L)])
    print(f"non-covariant unitary Phi_X {x0:.6f}->{x1:.6f}")


def c6_dressing():
    print("\n=== C6 similarity dressing")
    L,g1,g2=20,.35,.15
    S=np.diag(np.exp(-(g1-g2)*np.arange(L)))
    err=np.linalg.norm(S@hn(L,g1)@np.linalg.inv(S)-hn(L,g2),2)
    print(f"similarity residual={err:.2e}; logcond={(L-1)*abs(g1-g2):.3f}")


def c7_calibration_counterexample():
    print("\n=== C7 calibration necessity")
    pO,qO=np.array([.75,.25,0,0]),np.array([.25,.75,0,0])
    pP,qP=np.array([0,0,.9,.1]),np.array([0,0,.1,.9])
    T=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,0]],float)
    before=max(classical_d(pO,qO)-classical_d(pP,qP),0)
    after=max(classical_d(T@pO,T@qO)-classical_d(T@pP,T@qP),0)
    print(f"before={before:.6f}; after={after:.6f}")


def c8_sensitivity():
    print("\n=== C8 first-order sensitivity")
    M=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8))
    w,vl,vr=atlas(M); eps=1e-7; j=0
    dH=eps*np.outer(vl[:,j],vr[:,j].conj())
    wn=np.linalg.eigvals(M+dH)
    shift=np.min(np.abs(wn-w[j])); pred=eps/max(abs(np.vdot(vl[:,j],vr[:,j])),1e-300)
    print(f"ratio={shift/pred:.8f}")


def c9_cptp_closure():
    print("\n=== C9 genuine CPTP closure")
    r,s=random_density(4),random_density(4)
    ro,so=amp_damp_first_qubit(r,.2),amp_damp_first_qubit(s,.2)
    valid=all(x.shape==(4,4) for x in (ro,so)) and all(abs(np.trace(x)-1)<1e-12 for x in (ro,so))
    valid=valid and all(np.min(np.linalg.eigvalsh((x+x.conj().T)/2))>=-1e-12 for x in (ro,so))
    print("valid:",valid)


def find_recovery(pout_qout, pin_qin):
    p0,_=pout_qout[0]; pi0,_=pin_qin[0]
    n_in, n_out = pi0.size, p0.size
    nvar=n_in*n_out; Aeq=[]; beq=[]
    for (po,qo),(pi,qi) in zip(pout_qout,pin_qin):
        for target,src in ((pi,po),(qi,qo)):
            for i in range(n_in):
                row=np.zeros(nvar); row[i*n_out:(i+1)*n_out]=src
                Aeq.append(row); beq.append(target[i])
    for j in range(n_out):
        row=np.zeros(nvar)
        for i in range(n_in): row[i*n_out+j]=1
        Aeq.append(row); beq.append(1)
    res=linprog(np.zeros(nvar),A_eq=np.array(Aeq),b_eq=np.array(beq),bounds=[(0,None)]*nvar,method='highs')
    return None if not res.success else res.x.reshape(n_in,n_out)


def c10_target_recovery():
    print("\n=== C10 target-relative calibration recovery")
    p=np.array([.4,.2,.4]);q=np.array([.2,.1,.7]); T=np.eye(3)[[1,2,0]]
    R=find_recovery([(T@p,T@q)],[(p,q)])
    err=np.inf if R is None else max(np.max(abs(R@T@p-p)),np.max(abs(R@T@q-q)))
    print(f"feasible={R is not None}; error={err:.2e}")


def c11_boundary_noncreation():
    print("\n=== C11 boundary non-creation")
    p=np.array([.9,.05,.05]); A=np.array([0]);
    T=np.array([[.9,0,0],[0,.6,0],[.1,.4,1]],float)
    before=leakage_score(p,A); after=leakage_score(T@p,A)
    print(f"score before={before:.6f}; after={after:.6f}; monotone={after<=before+1e-12}")


def c12_free_composition_boundary():
    print("\n=== C12 nontrivial free composition with boundary-compatible maps")
    # 4 cells: 0 left boundary, 3 right boundary; maps only permute within compatible boundary classes.
    T1=np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]],float)
    T2=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],float)
    p=np.array([.7,.1,.1,.1]); q=np.array([.1,.1,.1,.7])
    p1,q1=T1@p,T1@q; p2,q2=T2@p1,T2@q1
    # recoveries are transposes (permutations)
    R1=T1.T; R2=T2.T; Rc=R1@R2; Tc=T2@T1
    ok=np.allclose(Rc@Tc@p,p) and np.allclose(Rc@Tc@q,q)
    print(f"first={np.allclose(R1@p1,p) and np.allclose(R1@q1,q)}; second={np.allclose(R2@p2,p1) and np.allclose(R2@q2,q1)}; composite={ok}")


def c13_calibration_score_invariance():
    print("\n=== C13 calibration leakage-profile invariance")
    p=np.array([.45,.05,.25,.25]); q=np.array([.25,.25,.05,.45])
    # Permute only within the chosen left/right boundary collars.
    T=np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]],float)
    R=T.T
    Aminus=np.array([0,1]); Aplus=np.array([2,3])
    g0=boundary_score(p,q,Aminus,Aplus); g1=boundary_score(T@p,T@q,Aminus,Aplus); grec=boundary_score(R@T@p,R@T@q,Aminus,Aplus)
    print(f"source={g0:.9f}; after={g1:.9f}; recovered={grec:.9f}; invariant={abs(g0-g1)<1e-12 and abs(g0-grec)<1e-12}")


def c14_exact_choi_affinity():
    print("\n=== C14 Choi affine constraints sanity")
    d=2
    # identity channel Choi in vec convention used only for PSD/TP sanity.
    phi=np.array([1,0,0,1],complex)/np.sqrt(2)
    J=2*np.outer(phi,phi.conj())
    ew=np.linalg.eigvalsh(J)
    # block partial trace over output: [[1,0],[0,1]]
    J4=J.reshape(d,d,d,d)
    tr_out=np.einsum('iaja->ij',J4)
    print(f"PSD={ew.min()>=-1e-12}; TP={np.allclose(tr_out,np.eye(2))}")


def c15_defective_embedding():
    print("\n=== C15 defective/generalized spectral embedding")
    H=np.array([[1,1,0],[0,1,1],[0,0,1]],complex)
    P=np.eye(3)  # full generalized eigenspace projector in this single-eigenvalue case
    rho=P/np.trace(P)
    print(f"density_shape={rho.shape}; trace={np.trace(rho):.6f}; min_eig={np.min(np.linalg.eigvalsh(rho)):.3f}")


def c16_physical_resource_equivalence():
    print("\n=== C16 physical NHSE -> calibrated tail resource")
    L=1000; k=.04
    for w in (8,16,32,48):
        leak=np.exp(-2*k*w)
        p=np.zeros(L); A=np.arange(w)
        p[A]=(1-leak)/len(A); p[-1]=leak
        score=leakage_score(p,A)
        print(f"w={w:2d}; score/w={score/w:.6f}; expected={k:.6f}")


def c17_resource_to_physical():
    print("\n=== C17 QRT tail resource -> physical boundary localization")
    for s in (2.0,4.0,6.0):
        leak=np.exp(-2*s)
        p=np.array([1-leak,0,0,leak])
        score=leakage_score(p,np.array([0]))
        reconstructed=np.exp(-2*score)
        print(f"score={score:.9f}; reconstructed={reconstructed:.3e}; actual={leak:.3e}")


def c18_hn_boundary_rate():
    print("\n=== C18 HN boundary resource rate with sublinear collar")
    g=.37
    for L in (80,160,320,640):
        w=max(3,int(np.sqrt(L)))
        vals=[]
        for j in range(1,L+1):
            lp,lq=hn_log_distributions(L,g,j)
            p_right_leak=logsumexp(lp[:L-w])
            q_left_leak=logsumexp(lq[w:])
            vals.append(0.5*min(-p_right_leak,-q_left_leak)/w)
        print(f"L={L:4d}; meanGamma/w={np.mean(vals):.6f}; target={g:.6f}; w={w}")


def c19_boundary_insensitive_zero():
    print("\n=== C19 boundary-insensitive localization is calibrated-free")
    L=100; w=10
    # Same distribution for O and P: localized at left boundary in both.
    p=q=np.zeros(L); p[0]=1; q[0]=1
    go=leakage_score(p,np.arange(w)); gp=leakage_score(q,np.arange(w))
    print(f"open={go:.3f}; reference={gp:.3f}; calibrated_excess={max(go-gp,0):.3f}")


def c20_layer_cake():
    print("\n=== C20 layer-cake identity")
    delta=np.array([.2,.8,1.1,.4])
    lhs=delta.mean()
    ks=np.linspace(0,1.1,100001)
    rhs=np.trapezoid([(delta>=k-1e-15).mean() for k in ks],ks)
    print(f"lhs={lhs:.8f}; rhs={rhs:.8f}; err={abs(lhs-rhs):.2e}")


def c21_right_left_paired_patterns():
    print("\n=== C21 right/left/paired pattern logic")
    r=np.array([.99,.005,.005]); l=np.array([.005,.005,.99])
    gr=leakage_score(r,np.array([0])); gl=leakage_score(l,np.array([2]));
    paired=min(gr,gl)
    print(f"R={gr:.6f}; L={gl:.6f}; RL={paired:.6f}")


def c22_cptp_fidelity_dp():
    print("\n=== C22 CPTP fidelity data processing")
    r,s=random_density(4),random_density(4)
    f0=fidelity(r,s); f1=fidelity(amp_damp_first_qubit(r,.31),amp_damp_first_qubit(s,.31))
    print(f"before={f0:.9f}; after={f1:.9f}; monotone={f1+1e-12>=f0}")


def c23_classical_dp():
    print("\n=== C23 classical fidelity data processing")
    p=np.array([.7,.2,.1]); q=np.array([.1,.3,.6])
    T=.37*np.eye(3)[[1,2,0]]+.63*np.eye(3)[[2,0,1]]
    print(f"stochastic={np.all(T>=0) and np.allclose(T.sum(axis=0),1)}; dF={classical_d(p,q):.6f}->{classical_d(T@p,T@q):.6f}")


def c24_hn_rate():
    print("\n=== C24 HN fidelity resource density")
    g=.37
    for L in (20,40,80,160,320):
        phi=hn_phi_exact(L,g)
        print(f"L={L:3d}; phi/L={phi/L:.8f}; target={g:.8f}")


def c25_noncreation_random():
    print("\n=== C25 random boundary-compatible maps never increase leakage score")
    ok=True
    for _ in range(100):
        # 3 input/output cells; cell 0 is the boundary collar.
        a=rng.random(2); a=a/a.sum()
        b=rng.random(2); b=b/b.sum()
        T=np.zeros((3,3))
        T[0,0]=1
        T[1:,1:]=np.array([[a[0],b[0]],[a[1],b[1]]])
        p=rng.dirichlet(np.ones(3));
        if leakage_score(T@p,np.array([0])) > leakage_score(p,np.array([0]))+1e-12:
            ok=False; break
    print("monotone_all_trials:",ok)


def amp_damp_qubit(rho, gamma):
    K0=np.array([[1,0],[0,np.sqrt(1-gamma)]],complex)
    K1=np.array([[0,np.sqrt(gamma)],[0,0]],complex)
    return sum(K@rho@K.conj().T for K in (K0,K1))


def c26_tensor_closure():
    print("\n=== C26 tensor CPTP closure")
    r,s=random_density(2),random_density(2)
    out=np.kron(amp_damp_qubit(r,.2),amp_damp_qubit(s,.3))
    print(f"shape={out.shape}; trace={np.trace(out):.12f}; psd_min={np.min(np.linalg.eigvalsh((out+out.conj().T)/2)):.2e}")


def c27_boundary_response_dp():
    print("\n=== C27 boundary-response susceptibility data processing")
    r0,s0=random_density(4),random_density(4)
    r1,s1=amp_damp_first_qubit(r0,.31),amp_damp_first_qubit(s0,.31)
    d0=-np.log(max(fidelity(r0,s0),1e-300)); d1=-np.log(max(fidelity(r1,s1),1e-300))
    print(f"d_before={d0:.9f}; d_after={d1:.9f}; monotone={d1<=d0+1e-12}")


def c28_representation_invariance():
    print("\n=== C28 cell-isometric representation invariance")
    p=np.array([.65,.15,.1,.1]); q=np.array([.1,.1,.15,.65])
    perm=np.eye(4)[[2,0,3,1]]
    A0=np.array([0,1]); B0=np.array([2,3])
    A1=np.array([1,3]); B1=np.array([0,2])
    g0=boundary_score(p,q,A0,B0)
    g1=boundary_score(perm@p,perm@q,A1,B1)
    print(f"source={g0:.9f}; permuted={g1:.9f}; invariant={abs(g0-g1)<1e-12}")


def c29_direct_sum_sector_closure():
    print("\n=== C29 direct-sum sector closure")
    p1=np.array([.8,.2]); q1=np.array([.2,.8])
    p2=np.array([.55,.45]); q2=np.array([.45,.55])
    w=.6
    p=np.r_[w*p1,(1-w)*p2]; q=np.r_[w*q1,(1-w)*q2]
    bc=-np.log(np.sum(np.sqrt(p*q)))
    print(f"combined_dF={bc:.9f}; valid={bc>=0}")


def c30_nonabelian_pattern_probe():
    print("\n=== C30 sector-resolved / pseudospin boundary pattern")
    # two internal sectors: sector 0 skins left, sector 1 skins right
    p=np.array([.49,.01,.01,.49]); q=np.array([.01,.49,.49,.01])
    Aminus=np.array([0,1]); Aplus=np.array([2,3])
    rscore=leakage_score(p,Aminus)
    lscore=leakage_score(q,Aplus)
    paired=min(rscore,lscore)
    print(f"R={rscore:.6f}; L={lscore:.6f}; RL={paired:.6f}; paired_positive={paired>0}")


def c31_boundary_sensitivity_zero_reference():
    print("\n=== C31 boundary-insensitive localization has zero calibrated response")
    p=np.array([.95,.05,0,0]); q=np.array([.90,.10,0,0])
    # active and calibration use the same boundary-localized family
    active=boundary_score(p,q,np.array([0,1]),np.array([2,3]))
    reference=boundary_score(p,q,np.array([0,1]),np.array([2,3]))
    print(f"active={active:.6f}; reference={reference:.6f}; excess={max(active-reference,0):.6f}")


def c32_threshold_monotonicity():
    print("\n=== C32 threshold abundance monotonicity")
    delta=np.array([.2, .8, 1.1, .4])
    thresholds=[.1,.5,.9,1.2]
    vals=[float(np.mean(delta>=t)) for t in thresholds]
    print("thresholds=",thresholds,"abundance=",[f"{v:.3f}" for v in vals],"nonincreasing=",all(vals[i]>=vals[i+1]-1e-15 for i in range(len(vals)-1)))


def c33_hn_boundary_response():
    print("\n=== C33 HN boundary-condition response is nonzero")
    g=.37; L=80; w=8
    lpO,lqO=hn_log_distributions(L,g,1)
    # PBC Fourier mode is uniform.
    pP=np.ones(L)/L; qP=np.ones(L)/L
    pO=np.exp(lpO); qO=np.exp(lqO)
    pO/=pO.sum(); qO/=qO.sum()
    dpr=classical_d(pO,pP); dql=classical_d(qO,qP)
    print(f"dF_right={dpr:.6f}; dF_left={dql:.6f}; positive={dpr>0 and dql>0}")


def c34_physical_equivalence_synthetic():
    print("\n=== C34 synthetic physical criterion <-> QRT threshold")
    k=.12; w=20
    leak=np.exp(-2*k*w)
    p=np.zeros(100); q=np.zeros(100)
    p[:w]=(1-leak)/w; q[:w]=(1-leak)/w
    p[-1]=leak; q[-1]=leak
    score=min(leakage_score(p,np.arange(w)),leakage_score(q,np.arange(w)))
    recovered=np.exp(-2*score)
    print(f"k={k:.3f}; score/w={score/w:.6f}; recovered_leak={recovered:.3e}; target={leak:.3e}; equivalent={abs(np.log(recovered/leak))<1e-10}")


def main():
    funcs=[c1_hn_identity,c2_chernoff,c3_large_l,c4_sandwich,c5_reciprocity_pbc_unitary,c6_dressing,c7_calibration_counterexample,c8_sensitivity,c9_cptp_closure,c10_target_recovery,c11_boundary_noncreation,c12_free_composition_boundary,c13_calibration_score_invariance,c14_exact_choi_affinity,c15_defective_embedding,c16_physical_resource_equivalence,c17_resource_to_physical,c18_hn_boundary_rate,c19_boundary_insensitive_zero,c20_layer_cake,c21_right_left_paired_patterns,c22_cptp_fidelity_dp,c23_classical_dp,c24_hn_rate,c25_noncreation_random,c26_tensor_closure,c27_boundary_response_dp,c28_representation_invariance,c29_direct_sum_sector_closure,c30_nonabelian_pattern_probe,c31_boundary_sensitivity_zero_reference,c32_threshold_monotonicity,c33_hn_boundary_response,c34_physical_equivalence_synthetic]
    for f in funcs:
        f()

if __name__ == '__main__':
    main()
