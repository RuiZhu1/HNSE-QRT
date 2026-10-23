# V31 文献核查（2026-10-23）

论文：`NHSE_QRT_V31.tex`
判定：**已知**＝已有文献给出同一结论；**部分**＝已有定性或特例结论，V31 给出精确或推广版本；**新**＝检索范围内没有找到先例；**冲突**＝已有文献与 V31 的表述矛盾。

检索范围：网页检索 arXiv、APS、Nature 系列、IOP、Springer 的摘要页，约 15 组关键词，包括 resource theory 与 non-Hermitian skin effect / non-Hermiticity / non-normality / biorthogonal / PT symmetry 的组合，(banded / k-) Toeplitz 本征向量局域的严格结果，互易系统中的趋肤效应，带噪声的 Toeplitz 矩阵。读过全文相关部分的：arXiv:2512.03757、2401.12626、2103.17148（主定理部分）、1910.02878（OKSS20，TRS† 部分）、2310.18627（补充材料）、2304.06926（模型部分）。其余条目只读了摘要页。网页检索做不了 arXiv 全文搜索，所以"新"的判定只在这个范围内成立。

## 0. 三条主要结论

1. **新颖性：没有找到先例。** 检索范围内没有"NHSE / 非厄米性 / 非正规性 / 双正交性"的资源论。最接近的三类工作都不处理非厄米生成元的本征数据：Wang–Wilde 2019（非对称可区分性，V31 已引）；Bu–Fan–Pati–Wu 2016（PT 型反幺正不对称性，态层面）；Miyazaki–Kuroiwa–Murao 2026-09-29（量子信道的时间反演破缺，动力学资源论）。
2. **最严重：V31 的资源看不到互易系统里的趋肤效应。** V31 定理 2 说互易且单谱时 R、L 的格点分布相同，于是配对得分恒为 0。但互易系统确有趋肤效应，二维几何依赖趋肤效应已有实验观测（PRL 131, 207201 (2023)）。我用 V31 自己的定义做了数值核实（第 2 节）。所以 V31 实际上是**定向（非互易）趋肤效应**的资源论，摘要第一句的 NHSE 定义和 "universal" 的说法都不成立。这是定义层面的问题，不是缺一个定理。
3. **Hypothesis H：严格结果只覆盖三对角链。** 带状 Toeplitz 的严格本征向量衰减结果都是针对半无限算子的；有限矩阵只有伪本征向量和本征值渐近。检索范围内没有有限带状 Toeplitz 矩阵（尤其复谱）本征向量边缘局域的现成定理。"Hypothesis" 这个标签是对的，证明它会是真正的新数学贡献。

## 1. 逐条核查

| # | V31 的内容 | 最接近的已有工作 | 判定 | 建议处理 |
|---|---|---|---|---|
| 0 | 定位：NHSE 的资源论 | Wang–Wilde 2019（已引）；Bu–Fan–Pati–Wu 2016（PT 不对称性，态）；Miyazaki–Kuroiwa–Murao 2026（信道的 T 破缺） | **新**（检索范围内） | 引言加一句相邻资源论，引后两篇，并说明区别：它们处理态或信道的反幺正对称性，V31 处理非厄米生成元的左右本征数据 |
| 1 | 定理 2(a)：互易且单谱 ⇒ R、L 的 collar 权重相同 ⇒ 无配对趋肤 | "一维互易体系没有（非 Z2 的）趋肤效应"是领域内常见说法（TRS†、TT*=+1 下 point-gap 绕数为零）；OKSS20 正文只处理 TT*=−1 | **部分**：本征向量层面、有限体系、任意维的表述似为新 | 保留；但见第 2 节，这个定理同时说明 V31 的资源对互易趋肤效应恒为零 |
| 2 | 定理 2(b)：U^T=−U ⇒ 没有单本征值 | OKSS20：TRS† 且 TT*=−1 时有 Kramers 简并，Z2 趋肤效应中一对 Kramers 伙伴分居两端 | **已知** | 写明是已知 Kramers 型简并的有限矩阵版本，引 OKSS20 |
| 3 | 定理 4：log κ ≥ s − ½log2，方向性残数 | Trefethen–Embree（Toeplitz 伪谱）、Budich–Bergholtz 2020（指数灵敏度传感）、Porras 2019 / Wanjura 2020（定向放大），均已引 | **部分**：定性联系已知，具体不等式是新的，但证明是初等的 | 维持 |
| 4 | 资源论结构（定理 8、9，命题 7） | Wang–Wilde 2019、Gour 2017、Takagi–Regula 2019 | **部分**：通用机制，用在这里是新的 | 维持 |
| 5 | 定理 10（生成元层面的自由操作） | 没有找到 | **新** | 维持 |
| 6 | 定理 13、命题 14（HN 链互不可转换、近似转换下界） | 没有找到 | **新** | 维持 |
| 7 | 定理 15（三对角链的规范下界，含任意实无序） | 相似变换法来自 Hatano–Nelson 1996；Ammari 等 ARMA 2024 用 Toeplitz 理论证明虚规范势共振子链的本征模在一端凝聚；Ammari 等 arXiv:2308.06124 研究随机缺陷下趋肤与 Anderson 局域化的竞争 | **部分**：方法已知；对任意实无序、带边缘振幅惩罚项的确定性下界似为新 | 必须引 Ammari 等两篇，并说明 V31 覆盖它们的三对角情形 |
| 8 | Hypothesis H 段："point-gap 绕数非零与 NHSE 等价" | Guo 等 2023（Commun. Phys.，arXiv:2304.06926）的反常趋肤效应：一维三带模型，某些耦合恰为零，开边界谱有随尺寸增长的多重简并 | **冲突**（细调情形，不在 V31 的单谱设定内） | 改为"对一般一维体系等价"，注明细调反例 |
| 9 | Hypothesis H 段："高维判据是谱面积非零 [ZYF22]" | ZYF22：二维及以上，趋肤 ⇔ PBC 谱面积非零，包括几何依赖趋肤（GDSE）；GDSE 已在互易二维晶格中实验观测 | **冲突** | 对配对得分这句话是错的：互易系统可以有非零谱面积和趋肤，但配对得分恒为零（定理 2）。见第 2 节 |
| 10 | Hypothesis H 的严格地位（"导入自非 Bloch 能带理论"） | 见第 3 节 | **开放问题** | 维持 "Hypothesis" 标签 |
| 11 | 物理解释中默认趋肤速率对扰动稳定 | Basak–Vogel–Zeitouni, Prob. Math. Phys. 4 (2023) 477：加 N^{−γ}（γ>1）的无结构随机噪声后，带状 Toeplitz 本征向量仍在绕数决定的那一端，但多数只局域在 N/log N 的尺度上 | **V31 未讨论** | Discussion 加一句：线性尺度 a(w)=w 下的趋肤速率在无结构的微小噪声下塌缩，与定理 4 的指数灵敏度一致；结构性无序（定理 15）不在此列 |
| 12 | 泄漏量分别用 R 和 L 定义，不用双正交密度 | 检索摘要显示：双正交重叠 ⟨L\|Π\|R⟩ 会抹掉边界信息，应只用左或只用右的量（arXiv:2302.09081，只读了检索摘要） | **支持 V31 的设计** | 可加一句引用作为设计理由 |

## 2. 互易趋肤效应：V31 资源的盲区（最重要）

**逻辑。** 只要 G^T = UGU†（U 与所有 collar 对易，例如 cell 内幺正）且单谱，V31 定理 2 给出 R_m 与 L_m 的格点分布相同，再由引理 1，对任何 collar 对配对得分都是 0。

**但互易系统确有趋肤效应：**
- 二维几何依赖趋肤效应：ZYF22 的理论；Fang–Hu–Zhou–Ding 2022（互易光子晶体，Nanophotonics）；Wang–Hu–Wang–Ma–Ding, PRL 131, 207201 (2023) 在互易二维晶格中实验观测。趋肤模只出现在与晶格对称性不匹配的边界上。
- 一维 TRS†（TT*=−1）的 Z2 趋肤效应（OKSS20）：谱简并，V31 明确不处理。
- Wang, PRB 109, L081108 (2024) 的"双向趋肤效应"：TRS† 体系的本征态同时局域在相对的两端（补充材料中 U=σx，UU*=+1）。

**数值核实**（`literature_check_v31_gdse.py`）。模型 H(k)=2cos kx + 2iγ cos ky（γ=0.6），对称耦合，加 ±0.05 的实在位无序以消除手征零模，所以 H^T=H 且谱是单的。占据数 occ(C)=Σ_m‖Π_C R_m‖²/|C| 对任何正交归一基都恰好等于 1。

| 样品 | 最小本征值间隔 | max \|\|R\|²−\|L\|²\| | V31 配对得分 | 对角边 collar（13% 格点）的占据数 | ≥50% 权重落在该 collar 的模式比例 |
|---|---|---|---|---|---|
| 正方形 30×30（对照） | 1.8e-2 | 3e-15 | 0 | 左边、下边均为 1.00 | 0% |
| 三角形 x+y≤29（465 个格点） | 6.6e-3 | 1e-13 | **恰为 0** | **3.98** | **53%**（正交性最多允许 25%） |

**后果。**
- 摘要第一句把 NHSE 定义为"大量本征模在边界聚集"。按这个定义 GDSE 是 NHSE，但 V31 的资源对它为零。V31 实际上是**定向（非互易）趋肤效应**的资源论。
- "universal" 的说法因此不成立，原因在资源的定义，不在缺定理。
- Hypothesis H 段引用 ZYF22 面积判据的那句话与定理 2 自相矛盾（#9）。

**两条出路（需要你决定）：**
- **(i) 收窄。** 标题、摘要和定义都改为"定向趋肤效应"，把 GDSE、Z2、双向趋肤列为范围之外。工作量小，但放弃 universal。
- **(ii) 扩展。** 把资源换成"右本征向量在边界的过度聚集"。一个自然的候选量是上面的占据数：任何正交归一基恰为 1；定向趋肤、互易 GDSE、Z2（用本征空间投影定义，不依赖基的选择）都使它远大于 1。现在的配对得分变成其中"定向"的那一部分，定理 4（配对 ⇒ 指数条件数和定向放大）仍然成立。这是核心定义层面的重构，要重新检查自由操作、单调量和定理 2 的角色（互易不再推出自由，只推出"无定向分量"）。

**建议：** 与"universal、不做牺牲"的目标相容的只有 (ii)。先花几天做可行性检查：占据数能否在 label-blind 自由操作下构成单调量，黄金法则是否成立，在 HN、GDSE、Z2 上数值是否正确。不行再退回 (i)。在定下来之前不要开始证 Hypothesis H，因为它是针对配对得分写的。

## 3. Hypothesis H 的严格地位

- **已证（有限矩阵）：** 三对角、t⁺t⁻>0 的链（V31 定理 15，任意实无序）；Ammari 等 ARMA 2024（三对角 Toeplitz 虚规范势链的边缘凝聚）。这一类都与 Hermitian 矩阵对角相似，谱是实的。
- **半无限算子：**
  - Ammari–Barandun–De Bruijn–Liu–Thalhammer（三对角 k-Toeplitz，J. Phys. A 2025，arXiv:2401.12626）：本征值曲线的总绕数为负时存在指数衰减的 ℓ² 本征向量；系数只要求非零，可以是复数，谱可以是复的。有限矩阵只给伪本征向量，有限链的局域只做了数值。
  - de Bruijn–Davies–Dupuy–Hiltunen 2025（arXiv:2512.03757）：带状和代数衰减的 Toeplitz 算子，半无限本征向量的衰减率由符号的根给出。有限矩阵只到伪本征向量和 Schmidt–Spitzer 极限谱；不处理左本征向量；常数不随 n 一致。
- **有限矩阵的本征值：** de Bruijn–Hiltunen 2026 两篇（arXiv:2602.09734，2610.13715）：实谱带状 Toeplitz 的 GBZ 严格基础和有限尺寸量子化规则，只涉及本征值。
- **可用工具：** Trench 本征向量公式（arXiv:1705.08067）、Schmidt–Spitzer 1960、Böttcher–Grudsky 2005 专著、Böttcher–Bogoya–Grudsky–Maximenko 2017 的本征渐近（主要是 Hermitian 情形）。
- **结论：** 有限带状 Toeplitz 矩阵（尤其复谱）的本征向量边缘局域，在检索范围内没有现成定理。证明它是真正的新贡献，但难。
- **可行的中间目标：** 存在 r 使 a(r·) 在单位圆上取实值的带状 Toeplitz 符号。这时 T_n(a) = D_r H D_r^{−1}，H 是 Hermitian 带状 Toeplitz，定理 15 的证明几乎原样推广，只需要 Hermitian 带状 Toeplitz 本征向量的边缘振幅下界（可用 Böttcher–Grudsky–Maximenko 的渐近）。这仍是实谱情形；复谱的一般情形保持开放。de Bruijn–Hiltunen 2026 处理的是更一般的极曲线情形，但只到本征值。

## 4. 对论文的影响（建议的修改）

1. 先决定第 2 节的 (i) 还是 (ii)。这决定标题、摘要第一句和定理 2 的角色。
2. 无论选哪条，都要删除或改写 Hypothesis H 段里"非零绕数与 NHSE 等价"和"高维判据是谱面积非零"两句（#8、#9）。
3. 补引用：Ammari 等 ARMA 2024；arXiv:2308.06124；k-Toeplitz（J. Phys. A 2025）；de Bruijn 等 2025；Basak–Vogel–Zeitouni 2023；Guo 等 2023；Wang PRB 2024；Wang–Hu–Wang–Ma–Ding PRL 2023；Bu–Fan–Pati–Wu 2016；Miyazaki–Kuroiwa–Murao 2026。
4. Discussion 加噪声脆弱性（#11）。
5. 定理 2(b) 标注为已知结论的有限矩阵版本（#2）。

## 5. 投稿前仍需人工核对

1. 用 Google Scholar 和 arXiv 全文检索再查一遍新颖性："resource theory" 加 "skin effect"，"non-Hermitian" 加 "free operations"，"non-normality" 加 "monotone"。网页检索覆盖不了全文。
2. Ammari 等 ARMA 2024 和 arXiv:2308.06124 的定理原文：矩阵类、是否允许无序、是否给出衰减率。这决定定理 15 能声称多少新意。
3. Wang–Hu–Wang–Ma–Ding PRL 2023 的耦合结构（是否 H^T=H 加在位损耗）。第 2 节的结论不依赖它，因为数值例子是自己构造的，但引用时要写准确。
4. Wang PRB 2024 的双向趋肤模型是否满足 V31 定理 2 的条件。补充材料中 U=σx，UU*=+1，但没有看到左本征向量的讨论。
5. 只有 arXiv 信息的条目要补期刊信息：Bu–Fan–Pati–Wu 2016、Guo 等 2023（Commun. Phys. 的卷号）、arXiv:2308.06124。
6. Basak–Vogel–Zeitouni 是否也给出左本征向量的局域（只读了前半部分，前半部分只处理右本征向量）。

## 6. 主要来源

- Ammari, Barandun, Cao, Davies, Hiltunen, Mathematical foundations of the non-Hermitian skin effect, Arch. Ration. Mech. Anal. (2024) — https://arxiv.org/abs/2306.15587
- Ammari, Barandun, Davies, Hiltunen, Liu, Stability of the non-Hermitian skin effect — https://arxiv.org/abs/2308.06124
- Ammari, Barandun, De Bruijn, Liu, Thalhammer, Spectra and pseudo-spectra of tridiagonal k-Toeplitz matrices and the topological origin of the NHSE, J. Phys. A (2025) — https://arxiv.org/abs/2401.12626
- de Bruijn, Davies, Dupuy, Hiltunen, Spectra and pseudospectra of non-Hermitian Toeplitz operators (2025) — https://arxiv.org/abs/2512.03757
- de Bruijn, Hiltunen, Mathematical foundation for the generalised Brillouin zone of m-banded Toeplitz operators (2026) — https://arxiv.org/abs/2602.09734
- de Bruijn, Hiltunen, Non-Bloch band theory for finite banded non-Hermitian Toeplitz matrices (2026) — https://arxiv.org/abs/2610.13715
- Basak, Vogel, Zeitouni, Localization of eigenvectors of non-Hermitian banded noisy Toeplitz matrices, Prob. Math. Phys. 4, 477 (2023) — https://arxiv.org/abs/2103.17148
- Okuma, Kawabata, Shiozaki, Sato, Topological origin of non-Hermitian skin effects, PRL 124, 086801 (2020) — https://arxiv.org/abs/1910.02878
- Zhang, Yang, Fang, Universal non-Hermitian skin effect in two and higher dimensions, Nat. Commun. 13, 2496 (2022) — https://arxiv.org/abs/2102.05059
- Wang, Hu, Wang, Ma, Ding, Experimental realization of geometry-dependent skin effect in a reciprocal two-dimensional lattice, PRL 131, 207201 (2023) — https://arxiv.org/abs/2302.06314
- Fang, Hu, Zhou, Ding, Geometry-dependent skin effects in reciprocal photonic crystals, Nanophotonics (2022) — https://arxiv.org/abs/2204.08866
- S.-X. Wang, Constraints of internal symmetry on the NHSE and bidirectional skin effect under TRS†, PRB 109, L081108 (2024) — https://arxiv.org/abs/2310.18627
- Guo et al., Anomalous non-Hermitian skin effect: the topological inequivalence of skin modes versus point gap, Commun. Phys. (2023) — https://arxiv.org/abs/2304.06926
- Bu, Fan, Pati, Wu, Resource theory of special antiunitary asymmetry (2016) — https://arxiv.org/abs/1610.09646
- Miyazaki, Kuroiwa, Murao, Dynamical resource theory of time-reversal symmetry breaking (2026-09-29) — https://arxiv.org/abs/2609.36408
- Cofactors and eigenvectors of banded Toeplitz matrices: Trench formulas via skew Schur polynomials — https://arxiv.org/abs/1705.08067
- Anomalous skin effects in disordered systems with a single non-Hermitian impurity — https://arxiv.org/abs/2302.09081
