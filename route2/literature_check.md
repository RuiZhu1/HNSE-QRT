# 路线二：文献与新颖性核查（2026-10-16）

论文：`paper/nonreciprocity_resource.tex`
判定：**已知**＝已有文献给出同一结论；**部分**＝已有定性或特例结论，本文给出精确或推广版本；**新**＝检索范围内没有找到先例。

检索范围：arXiv / APS / Optica / NJP / SciPost 摘要，关键词包括 resource theory + nonreciprocity / time-reversal / PT asymmetry，nonreciprocity bandwidth bound，minimal gyrator，topological amplification，quantum-limited directional amplifier，transfer matrix skin effect。全文只读了部分论文，结论依据的是摘要和关键段落。

## 1. 逐条核查

| # | 本文结论 | 最接近的已有工作 | 判定 | 论文中的处理 |
|---|---|---|---|---|
| 0 | “非互易性从未被当作资源理论处理” | Bu–Fan–Pati–Wu 2016（PT 型反幺正不对称性，态层面）；**Miyazaki–Kuroiwa–Murao 2026-09（量子信道时间反演破缺的动力学资源理论）** | **原说法不成立** | 已删掉。改为：已有理论处理的是态/信道的反幺正对称性；本文处理的是 Lorentz 互易 S=Sᵀ、色散对象、网络组合、增益与噪声，这一设定下没有找到先例 |
| 1 | 预算定理 𝒩₁ ≤ Σβ（带宽平均） | Mann–Sounas–Alù 2019（非互易腔的时间–带宽极限）；Bode–Fano 类积分约束 | **新**（没有找到带宽积分形式的非互易预算界） | 声明为新；引用 MSA19 作为背景 |
| 2 | 隔离–带宽推论 | 同上；器件工程中的经验折中 | **新**（作为定理） | 同上 |
| 3 | Cost 是忠实单调量；𝒩₁ 不是单调量 | 经典网络综合中的最少回转器数目（Anderson–Vongpanitlerd 1973） | **部分**：回转器计数是整数版的类比；连续的 Cost 和 𝒩₁ 的非单调反例没有找到先例 | 引用 AV73 作为经典类比 |
| 4 | 串联网络中 S21/S12 可乘、det T = det S21/det S12、互易 ⇒ 辛结构 | 微波工程教材：互易二端口 AD−BC=1（Pozar） | **已知**（成分） | 明确标注为经典成分，引用 Poz12 |
| 5 | 环路原理：纯相位非互易必须有环路才起作用 | Ranzani–Aumentado 2015（耦合模图论：非互易需要带非平凡相位的环）；Metelmann–Clerk 2015（相干与耗散路径干涉） | **部分**：定性结论已知；MZI 中 π/Δ 个元件得到完美宽带隔离器、效率 1/π 是定量的新结果 | 引用 RA15、MC15，并在晶格段落注明一致 |
| 6 | 坍缩定理（互易增益 + 任意弱非互易 ⇒ 完美定向放大器） | 工程上“隔离器 + 放大器”的常识 | **部分/平凡**：作为资源理论的 no-go 陈述是新的，数学上是初等的 | 维持原表述 |
| 7 | 有增益时串联规范通量守恒、固定频率下是完全不变量；串联定向放大需 ∏≥√G | 同 #4 | **部分** | 维持原表述 |
| 8 | 噪声代价定理 𝒩₁ ≤ Σβ(1+νᵢ)，且可取等号 | Kamal–Clarke–Devoret 2011（无噪声非互易）；Metelmann–Clerk 2015；Malz et al. 2018（量子极限定向放大器） | **新**：没有找到把带宽平均非互易与到达非互易元件的噪声量子数联系起来的不等式。已有工作与本文**一致**：它们在输出端达到量子极限，本文环形例子同样显示外部附加噪声可以为零，代价在网络内部支付 | 加了一段说明一致性；并指出参量器件（增益和非互易来自同一耦合）是非无源器件，不在定理覆盖范围内，列为开放问题 (vii) |
| 9 | 鲁棒稳定 ⇔ r²\|S12 S21\|<1；增益率 ≤ 通量率 | Rollett 1962（无条件稳定）；Xue et al. 2021、Wanjura et al. 2020（OBC 定向放大器在 PBC 下不稳定） | **部分**：稳定性条件是经典的，“闭合边界即不稳定”的思想已知；与规范通量守恒结合得到的“增益率 ≤ 通量率”是新的表述 | 引用 Rol62、Xue21、WBN20 |
| 10 | 网络绕数 w 在无源、互易（含增益）、单通道纯相位单元中为零 | Kunst–Dwivedi 2019（转移矩阵判据：\|det T\|≠1 ⇔ 趋肤效应）；Porras 2019、Wanjura 2020/2021、Brunelli 2023（绕数 ↔ 定向放大，需要增益） | **部分**：情形 (iii) 是 Kunst–Dwivedi 判据的散射网络版本；“需要增益”已知；情形 (ii)（互易网络任意增益下 w=0，来自辛配对）在检索中没有找到 | 情形 (iii) 引用 KD19；绕数–放大对应引用 Por19、WBN20、WBN21、BWN23 |
| 11 | 数值：链稳定阈值 2γ=2log g；w=−1 区域的前向增长率 = log\|μ₁\| | Wanjura 2020（解析增益 ∝ \|z\|^{−2ν(N+1)}）、Brunelli 2023（\|ν\| 个边缘奇异向量） | **部分**：网络版本的数值验证 | 已标为“数值观察，未证明”（开放问题 v） |

## 2. 对论文的影响

- **核心卖点保留**：预算定理、噪声代价定理和 Cost 单调量，在检索范围内没有找到先例。论文现在只对这三项声明“据我们所知是新的”。
- **降级为“重述/精确化”**：环路原理、规范通量守恒、稳定性定理和绕数定理。论文中已明确写成“已知的对应在散射网络语言中的精确化”。
- **新增限制**：参量/时间调制器件不在噪声定理范围内（开放问题 vii）；与 Miyazaki 等的信道层面单调量之间的关系待研究（开放问题 viii）。
- 参考文献从 25 篇增加到 38 篇，按首次引用顺序重排。论文从 7 页变为 8 页。

## 3. 投稿前仍需人工核对

1. **Miyazaki–Kuroiwa–Murao (arXiv:2609.36408)**：2026-09-29 才挂出，只读了摘要。需要通读，确认他们的框架是否覆盖线性散射网络或带宽。他们提到“即便对经典信道也存在通用黄金单位”，可能与本文的蒸馏效率问题（开放问题 iii）相关。
2. **Anderson–Vongpanitlerd** 中最少回转器数目的精确表述（是否为 ½ rank(Z−Zᵀ)），确认后可以写得更具体。
3. **Xue et al.** 的期刊信息（PRB 103, L241408 (2021)）来自记忆，arXiv 页面没有列出期刊号，需要核对。
4. 一篇非互易教程（Miller & Monticone, arXiv:2605.24738）和 Khurgin (arXiv:2211.11604) 还没有细读，可能含有隔离–带宽方面的论述。
5. 建议用 Google Scholar 查一遍 “nonreciprocity sum rule” / “gyrotropic sum rule bandwidth”：如果存在 Bode–Fano 型的非互易积分界，需要与定理 1 对比。

## 4. 主要来源

- Miyazaki, Kuroiwa, Murao, arXiv:2609.36408 — https://arxiv.org/abs/2609.36408
- Bu, Fan, Pati, Wu, arXiv:1610.09646 — https://arxiv.org/abs/1610.09646
- Wanjura, Brunelli, Nunnenkamp, Nat. Commun. 11, 3149 (2020) — https://arxiv.org/abs/1909.11647
- Wanjura, Brunelli, Nunnenkamp, PRL 127, 213601 (2021) — https://arxiv.org/abs/2010.14513
- Brunelli, Wanjura, Nunnenkamp, SciPost Phys. 15, 173 (2023) — https://arxiv.org/abs/2207.12427
- Porras, Fernández-Lorenzo, PRL 122, 143901 (2019) — https://arxiv.org/abs/1812.01348
- Xue et al., arXiv:2004.09529 — https://arxiv.org/abs/2004.09529
- Ranzani, Aumentado, NJP 17, 023024 (2015)；Metelmann, Clerk, PRX 5, 021025 (2015)；Kunst, Dwivedi, PRB 99, 245116 (2019)；Mann, Sounas, Alù, Optica 6, 104 (2019)；Kamal, Clarke, Devoret, Nat. Phys. 7, 311 (2011)；Malz et al., PRL 120, 023601 (2018)
