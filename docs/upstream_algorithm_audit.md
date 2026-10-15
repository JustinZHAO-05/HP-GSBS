# 官方算法审计

审查日期 2026-10-16。源码固定为 [lgeerligs/statesegmentation](https://github.com/lgeerligs/statesegmentation) commit `a75d2f144e740ca347b1a3c5035d1936ce192f0a`。PyPI 0.0.6 的 gsbs.py 与该文件逐行比较相同（Git setup.py 版本号仍为 0.0.4）。本包依赖 PyPI 0.0.6；reference 每次记录实际源码 SHA256。官方包不修改。

论文：[2021](https://doi.org/10.1016/j.neuroimage.2021.118085)，[2022](https://doi.org/10.7554/eLife.77430)。2021 描述逐边界贪心与独立的状态数估计。2022 全文通过原 BBBD 缓存 `data/external/sources/primary_literature/states_gsbs_2022.txt` 核对，方法段 76–80 和附录 164–169 明确双边界、t-distance 选择与强度顺序微调；文中第三项改进涉及 fMRI HRF 去卷积，不适用于本 EEG 加速工程。blockwise 的精确操作定义来自源码。2021 出版商与 bioRxiv 全文端点返回 403，目前完成摘要、引用与算法源码核对，未冒称完整原文已读。

另核对了 [2020 原始预印本方法全文](https://www.researchgate.net/publication/341107327_Detecting_neural_state_transitions_underlying_event_segmentation)（DOI 10.1101/2020.04.30.069989）：平均空间相关匹配、逐轮贪心插入、within/adjacent 相关分布的 t-distance，以及最大 t-distance 选 K。该版本不能代替 2021 期刊最终全文；实现细节以固定官方源码为准。

下表 T=时间点，P=特征，K=当前状态数，C1/C2=合法单/双候选数；复杂度为一次调用的主要上界。

|函数|输入 → 输出 / 目标|时间|额外空间|调用频次|
|---|---|---|---|---|
|fit|X,Y,参数 → 原位 all_bounds/tdists|最坏 O(Kmax T³P)，另有微调|O(T²+TP+Kmax T)|每记录一次|
|_wdists|deltas,states,X,Z,boundopt → 长 T 评分，非法=-1|O(C1 TP)|O(TP+T)|每轮各 block、全局及每个微调边界|
|_wdists_state|上述及 stateopt → T×T 评分或 None|O(C2 TP)，首轮 C2=O(T²)|O(T²+TP)|每轮各 block 和全局|
|_wdists_blocks|分段和 blocksize → 单/双评分|各局部搜索之和+全局重评|O(T²+TP)|每轮一次|
|_finetune|带插入编号边界、X,Z、范围及顺序 → 更新编号边界|O(TP+K P)+最多 K 次 _wdists|O(TP+T)|k>2 且 finetune!=0 每轮一次|
|_tdist|deltas、相关上三角向量、dmin mask → Welch t|O(T²)|O(T²)|每轮单/双候选及微调后，2–3 次|
|_zscore|任意行×特征矩阵 → 每行样本标准化|O(TP)|O(TP)|候选内重复调用；ddof=1|
|_states|deltas → cumsum+1|O(T)|O(T)|搜索、t-distance、微调及输出|
|get_bounds|K → all_bounds[K] 视图|O(1)|O(1)|按需|
|get_deltas/get_states|K → 布尔边界/1-based状态|O(T)|O(T)|按需|
|get_state_patterns|K,X → 实际状态数×P 原始均值|O(KT+TP)|O(KP+T)|按需|
|get_strengths|K,X → 长 T，相邻均值 Pearson 距离|O(KT+TP+KP)|O(KP+T)|每轮微调及按需|

## 搜索路径必须保留的细节

1. 首轮没有 block 划分，搜索全部合法 i<j，且两者位于同一旧状态内部。空间匹配决定候选位置，**t-distance 严格 `>`** 决定是否选双边界；相等与 NaN 比较均选择单边界。
2. block 累积整个状态，只有长度 **大于** blocksize（不是大于等于）或到最后状态才结算。局部分数使用 block 长度作分母；局部胜者再全局评分。
3. `_wdists_state` 的 stateopt 筛选是 `np.sum(equal(stateopt, [[i,j]])) >= 2`，并非按行判断成对相等。重复首坐标或跨行坐标可以放行额外候选。本包保留此现存行为，不能改成“更正确”的候选集合。
4. 无候选双矩阵仅当所有值均为 -1 才返回 None；argmax 采用 C-order 首个最大值。NaN 会被 numpy.argmax 选择；对未定义输入，本包提前拒绝。
5. 双插入编号均为 k+1；all_bounds[k:k+2] 相同，tdists[k]=tdists[k-1]，tdists[k+1] 才存新 t 值。最后一次可以达到 kmax+1 状态。
6. 微调在 k>2 才执行。strengths 从刚写入 all_bounds 的实际状态数索引计算。顺序是 np.argsort，零强度被跳过；逐边界删除、重搜、恢复旧编号。负 finetune 可跨越其他状态。不可并行化这些依赖。
7. t-distance 使用 `cov(z)` 与 `triu(...,dmin)`，within 样本数<2 返回 0，否则 SciPy Welch，between 小样本可能 NaN。y 只用于此处，不用于搜索或 strengths。
8. nstates 是整个 tdists 的 argmax，包含初始零槽。不能擅自从 2 开始选择或替换 BBBD 自定义峰值规则。
