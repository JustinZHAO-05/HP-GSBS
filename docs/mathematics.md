# 数学与复杂度

对每行作空间样本标准化：`Z[t]=(X[t]-mean(X[t]))/std(X[t],ddof=1)`。这不代替上游 channel-wise EEG 标准化。

令 A 为去空间均值后 X 的时间前缀，B 为 Z 的时间前缀，二者包含首行零。区间均值的中心化方向为 `A[b]-A[a]`，除以区间长度的因子在标准化时约掉。因此

`F(a,b)=dot(B[b]-B[a], A[b]-A[a]) / (T sqrt(P-1) ||A[b]-A[a]||)`。

原匹配目标为所有状态 F 之和。单/双候选只替换所属旧状态的贡献，不重新构造整条 xmeans。局部 block 分数乘 `T / block_length`，使其与官方局部调用分母相同；最终全局候选仍按官方 stateopt 实际筛选。

Full Q 的直接前缀实现时间 O(T²P)、存储 O(T²+TP)，中间仅 tile_size×P；lazy 实现只保存 O(TP) 前缀，逐行按需算分，重复请求可能多算，未声称整个拟合降到线性内存。CUDA Gram 计算 `AA=A Aᵀ`、`AB=A Bᵀ`；分子是 AB 对角项之和减 AB[a,b] 和 AB[b,a]，范数平方来自 AA。短区间用直接前缀差重算以减少 Gram 相减消去。Triton 融合前缀差、点积与范数，直接写 Q，不生成 T×T×P。

Full Q 后：单候选 O(1)，双候选 O(1)，每轮遍历候选 O(T+Σ Ls²)；仍保留官方 blockwise 筛选。Numba 编译合法性检查、评分和原始行优先候选布局。控制器、微调顺序与 K 索引不改动。

## t-distance

`R=cov(Z)`；理论上为 `ZZᵀ/(P-1)`，为保留 NumPy cov 的细节，实现先去掉 Z 残余行均值。只保留 `column-row >= dmin` 的位置，分别为计数、R、R² 建立带零边框二维前缀。每个状态取 within 矩形，每对相邻状态取 between 矩形；dmin 的三角限制已经编码在前缀内。

各分布 `mean=S/n`，`var=(SS-S²/n)/(n-1)`，Welch `t=(mean_w-mean_b)/sqrt(var_w/n_w+var_b/n_b)`。预计算 O(T²P+T²)、空间 O(T²)，每次查询 O(K)，替代原每次 O(T²) cdist/mask/ttest。within<2 返回 0；between<2 为 NaN。接近退化方差时调用原 two-pass SciPy ttest；接近单/双决策平局时也用原相关矩阵重评。

## 浮点与退化边界

不加 epsilon，不改 ddof，不填补无效行。P=1、NaN/Inf、空间常数行明确拒绝。输入有效但某个区间均值相消成常数时，候选矩阵出现非有限数会调用官方评分重新计算。接近最大分数的候选按官方 xmeans 的运算顺序重评；这是保守的实用保护，阈值不是任意输入上的严格误差证明。测试通过的 FP64 域与普遍数学等价必须区分。FP32 始终标记 experimental，不能因最终 K 相同就视为科学等价。

## v0.2 归约、局部更新与内存

默认 streaming reducer 仍遍历全部合法候选并保留官方 C-order/tie/sentinel/NaN/stateopt 语义；避免的是完整双切 score matrix 的物化，而不是剪枝候选。Window-local refinement 仅访问实际微调搜索窗口，依次提交；active-block cache 只复用未改变状态的局部候选。boundary birth label 和依赖前次移动的后续微调保持。Full Q/Prefix2D 存储仍为 O(T²)。

显式 raw-prefix streaming 用有界 chunk 计算区间分数；Feature-Moments TD 用特征矩及 tiled 边界修正获得 count/sum/squared-sum，不构造全 R/prefix。它仍使用无偏方差和 Welch 定义，但对近退化统计保守拒绝，不夹紧方差。工作/回退/内存预算超限抛错，不减少科学候选。其推导与原型历史见 [算法审查](v02_algorithm_review_CN.md)，当前成熟度与 false-refusal 边界见 [API scope](v02_api_scope.md)。
