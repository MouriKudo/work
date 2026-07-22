# 肺结节候选分类项目：方法说明

## 1. 文档范围与实现口径

本文档描述本仓库当前实现的肺结节候选二分类流程，包括 `.mhd` CT 读取、候选 patch 预处理、ResNet18 基线、PBIP-Lite 原型融合、类别感知原型损失、训练、推理和统计评估。文档中的默认值以正式三随机种子实验和仓库现有配置为准；若命令行参数、运行目录中的 `config.json` 与本文档不一致，应以该次运行保存的配置为准。

本项目解决的是**给定候选位置后的结节/非结节分类**，不是完整 CT 上的端到端结节检测。预处理阶段对负候选执行了 1:3 采样，因此仓库报告的候选级指标和 sampled-candidate FROC 不能表述为官方 LUNA16 全候选 FROC。

工程中的命名约定如下：

| 工程名称 | 原型 logit 融合 | 原型损失权重 | 说明 |
|---|---:|---:|---|
| `resnet18_noaug` | 否 | 不适用 | 无增强 ResNet18 |
| `resnet18_strong` | 否 | 不适用 | 强增强 ResNet18，也是原型库特征模型 |
| `pbip_lite` | 是 | $\beta=0$ | 只进行轻量原型证据融合 |
| `pbip_contrast` / 报表中的 `pbip_full` | 是 | $\beta=0.05$ | 在融合基础上加入类别感知原型损失 |

除非专门讨论消融实验，本文用“PBIP-Lite 家族”统称后两种配置，并显式给出 $\beta$ 以消除歧义。

## 2. 总体流程

```text
.mhd/.raw CT + candidates.csv + annotations.csv
                    │
                    ▼
        SimpleITK 读取、坐标转换与肺窗归一化
                    │
                    ▼
      3 张相邻轴位切片组成 3×64×64 的 2.5D patch
                    │
          全部正例 + 1:3 随机负例
                    │
                    ▼
       subset 0–7 / 8 / 9 → train / val / test
                    │
                    ▼
          强增强 ResNet18 监督训练与特征提取
                    │
                    ▼
       分类别球面 K-means + 类内代表样本原型库
                    │
                    ▼
 ResNet18 分类 logit + 正负原型余弦证据 → 融合 logit
                    │
                    ▼
 BCE + β·L_proto；验证集选择模型与判别阈值；固定测试
```

## 3. 数据预处理流水线

### 3.1 数据输入与 SimpleITK 读取

LUNA16 的每个 CT 序列由 `.mhd` 元数据头和配套体素文件组成。预处理脚本使用 `SimpleITK.ReadImage` 读取图像，再通过 `GetArrayFromImage` 转换为 NumPy 数组：

```python
image = sitk.ReadImage(str(mhd_path))
volume_hu = sitk.GetArrayFromImage(image)  # [z, y, x]
origin_xyz = image.GetOrigin()             # 世界坐标原点，mm
spacing_xyz = image.GetSpacing()           # 体素间距，mm/voxel
```

SimpleITK 返回的空间元数据顺序为 $(x,y,z)$，NumPy 数组索引顺序为 $(z,y,x)$。世界坐标 $(x_w,y_w,z_w)$ 在当前实现中按下式转换为最近体素中心：

$$
i_x=\operatorname{round}\left(\frac{x_w-o_x}{s_x}\right),\quad
i_y=\operatorname{round}\left(\frac{y_w-o_y}{s_y}\right),\quad
i_z=\operatorname{round}\left(\frac{z_w-o_z}{s_z}\right),
$$

其中 $\mathbf{o}$ 是原点，$\mathbf{s}$ 是 spacing。当前 LUNA16 流程假定 `.mhd` 方向矩阵与坐标轴对齐，未显式应用 direction cosine；若接入方向矩阵非单位阵的外部数据，应改用 SimpleITK 的物理点变换接口并做一致性测试。

### 3.2 正负标签与候选匹配

对 `candidates.csv` 中每个候选点，脚本在同一 `seriesuid` 的 `annotations.csv` 标注中查找欧氏距离不超过 5 mm 的结节中心：

$$
d(\mathbf{x},\mathbf{a})=
\sqrt{(x-a_x)^2+(y-a_y)^2+(z-a_z)^2}.
$$

若存在 $d\leq 5\ \mathrm{mm}$ 的标注，则候选标签为 $y=1$；否则为 $y=0$。越出 CT 有效索引范围的候选被跳过。元数据保留 `seriesuid`、subset、世界/体素坐标、类别、HU 均值/标准差和 patch 文件名，便于追踪与患者级划分。

### 3.3 2.5D patch 提取

以候选中心层 $i_z$ 为中心，分别提取 $i_z-1$、$i_z$ 和 $i_z+1$ 三张轴位切片，每张裁剪为 $64\times64$。在 CT 首尾层，层索引会裁剪到有效范围；在平面边界处，当前实现使用 0 HU 补齐后再做肺窗归一化。最终输入为：

$$
\mathbf{X}=\operatorname{stack}
\left(I_{i_z-1},I_{i_z},I_{i_z+1}\right)
\in\mathbb{R}^{3\times64\times64}.
$$

这种三相邻层方案把切片通道作为 ResNet18 的三个输入通道，计算量明显低于完整 3D CNN，同时保留有限的层间上下文。当前主流程不执行各向同性重采样，因此不同扫描的 z-spacing 仍会影响三层覆盖的实际物理厚度；这是外部验证时必须记录的协议差异。

### 3.4 肺窗与归一化

默认肺窗窗宽 $W=1500$ HU、窗位 $L=-600$ HU，对应：

$$
h_{\min}=L-\frac{W}{2}=-1350,\qquad
h_{\max}=L+\frac{W}{2}=150.
$$

对原始 HU 值 $h$，归一化为：

$$
I(h)=\frac{\operatorname{clip}(h,h_{\min},h_{\max})-h_{\min}}
{h_{\max}-h_{\min}}\in[0,1].
$$

预处理结果以 `float32` 的 `.npy` 文件保存，不再执行 ImageNet 均值/标准差归一化。窗宽窗位偏移在鲁棒性实验中作为独立退化类型处理，不与基础预处理混用。

### 3.5 1:3 正负采样

所有正候选均保留。若负候选数大于正候选数的三倍，则使用固定随机种子 42 从全体负候选中采样：

$$
N_{-}^{\mathrm{keep}}=\min(N_-,\lfloor3N_+\rfloor).
$$

正式实验使用 `random` 策略。仓库还实现了 `hard_negative` 备选策略，但它不是主实验配置。1:3 采样降低了类别不平衡和训练成本，也改变了测试集先验分布，因此报告结果必须标注为 sampled-candidate classification，不能直接推断临床每扫描假阳性负担。

### 3.6 CT 级数据划分

数据按 LUNA16 subset 划分：subset 0–7 为训练集，subset 8 为验证集，subset 9 为测试集。一个 `seriesuid` 只属于一个 subset，因此同一 CT 的候选不会跨集合出现。

当前已缓存的 `metadata.csv` 统计如下；这些数字来自实际工程文件，而不是论文设定值：

| 集合 | 候选数 | 正候选数 | CT 数 |
|---|---:|---:|---:|
| 训练集 | 3553 | 916 | 689 |
| 验证集 | 462 | 114 | 87 |
| 测试集 | 497 | 98 | 87 |

外部 LIDC-IDRI 评估必须先进行病例去重、标签映射和两类完整性检查；若无法得到独立且同时包含正负类的外部集合，流程自动回退到 LUNA16 固定测试集，并在数据源清单中记录回退原因。

### 3.7 训练增强

PBIP-Lite 与强增强 ResNet18 使用同一套训练增强：水平翻转 $p=0.5$、垂直翻转 $p=0.5$、$\pm30^\circ$ 随机旋转、亮度/对比度/饱和度抖动 $(0.2,0.2,0.1)$、尺度范围 $[0.7,1.0]$ 的随机裁剪回 $64\times64$，以及核大小 3、$\sigma\in[0.1,1.0]$ 的高斯模糊。验证集和测试集不应用随机增强。

## 4. ResNet18 分类网络

### 4.1 结构调整

主干基于 torchvision ResNet18。为适配 $64\times64$ 小尺寸 CT patch，初始 $7\times7$、stride 2 卷积替换为 $3\times3$、stride 1 卷积，并移除初始最大池化。其余残差阶段保持 ResNet18 的 `[2,2,2,2]` BasicBlock 配置。

| 模块 | 输出通道 | 空间尺寸（典型） | 说明 |
|---|---:|---:|---|
| 输入 | 3 | $64\times64$ | 相邻三层肺窗 patch |
| `conv1` | 64 | $64\times64$ | $3\times3$, stride 1, padding 1 |
| `layer1` | 64 | $64\times64$ | 2 个 BasicBlock |
| `layer2` | 128 | $32\times32$ | 首块 stride 2 |
| `layer3` | 256 | $16\times16$ | 首块 stride 2 |
| `layer4` | 512 | $8\times8$ | 首块 stride 2 |
| 全局平均池化 | 512 | $1\times1$ | 得到 512 维特征 |
| Dropout + Linear | 1 | — | dropout 0.3，输出二分类 logit |

除被替换的首卷积和分类头外，ResNet18 可继承 ImageNet-1K V1 预训练权重。令特征提取器为 $f_\theta$，则：

$$
\mathbf{z}=f_\theta(\mathbf{X})\in\mathbb{R}^{512},\qquad
b(\mathbf{X})=\mathbf{w}^{\top}\mathbf{z}+b_0.
$$

强增强 ResNet18 先独立训练；每个随机种子以验证 AUC 最高的 checkpoint 作为该 seed 的原型库特征模型和 PBIP 初始化权重。

## 5. PBIP-Lite 原型库

### 5.1 训练集特征与球面 K-means

对强增强 ResNet18 的最佳 checkpoint，关闭随机增强并提取全部训练候选的 512 维特征。对每个类别 $c\in\{0,1\}$ 分别归一化：

$$
\tilde{\mathbf{z}}_i=
\frac{\mathbf{z}_i}{\lVert\mathbf{z}_i\rVert_2+\varepsilon}.
$$

在单位球面上执行 $K=3$ 的 K-means。第 $t$ 次迭代中的分配与中心更新为：

$$
a_i=\arg\max_k\tilde{\mathbf{z}}_i^{\top}\boldsymbol{\mu}_k,
$$

$$
\boldsymbol{\mu}_k\leftarrow
\frac{\sum_{i:a_i=k}\tilde{\mathbf{z}}_i}
{\left\lVert\sum_{i:a_i=k}\tilde{\mathbf{z}}_i\right\rVert_2+\varepsilon}.
$$

每个簇选取与中心余弦相似度最高的 $N=20$ 个真实训练样本作为代表原型，而不是把聚类中心本身作为可视原型。默认情况下，每个类别最多有 $K\times N=60$ 个原型，正负两类合计最多 120 个。原型库保存源 checkpoint 路径及 SHA-256、特征、标签、病例标识、patch 文件名、聚类中心和簇统计，从而支持可追溯的病例检索。

### 5.2 类别感知原型证据

推理时对当前特征和原型分别做 $L_2$ 归一化。设第 $j$ 个原型为 $\mathbf{p}_j$、其标签为 $y_j$，则：

$$
s_j=\cos(\mathbf{z},\mathbf{p}_j)
=\frac{\mathbf{z}^{\top}\mathbf{p}_j}
{\lVert\mathbf{z}\rVert_2\lVert\mathbf{p}_j\rVert_2}.
$$

对每个类别独立选取相似度最高的 $k_c=\min(k,M_c)$ 个原型，默认 $k=20$，并用温度 $\tau=0.2$ 缩放：

$$
q_c(\mathbf{X})=
\frac{1}{\tau k_c}
\sum_{j\in\operatorname{TopK}\{s_j:y_j=c\}}s_j.
$$

类别原型 logits 为 $\mathbf{q}=[q_0,q_1]$，二分类原型证据为：

$$
r(\mathbf{X})=q_1(\mathbf{X})-q_0(\mathbf{X}).
$$

这种分别汇总正负类证据的方式避免把“与负原型相似”错误地当作正类证据。

### 5.3 余弦证据融合

最终 logit 是基础分类 logit 与原型 logit 的凸组合：

$$
\ell(\mathbf{X})=(1-\alpha)b(\mathbf{X})+\alpha r(\mathbf{X}),
\qquad \alpha=0.3,
$$

$$
P(y=1\mid\mathbf{X})=\sigma\bigl(\ell(\mathbf{X})\bigr).
$$

$\alpha=0$ 退化为原始 ResNet18 分类头；$\alpha=1$ 只依赖原型证据。主实验固定 $\alpha=0.3$，避免在测试集上调参。

## 6. 损失函数

### 6.1 二元交叉熵

对 batch 中第 $i$ 个样本，融合 logit 为 $\ell_i$，标签为 $y_i\in\{0,1\}$。分类损失为带 logits 的 BCE：

$$
\mathcal{L}_{\mathrm{BCE}}
=-\frac{1}{B}\sum_{i=1}^{B}
\left[y_i\log\sigma(\ell_i)+(1-y_i)\log(1-\sigma(\ell_i))\right].
$$

实现使用 `torch.nn.BCEWithLogitsLoss`，由数值稳定的 log-sum-exp 形式完成计算。

### 6.2 类别感知原型对比损失

本项目的 $\mathcal{L}_{\mathrm{proto}}$ 是对正负两类原型证据 $[q_{i,0},q_{i,1}]$ 做交叉熵：

$$
\mathcal{L}_{\mathrm{proto}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log\frac{\exp(q_{i,y_i})}
{\exp(q_{i,0})+\exp(q_{i,1})}.
$$

它通过标签提升本类原型的聚合证据、抑制异类原型证据，属于类别感知的原型对比监督。它不是需要显式构造样本对的标准 SupCon 损失；该区别在复现实验或与文献比较时必须保留。

总损失为：

$$
\boxed{
\mathcal{L}=\mathcal{L}_{\mathrm{BCE}}
+\beta\mathcal{L}_{\mathrm{proto}}
}
$$

其中纯 PBIP-Lite 取 $\beta=0$，带原型对比监督的完整版本取 $\beta=0.05$。消融实验还评估 $\beta\in\{0.01,0.05,0.10,0.20\}$。

## 7. 训练、选择与推理协议

### 7.1 训练顺序

1. 使用当前 seed 训练无增强和强增强 ResNet18。
2. 仅根据强增强 ResNet18 的验证 AUC 选择最佳 checkpoint。
3. 用该 checkpoint 对训练集提取特征，并为该 seed 单独构建原型库。
4. 以同一强增强 checkpoint 初始化 PBIP 主干和分类器。
5. 分别训练 $\beta=0$ 与 $\beta=0.05$ 的配置。
6. 仍只根据验证 AUC 保存 PBIP 最佳 checkpoint。
7. 最终报告阶段在验证集上选择最大 F1 阈值，再冻结阈值评估测试集或外部集。

每个 seed 的原型库只使用该 seed 的训练特征，不能跨 seed 共用，以免把模型差异与原型差异混在一起。

### 7.2 优化器与模型选择

PBIP 使用 AdamW，学习率 $10^{-4}$、weight decay $10^{-4}$，训练 20 epochs；学习率由 `CosineAnnealingLR(T_max=20)` 调度。训练 batch size 为 64，启用 CUDA 时默认使用自动混合精度。每轮完成后在验证集计算 AUC、PR-AUC 和 F1，checkpoint 选择指标固定为验证 AUC。

训练脚本内部展示的 F1 使用 0.5 阈值，仅用于监控。正式测试阈值 $t^*$ 由验证集最大 F1 确定：

$$
t^*=\arg\max_{t\in\mathcal{T}_{val}}
F_1\left(y_{val},\mathbb{I}[p_{val}\geq t]\right).
$$

若多个阈值达到相同 F1，选择最接近 0.5 的阈值。测试阶段使用 $\hat y=\mathbb{I}[p\geq t^*]$，禁止根据测试标签重新选择阈值。

## 8. 全部默认超参数

### 8.1 预处理与划分

| 参数 | 默认值 | 实现含义 |
|---|---:|---|
| 输入格式 | `.mhd` + 体素文件 | SimpleITK 读取 |
| patch 大小 | $3\times64\times64$ | 三张相邻轴位切片 |
| 窗宽 / 窗位 | 1500 / -600 HU | 输出归一化到 $[0,1]$ |
| 标注匹配容差 | 5 mm | 候选中心到结节标注中心 |
| 负/正上限 | 3:1 | 全部正例 + 固定随机负例 |
| 负采样种子 | 42 | `pandas.sample` |
| 训练 subsets | 0–7 | CT 级隔离 |
| 验证 subset | 8 | 模型与阈值选择 |
| 测试 subset | 9 | 固定测试 |

### 8.2 模型与原型库

| 参数 | 默认值 | 说明 |
|---|---:|---|
| 主干 | ResNet18 | torchvision 实现 |
| 输入首卷积 | $3\times3$, stride 1 | 初始 maxpool 移除 |
| 特征维数 | 512 | 全局平均池化后 |
| Dropout | 0.3 | 二分类头 |
| 类别数 | 2 | 非结节 / 结节 |
| 每类簇数 $K$ | 3 | 消融值 1、3、5 |
| 每簇代表数 $N$ | 20 | 真实训练样本原型 |
| K-means 最大迭代 | 100 | 球面 K-means |
| 推理类内 Top-k | 20 | 分类别汇总 |
| 温度 $\tau$ | 0.2 | 原型类别 logits 缩放 |
| 融合系数 $\alpha$ | 0.3 | 基础/原型 logit 融合 |

### 8.3 训练与统计

| 参数 | ResNet18 | PBIP-Lite 家族 |
|---|---:|---:|
| epochs | 20（正式矩阵覆盖脚本默认值） | 20 |
| batch size | 64 | 64 |
| 优化器 | AdamW | AdamW |
| 学习率 | $10^{-3}$ | $10^{-4}$ |
| weight decay | $10^{-4}$ | $10^{-4}$ |
| 调度器 | CosineAnnealingLR | CosineAnnealingLR |
| 强增强 | 可选 | 默认启用 |
| AMP | 默认启用 | 默认启用 |
| workers | 4 | 训练 4、验证/测试 0 |
| 随机种子 | 0、1、2 | 0、1、2 |
| $\beta$ | 不适用 | 0 或 0.05 |
| checkpoint 指标 | 最大验证 AUC | 最大验证 AUC |
| 判别阈值 | 验证集最大 F1 | 验证集最大 F1 |
| Bootstrap | 1000 次 | 1000 次 |
| 置信水平 | 95% | 95% |
| Bootstrap 单位 | `seriesuid` | `seriesuid` |

## 9. PyTorch 风格伪代码

### 9.1 预处理

```python
def preprocess_scan(mhd_path, candidates, annotations):
    image = sitk.ReadImage(str(mhd_path))
    volume = sitk.GetArrayFromImage(image)  # [z, y, x], HU
    origin, spacing = image.GetOrigin(), image.GetSpacing()

    records = []
    for candidate in candidates.for_same_series(mhd_path.stem):
        label = int(min_distance_mm(candidate, annotations) <= 5.0)
        x, y, z = world_to_voxel(candidate.xyz, origin, spacing)
        if not inside_volume(x, y, z, volume.shape):
            continue

        slices = []
        for dz in (-1, 0, 1):
            plane = crop_or_pad(volume[clip(z + dz)], center=(x, y), size=64)
            plane = clip_and_normalize(plane, low=-1350, high=150)
            slices.append(plane)
        patch = torch.from_numpy(stack(slices)).float()  # [3, 64, 64]
        save_patch_and_metadata(patch, label, candidate)

    keep(all_positive_records)
    keep(random_sample(negative_records, at_most=3 * num_positive, seed=42))
```

### 9.2 原型库构建

```python
@torch.no_grad()
def build_prototype_bank(feature_model, train_loader, K=3, N=20):
    features, labels, provenance = extract_512d_features(feature_model, train_loader)
    bank = []
    for class_id in (0, 1):
        z = F.normalize(features[labels == class_id], dim=1)
        assignments, centers = spherical_kmeans(z, n_clusters=K)
        for cluster_id in range(K):
            members = z[assignments == cluster_id]
            similarity = members @ centers[cluster_id]
            selected = similarity.topk(min(N, len(members))).indices
            bank.extend(real_training_samples(members[selected], provenance))
    return bank  # 保存特征、标签、病例、patch 路径和 checkpoint 哈希
```

### 9.3 PBIP-Lite 前向传播与训练

```python
class PBIPLite(nn.Module):
    def forward(self, x):
        feature = self.backbone(x)                       # [B, 512]
        base_logit = self.classifier(feature).squeeze(1) # [B]

        z = F.normalize(feature, dim=1)
        cosine = z @ self.prototypes.T                   # [B, M]

        class_logits = []
        for class_id in (0, 1):
            s = cosine[:, self.prototype_labels == class_id]
            evidence = s.topk(min(self.top_k, s.size(1)), dim=1).values.mean(1)
            class_logits.append(evidence / self.temperature)
        class_logits = torch.stack(class_logits, dim=1) # [B, 2]

        proto_logit = class_logits[:, 1] - class_logits[:, 0]
        fused_logit = (1 - self.alpha) * base_logit + self.alpha * proto_logit
        return fused_logit, class_logits

for x, y, _ in train_loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        fused_logit, class_logits = model(x.to(device))
        loss_bce = F.binary_cross_entropy_with_logits(fused_logit, y.float())
        loss_proto = F.cross_entropy(class_logits, y.long())
        loss = loss_bce + beta * loss_proto
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 9.4 固定阈值推理

```python
model = load_checkpoint_and_matching_prototype_bank()
val_prob = sigmoid(collect_logits(model, val_loader))
threshold = argmax_f1_threshold(val_labels, val_prob)

test_prob = sigmoid(collect_logits(model, fixed_test_loader))
test_pred = (test_prob >= threshold).long()
metrics = compute_metrics(test_labels, test_prob, test_pred)
ci = cluster_bootstrap(metrics, cluster=seriesuid, iterations=1000, level=0.95)
save_predictions_metrics_and_source_manifest()
```

## 10. YAML 配置样本

完整训练样本见 [`config_samples/train_pbip_lite.yaml`](config_samples/train_pbip_lite.yaml)，固定测试/外部推理样本见 [`config_samples/inference_pbip_lite.yaml`](config_samples/inference_pbip_lite.yaml)。二者是统一的配置契约样本，字段值与当前正式实验一致。

当前 `train.py`、`prototype_bank.py`、`pbip_train.py` 和 `metrics.py` 主要使用命令行参数；配置中的同名字段应转换为对应 CLI 参数。仓库中已经直接读取 YAML 的流程可参考 `src/configs/ablation.yaml`、`src/configs/robustness.yaml` 和 `src/configs/robust_pbip.yaml`。不得把文档 YAML 是否被某个入口直接解析与模型方法本身混为一谈。

训练配置核心片段：

```yaml
prototype_bank:
  clustering: spherical_kmeans
  clusters_per_class: 3
  representatives_per_cluster: 20
  inference_top_k_per_class: 20
  temperature: 0.2
fusion:
  alpha: 0.3
loss:
  classification: BCEWithLogitsLoss
  prototype: class_aware_prototype_cross_entropy
  prototype_weight_beta: 0.05
optimization:
  epochs: 20
  batch_size: 64
  learning_rate: 0.0001
  weight_decay: 0.0001
reproducibility:
  seeds: [0, 1, 2]
```

推理配置核心片段：

```yaml
external_validation:
  require_binary_labels: true
  require_patient_or_scan_deduplication: true
  fallback_when_lidc_invalid: luna16_fixed_test
decision:
  threshold_source: validation_maximum_f1
  allow_test_threshold_tuning: false
statistics:
  bootstrap_iterations: 1000
  confidence_level: 0.95
  bootstrap_unit: seriesuid
```

## 11. 实现文件映射与复现命令

| 方法环节 | 实现文件 |
|---|---|
| `.mhd` 读取、肺窗、patch 和负采样 | `src/preprocess.py` |
| CT 级 subset 划分 | `src/split_data.py` |
| Dataset 与 DataLoader | `src/luna16_dataset.py` |
| ResNet18 与增强 | `src/train.py` |
| 原型库构建 | `src/prototype_bank.py` |
| PBIP-Lite、融合和损失 | `src/pbip_train.py` |
| 四方法 × 三 seed 调度 | `src/run_experiments.py` |
| 阈值选择与候选级指标 | `src/metrics.py` |
| 外部/固定测试 | `src/evaluate_external.py` |
| Bootstrap 与差异检验 | `src/stats_test.py` |

```powershell
# 1. 预处理与 CT 级划分
.\.venv\Scripts\python.exe src\preprocess.py --neg_ratio 3 --neg_strategy random
.\.venv\Scripts\python.exe src\split_data.py

# 2. 四方法、三随机种子正式实验
.\.venv\Scripts\python.exe src\run_experiments.py `
  --seeds 0,1,2 --epochs 20 --batch_size 64 `
  --alpha 0.3 --contrast_beta 0.05

# 3. 单个 PBIP checkpoint 的验证阈值选择与固定测试
.\.venv\Scripts\python.exe src\metrics.py `
  --model pbip `
  --checkpoint runs\experiments_v2\seed_0\pbip_contrast\best_model.pth `
  --prototype_bank runs\experiments_v2\seed_0\prototype_bank\prototype_bank.pkl `
  --output_dir runs\experiments_v2\best_evaluation
```

## 12. 复现与报告边界

- 原型库、初始化 checkpoint 和 seed 必须一一匹配；加载 checkpoint 时同时核对原型库来源和哈希。
- 模型选择只使用验证 AUC，判别阈值只使用验证 F1；测试标签不得参与任何选择。
- 多随机种子报告均值与样本标准差；置信区间优先按 `seriesuid` 做 CT 簇 Bootstrap，避免把同一 CT 的候选误当作完全独立样本。
- Grad-CAM 目前是后验可视化，不参与损失函数；没有像素级病灶真值时，不把热图视觉一致性表述为定量定位准确率。
- LIDC-IDRI 与 LUNA16 存在病例来源重叠的可能。外部验证必须完成病例级去重；无法去重或数据不满足二分类评估条件时，只能报告 LUNA16 固定测试结果及明确的回退原因。
- 当前三层输入未执行 spacing 统一和方向矩阵变换，部署到异构 CT 协议前应补充物理空间重采样与方向一致性验证。

