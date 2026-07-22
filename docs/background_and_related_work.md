# 肺结节候选分类项目：背景与相关工作

## 1. 文档目的与任务边界

肺癌筛查和临床胸部 CT 会产生大量需要复核的肺结节候选。计算机辅助检测系统通常先生成高敏感度候选，再通过分类器去除血管交叉、胸膜结构、瘢痕和噪声等假阳性。本项目聚焦第二阶段：对已经给定空间坐标的候选 patch 进行“结节/非结节”二分类，并进一步研究原型证据、可解释性和图像退化鲁棒性。

需要区分三个经常被混写的任务：

1. **候选分类/假阳性削减**：输入候选位置，输出结节或非结节；本项目属于该任务。
2. **完整结节检测**：从整幅 CT 直接输出结节位置，需要以每扫描假阳性和 FROC/CPM 为主要评价。
3. **结节性质判断**：对已确认结节判断良恶性、亚型或随访风险，其标签定义和临床终点均不同。

本文从肺结节候选分类、原型学习、Grad-CAM 可解释性、模型鲁棒性四个方向整理研究脉络，并说明本项目的工程定位。文献以原始论文、正式会议页面、出版社或 PubMed/PMC 记录为准；近年文献用于说明趋势，不代表穷尽式综述。

## 2. 肺结节候选分类研究现状

### 2.1 公共数据与评价体系

LIDC-IDRI 建立了公开胸部 CT 与多放射科医师两阶段标注体系，包含 1018 个病例，并保留观察者间差异而非强制共识 [1]。它为检测、分割、性质评价和算法复现提供了基础，但也带来标签共识、同一病灶多标注合并和患者去重等方法学问题。

LUNA16 从 LIDC-IDRI 中筛选符合层厚等条件的 888 个 CT，并提供完整检测和假阳性削减两个赛道 [3]。LUNA16 的价值不仅在数据规模，还在于统一了候选、参考标准与 FROC 评价。其挑战结果显示，基于卷积网络的候选分类器以及多系统融合能够显著减少假阳性；然而，官方 FROC 依赖完整候选集合和每扫描假阳性，不能用经过负采样的普通二分类测试集替代。

因此，公开数据上的高 AUC 只回答“在当前候选分布中能否区分两类”，不等价于完整 CAD 系统在临床中的敏感度、读片时间收益或每扫描误报数。外部验证还必须检查 LIDC-IDRI 与 LUNA16 的病例来源重叠，避免把同源数据误称为真正独立外部队列。

### 2.2 从多视图 2D 到 3D 上下文

早期深度学习候选削减常把 CT 候选投影到多个 2D 视图。Setio 等提出多视图 CNN，对不同方向的二维 patch 分流提取特征后融合，证明了数据增强、dropout 和视图互补对假阳性削减的价值 [2]。这一类方法计算成本较低，并能利用成熟的 2D 预训练网络，但视图选择和切片间距会影响实际三维上下文。

随后研究转向 3D CNN。Dou 等使用多层级上下文 3D CNN，在不同感受野上学习候选局部结构并在 LUNA16 假阳性削减赛道取得领先结果 [4]。DeepLung 将 3D 检测与结节分类组合为完整系统，说明体积卷积和端到端特征对 CT 任务的潜力 [5]。LungSeek 等后续工作继续探索 3D 残差网络、选择性卷积核和多尺度融合，以适应结节大小与形态差异 [7]。

3D 网络能直接建模体积关系，但显存占用、重采样误差、扫描层厚差异和训练样本量成为新约束。2.5D 方法用相邻切片作为通道，在单层 2D 与完整 3D 之间折中。本项目采用三张相邻轴位切片与残差学习主干 ResNet18 [9]，定位是可复现、计算受控的候选分类基线，而不是声称替代完整 3D 检测器。

### 2.3 从单任务分类到临床验证

肺结节研究还逐步从“是否为结节”扩展到结节类型、良恶性和管理建议。Ciompi 等使用多流、多尺度网络对临床相关结节类型进行分类 [6]。这类研究提示多尺度形态与上下文对风险判断重要，但不能直接与候选假阳性削减结果横向比较，因为输入对象、标签和终点均不同。

近年的重点进一步转向真实临床队列和外部验证。例如 Hendrix 等在非筛查胸部 CT 上评估可随访良性结节、原发肺癌和转移灶，并使用多位胸部放射科医师与长期结局构建参考标准 [8]。这代表研究评价从单一公开数据集向多中心、异构协议、临床终点和可部署性迁移。

### 2.4 当前仍未解决的问题

- 高敏感度候选生成会产生大量难负例，单纯随机负采样无法完全覆盖血管和胸膜等混淆结构。
- 公开数据常存在扫描协议、重建核、层厚和厂商偏差，内部划分难以估计跨中心性能。
- 候选级 Accuracy/AUC 容易被类别先验和采样策略影响，需同时报告 PR-AUC、F1、敏感度、每扫描假阳性及置信区间。
- 同一 CT 内多个候选相关，统计推断不应把所有候选视为完全独立样本。
- LIDC-IDRI 的多观察者标注本身具有不确定性；二值化策略会改变任务定义。

## 3. 原型学习研究现状

### 3.1 度量学习中的类别原型

Prototypical Networks 将每个类别的支持集嵌入均值作为原型，并按查询样本到类别原型的距离完成少样本分类 [10]。它建立了清晰的归纳偏置：同类样本在嵌入空间中聚集，分类由与类别代表的距离决定。该思想后来被扩展到多原型、层次原型、开放集识别和类别不平衡场景。

单一类别均值难以覆盖肺结节内部的多模态形态，例如实性、亚实性、不同尺寸或胸膜邻近结节。多原型设计可以为一个类别保留多个局部模式，但也引入原型数、聚类稳定性、异常样本和病例重复代表等问题。原型数越多不一定越好：过少会欠拟合类内结构，过多会记忆训练样本并增加推理成本。

### 3.2 原型与监督对比学习

监督对比学习利用标签将同类表示拉近、异类表示推远，在有限数据、自然腐蚀和超参数稳定性方面显示出优势 [12]。原型损失可被视为更紧凑的类别级对比约束：样本不是与 batch 内全部样本比较，而是与类别代表比较。

本项目使用的 $L_{proto}$ 是对“负类原型证据/正类原型证据”做交叉熵，不是标准 SupCon 的样本对损失。其优点是计算量随原型数而非 batch 内样本对数量增长，并与最终正负原型证据一致；局限是性能依赖原型库质量，且静态原型可能滞后于训练过程中不断变化的特征空间。

### 3.3 可解释的案例型原型

ProtoPNet 提出“this looks like that”的原型部件推理，用训练图像中的原型部件为分类提供案例证据 [11]。ProtoTree 将原型与决策树结合，形成可追踪的层次决策路径 [13]。PIP-Net 进一步强调稀疏、直观的原型评分和原型与人类视觉概念的一致性 [14]。这些工作推动原型从纯粹度量工具走向“可解释性内生”的模型设计。

但“原型来自真实训练样本”并不自动保证临床可解释：

- 潜空间的相似性可能依赖扫描仪、窗位或边缘伪影，而非病灶形态。
- 一个全局 512 维原型对应整张 patch，不等价于经过定位验证的解剖部件。
- 最近邻病例可能与查询病例来自相同 CT、同一患者或高度相似的重建版本，必须去重。
- 原型库反映训练集覆盖范围，少数亚型和外部人群可能没有合适代表。

本项目的 PBIP-Lite 与经典 few-shot Prototypical Networks、ProtoPNet 均不同：它不采用 episodic few-shot 训练，也不学习空间部件原型；它从强增强 ResNet18 的训练特征中按类别做球面 K-means，选择真实病例代表，再把正负类 Top-k 余弦证据与基础 logit 融合。其“解释性”主要表现为病例级检索和正负证据分解，仍需结合 Grad-CAM、去重和人工审阅验证。

## 4. AI 可解释性与 Grad-CAM 研究现状

### 4.1 后验类激活图

Grad-CAM 使用目标类别分数对卷积特征图的梯度作为通道权重，生成类别相关的粗粒度空间热图，并且不需要修改网络或重新训练 [15]。设末层卷积特征图为 $A^k$、目标分数为 $y^c$，则：

$$
\alpha_k^c=\frac{1}{Z}\sum_i\sum_j
\frac{\partial y^c}{\partial A_{ij}^k},\qquad
L_{\mathrm{GradCAM}}^c=operatorname{ReLU}
\left(\sum_k\alpha_k^cA^k\right).
$$

Grad-CAM++ 对梯度权重做更细化的组合，以改善多实例或不完整覆盖场景 [16]。后续工作还发展了无梯度 CAM、扰动解释和针对检测/嵌入模型的适配方案。

### 4.2 医学影像中的价值

在肺结节候选分类中，热图可用于检查模型是否关注候选中心、邻近血管、胸壁或图像边界。把样本按 TP、FP、FN、TN 分组，比只展示高置信正确样本更有诊断价值：

- TP 可观察模型是否关注结节主体及其边缘；
- FP 可定位血管交叉、胸膜或噪声诱发的错误关注；
- FN 可发现微小、低对比度或非典型结节是否被忽略；
- TN 可检查模型抑制非结节结构时使用的证据。

本项目对阳性融合 logit 相对于 ResNet18 最后残差块反向传播，并把热图叠加到三层输入的中心层。该设计适合候选级错误分析，但热图的空间分辨率受最后卷积层和上采样限制。

### 4.3 解释并不等于因果或定位真值

Adebayo 等提出模型参数与标签随机化 sanity checks，指出部分显著图可能主要反映输入边缘或网络结构，而非已学习参数 [17]。Gupta 等进一步从完整性和可靠性角度讨论显著图的内在评价，强调仅用保留高显著像素后的预测变化仍不足以证明因果解释 [18]。

因此，Grad-CAM 的彩色覆盖不能单独证明模型使用了临床合理证据。本项目遵循以下表述边界：

- 没有像素级病灶分割真值时，只报告定性覆盖和样本类别，不声称定位准确率。
- 解释性分析必须同时包含正确与错误样本，避免展示偏差。
- 热图应与模型置信度、原型检索结果和原始 CT 联合审阅。
- 若开展后续定量研究，应加入随机化检验、插入/删除曲线、病灶掩膜重叠和多医师评分。

## 5. 模型鲁棒性研究现状

### 5.1 从对抗扰动到常见退化

鲁棒性研究既包括人为优化的对抗扰动，也包括临床更常见的噪声、模糊、压缩、分辨率下降和强度偏移。ImageNet-C/ImageNet-P 建立了多种常见腐蚀、多个严重度等级和相对性能下降的标准化评价思路 [19]。这种“固定腐蚀类型 × 逐级强度”的设计比只在干净测试集报告单点 AUC 更容易暴露模型失效边界。

医学影像需要把通用退化映射到成像物理和工作流：低剂量 CT 会增加量子噪声，重建核会改变纹理与锐度，层厚和重采样影响微小结节，窗宽窗位改变可见对比度，PACS 导出或远程传输可能引入压缩与降采样。MedMNIST-C 将数字伪影、噪声、模糊、颜色/强度和任务特定腐蚀扩展到多种医学影像，并采用五级严重度，说明面向模态的定向增强通常比泛化的自然图像增强更有意义 [22]。

### 5.2 域偏移与外部泛化

Zech 等在跨医院胸片实验中发现，模型能够识别医院来源，并且内部性能优势不一定迁移到外部医院 [20]。这说明模型可能利用设备、流程或患病率相关的混杂信号。CT 场景还会受到厂商、剂量、层厚、造影剂、重建算法和人群差异影响，因此“外部验证”必须是病例独立、来源明确且预处理固定的评估。

针对肺结节 CT，Shen 等直接研究了图像噪声对深度学习良恶性分类的影响，表明即使干净测试 AUC 较高，现实噪声仍可能改变部分样本预测 [21]。这类研究把鲁棒性从抽象安全问题转化为 CT 剂量和噪声模型下可测量的稳定性问题。

### 5.3 本项目的六类压力测试

本项目在同一固定测试候选和冻结阈值下评估六类退化，每类设 1–5 级：

| 退化 | 模拟问题 | 五级范围 |
|---|---|---|
| 低对比度 | 病灶与肺实质对比下降 | 因子 0.90 → 0.30 |
| 高斯噪声 | 低剂量或重建噪声 | $\sigma$ 0.01 → 0.10 |
| 高斯模糊 | 分辨率/运动或平滑重建 | 核 3 → 9，$\sigma$ 0.5 → 2.0 |
| 窗宽窗位偏移 | 阅片/导出强度设置变化 | WW +100→+700，WL +25→+250 HU |
| JPEG 压缩 | 导出、传输或截图压缩 | quality 90 → 15 |
| 下采样 | 低空间分辨率与重采样 | scale 0.85 → 0.25 |

报告 AUC 和 F1 相对干净数据的变化。AUC 主要反映排序稳定性；F1 同时受冻结判别阈值和概率校准漂移影响。二者联合能够区分“排序仍稳定但操作点失准”和“表征本身失效”两类问题。

压力测试仍不是临床外部验证。合成高斯噪声不能完整替代真实低剂量投影与重建过程，JPEG 也不是 DICOM 原生压缩的全部情形。最可靠的结论应同时来自合成腐蚀曲线、真实跨协议/跨中心数据和亚组分析。

## 6. 本项目要解决的核心痛点

| 核心痛点 | 风险 | 本项目的应对 | 当前边界 |
|---|---|---|---|
| 候选假阳性多、类别不均衡 | 模型偏向负类或训练成本过高 | 全部正例 + 1:3 负采样；报告 PR-AUC/F1 | 采样测试集不能替代官方全候选 FROC |
| 单一线性头难以表达类内多模态 | 对不同形态结节或难负例覆盖不足 | 分类别球面 K-means、多原型 Top-k 余弦证据 | 静态原型依赖训练集覆盖，尚未在线更新 |
| 黑盒输出难以审阅 | 无法判断模型是否利用病灶或伪影 | TP/FP/FN/TN Grad-CAM + Top-3 病例检索 | 尚无像素级解释准确率和临床读者试验 |
| 干净测试性能掩盖退化失效 | 噪声、模糊、窗位或分辨率变化导致性能下降 | 六类退化 × 五级 AUC/F1 曲线 | 合成退化不能代替真实扫描协议变化 |
| 单次实验结论不稳定 | 随机初始化与采样导致偶然差异 | seed 0/1/2、均值±标准差、统计检验 | 三个 seed 的检验功效仍有限 |
| 候选间相关导致过窄 CI | 同一 CT 的多个候选被当作独立样本 | 按 `seriesuid` 做 CT 簇 Bootstrap | 外部集还需可靠患者/扫描标识 |
| 外部数据重叠或标签不完整 | 数据泄漏、错误地声称外部泛化 | LIDC 优先、去重与两类检查；失败时记录并回退 LUNA16 固定测试 | 当前 LIDC 本地样本不足，不能声称正式 LIDC 外部验证 |

## 7. 项目技术定位

本项目不是提出新的通用原型学习理论，而是把轻量、可追溯的原型证据整合进可复现的肺结节候选分类工程。其技术组合具有以下特点：

1. **计算受控**：以 2.5D ResNet18 代替大规模 3D 网络，适合有限显存和多随机种子实验。
2. **证据分解**：分别计算正负类原型证据，再与基础分类 logit 融合，避免无类别聚合的语义混乱。
3. **病例可追溯**：原型来自真实训练 patch，并保存病例、簇、相似度和源 checkpoint 信息。
4. **解释互补**：原型检索回答“像哪些训练病例”，Grad-CAM 回答“当前 patch 的哪些区域影响预测”；两者并非同一种解释。
5. **评价完整**：主指标、三 seed 稳定性、CT 簇置信区间、差异检验、退化曲线和固定测试共同约束结论。
6. **不确定性诚实**：当 LIDC-IDRI 无法去重或不满足二分类评估条件时，不生成伪外部结果，而是回退并明确记录数据源。

后续研究优先级应是：建立真正独立且有可靠病例标识的外部 CT 队列；在物理空间统一 spacing 与方向；补充多尺度或 3D 表征；对原型进行病例级去重和临床语义标注；用病灶掩膜、随机化检验及医师评分定量验证解释；在真实低剂量/多厂商协议上校准鲁棒性曲线。

## 8. 参考文献

1. Armato SG III, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans. *Medical Physics*. 2011;38(2):915–931. [doi:10.1118/1.3528204](https://doi.org/10.1118/1.3528204)

2. Setio AAA, Ciompi F, Litjens G, et al. Pulmonary nodule detection in CT images: false positive reduction using multi-view convolutional networks. *IEEE Transactions on Medical Imaging*. 2016;35(5):1160–1169. [doi:10.1109/TMI.2016.2536809](https://doi.org/10.1109/TMI.2016.2536809)

3. Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge. *Medical Image Analysis*. 2017;42:1–13. [doi:10.1016/j.media.2017.06.015](https://doi.org/10.1016/j.media.2017.06.015)

4. Dou Q, Chen H, Yu L, Qin J, Heng PA. Multilevel contextual 3-D CNNs for false positive reduction in pulmonary nodule detection. *IEEE Transactions on Biomedical Engineering*. 2017;64(7):1558–1567. [doi:10.1109/TBME.2016.2613502](https://doi.org/10.1109/TBME.2016.2613502)

5. Zhu W, Liu C, Fan W, Xie X. DeepLung: deep 3D dual path nets for automated pulmonary nodule detection and classification. In: *2018 IEEE Winter Conference on Applications of Computer Vision*. 2018:673–681. [doi:10.1109/WACV.2018.00079](https://doi.org/10.1109/WACV.2018.00079)

6. Ciompi F, Chung K, van Riel SJ, et al. Towards automatic pulmonary nodule management in lung cancer screening with deep learning. *Scientific Reports*. 2017;7:46479. [doi:10.1038/srep46479](https://doi.org/10.1038/srep46479)

7. Zhang H, Zhang H. LungSeek: 3D Selective Kernel residual network for pulmonary nodule diagnosis. *The Visual Computer*. 2023;39:679–692. [doi:10.1007/s00371-021-02366-1](https://doi.org/10.1007/s00371-021-02366-1)

8. Hendrix W, Hendrix N, Scholten ET, et al. Deep learning for the detection of benign and malignant pulmonary nodules in non-screening chest CT scans. *Communications Medicine*. 2023;3:156. [doi:10.1038/s43856-023-00388-5](https://doi.org/10.1038/s43856-023-00388-5)

9. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016:770–778. [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

10. Snell J, Swersky K, Zemel R. Prototypical Networks for Few-shot Learning. In: *Advances in Neural Information Processing Systems 30*. 2017. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)

11. Chen C, Li O, Tao D, Barnett A, Rudin C, Su JK. This Looks Like That: deep learning for interpretable image recognition. In: *Advances in Neural Information Processing Systems 32*. 2019. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html)

12. Khosla P, Teterwak P, Wang C, et al. Supervised Contrastive Learning. In: *Advances in Neural Information Processing Systems 33*. 2020. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)

13. Nauta M, van Bree R, Seifert C. Neural Prototype Trees for interpretable fine-grained image recognition. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2021:14933–14943. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.html)

14. Nauta M, Schlötterer J, van Keulen M, Seifert C. PIP-Net: Patch-Based Intuitive Prototypes for interpretable image classification. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2023:2744–2753. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Nauta_PIP-Net_Patch-Based_Intuitive_Prototypes_for_Interpretable_Image_Classification_CVPR_2023_paper.html)

15. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: *Proceedings of the IEEE International Conference on Computer Vision*. 2017:618–626. [CVF Open Access](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)

16. Chattopadhay A, Sarkar A, Howlader P, Balasubramanian VN. Grad-CAM++: generalized gradient-based visual explanations for deep convolutional networks. In: *2018 IEEE Winter Conference on Applications of Computer Vision*. 2018:839–847. [doi:10.1109/WACV.2018.00097](https://doi.org/10.1109/WACV.2018.00097)

17. Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity Checks for Saliency Maps. In: *Advances in Neural Information Processing Systems 31*. 2018. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)

18. Gupta A, Saunshi N, Yu D, Lyu K, Arora S. New definitions and evaluations for saliency methods: staying intrinsic, complete and sound. In: *Advances in Neural Information Processing Systems 35*. 2022. [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d6383e7643415842b48a5077a1b09c98-Abstract-Conference.html)

19. Hendrycks D, Dietterich TG. Benchmarking neural network robustness to common corruptions and perturbations. In: *International Conference on Learning Representations*. 2019. [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)

20. Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine*. 2018;15(11):e1002683. [doi:10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683)

21. Shen C, Tsai MY, Chen L, et al. On the robustness of deep learning-based lung-nodule classification for CT images with respect to image noise. *Physics in Medicine & Biology*. 2020;65(24):245037. [doi:10.1088/1361-6560/abc812](https://doi.org/10.1088/1361-6560/abc812)

22. Di Salvo F, Doerrich S, Ledig C. MedMNIST-C: comprehensive benchmark and improved classifier robustness by simulating realistic image corruptions. In: *ADSMI 2024: Workshop on Advancing Data Solutions in Medical Imaging AI*. 2024. [arXiv:2406.17536](https://arxiv.org/abs/2406.17536)
