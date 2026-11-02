# 手势 × 身份双任务：总览（唯一入口）

> 整合日期：2026-10-29
> 本文件是该课题唯一的文档。它合并并取代了此前的文献综述（9/2）、数据划分方案（8/1）、两版实验设计与实现计划（7/2、7/7）以及实现过程记录；这些旧文件已删除。
> 实现细节以代码与测试为准；解耦方案 Route A 的设计见 `models/disentangle.py` 的注释。

---

## 0. 一页速览

| 项 | 状态 |
|---|---|
| 研究问题 | 手势识别与身份识别在共享表征下是否**相互竞争**；若是，如何缓解 |
| 数据划分 | ✅ 闭集分块 + trial 级隔离带（§3） |
| 硬共享定性实验 | ✅ 已跑（7/7）；⚠️ **结论需复核**——原文档按不同 epoch 取数，统一口径后差距落在噪声带边缘（§4.3） |
| 文献证据 | ✅ 已整理；⚠️ 均为**间接证据**，无文献直接做过"单任务 vs 联合"的跨用户对照（§5） |
| 理论论证 | 🟡 仅有思路，未写成正式推导（§6） |
| 解耦方案 Route A | 🟡 代码已写（8/3），**未训练过**，单测未在当前环境跑通（缺 pytest）（§7） |
| 下一步 | 统一口径 + 多种子复跑基线 → 换跨用户协议 → λ 扫描 → 跑 Route A（§8） |

**当前能站住的表述**：身份信息是跨用户手势泛化的干扰因子（文献）；subject-overlapping 随机划分下的"双高"不可信（文献 + 已核实全文）。
**当前还站不住的表述**：本项目数据上已证实"双向负迁移"。

---

## 1. 研究问题与目标

- 任务：手势分类（8 类）+ 闭集身份识别（8 人，正式版 12 人）。
- 输入（方案 A，当前）：第 1 阶段预测的骨架关键点 `data/mmVR/kpt_output/<name>.npy`，形状 `(30, 42, 3)`。
  方案 B（后续）：用 `pose_memory`（512 维，含毫米波特征）做端到端多任务。
- 决策逻辑（来自设计文档 §1）：互促 → 复现学长的硬参数共享路线；竞争 → 走特征解耦 / 对抗 GRL。
- 学长论文（`王琛学长论文.docx`，Wi-Fi 步态 + 行走方向，硬参数共享多任务）§2.5 指出：硬参数共享要求任务高度相关，不相关任务会相互产生负面影响。可作为"可能负迁移"的铺垫引用，但注意其结论方向是多任务**提升**性能。

---

## 2. 数据与已知泄漏问题

list 行格式：`file_name,person_id,scene_id,gesture_id,trial_id`；12 人 × 3 场景 × 8 手势 × 20 trial，`kpt_output` 共 5206 个文件。

### 2.1 补零泄漏（致命，已临时规避）

原 `PN_Dataset` 对单手手势（label 0–5）补零凑成双手形状，**补零侧由 `people_id` 决定**：`{02,04,05,12}` 补右侧，其余补左侧。模型只看"零在哪侧"即可把人分成两组 → 身份被直接写进输入布局。

- 当前规避：删除这 4 人 → 8 人，统一补左侧（`dataset/dataset_identity.py`）。
- **正式版必改**：补零与 people_id 解耦（统一固定补零 / 查真实左右手 / 只用双手样本），恢复 12 人。

### 2.2 repeat 近邻泄漏（已缓解）

原论文划分 test = trial 05/10/15/20，与相邻训练 trial 同人、同场景、同手势、同采集条件，测试偏乐观。→ 用 §3 的分块划分替代。

---

## 3. 划分协议：闭集分块 + trial 级隔离带

在每个组 `(person_id, scene_id, gesture_id)` 内按 trial_id 划分：

```text
train: 01-06, 15-20
gap:   07-08, 13-14   （故意弃用，不是验证集）
test:  09-12
```

- 划分单元：整个 trial（一个 `.npy`），禁止帧级切分。
- 实现：`dataset/dataset_identity.py` 中的 `TRAIN_TRIALS / GAP_TRIALS / TEST_TRIALS` 常量，不生成额外 list 文件。
- 规模：12 人时 train 3098 / gap 1046 / test 1062；**当前 8 人时 train ≈ 2029 / test ≈ 701**。
- 性质：闭集，所有人、场景、手势在训练中都出现。**不可**描述为跨人或跨场景泛化。
  英文表述：*closed-set block-wise repeated-trial split with a trial-level gap*。

可选对照基线：① 原 repeat 划分；② 分块无隔离带；③ 分块 + 隔离带（推荐）。

开放问题：`scene_id` 的语义（场景 / 姿态 / 采集条件）；是否做 5 折区块交叉验证；隔离带宽度；是否设独立验证集做 early stopping。

---

## 4. 实验一：硬参数共享定性验证

### 4.1 设计

- 模型：共享 ResNet-50 backbone（2048 维）+ 独立线性头（`models/multitask_resnet.py`，`train_multitask.py`）。
- 三个实验同数据、同划分、同超参，仅改 `--task gesture | identity | dual`；dual 时 λ（`--identity_loss_weight`）= 1。
- 判定规则（噪声带 ±0.5~1%）：
  - 互促：`G_dual ≥ G₀` 且 `I_dual ≥ I₀`
  - 竞争：任一显著下降（跷跷板或双降）
  - 中性：均在噪声内

### 4.2 原文档报告的结果（9/2 综述 §4）

| 实验 | 手势 acc | 身份 acc |
|---|---|---|
| gesture | 85.164 | — |
| identity | — | 97.432 |
| dual | 80.884 | 94.579 |

原结论："手势 −4.28 pp、身份 −2.85 pp，双向负迁移，明确竞争"。

### 4.3 ⚠️ 复核（2026-10-29，基于 `experiments/train_mt_*_test.log`）

上表三行取自**不同口径的 epoch**：

| 实验 | 原文档所取 epoch | 该 epoch 性质 | 最佳 epoch（按 score） | 最后 epoch（60） |
|---|---|---|---|---|
| gesture | ep52 → 85.164 | 非最佳、非最后 | 85.735 | 80.029 |
| identity | ep60 → 97.432 | 最后 | 97.860 | 97.432 |
| dual | ep46 → 80.884 / 94.579 | 非最佳、非最后 | 84.879 / 96.576 | 84.165 / 96.576 |

- **统一按最佳 epoch**：手势 −0.86 pp，身份 −1.28 pp → 位于噪声带边缘，介于"中性"与"弱竞争"之间。
- **统一按最后 epoch**：手势 dual（84.17）反而高于单任务（80.03）。
- `train_mt_gesture_test.log` 内含两次运行（第一次仅 15 epoch 中断），上表取第二次。
- 单种子、无方差估计。

**结论**：在当前闭集协议下，"双向负迁移"**未被可靠证实**。这与 §6 理论预测（闭集下竞争本应较弱）一致，说明当前协议不是暴露竞争的合适协议。

### 4.4 历史 λ 扫描（不可用）

7/6 跑过 λ = 0.3 / 1 / 3（`experiments/train_mt_dual_l*.log`），但在改用分块划分之前，test 手势仅 30–40%，与现结果不可比，作废。

---

## 5. 文献证据（按证明力分级）

> 总判断：文献**没有直接证明**手势与身份在共享模型中负迁移——没有工作做过"跨用户协议下单任务 vs 联合训练"的对照。文献能支撑的是"为什么预期会竞争"；直接证明需要本项目的受控实验来补，这也是论文可占的贡献点。
> 标 **[已核实全文]** 者已读正文；其余为标题/摘要级，引用前需读全文确认消融细节。

### 5.1 去除身份信息 → 跨用户手势变好（最强的间接证据）

- **Adversarial Deep Feature Extraction Network for User-Independent HAR**（arXiv 2110.12163）：对抗 + 梯度反转剥离受试者身份，提升跨用户识别。
- **Redefining Domains for Robust WiFi-based Gesture Recognition**（arXiv 2601.03825）：对抗移除用户/域身份，得到域无关手势识别。

逻辑：压掉身份信息有益 ⇒ 身份信息对手势泛化有害。
缺口：证明的是"身份**信息**有害"，不是"身份**任务**有害"；后者是合理推论但未被直接验证。

### 5.2 动作信息干扰身份识别（反方向证据）

- **ABNet**（arXiv 2403.17360）、**Disentangling Q-Former**（arXiv 2507.07262）：动作线索造成 "identity bias"，需解耦。
- **Dual-Branch Adversarial Feature Disentanglement, cross-subject EMG**（arXiv 2506.08555）：把特征拆为模式相关 / 受试者相关。

与 5.1 合起来构成"两个方向都有干扰"。缺口：前两篇为视觉活动-生物识别，迁移到雷达手势需要论证。

### 5.3 身份隐私混淆（⚠️ 原综述表述过强，已更正）

- **ImmCOGNITO**（arXiv 2602.07139）：降低身份识别准确率、同时保持手势性能。

原综述称其"证明存在此消彼长的权衡"。更正：如果它能在**不损手势**的前提下去除身份，恰恰说明两类信息**可分离**、手势不依赖身份。它支持"解耦可行"（§6 机制 2、§7 Route A 的动机），**不能**作为竞争存在的证据。

### 5.4 排除反例："双高"是评估协议的产物

- **GesturePrint**（arXiv 2408.05358 / ICDCS 2024）**[已核实全文]**：98.87% 手势 / 99.78% 身份，但为 8:2 随机划分 + 5 折，所有人同时出现在训练与测试，无留一人验证；跨距离/环境降到约 93%/87%。
- **JRF-CMAF, 4D Joint Framework**（Sensors 2025, 25, 7249；`sensors-25-07249-v2.pdf`）**[已核实全文]**：99.76/97.57/96.84%，主实验划分比例未说明，无留一人验证；唯一跨受试测试是开集拒识附加实验。

逻辑：排除"二者互促"的反例。缺口：排除反例不等于正面证明竞争。

### 5.5 方法学：随机划分导致虚高

- **On the Evaluation Protocol of UAV gesture recognition**（arXiv 2602.17854）**[已核实全文]**：帧级随机划分混入同一受试者 → "substantially overestimate"；subject-independent 划分是 "fundamental prerequisite"。
- **Critical Analysis of Data Leakage in WiFi-CSI HAR**（PMC11124867 / Sensors 24(10):3159）
- **Revealing Overestimated Performance in WiFi-CSI HAR**（MDPI Signals 6(4):59）

### 5.6 待补（未核实）

通用多任务负迁移：Standley et al. 2020 *Which Tasks Should Be Learned Together?*；Yu et al. 2020 *Gradient Surgery for Multi-Task Learning (PCGrad)*。

### 5.7 查新结果（2026-10-30，三路调研；PDF 存于 `D:\tmp\lit\`、`D:\tmp\nov\pdf\`）

**已被占用（不能作为创新点，只能作为先例引用）：**

| 要素 | 已有工作 |
|---|---|
| 双向 GRL 双分支（手势头 + 身份头互相对抗），跨被试手势 | Niu & Furui, EMBC 2025, arXiv 2506.08555 [已核实全文]：原始 EMG 输入；**身份 acc 从未报告**（仅 t-SNE）；4 折跨被试，平均约 +2.5 pp，无显著性检验 |
| 拆分潜空间 + 双向对抗 + 重建 | Han et al., DA-cAE, IEEE SPL 2020, arXiv 2008.11426 [已核实全文]：生理信号，身份不是输出 |
| 两个子空间 + 两个任务 | Yuan et al., arXiv 2607.14744 [已核实全文]：HD-sEMG，手势**不跨用户** |
| 骨架上联合识别动作 + 闭集身份（内容/风格框架） | Wang & Wang, Pattern Recognition 2018（仅摘要）：结论是多任务**互促**，可作为"闭集下不竞争"的佐证 |
| 骨架上对抗去除身份 | Anonymization for Skeleton Action Recognition, arXiv 2111.15129（仅摘要）等 |
| 毫米波 / WiFi 联合手势 + 身份 | GesturePrint、JRF-CMAF [已核实全文]、WiHF (INFOCOM 2020)、WiDual (TMC 2025)、WiGesID (JSAC 2022)（后三者仅摘要）：都是共享或融合特征，无解耦，未见跨用户手势 + 身份同时报告 |

**未发现被占用：**
1. 在**雷达 + IMU 估计出的手部姿态中间表示**上做手势 / 身份解耦（雷达"关键点 → 手势"路线的 mmEgoHand、mmHand 都没有身份分支）。
2. 一个模型同时给出**闭集身份 + 留一人手势**，并且两项都作为主指标评估。
3. 受控地展示任务关系随协议翻转：随机划分下双赢 → 跨用户下竞争。

**判断：** 损失设计（双向 GRL / 子空间 / 重建）不是创新点。可争取的是"问题设定 + 表示层级 + 实证发现"的组合，强度中等，前提是留一人实验证明干扰存在。

**动机证据：** 原型论文 mmEgoHand（2501.13805）的留一人手势准确率仅约 62–63%，跨用户手势正是基线的已知瓶颈。

**仍需核实：** WiHF 的检索片段提到 "adversarial architecture"，它又是跨域手势 + 用户识别，可能是最近的撞车对象，必须读全文。

---

## 6. 理论论证思路（草稿，未正式推导）

核心表述：**不能证明二者本质互斥；能论证的是"在共享表征 + 追求跨用户泛化"的条件下竞争。**

**机制 0（先承认）：信息上不冲突。** X = f(G, S, ε)，数据设计中每人做全部手势，G ⟂ S。理论上存在 Z = [Z_G, Z_I] 让两个头各取所需。所以竞争只能来自下面的具体机制。

**机制 1（主论证）：身份任务推高跨用户误差上界。** 把每个用户看作一个域，用 Ben-David 域适应上界：

```
ε_新用户(h) ≤ ε_训练用户(h) + ½·d_HΔH(Z_训练用户, Z_新用户) + λ*
```

DANN 用"域判别器 + 梯度反转"压小中间的分布差异项；**身份头就是一个没有梯度反转的域判别器**，它推动各用户在 Z 中可分，方向正好相反。注意：从"训练用户之间可分"推到"与新用户的分布差异变大"是启发式的，需如实说明。

**推论（可证伪预测）**：闭集时目标域就是训练用户，分布差异项接近 0，身份信息甚至可能帮助手势（按人解读动作风格）→ 预测闭集下竞争弱、跨用户下竞争强。§4.3 的复核结果与前半句一致。

**机制 2：两个任务对 I(Z;S) 的要求相反。**（参考 Achille & Soatto 2018）跨用户手势的理想表示 I(Z;S) → 0；身份要求 I(Z;S) → H(S)。共用同一个 Z 时必须妥协；拆分私有子空间则可各自满足 → 直接对应 §7 Route A。物理直觉：身份线索（手的尺寸、速度曲线、习惯）就体现在"手势怎么做"之中，与手势内容混在同一组运动特征里。

**机制 3：梯度冲突（可测量）。** L = L_G + λL_I，身份更新一步对手势损失的一阶影响是 ΔL_G ≈ −ηλ⟨∇L_G, ∇L_I⟩；余弦持续 < 0 即负迁移（PCGrad）。

**对应的实验证据**：
1. 手势用留一人交叉验证，身份在训练用户的保留 trial 上做闭集评估 → 验证机制 1 的预测。
2. 记录共享 backbone 最后一层上两个任务梯度的余弦曲线 → 机制 3。
3. 冻结特征训练身份探针，对比单任务 / 双任务 / 解耦模型的特征中残留多少身份信息 → 机制 2。

---

## 7. 缓解方案：Route A（有界双向 GRL + 私有子空间）

代码（8/3，均为新增文件，不改动现有代码）：`models/disentangle.py`、`models/multitask_resnet_disent.py`、`train_multitask_disent.py`、`tests/test_disentangle.py`。

结构：

```text
ResNet-50 → 2048 → Linear+LayerNorm+ReLU → z = [f_G(320) | f_I(320) | f_S(128)]
手势头读 cat(f_G, f_S)          身份头读 cat(f_I, f_S)
D_I ← GRL(f_G)：把身份推出 f_G   D_G ← GRL(f_I)：把手势推出 f_I
重建头：z → 2048 维特征（MSE），只允许重新分配信息、不允许删除信息
```

关键设定：
- 判别器为 2 层 MLP（hidden 256），表达能力大于线性任务头，满足 DANN 的 H_y ⊂ H_d 条件。
- 对抗梯度默认不回传 backbone（`adv_detach_backbone=True`），只作用在瓶颈投影上。
- λ 调度：前 40 epoch 不启用对抗，之后用 40 epoch 按 DANN 退火升到 `lambda_max = 0.3`。
- floor gate：判别器准确率降到接近随机水平（1/类别数 + margin）时自动减弱对抗。
- `LambdaController`：身份准确率连续 5 epoch 低于 94% 时 λ 减半，只降不升。
- 消融开关 `--disent_method`：`none`（等价于基线）/ `split`（只拆不对抗）/ `bigrl` / `bigrl_g` / `bigrl_i`。

运行：

```bash
python train_multitask_disent.py --dataset_root_kpt ./data/mmVR \
    --task dual --disent_method bigrl --disent_lambda_max 0.3
```

状态：**从未训练过**；单测因当前环境缺 pytest 未跑。

---

## 8. 待办（按优先级）

1. **复核基线**：固定取数口径（固定最后一轮，或用独立验证块选模型），三个实验各跑 ≥3 个种子，报告均值 ± 标准差。
2. **换协议**：手势留一人交叉验证 + 身份闭集，检验机制 1 的预测。
3. **λ 扫描**（0.3 / 1 / 3），排除"权重没调好"被误判为竞争。
4. **诊断量**：梯度余弦曲线、身份探针。
5. **跑 Route A** 的消融：`none / split / bigrl / bigrl_g / bigrl_i`。
6. **修补零泄漏**，恢复 12 人。
7. **方案 B**：`pose_memory` 端到端多任务。
8. 文献：读 §5.1、§5.2 全文核实消融细节；所有 arXiv 编号投稿前用 DOI 核实。

---

## 9. 文件索引

| 类别 | 路径 |
|---|---|
| 数据集 | `dataset/dataset_identity.py`（分块 + 双标签 + 删 4 人） |
| 基线模型 / 训练 | `models/multitask_resnet.py`、`train_multitask.py` |
| 解耦模型 / 训练 | `models/disentangle.py`、`models/multitask_resnet_disent.py`、`train_multitask_disent.py` |
| 测试 | `tests/test_dataset_identity.py`、`test_multitask_resnet.py`、`test_multitask_config.py`、`test_train_multitask_epoch.py`、`test_disentangle.py` |
| 实验日志 | `experiments/train_mt_{gesture,identity,dual}_test.log`（分块划分）；`train_mt_dual_l*.log`（旧划分，作废） |
| 参考资料 | `sensors-25-07249-v2.pdf`（JRF-CMAF）、`王琛学长论文.docx` / `wangchen_extract.txt` |

## 10. 引用清单

| 简称 | 出处 | 角色 |
|---|---|---|
| Adversarial DFE Net | arXiv 2110.12163 | 间接支持竞争（去除身份有益） |
| WiFi Redefining Domains | arXiv 2601.03825 | 间接支持竞争（去除身份有益） |
| Dual-Branch Adv. Disentangle (EMG) | arXiv 2506.08555 | 间接支持（解耦） |
| ABNet | arXiv 2403.17360 | 间接支持（identity bias） |
| Disentangling Q-Former | arXiv 2507.07262 | 间接支持（解耦） |
| ImmCOGNITO | arXiv 2602.07139 | 支持"可解耦"，**不**作为竞争证据 |
| GesturePrint | arXiv 2408.05358 / ICDCS 2024 | 双高反例（协议有泄漏）[已核实全文] |
| JRF-CMAF (4D) | Sensors 2025, 25, 7249 | 双高反例（协议有泄漏）[已核实全文] |
| UAV eval-protocol critique | arXiv 2602.17854 | 泄漏批判 [已核实全文] |
| WiFi-CSI leakage | PMC11124867 / Sensors 24(10):3159 | 泄漏批判 |
| WiFi-CSI overestimation | MDPI Signals 6(4):59 | 泄漏批判 |
| Standley et al. 2020 / PCGrad (Yu et al. 2020) | — | 通用负迁移（待核实） |
| Ben-David et al. 2010 / DANN (arXiv 1505.07818) / Achille & Soatto 2018 | — | §6 理论依据（待核实具体出处） |
