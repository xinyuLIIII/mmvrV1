# 本地改动 vs GitHub 远端（origin/main）对照

> 用途：记录 `D:\mmvrV1` 本地工作区相对 GitHub 仓库多出/改动了什么，以及在新机器上复现当前工作状态需要同步哪些东西。
> 本文件需要**持续维护**：每次同步或有新改动时，刷新上面的基线信息，并在文末「变更记录」追加一条。

## 0. 基线信息

| 项 | 值 |
|---|---|
| 远端 | `origin` = `https://github.com/xinyuLIIII/mmvrV1.git` |
| 基线 `origin/main` | `d5d5e940bfd8cfcb73a0aaf8c2ba3d278a44433c`（2026-07-06 21:09 +0800，`revert: restore code to 2026-07-02 version`） |
| 本地 `HEAD`（分支 `main`） | `d5d5e94`，与 `origin/main` **相同**（ahead 0 / behind 0） |
| 快照时间 | 2026-10-30 21:04 +0800（首版）；**2026-10-30 晚些时候刷新**（补入 LOSO pilot 改动，相关文件 mtime 21:09–21:12） |
| 交接包 | `handoff/`（未跟踪，不进补丁）：`mmvrV1_local_changes.patch`（本文件 §2 中需同步的全部改动，基于 `d5d5e94`）+ `HANDOFF_FOR_AI.md`（写给新机器上 AI 的操作说明）；用 `bash scripts/make_handoff.sh` 重新生成 |
| fetch 状态 | ✅ 2026-10-30 21:09 `git fetch origin` 成功，远端 `main` 仍为 `d5d5e94`，无新提交。（若 fetch 连不上 github.com:443，可临时走系统代理：`git -c http.proxy=http://127.0.0.1:7897 fetch origin`，不改 git 配置） |

### 如何重新生成对照

```bash
cd /d/mmvrV1
bash scripts/diff_vs_remote.sh --fetch     # 只读：fetch + 列 ahead/behind、tracked 改动、untracked 文件(大小/mtime)

# 或手动：
git fetch origin
git log --oneline origin/main..HEAD        # 本地领先的提交
git log --oneline HEAD..origin/main        # 本地落后的提交
git diff --ignore-cr-at-eol --stat origin/main   # tracked 改动（忽略 CRLF）
git diff --ignore-cr-at-eol origin/main -- config.py
git status --porcelain --untracked-files=all     # 未跟踪文件
```

> 注意：本机 `core.autocrlf=true`，工作区文件是 CRLF，而 `HEAD` 里的 `config.py` 是 CRLF/LF 混合。直接 `git diff HEAD -- config.py` 会显示近 140 行"改动"，其实绝大多数只是行尾差异，务必加 `--ignore-cr-at-eol` 看真实改动。

---

## 1. 提交层面

本地与 `origin/main` 无提交差异。最近几次提交（均已在远端）：

| 提交 | 时间 | 内容 |
|---|---|---|
| `70a6f65` 增加双任务检测分支 | 07-06 10:10 | 新增 `dataset/dataset_identity.py`、`models/multitask_resnet.py`、`train_multitask.py`，`config.py` 加 `--task/--identity_loss_weight/--test_size` |
| `68836a5` 更新双任务 | 07-06 20:03 | 仅改 `train_multitask.py`（+6/−2） |
| `d5d5e94` revert: restore code to 2026-07-02 version | 07-06 21:09 | **把上面两次提交整体撤回**：从仓库删除 `dataset/dataset_identity.py`、`models/multitask_resnet.py`、`train_multitask.py`，`config.py` 恢复到 07-02 版本 |

**结论**：远端仓库目前**不包含任何双任务（手势×身份）代码**。这些文件在 revert 之后以 *untracked* 形式留在本地，并在 07-07 / 08-03 / 10-29 继续被重写和扩展——本地版本与 `70a6f65/68836a5` 中的版本**都不相同**（见 §3.1）。

---

## 2. 汇总表

状态说明：新增 = untracked（远端没有）；修改 = tracked 且内容不同。
"LOSO" = 21:04 快照之后由 LOSO pilot agent 改动/新增（该 agent 已被用户主动停止，不会再在本机继续改，详见 §3.8）。

| 文件 | 状态 | 类别 | 作用 | 新环境是否需要同步 | 备注 |
|---|---|---|---|---|---|
| `config.py` | 修改 | code | 新增 `--task {gesture,identity,dual}`（默认 `dual`）、`--identity_loss_weight`（默认 1.0）；LOSO 加了 `--loso_person`、`--loso_out`、`--loso_save_ckpt` | ✅ 需要 | LOSO（21:09 改）；双任务训练与 `tests/test_multitask_config.py` 依赖 |
| `dataset/dataset_identity.py` | 新增 | code | 双任务数据集：kpt 骨架 + 手势标签 + 身份标签，trial 分块划分；LOSO 加了 `persons/trials` 参数和 `build_loso_datasets` | ✅ 需要 | LOSO（21:09 改） |
| `models/multitask_resnet.py` | 新增 | code | 共享 ResNet50 + 手势头/身份头（硬共享） | ✅ 需要 | |
| `train_multitask.py` | 新增 | code | 双任务/单任务训练入口 | ✅ 需要 | 10-29 版，LOSO **没有改**它 |
| `models/disentangle.py` | 新增 | code | 解耦 Route A：GRL、判别器、`DisentangleNeck`、λ 调度、参数注册 | ✅ 需要 | 08-03 写成，未训练过；LOSO 没改 |
| `models/multitask_resnet_disent.py` | 新增 | code | 带解耦颈部的双任务模型；`method='none'` 时等价于 `MultiTaskResNet` | ✅ 需要 | LOSO 没改 |
| `train_multitask_disent.py` | 新增 | code | Route A 训练入口（自己往 `config.parser` 注册解耦参数）；LOSO 加了 `--loso_person` 分支 | ✅ 需要 | LOSO（21:09 改）；LOSO 实验唯一的训练入口 |
| `scripts/run_loso_pilot.py` | 新增 | code | LOSO 3 折 × 3 seed × A/B/C 的可续跑驱动脚本 | ✅ 需要 | LOSO（21:12 新增），未运行过 |
| `scripts/summarize_loso_pilot.py` | 新增 | code | 汇总 `results.csv`：均值±标准差、B−A / C−A 配对差 | ✅ 需要 | LOSO（21:12 新增），未运行过 |
| `train_cls_remote.py` | 新增 | code | `train_cls.py` 的变体：可选 `--cls_model resnet50/skeleton_gcn`、`--cls_run_name`，记录 GCN 超参 | 🟡 可选 | 07-02 写成；GCN stage2 分类用。远端 `train_cls.py` 只用 resnet50 |
| `tests/test_dataset_identity.py` | 新增 | test | 用临时 fixture 测 trial 划分、删人、补零；LOSO 加了 `test_loso_split_holds_out_person_and_uses_seen_val_block` | ✅ 需要 | LOSO（21:10 改）；新测试**还没跑过** |
| `tests/test_multitask_config.py` | 新增 | test | 测 `--task` 默认 `dual`、`--identity_loss_weight` 默认 1 | ✅ 需要 | |
| `tests/test_multitask_resnet.py` | 新增 | test | 测三种 task 的头是否正确构建 | ✅ 需要 | |
| `tests/test_train_multitask_epoch.py` | 新增 | test | 测 `run_multitask_epoch` 在 dual/gesture 下能跑并正确计数 | ✅ 需要 | |
| `tests/test_disentangle.py` | 新增 | test | GRL 梯度取反、λ 上界、floor gate、解耦 epoch 能跑等 | ✅ 需要 | |
| `docs/gesture_identity_overview.md` | 新增 | doc | 双任务课题**唯一**总览文档（10-29 整合，10-30 更新） | ✅ 需要 | |
| `docs/LOCAL_CHANGES_vs_REMOTE.md` | 新增 | doc | 本文件 | ✅ 需要 | |
| `scripts/diff_vs_remote.sh` | 新增 | env | 只读刷新本文件用的对照脚本 | ✅ 建议 | |
| `CLAUDE.md` | 新增 | doc | 给 Claude Code 的仓库说明 | 🟡 建议 | 只描述 stage1/stage2 姿态部分，没写双任务 |
| `.merge_backup/`（4 个文件） | 新增 | code（备份） | 07-06 10:16 合并时留下的旧版双任务代码备份 | ❌ 不需要 | 见 §3.6 |
| `data/archive.zip` | 新增 | data | mmVR 原始数据集压缩包（23.8 GB，20824 个文件） | ⚠️ 需要，但**不进 git** | 见 §4 |
| `sensors-25-07249-v2.pdf` | 新增 | doc（参考文献） | Sensors 2025 论文 PDF（4.5 MB） | ❌ 不进 git | 网盘/本地文献库 |
| `王琛学长论文.docx` | 新增 | doc（参考） | 学长硕士论文（Wi-Fi 步态 + 行走方向，多任务）21 MB | ❌ 不进 git | 私人资料 |
| `wangchen_extract.txt` | 新增 | doc（派生） | 上面 docx 的纯文本抽取（153 KB） | ❌ 不进 git | 可随时从 docx 重新抽取 |
| `experiments/loso_pilot/` | 不存在 | 实验产物 | LOSO 驱动脚本的输出目录 | ❌ 不用拷 | 本机**从未运行**，目录还没生成；在新机器上跑时自动创建，`experiments/` 已被 `.gitignore` 忽略 |

---

## 3. 分区说明

### 3.1 双任务手势 × 身份（硬共享基线）

**`dataset/dataset_identity.py`**（10-30 21:09 版，含 LOSO 改动，见 §3.8）
- `MultiTaskDataset(dataset_root, mode, persons=None, trials=None)`：读 `<dataset_root>/kpt_output/<name>.npy`（stage1 预测出的骨架），返回 `(kpt[30,42,3], gesture_label, identity_label)`。
- list 文件位置：若 `dataset_root` 末级是 `mmVR`，就去上一级找 `train_list.txt/eval_list.txt`（即 `./data/`），否则用 `dataset_root` 本身。两个 list **合并**后再按 trial 划分。
- 划分：去掉 4 个人 `DROP_PERSONS = {02,04,05,12}`，剩 8 人映射成身份 0–7；trial `01–06,15–20` 为训练，`09–12` 为测试，`07,08,13,14` 为隔离带（丢弃）。
- 单手手势（label 0–5）在左侧**固定**补 21 个零关节，用来规避之前"补零位置泄漏身份"的问题（见总览文档 §2.1）。
- 与远端历史版本（`70a6f65`）的区别：旧版用 `sklearn.GroupShuffleSplit` 按 (person, scene, gesture) 组随机切分并带 `--test_size`；新版改为确定性的 trial 分块，**不再依赖 sklearn**，`--test_size` 参数也随之去掉。

**`models/multitask_resnet.py`**（07-07 版）
- `MultiTaskResNet(task)`：`models/ResNet.resnet50(include_top=False)` → avgpool → 2048 维 → `head_gesture`/`head_identity`，按 `task` 只建需要的头；`forward` 返回 `{'gesture': logits|None, 'identity': logits|None}`。
- 依赖远端已有的 `models/ResNet.py`。

**`train_multitask.py`**（10-29 版，LOSO 没改）
- `run_multitask_epoch(...)`：dual 时 loss = CE(gesture) + λ·CE(identity)，λ = `--identity_loss_weight`。分别统计两个任务的准确率。
- `main()`：`model_name = train_mt_{task}`，权重保存到 `./experiments/param/<model_name>/best.pth`，日志写到 `./experiments/<model_name>_*.log` 和 `logs/<model_name>/`（TensorBoard）。
- 数据用 `--dataset_root_kpt`。**注意**：默认值是 `./data`，但数据集类需要指向含 `kpt_output/` 的目录，实际要传 `--dataset_root_kpt ./data/mmVR`。
- 依赖远端已有的 `utils/train_runtime.py`、`utils/misc.py`；需要 `tensorboardX`、`tqdm`。
- 写死了 `os.environ['CUDA_VISIBLE_DEVICES'] = "0"`。
- 与 `68836a5` 版的区别：旧版的 `resolve_model_name`（dual 按 λ 起目录名，比如 `train_mt_dual_l0.3`）、`compute_loss`/`update_metrics` 辅助函数、随机种子设置，在当前版里都**没有了**。所以当前版跑多组 λ 时，输出目录会**互相覆盖**（都叫 `train_mt_dual`）。`experiments/train_mt_dual_l{0.3,1,3}_*.log` 是 07-06 用旧版跑出来的。

### 3.2 解耦模块（Route A：有界双向 GRL）

- **`models/disentangle.py`**（08-03）：可单独使用，不改动任何已有文件。包含 `grad_reverse`（DANN 梯度反转）、`Discriminator`、`DisentangleNeck`（2048 → f_G | f_I | f_S 三个子空间 + 重构锚点）、`DisentConfig`、`add_disent_args(parser)`、`config_from_args`、`dann_anneal`、`scheduled_lambda`、`floor_gate`、`LambdaController`。
- **`models/multitask_resnet_disent.py`**：`DisentMultiTaskResNet`，`cfg.method == 'none'` 时数值上等价于 `MultiTaskResNet`，所以基线也能用这个文件复现。
- **`train_multitask_disent.py`**：Route A 入口。它会在 `config.parser` 上自己注册解耦参数，所以**不需要改 `config.py`**。示例：
  `python train_multitask_disent.py --dataset_root_kpt ./data/mmVR --task dual --disent_method bigrl --disent_lambda_max 0.3`
- 状态：代码已经写好，**还没训练过**（见总览文档 §7）。

### 3.3 Stage2 分类变体

- **`train_cls_remote.py`**：比远端 `train_cls.py` 多了 `build_classification_model`，可以选 `resnet50` 或 `skeleton_gcn`（`models/SkeletonGCN.py`，远端已有），并且多了 `--cls_run_name` 和 GCN 超参记录。这些参数在远端 `config.py` 里已经有了，所以这个文件在远端代码上能直接跑。看名字应该是在远程服务器上用的版本，和双任务无关。要复现 GCN 分类结果时才需要同步。

### 3.4 配置

- **`config.py`**（相对 `origin/main` 忽略行尾后多了 8 行，21:09 最后修改）：
  ```python
  parser.add_argument('--task', default='dual', choices=['gesture', 'identity', 'dual'])
  parser.add_argument('--identity_loss_weight', type=float, default=1.0)
  # LOSO（留一人）：设置后 train_multitask_disent.py 用 build_loso_datasets 划分
  parser.add_argument('--loso_person', default=None, help=...)   # 留出的 person_id，如 07
  parser.add_argument('--loso_out', default='', help=...)        # 单次运行的结果 JSON 路径
  parser.add_argument('--loso_save_ckpt', action='store_true', default=False)
  ```
  后 3 个参数只有 `train_multitask_disent.py` 会读；`train_multitask.py` 不受影响。
- 行尾：工作区是 CRLF，仓库里是混合的。如果要提交，建议顺便统一行尾，否则 diff 会很吵。

### 3.5 测试

| 测试文件 | 依赖 | 是否需要真实数据 |
|---|---|---|
| `tests/test_dataset_identity.py` | `dataset/dataset_identity.py`、numpy、torch（LOSO 测试另需 pytest） | 否（`tmp_path` 里造 fixture） |
| `tests/test_multitask_config.py` | `config.py`（需要上面两个新参数） | 否 |
| `tests/test_multitask_resnet.py` | `models/multitask_resnet.py` | 否 |
| `tests/test_train_multitask_epoch.py` | `train_multitask.run_multitask_epoch` | 否 |
| `tests/test_disentangle.py` | `models/disentangle.py`、`models/multitask_resnet_disent.py` | 否 |

本机的系统 Python 3.12 **没装 torch 和 pytest**，所以这些测试在当前环境里还没跑通过（总览文档也有记录）。LOSO 新增的 `test_loso_split_holds_out_person_and_uses_seen_val_block` 同样**从未执行过**。

另外，`tests/` 下远端原有的 `test_mamba_model.py` 依赖已废弃的 `mamba_ssm`，在没装 mamba 的机器上跑 `pytest tests/` 时它失败是正常的，不影响双任务 / LOSO。

### 3.6 `.merge_backup/`

- 内容：`config.py`、`dataset/dataset_identity.py`、`models/multitask_resnet.py`、`train_multitask.py`，时间都是 2026-07-06 10:16，也就是 `70a6f65` 提交后几分钟。
- 看起来是当时合并/拉取前对本地版本做的备份。内容和 `70a6f65`、`68836a5`、当前工作区**都不完全一样**，比如 backup 里的 `config.py` 有 `--test_size`，`dataset_identity.py` 用的是 `GroupShuffleSplit` + `EXCLUDED_PERSONS`。
- 结论：只是历史快照，当前代码不依赖它，**不需要同步**。可以留在本机作参考，确认没用后再删（建议加进 `.gitignore`）。

### 3.7 文档

- `docs/gesture_identity_overview.md`：双任务课题唯一的总览（研究问题、数据泄漏、划分方案、已有结果和需要复核的地方、文献、Route A、下一步）。**需要同步**。
- `CLAUDE.md`：仓库说明，只覆盖姿态估计主线。建议同步，也可以提交到仓库。
- 注意 `.gitignore` 里有 `docs/plans/`，但 `docs/` 下其他文件不会被忽略。

### 3.8 留一人（LOSO）3 折对比实验

> 状态：**代码已写完，但没测过、也没跑过**。新增的单元测试从未执行；`experiments/loso_pilot/` 不存在。LOSO agent 被用户主动停止，不会在本机继续；实验改到另一台机器上跑（步骤见 §6 的 7–11 步）。

**目的**：回答"加身份任务 / 加解耦，会不会提升跨用户手势识别"。
- A = 只做手势（`--task gesture --disent_method none`）
- B = 手势 + 身份硬共享（`--task dual --disent_method none`）
- C = 手势 + 身份 + bigrl 解耦（`--task dual --disent_method bigrl`）
- 3 个留出者 × 3 个 seed（0/1/2）× 3 个模型 = **27 次运行**。
- **选 epoch 只看见过的人的验证块**（val 手势准确率最高的那个 epoch），留出者的数据绝不参与选模型。`LambdaController` 也只看 val 身份准确率。
- 报告两个指标：留出者的**手势**准确率（跨用户），以及见过的人在验证块上的**身份**准确率（闭集，仅 B/C）。

**划分**（`dataset/dataset_identity.py: build_loso_datasets(dataset_root, heldout_person)` → `(train, val, heldout)`）：
- 留出者：`FOLDS = ('01', '07', '11')`，即 8 个保留者排序后的第 0/3/7 位，覆盖 person_id 的低、中、高段。必须是 `IDENTITY_MAP` 中的人，否则抛 `ValueError`（例如删掉的 `02`）。
- train = 其余 7 人的 `TRAIN_TRIALS`（01–06、15–20）
- val = 其余 7 人的 `TEST_TRIALS`（09–12），和 train 之间隔着 `GAP_TRIALS`（07、08、13、14）
- heldout = 留出者的**全部** trial（`ALL_TRIALS`，包括隔离带）
- 身份头仍是 8 类，留出者那一类在训练中不会出现。
- 实现方式：`MultiTaskDataset` 新增 `persons=None`（只保留这些人）和 `trials=None`（覆盖按 mode 选择的 trial 集合）两个可选参数；不传时行为和原来一样。

**训练入口的改动**（`train_multitask_disent.py`）：
- 传了 `--loso_person` 就走 LOSO 分支：用 `build_loso_datasets`；`random`/`numpy` 也按 `--seed` 设种子；每个 epoch 额外评估 heldout，在日志里加 `heldout_gesture_acc`。
- 不保存 `best.pth`/`best_constrained.pth`；按 val 手势准确率记录 `selected`；加 `--loso_save_ckpt` 才保存 `experiments/param/<model_name>/best_val_gesture.pth`。
- 结束时把 `{model_name, task, method, loso_person, seed, epochs, train/val/heldout_size, selected, final, history}` 写到 `--loso_out` 指定的 JSON。
- 不传 `--loso_person` 时，行为和原来的 Route A 一样。

**驱动脚本** `scripts/run_loso_pilot.py`：
```bash
python scripts/run_loso_pilot.py [--epochs 60] [--jobs 4] [--threads N] \
    [--adv_start N] [--adv_ramp N] [--dataset_root_kpt ./data/mmVR] [--only A_p01_s0,B_p07_s1]
```
- `--epochs` 默认 60；`--adv_start`/`--adv_ramp` 默认都是 `round(epochs*40/60)`（保持原来 60 轮里第 40 轮开始对抗、爬坡 40 轮的比例，只影响 C）。
- `--jobs` 是并行子进程数（默认 4）；`--threads` 是每个 job 的 `OMP/MKL_NUM_THREADS`（默认 `cpu_count // jobs`）；`--only` 只跑列出的 run id（格式 `<模型>_p<人>_s<seed>`）。
- 每次运行都调用 `train_multitask_disent.py --loso_person <p> --seed <s> --disent_run_suffix _loso_p<p>_s<s>`。公共参数写死为：`--device <见下> --batch_size 32 --lr 1e-4 --weight_decay 1e-4 --lr_drop 200 --clip_max_norm 0.1 --identity_loss_weight 1.0 --num_workers 0 --no_pin_memory --disent_lambda_max 0.3`，同时设置 `TQDM_DISABLE=1`。
  - `--device` 默认 `auto`：有 CUDA 就用 `cuda`，否则用 `cpu`；也可以显式传 `--device cuda` / `cuda:N` / `cpu`。实际使用的设备会打印在驱动日志第一行（`device=...`）。
- 可以续跑：已有 `runs/<run_id>.json` 的运行会被跳过；中断后重新执行同一条命令即可。执行顺序是 seed 优先（先把 seed 0 的 9 个 (模型, 折) 跑完）。
- 输出：
  - `experiments/loso_pilot/runs/<run_id>.json`：每次运行的结果（写完 `.partial` 后再改名；会加 `wall_sec`）
  - `experiments/loso_pilot/logs/<run_id>.log`：子进程 stdout/stderr（第一行是完整命令）
  - `experiments/loso_pilot/results.csv`：每完成一次运行就根据全部 JSON 重建。列：`model, task, method, fold_person, seed, gesture_acc_heldout, identity_acc_val, selected_epoch, val_gesture_acc, *_final, epochs, train/val/heldout_size, wall_sec`
  - 另外每次运行还会照常生成 `experiments/train_mt_<task>_<tag>_loso_p<p>_s<s>_*.log` 和 `logs/<同名>/`（TensorBoard），共 27 组。

**汇总脚本** `scripts/summarize_loso_pilot.py`：
```bash
python scripts/summarize_loso_pilot.py [--csv experiments/loso_pilot/results.csv] [--final]
```
- 按模型 × 折以及全部输出留出者手势准确率的均值 ± 标准差（ddof=1），并给出"各折均值的均值"；B/C 另外输出 val 身份准确率。
- 按 (fold, seed) 配对计算 B−A、C−A 的差值：每折和全部的均值 ± 标准差、正差值个数、配对 t（装了 scipy 才输出 p 值）。脚本里也写明：只有 3 折，p 值只能作为描述。
- `--final` 用最后一个 epoch 的列，不用 val 选出的 epoch。

---

## 4. 数据与大文件

| 文件 | 大小 | 是什么 | 建议 |
|---|---|---|---|
| `data/archive.zip` | 23,828,308,253 B（约 22.2 GiB），20824 个文件 | mmVR 数据集，解压后是 `mmVR/{imu,kpt_gt,kpt_output,mmwave}/` | **不要进 git**（GitHub 单文件上限 100 MB）。用网盘、移动硬盘或 `rsync/scp` 传到新机器，解压到 `data/` 下得到 `data/mmVR/...` |
| `data/mmVR/`（已解压） | — | 本机已解压；`kpt_output/` 有 5206 个文件 | 已被 `.gitignore` 忽略（`/data/mmVR/`） |
| `data/mmwave_cfar` | 33 B | **git 里存的符号链接**，指向 `/root/autodl-tmp/mmVR/mmwave_cfar`（AutoDL 服务器路径）。Windows 上 `core.symlinks=false`，所以它只是个纯文本文件 | 只有 `--cfar_mode os2d` 时才会用到；新机器上需要自己跑 `scripts/precompute_mmwave_cfar.py` 生成，或者改成真正的目录 |
| `data/train_list.txt`、`data/eval_list.txt` | 小 | 远端已有（tracked） | 不用另外处理 |
| `sensors-25-07249-v2.pdf` | 4.5 MB | 参考论文 | 不进 git，放文献库 |
| `王琛学长论文.docx` | 21 MB | 学长学位论文（私人资料） | 不进 git（版权/隐私）；只在需要引用时私下传 |
| `wangchen_extract.txt` | 153 KB | docx 的文本抽取 | 不进 git；需要时从 docx 重新抽 |
| `2501.13805v2.pdf` | 3.1 MB | 原型论文 | 已经在远端（tracked），不用处理 |
| `experiments/`、`logs/` | — | 训练日志、权重、混淆矩阵 | 已被 `.gitignore` 忽略；如果要保留结果（比如 07-06/07-07 的 `train_mt_*` 日志），单独打包备份 |

建议在 `.gitignore` 里补上：`data/archive.zip`、`*.docx`、`wangchen_extract.txt`、`sensors-*.pdf`、`.merge_backup/`，免得以后误提交（本次没改，等用户决定）。

---

## 5. 环境

- 仓库约定的环境：`environment_py310.yaml` + `requirements_py310.txt`，安装脚本 `scripts/setup_mmvr_py310.sh`（conda，Python 3.10，PyTorch 2.5.1 + CUDA 11.8，torchvision 0.20.1；pip 装 tqdm、tensorboardX、opencv、PyYAML、pytest 8.3.5、einops、seaborn、timm）。
  - 这个脚本还会装 mamba/causal-conv1d 的 **Linux x86_64 wheel**，Windows 上装不上。mamba 部分已经废弃，Windows 上可以跳过。
- 双任务 / 解耦代码实际用到：`torch`、`numpy`、`tensorboardX`、`tqdm`（加上远端 `utils/` 依赖的 `h5py`、`scipy` 等）。当前版本**不需要 `sklearn`**（旧版 `GroupShuffleSplit` 需要）。
- 本机现状：系统 Python 3.12.10（没有 conda），**没装 torch / pytest**，所以本机目前跑不了训练和测试。
  - LOSO agent 曾经在 `D:\mmvr_venv` 建了一个 venv，装了 torch 2.15.0+cpu 和 torchvision（CPU wheel），其他依赖**没装完**它就被停止了。⚠️ 本次刷新时检查过，**`D:\mmvr_venv` 已经不存在**（可能被删了，也可能创建没有完成），所以本机目前没有能用的环境。实验反正改到新机器上跑，不必修复。
- **LOSO / 双任务在新机器上需要的包**（按仓库实际 import 核对过）：
  | 包 | 谁用 | 是否必需 |
  |---|---|---|
  | `torch`、`torchvision`、`numpy` | 全部代码（torchvision 被 `utils/misc.py`、`models/backbone.py` import） | ✅ |
  | `tqdm`、`tensorboardX` | `train_multitask*.py` | ✅ |
  | `pytest` | `tests/`（LOSO 测试里有 `pytest.raises`） | ✅ |
  | `scipy` | `utils/matcher.py`；`summarize_loso_pilot.py` 用它算 p 值（没装也能跑，只是不输出 p） | ✅ |
  | `h5py` | `utils/mmwave_cfar_dataset.py`（姿态数据集 `dataset/datasets.py` 用）、`tests/test_dataset_runtime.py`；LOSO 训练本身用不到，但跑全量 `pytest tests/` 需要 | ✅ |
  | `matplotlib`、`pandas` | `utils/misc.py`（训练入口会 import `utils.misc`）、`utils/loss.py` | ✅ |
  | `einops`、`timm`、`seaborn`、`opencv-python`、`PyYAML` | 仓库 `.py` 里**没有直接 import**；写在 `requirements_py310.txt`/安装脚本里 | 🟡 建议照装，保证和仓库约定的环境一致（`tests/test_env_files.py` 只检查环境文件的内容，不检查是否安装） |
  | `sklearn` | 只有 `.merge_backup/` 里的旧代码用 | ❌ |
  | `mamba_ssm` | 已废弃的 mamba 代码 | ❌ |
  一行装完（CPU 机器）：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install numpy tqdm tensorboardX pytest scipy h5py matplotlib pandas einops timm seaborn opencv-python PyYAML`
- 和机器相关的地方：
  - `train_multitask.py`、`train_multitask_disent.py`、`train_cls_remote.py` 都写死了 `CUDA_VISIBLE_DEVICES="0"`。
  - `scripts/run_loso_pilot.py` 的 `--device` 默认 `auto`（有 GPU 用 GPU，否则 CPU）；`--num_workers 0` 仍是写死的（见 §3.8）。
  - `--device` 默认 `cuda`；`--amp` 只在 CUDA 下有效。
  - `data/mmwave_cfar` 符号链接指向 AutoDL 路径 `/root/autodl-tmp/...`。
  - `utils/loss.py:save_as_npy` 里有写死的 `/home/lvyizhe/...`（远端已有，默认不会调用）。
  - Windows 下 `core.autocrlf=true`，所以有行尾差异。

---

## 6. 新环境同步步骤（Checklist）

> 前提：LOSO agent 已停止，本机代码不会再变。同步前仍建议重新跑一次 `bash scripts/diff_vs_remote.sh --fetch`，确认文件列表和 §2 一致。

- [ ] 1. 克隆：`git clone https://github.com/xinyuLIIII/mmvrV1.git && cd mmvrV1`（确认 `HEAD` 是 `d5d5e94` 或更新）
- [ ] 2. 从本机拷贝代码（保持相对路径）：
  ```
  config.py
  dataset/dataset_identity.py
  models/multitask_resnet.py  models/disentangle.py  models/multitask_resnet_disent.py
  train_multitask.py  train_multitask_disent.py  [train_cls_remote.py 可选]
  scripts/run_loso_pilot.py  scripts/summarize_loso_pilot.py
  tests/test_dataset_identity.py  tests/test_multitask_config.py  tests/test_multitask_resnet.py
  tests/test_train_multitask_epoch.py  tests/test_disentangle.py
  docs/gesture_identity_overview.md  docs/LOCAL_CHANGES_vs_REMOTE.md
  scripts/diff_vs_remote.sh  CLAUDE.md
  ```
  更推荐的做法：在本机把这些文件提交到一个新分支再推送（需要用户确认），新机器直接 checkout 这个分支。
- [ ] 3. 不要拷：`.merge_backup/`、`*.docx`、`wangchen_extract.txt`、PDF。
- [ ] 4. 数据：把 `data/archive.zip` 传过去，在仓库根目录执行 `unzip data/archive.zip -d data/`，确认 `data/mmVR/kpt_output/` 有 5206 个 `.npy`，`data/train_list.txt` / `data/eval_list.txt` 存在。
- [ ] 5. （只有 CFAR 实验需要）删掉/替换 `data/mmwave_cfar` 链接，跑 `python scripts/precompute_mmwave_cfar.py --dataset_root ./data --cfar_mode os2d ...`，参数要和训练时一致。
- [ ] 6. 环境：Linux 上用 `bash scripts/setup_mmvr_py310.sh`。Windows 或只跑双任务 / LOSO 时，可以用 conda/venv 建 Python 3.10 环境，装 torch + torchvision（GPU 版或 CPU 版都行），再装 §5 表里列的包：`pip install numpy tqdm tensorboardX pytest scipy h5py matplotlib pandas einops timm seaborn opencv-python PyYAML`。装完后检查：`python -c "import torch, torchvision, tqdm, tensorboardX, scipy, h5py, matplotlib, pandas, pytest"`。
- [ ] 7. **先跑测试**（LOSO 代码从未测过）：
  ```bash
  python -m pytest tests/test_multitask_config.py tests/test_multitask_resnet.py \
      tests/test_dataset_identity.py tests/test_train_multitask_epoch.py tests/test_disentangle.py -q
  python -m pytest tests -q   # 全量（含远端原有测试；test_mamba_model.py 没装 mamba 时失败是正常的）
  ```
  如果 `test_loso_split_holds_out_person_and_uses_seen_val_block` 或其他双任务 / 解耦测试失败，**先修好再开跑**，不要带着失败的测试开始 27 次运行。
- [ ] 8. 冒烟训练（常规双任务）：`python train_multitask.py --dataset_root_kpt ./data/mmVR --task dual --epoch 1`
- [ ] 9. **LOSO 1 epoch 冒烟 + 测时间**：先只跑一次运行，确认流程走通，并估算总耗时：
  ```bash
  python scripts/run_loso_pilot.py --epochs 1 --jobs 1 --only C_p01_s0
  ```
  检查 `experiments/loso_pilot/runs/C_p01_s0.json` 是否生成（日志在 `experiments/loso_pilot/logs/C_p01_s0.log`），看 `wall_sec` 和日志里的 `train_size / val_size / heldout_size` 是否合理（heldout 非空，val 非空）。
  - 冒烟结果**一定要删掉**：`rm -r experiments/loso_pilot`。驱动脚本会跳过已有 JSON 的运行，不删的话这个 1 epoch 的结果会被当成正式结果。
  - 粗估总耗时 ≈ `wall_sec × 正式 epochs × 27 / jobs`（C 带判别器，会比 A 稍慢）。据此决定 `--epochs`、`--jobs`、`--threads`。
- [ ] 10. **正式跑 LOSO**（可续跑，中断后重新执行同一条命令即可）：
  ```bash
  nohup python scripts/run_loso_pilot.py --epochs 60 --jobs 4 > loso_driver.log 2>&1 &
  ```
  设备默认自动选择（有 GPU 用 GPU），看 `loso_driver.log` 第一行的 `device=` 确认。GPU 上 4 个 job 共享一张卡，显存不够就减小 `--jobs`。
- [ ] 11. 汇总：
  ```bash
  python scripts/summarize_loso_pilot.py            # 按 val 选出的 epoch
  python scripts/summarize_loso_pilot.py --final    # 对照：最后一个 epoch
  ```
  把结论写进 `docs/gesture_identity_overview.md`，再把 `experiments/loso_pilot/`（JSON/CSV/日志）打包备份。
- [ ] 12. 回到本文件，在「变更记录」里加一条同步记录。

---

## 7. 变更记录（Changelog）

### 2026-10-30（交接包）
- 新增：`scripts/make_handoff.sh` —— 用临时 index 生成 `handoff/mmvrV1_local_changes.patch`（排除 `.merge_backup/`、`data/`、PDF/docx、`wangchen_extract.txt`、`handoff/`、`experiments/`、`logs/`）+ `MANIFEST.txt`，并刷新 `handoff/HANDOFF_FOR_AI.md` 的版本块；新机器上在 `d5d5e94` 的干净克隆里 `git apply` 即可，替代 §6 第 2 步的逐个拷文件。

### 2026-10-30（同日晚些时候，LOSO 刷新）
- 基线不变：origin/main = HEAD = `d5d5e94`，ahead/behind = 0/0。
- LOSO pilot agent 在 21:09–21:12 之间做了改动，之后被用户主动停止（不会再在本机继续，实验改到另一台机器上跑）：
  - 修改：`config.py` —— 加 `--loso_person`、`--loso_out`、`--loso_save_ckpt`（相对远端现在多 8 行）。
  - 修改：`dataset/dataset_identity.py` —— `MultiTaskDataset` 加 `persons`/`trials` 可选参数；新增 `build_loso_datasets`（train=其余 7 人的训练块，val=其余 7 人的测试块，heldout=留出者全部 trial）。
  - 修改：`train_multitask_disent.py` —— `--loso_person` 分支：评估 heldout，按 val 手势准确率选 epoch，输出 JSON 结果。
  - 新增：`scripts/run_loso_pilot.py`（27 次运行的可续跑驱动）、`scripts/summarize_loso_pilot.py`（均值±标准差 + 配对差）——都需要同步。
  - 新增测试：`tests/test_dataset_identity.py::test_loso_split_holds_out_person_and_uses_seen_val_block`。
  - 没改：`train_multitask.py`、`models/*`、其他测试。
- 状态：LOSO 代码**没测过也没跑过**；`experiments/loso_pilot/` 不存在。
- 环境：LOSO agent 说在 `D:\mmvr_venv` 建了 venv（torch 2.15.0+cpu + torchvision），其他依赖没装完；本次检查时该目录**已经不存在**。§5 补了按实际 import 核对过的依赖表。
- 修改：`scripts/run_loso_pilot.py` —— 原来写死 `--device cpu`，改为新增 `--device`（默认 `auto`：有 CUDA 用 cuda，否则 cpu），并把设备写进驱动日志。
- 文档：新增 §3.8 LOSO 小节；§2 去掉"进行中"标记；§6 增加 LOSO 步骤（先测试 → 1 epoch 冒烟测时间 → 正式跑 → 汇总）。`scripts/diff_vs_remote.sh` 能自动列出新的 untracked 文件，不需要改。

### 2026-10-30
- 创建本文件和 `scripts/diff_vs_remote.sh`（只读对照脚本）。
- 基线：`origin/main` = `HEAD` = `d5d5e94`（21:04 首次 fetch 失败；21:09 联网后 fetch 成功，确认远端无新提交）。
- 记录状态：1 个 tracked 修改（`config.py` +2 个参数）；22 个 untracked 条目（双任务代码/测试、解耦 Route A、`train_cls_remote.py`、文档、`.merge_backup/`、数据 zip、参考资料）。
- 快照时 LOSO pilot agent 正在并行改动 `train_multitask*.py`、`config.py`、`dataset/dataset_identity.py`、`tests/`，同步前要重新检查。

<!-- 新条目模板（复制到上方，最新的放最前面）：
### YYYY-MM-DD
- 基线：origin/main = <hash>（<日期>），HEAD = <hash>，ahead/behind = <n>/<m>
- 新增：<文件> —— <作用>，是否需要同步
- 修改：<文件> —— <语义上改了什么>
- 删除/已提交到远端：<文件>
- 数据/环境变化：<...>
- 备注：<...>
-->
