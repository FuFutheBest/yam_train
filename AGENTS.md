# AGENTS.md

## 任务目标

为 i2rt YAM 双臂主从遥操作机器人完成网络训练部分。用户电脑负责开发，训练在**服务器**运行，推理在 `../yam_infer`。

## 机器

见 [SEVERS.md](SEVERS.md)。ssh 前向用户确认（除非用户要求）。

## 分析依据

- 数据采集项目：`~/Dev/MVIG/SuperInference-feat-i2rt_NEDF3/`（只读；不能在用户电脑运行，不修改其中文件）
- 推理项目：`~/Dev/MVIG/yam_infer/`（`pi05` 分支有 pi0.5 客户端）
- 训练数据：`../dataset/<task>/`（任务说明见各自 README.md）
- **不要参考** `~/Dev/MVIG` 下和服务器上的其他尝试（dp / pi0 / pi0.5 / act 等目录），独立分析决定方案。

## 文档（`docs/`）

| 文档 | 内容 |
|---|---|
| `01_background.md` | 背景知识（用户不懂具身 / 神经网络，用到的新概念写在这里） |
| `02_repo.md` | 现在的仓库结构、配置、命令、部署接口 |
| `VENDORED.md` | 拷贝的源码与本地改动 |
| `HISTORY.md` | 演进历史、代码变化、各轮实验结果（新结果追加到这里） |

- 风格：mermaid 图 + 表格优先，避免大段文字。
- 01 / 02 只描述**当前**状态；代码改动后同步更新，旧内容移到 HISTORY.md。

## 仓库要点（详见 docs/02_repo.md）

| 部分 | 位置 | 运行环境 |
|---|---|---|
| 数据转换 | `yam_train/data/`（`convert` → `data/<name>/`） | `.venv` |
| ACT（主线 act_ref） | `configs/*.yaml`、`yam_train/{detr,policy,train}.py`、`tools/` | `.venv`（`scripts/setup_venv.sh`；本机加 `cpu`） |
| pi0.5 | `yam_train/pi05/`、`configs/pi05/` | openpi 的 uv 环境：`source scripts/pi05_env.sh` 后用 `$PI05_PY` |
| 脚本 | `scripts/`（`train_all.sh`、`train_pi05.sh`、`pi05_after_act.sh` 等） | 服务器 |

### ACT 约定

- **一切按源码 `tonyzhaozh/act`**。偏离只有两处：新增 `data.action_mode: delta`；`data.action_shift` 用 0（源码 1）。
- 新的改动必须做成配置项，默认值保持源码行为。
- `yam_train/detr/` 原样拷贝，不改、不 lint；改动记 `docs/VENDORED.md`。
- 配置：`act_ref.yaml` → `long.yaml` → `{block_bowl,blue_cup,fold_towel,unroll_cylinder}_{abs,delta}.yaml`（`_base_` 继承）。
- 不放推理代码；推理端需要对齐的项列在 docs/02_repo.md。

### pi0.5 约定

- openpi 固定 `215abfb`（`scripts/setup_openpi.sh` 装到 `../openpi`）；本机静态检查用 `tmp/openpi`（已写入 pyrightconfig extraPaths）。
- 配置在 `yam_train/pi05/config.py::CONFIGS`：`pi05_yam`（积木放碗 + 蓝杯，`data/nedf3_30hz_v3`，右臂 delta）、`pi05_yam3`（蓝杯 100 + 叠毛巾 + 拔盖子，`data/nedf3_30hz_r3`，两臂 delta）；各有 `_split`（去掉 `configs/pi05/val_episodes*.json`，与 ACT 80/20 验证集相同）。

## 服务器目录

| 目录 | 内容 |
|---|---|
| `~/output/fufu/yam_train_r3` | 第三轮 ACT 训练 |
| `~/output/fufu/yam_train_pi05` | pi0.5：`lerobot/`、`runs_pi05/`、`models/`、`logs/`；openpi 在 `../openpi`，`pi05_base` 在 `../.cache/openpi` |

服务器目录不是 git 仓库，代码用 tar 同步。

## 工作约定

- 用户不懂具身 / 神经网络 → 独立探索、自主设计，并在 01_background.md 解释用到的知识。
- 训练后可能继续调整：保持代码、配置、日志可复现。
- 不提交 `data/`、`runs/`、`models/*.pt`、`.venv/`、`tmp/`。

## 已交付模型（`../yam_infer/models/`）

| 模型 | 来源 |
|---|---|
| `act_v2_all.pt`、`act_v2_delta_s15k.pt` | 旧方案 act_v2（提交 b258d75），训练参数见 docs/HISTORY.md |
| `pi05_yam/` | `pi05_yam` run1 14999 步 |
