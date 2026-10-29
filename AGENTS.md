# AGENTS.md

## 任务目标

为 i2rt yam 双臂主从遥操作机器人，从完成网络训练部分代码。 用户电脑负责开发，最终在**服务器电脑**运行。

## 机器区分

见 [SEVERS.md](SEVERS.md)。

## 分析依据

- 数据采集项目：`~/Dev/MVIG/SuperInference-feat-i2rt_NEDF3/`（采集端，不能在用户机器运行）
- 推理项目：`~/Dev/MVIG/yam_infer/`
- 用户的 prompt
- **不要参考** `~/Dev/MVIG` 下和服务器上的其他尝试（dp / pi0 / pi0.5 / act 等目录），独立分析决定方案。

## 工作约定

- 用户不懂具身/神经网络 → 需要我独立探索、自主设计架构，并在文档中解释用到的知识。
- 文档风格：mermaid 图 + 表格优先，避免大段 README 式文字。
- 训练后网络可能还要调整，后续在同一会话继续迭代；保持代码、配置、日志可复现。
- 不修改源项目 `SuperInference-feat-i2rt_NEDF3` 中的文件（只读引用其代码与数据）。

## 仓库结构

```text
yam_train/
├── configs/           act_ref.yaml（源码全部默认值）+ {block_bowl,blue_cup}_{abs,delta}.yaml（_base_ 继承）
├── yam_train/
│   ├── detr/          从 tonyzhaozh/act 原样拷贝的网络（不要改；改动记 docs/VENDORED.md）
│   ├── policy.py      源码 ACTPolicy + 优化器
│   ├── config.py      YAML 读取（_base_ 继承、--set a.b=c 覆盖）
│   ├── data/          nedf3.py 读录制 / convert.py 转数据（原始 JPEG）/ dataset.py = 源码 utils.py
│   ├── train.py       = 源码 train_bc，单卡，一个任务一个模型
│   └── tools/         eval_offline.py（时间集成开环评估）、export.py
├── scripts/setup_venv.sh  建 .venv（默认 torch 2.7.1+cu126，阿里云镜像；`cpu` 参数给无 GPU 机器）
└── docs/              07 当前主线 act_ref；06 旧方案与 ACT 对比；01–05 旧方案 act_v2；VENDORED.md
```

- 主线原则：**一切按 ACT 源码**。偏离只有两处：新增 `data.action_mode: delta`；`data.action_shift` 用 0（源码 1，理由见 docs/07）。新的改动必须做成配置项，默认值保持源码行为。
- 不放推理代码（在 `../yam_infer`）；推理端需要的改动列在 docs/07_act_ref.md 末尾。
- 本机 `.venv` 由 `bash scripts/setup_venv.sh cpu` 建立（torch CPU，只做静态检查/小测试）。`../opus_inference_train` 已弃用。
- 训练数据：`../dataset/{put_block_in_bowl,put_blue_block_in_blue_cup}`（任务说明见各自 README.md）；`data/nedf3_30hz` 是 `convert --raw ../dataset` 的结果（不进 git）。

## 已交付模型的来源（`../yam_infer/models/`，旧方案 act_v2，代码见 git 提交 b258d75）

| 模型                   | 训练参数（`torchrun --nproc_per_node 4 -m yam_train.train --data DATA ...`） | 说明                       |
| ---------------------- | ---------------------------------------------------------------------------- | -------------------------- |
| `act_v2_all.pt`        | `--out runs/act_v2_all --steps 15000 --action-mode delta --train-split all`  | 全部 81 条，step 15000     |
| `act_v2_delta_s15k.pt` | `--out runs/act_v2_delta --steps 30000 --action-mode delta`                  | 留 8 条验证，取 step 15000 |

当时用 `opus` 包名训练；checkpoint 里只有张量和 dict，不依赖包名，可直接用本仓库加载。
