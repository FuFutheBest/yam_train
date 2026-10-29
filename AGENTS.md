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
├── yam_train/         训练代码（源自 ../opus_inference_train @5e547c7，只保留训练相关）
│   ├── data/          nedf3.py 读原始录制 / convert.py 转 npy / dataset.py 切 chunk
│   ├── model/act.py   ACT（ResNet18 + Transformer + CVAE）；改结构要同步 yam_infer
│   ├── train.py       DDP 训练（metrics.jsonl / curves.png / tb / ckpt）
│   └── tools/         eval_offline.py 验证集开环评估、export.py 导出交付模型
├── scripts/setup_venv.sh  建 .venv（默认 torch 2.7.1+cu126，阿里云镜像；`cpu` 参数给无 GPU 机器）
└── docs/              01 数据 / 02 模型 / 03 训练 / 05 报告
```

- 不放推理代码（在 `../yam_infer`）。
- 本机 `.venv` 目前是指向 `../opus_inference_train/.venv` 的软链接；`setup_venv.sh` 遇到软链接会拒绝，需先 `rm .venv`。

## 已交付模型的来源（`../yam_infer/models/`）

| 模型 | 训练参数（`torchrun --nproc_per_node 4 -m yam_train.train --data DATA ...`） | 说明 |
|---|---|---|
| `act_v2_all.pt` | `--out runs/act_v2_all --steps 15000 --action-mode delta --train-split all` | 全部 81 条，step 15000 |
| `act_v2_delta_s15k.pt` | `--out runs/act_v2_delta --steps 30000 --action-mode delta` | 留 8 条验证，取 step 15000 |

当时用 `opus` 包名训练；checkpoint 里只有张量和 dict，不依赖包名，可直接用本仓库加载。
