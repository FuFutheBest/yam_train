# Vendored code

来源：`tonyzhaozh/act` @ `742c753c0d4a5d87076c8f69e5628c79a8cc5488`（本机 `../policy/act`），MIT（`yam_train/detr/LICENSE`）；DETR 部分版权 Facebook。

| 本仓库 | 来源 | 本地改动 |
|---|---|---|
| `yam_train/detr/models/{backbone,detr_vae,position_encoding,transformer}.py` | `detr/models/` 同名 | 删 `import IPython` / `e = IPython.embed`；`from util.misc` → `from ..util.misc` |
| `yam_train/detr/models/__init__.py` | 同名 | 只保留 `build_ACT_model`（去掉 CNNMLP） |
| `yam_train/detr/util/misc.py` | `detr/util/misc.py` | 无 |
| `yam_train/policy.py` | `policy.py::ACTPolicy`、`kl_divergence`；`detr/main.py::build_ACT_model_and_optimizer` | 从配置 dict 建模型，不读 `sys.argv`；加 `predict()`（= 推理分支） |
| `yam_train/data/dataset.py` | `utils.py` | 重写成读本项目数据；逻辑逐项对齐（见 07_act_ref.md） |
| `yam_train/train.py` | `imitate_episodes.py::main/train_bc` | 读 YAML；额外 z=0 评估 |

`yam_train/detr/` 不做 lint / 类型检查（`ruff.toml`、`pyrightconfig.json` 已排除），保持与上游可 diff：

```bash
diff -r ../policy/act/detr/models yam_train/detr/models
```
