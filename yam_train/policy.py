"""ACT policy = tonyzhaozh/act `policy.py::ACTPolicy` + `detr/main.py::build_ACT_model_and_optimizer`.

Same behaviour as the source; the only change is that the model/optimizer are built from an explicit
dict (our config `policy:` section) instead of `argparse.parse_args()` on the process command line.
"""

from __future__ import annotations

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from yam_train.detr.models import build_ACT_model

# defaults of act/detr/main.py::get_args_parser (the ones ACT actually reads)
SOURCE_DEFAULTS = {
    "lr": 1e-4, "lr_backbone": 1e-5, "weight_decay": 1e-4,
    "backbone": "resnet18", "dilation": False, "position_embedding": "sine",
    "enc_layers": 4, "dec_layers": 6, "dim_feedforward": 2048, "hidden_dim": 256, "dropout": 0.1,
    "nheads": 8, "num_queries": 400, "pre_norm": False, "masks": False,
}


def make_args(policy_cfg: dict, camera_names: list[str]) -> argparse.Namespace:
    """policy_cfg uses the source argument names; `chunk_size` maps to `num_queries` like imitate_episodes.py."""
    args = dict(SOURCE_DEFAULTS)
    for k, v in policy_cfg.items():
        if k == "chunk_size":
            args["num_queries"] = v
        elif k != "kl_weight":
            args[k] = v
    args["camera_names"] = list(camera_names)
    return argparse.Namespace(**args)


def build_ACT_model_and_optimizer(args: argparse.Namespace):
    model = build_ACT_model(args)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


class ACTPolicy(nn.Module):
    def __init__(self, policy_cfg: dict, camera_names: list[str]):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(make_args(policy_cfg, camera_names))
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = policy_cfg["kl_weight"]

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            assert is_pad is not None, "is_pad is required together with actions"
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, _is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, _dim_wise_kld, _mean_kld = kl_divergence(mu, logvar)
            loss_dict = {}
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state)  # no action, z = 0
            return a_hat

    def predict(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Inference (z = 0): normalised action chunk (B, chunk_size, 14)."""
        a_hat = self(qpos, image)
        assert isinstance(a_hat, torch.Tensor)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
