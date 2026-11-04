"""pi0.5 (openpi) fine-tuning for the YAM bimanual robot.

Runs inside the openpi virtualenv (see scripts/setup_openpi.sh), not in yam_train's own .venv:

    yam_policy.py   YamInputs / YamOutputs: robot data <-> model inputs (used for training AND serving)
    config.py       openpi TrainConfigs for YAM (registered into openpi's config table)
    to_lerobot.py   data/nedf3_30hz (yam_train.data.convert) -> LeRobot dataset (openpi's training format)
    cli.py          train / norm-stats / serve / eval entry points (thin wrappers around openpi's scripts)
    eval_offline.py open-loop evaluation on held-out demos (same metrics as the act_ref grasp test)

See docs/02_repo.md.
"""
