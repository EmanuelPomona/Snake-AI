from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import STATE_SIZE


class LinearQNet(nn.Module):
    def __init__(self, input_size=STATE_SIZE, hidden_size=256, output_size=3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


def save_checkpoint(
    path,
    model,
    optimizer,
    episode,
    best_score,
    epsilon,
    target_model=None,
    extra=None,
):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "episode": episode,
        "best_score": best_score,
        "epsilon": epsilon,
    }

    if target_model is not None:
        checkpoint["target_model_state_dict"] = target_model.state_dict()

    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(path, model, optimizer=None, target_model=None):
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint_input_size = checkpoint.get("input_size")
    if checkpoint_input_size is None and "linear1.weight" in checkpoint["model_state_dict"]:
        checkpoint_input_size = checkpoint["model_state_dict"]["linear1.weight"].shape[1]
    model_input_size = model.linear1.in_features

    if checkpoint_input_size is not None and checkpoint_input_size != model_input_size:
        raise ValueError(
            f"Checkpoint input_size={checkpoint_input_size} does not match "
            f"current model input_size={model_input_size}. Start a new checkpoint "
            "directory for this state representation or load a matching model."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if target_model is not None:
        target_state = checkpoint.get("target_model_state_dict")
        if target_state is None:
            target_model.load_state_dict(model.state_dict())
        else:
            target_model.load_state_dict(target_state)

    return checkpoint
