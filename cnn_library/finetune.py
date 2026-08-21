import time
from dataclasses import dataclass
from typing import Literal

from torch import nn
from torch.optim import Adam
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from cnn_library.train import _run_epoch, get_loaders, get_scheduler

NUM_CLASSES = 100


def build_resnet():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


@dataclass
class FinetuneConfig:
    """Hyperparameters for the finetuning run in `finetune.py`."""

    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.0
    scheduler: Literal["ReduceLROnPlateau", "CosineAnnealingLR"] = "ReduceLROnPlateau"
    label_smoothing: float = 0.0
    train_transform: T.Compose = None
    mixup_alpha: float = 0.0
    epochs: int = 1
    split_validation: bool = True
    batch_size: int = 128


def finetune(
    model,
    device,
    config: FinetuneConfig,
):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    backbone_params = [
        p for name, p in model.named_parameters() if not name.startswith("fc.")
    ]
    head_params = list(model.fc.parameters())

    for param in backbone_params:
        param.requires_grad = config.lr_backbone != 0
    for param in head_params:
        param.requires_grad = config.lr_head != 0

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    train_loader, val_loader, _ = get_loaders(
        config.train_transform, config.split_validation, config.batch_size
    )

    param_groups = []
    if config.lr_backbone != 0:
        param_groups.append({"params": backbone_params, "lr": config.lr_backbone})
    if config.lr_head != 0:
        param_groups.append({"params": head_params, "lr": config.lr_head})

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = Adam(param_groups, weight_decay=config.weight_decay)
    scheduler = get_scheduler(config.scheduler, optimizer, config.epochs)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    epoch_durations = []

    for epoch in range(config.epochs):
        start = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
            mixup_alpha=config.mixup_alpha,
        )
        val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, device=device)
        duration = time.perf_counter() - start
        epoch_durations.append(duration)

        if scheduler is not None:
            if config.scheduler == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            elif scheduler:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        avg_duration = sum(epoch_durations) / len(epoch_durations)
        eta = avg_duration * (config.epochs - epoch - 1)
        print(
            f"epoch {epoch + 1}/{config.epochs} loss={train_loss:.4f} accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f} "
            f"time={duration:.1f}s eta={eta / 60:.1f}min"
        )

    return model, history
