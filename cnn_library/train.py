from dataclasses import dataclass
import time

import torch
from torchvision.datasets import FGVCAircraft
from typing import Literal
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms as T


@dataclass
class TrainConfig:
    """Hyperparameters for the training run in `train.py`."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    scheduler: Literal["ReduceLROnPlateau", "CosineAnnealingLR"] = "CosineAnnealingLR"
    label_smoothing: float = 0.0
    train_transform: T.Compose = None


def get_loaders(train_transform: T.Compose, split_validation: bool, batch_size: int):
    base_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    trainloader = DataLoader(
        batch_size=batch_size,
        shuffle=True,
        dataset=FGVCAircraft(
            root="./data",
            split="train" if split_validation else "trainval",
            annotation_level="variant",
            download=True,
            transform=train_transform,
        ),
    )
    valloader = DataLoader(
        batch_size=batch_size,
        shuffle=True,
        dataset=FGVCAircraft(
            root="./data",
            # if validation isnt split from train, use test for reporting
            split="val" if split_validation else "test",
            annotation_level="variant",
            download=True,
            transform=base_transform,
        ),
    )
    testloader = DataLoader(
        batch_size=batch_size,
        shuffle=True,
        dataset=FGVCAircraft(
            root="./data",
            split="test",
            annotation_level="variant",
            download=True,
            transform=base_transform,
        ),
    )

    return trainloader, valloader, testloader


def _run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    model.to(device)
    model.train(optimizer is not None)

    total_loss = 0.0
    correct = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

    dataset_size = len(loader.dataset)
    return total_loss / dataset_size, correct / dataset_size


def evaluate(model, train_transform=None, batch_size=64, device="cuda"):
    """Computes loss and accuracy on the test split"""

    _, _, loader = get_loaders(train_transform, False, batch_size)
    criterion = nn.CrossEntropyLoss()
    return _run_epoch(model, loader, criterion, device=device)


def get_scheduler(
    name: Literal["ReduceLROnPlateau", "CosineAnnealingLR"], optimizer, epochs
):
    if name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    elif name == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "OneCycleLR":
        return OneCycleLR(
            optimizer,
            max_lr=[lr_backbone, lr_head],
            total_steps=steps_per_epoch * num_epochs,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=10000,
            epochs=num_epochs,
        )
    else:
        return None


def train(
    model,
    split_validation: bool,
    device,
    config: TrainConfig,
    epochs=1,
):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    train_loader, val_loader, _ = get_loaders(
        config.train_transform, split_validation, 128
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_scheduler(config.scheduler, optimizer, epochs)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    epoch_durations = []

    for epoch in range(epochs):
        start = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer, device=device
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
        eta = avg_duration * (epochs - epoch - 1)
        print(
            f"epoch {epoch + 1}/{epochs} loss={train_loss:.4f} accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f} "
            f"time={duration:.1f}s eta={eta / 60:.1f}min"
        )

    return model, history
