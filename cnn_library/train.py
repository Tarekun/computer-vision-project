import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from cnn_library.data import FGVCAircraftDataset


def _run_epoch(model, loader, criterion, optimizer=None, device="cuda"):
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


def evaluate(model, root, transform=None, batch_size=64, device="cuda"):
    """Computes loss and accuracy on the test split"""

    loader = DataLoader(
        FGVCAircraftDataset(root, split="test", transform=transform),
        batch_size=batch_size,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    return _run_epoch(model, loader, criterion, device=device)


def train(
    model,
    loader,
    epochs,
    learning_rate=1e-3,
    optimizer=None,
    scheduler=None,
    device="cuda",
    callback=None,
):
    """Trains on the merged train+val split (`trainval`) for `epochs` epochs.

    `optimizer` defaults to `Adam(model.parameters(), lr=learning_rate)` when
    not given, so callers who want a different optimizer (e.g. `SGD` with
    momentum, or per-group learning rates) can build one themselves.
    `scheduler.step()` runs after every epoch, if a scheduler is given.
    `callback(epoch, loss, accuracy)` also runs after every epoch, useful for
    collecting a training history to plot.
    """

    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    epoch_durations = []
    for epoch in range(epochs):
        start = time.perf_counter()
        loss, accuracy = _run_epoch(
            model, loader, criterion, optimizer=optimizer, device=device
        )
        if scheduler is not None:
            scheduler.step()
        if callback is not None:
            callback(epoch, loss, accuracy)
        duration = time.perf_counter() - start
        epoch_durations.append(duration)

        avg_duration = sum(epoch_durations) / len(epoch_durations)
        eta = avg_duration * (epochs - epoch - 1)
        print(
            f"epoch {epoch + 1}/{epochs} loss={loss:.4f} accuracy={accuracy:.4f} "
            f"time={duration:.1f}s eta={eta / 60:.1f}min"
        )

    return model
