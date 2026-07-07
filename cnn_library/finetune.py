"""Part 2 of the assignment: fine-tune torchvision's ImageNet-pretrained
ResNet-18 on FGVC-Aircraft.

2A trains with the exact hyperparameters used for the best Part 1 model
(Adam, lr=1e-3, 20 epochs — see `assignment_module_two.ipynb`, cell 8).

2B then tweaks the optimizer to fit the fact that we're fine-tuning a
pretrained backbone rather than training from scratch:
- SGD with momentum + a step decay schedule, following PyTorch's own
  transfer-learning tutorial
  (https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html),
  which reports this combination working better than Adam for fine-tuning
  a pretrained conv net;
- a much smaller learning rate (1e-4 vs 1e-3), since large updates on
  pretrained weights risk catastrophic forgetting of the ImageNet features
  (Yosinski et al., "How transferable are features in deep neural
  networks?", NeurIPS 2014);
- mild weight decay, since FGVC-Aircraft's ~6700 trainval images are far
  fewer than ImageNet and the model is prone to overfitting.

`TRANSFORM` is left at `None` for both parts for now, so results are
comparable and any accuracy change is attributable to the optimizer alone.
"""

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

import matplotlib.pyplot as plt

from cnn_library.data import FGVCAircraftDataset
from cnn_library.train import evaluate, train

DATA_ROOT = "data/fgvc-aircraft"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 100

BATCH_SIZE = 64
TRANSFORM = None


def build_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def make_history():
    history = {"loss": [], "accuracy": []}

    def callback(epoch, loss, accuracy):
        history["loss"].append(loss)
        history["accuracy"].append(accuracy)

    return history, callback


def plot_history(history, test_loss, test_accuracy, title, out_path):
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, history["loss"], label="train")
    ax_loss.axhline(test_loss, color="tab:orange", linestyle="--", label="test")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["accuracy"], label="train")
    ax_acc.axhline(test_accuracy, color="tab:orange", linestyle="--", label="test")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_loader():
    dataset = FGVCAircraftDataset(DATA_ROOT, split="trainval", transform=TRANSFORM)
    print(f"Loaded {len(dataset)} images")
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def run_2a():
    """Fine-tunes with Part 1's best hyperparameters: Adam, lr=1e-3, 20 epochs."""

    model = build_model()
    history, callback = make_history()

    model = train(
        model,
        make_loader(),
        epochs=20,
        learning_rate=1e-3,
        device=DEVICE,
        callback=callback,
    )

    test_loss, test_accuracy = evaluate(
        model, DATA_ROOT, transform=TRANSFORM, batch_size=BATCH_SIZE, device=DEVICE
    )
    print(f"[2A] test loss={test_loss:.4f} accuracy={test_accuracy:.4f}")
    plot_history(history, test_loss, test_accuracy, "Part 2A: baseline fine-tuning", "finetune_2a.png")

    return model, test_loss, test_accuracy


def run_2b():
    """Fine-tunes with hyperparameters tweaked for transfer learning: SGD with
    momentum, a lower learning rate, weight decay, and a step decay schedule."""

    model = build_model()
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    history, callback = make_history()

    model = train(
        model,
        make_loader(),
        epochs=20,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        callback=callback,
    )

    test_loss, test_accuracy = evaluate(
        model, DATA_ROOT, transform=TRANSFORM, batch_size=BATCH_SIZE, device=DEVICE
    )
    print(f"[2B] test loss={test_loss:.4f} accuracy={test_accuracy:.4f}")
    plot_history(history, test_loss, test_accuracy, "Part 2B: tuned fine-tuning", "finetune_2b.png")

    return model, test_loss, test_accuracy


if __name__ == "__main__":
    print("=== Part 2A: fine-tuning with Part 1's hyperparameters ===")
    run_2a()

    print("\n=== Part 2B: fine-tuning with tuned hyperparameters ===")
    run_2b()
