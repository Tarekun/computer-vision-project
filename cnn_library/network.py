from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from typing import Literal, Optional, Tuple, Union


def _as_hw(size):
    return (size, size) if isinstance(size, int) else tuple(size)


class Stem(Module):
    """Resizes the input to a canonical size, then downsamples with
    `conv -> batchnorm -> activation -> maxpool` blocks.

    With the default `canonical_size=256` and three stride-2 poolings,
    output is 256 / 2**3 = 32 pixels per side, at `channels[-1]` channels.
    """

    def __init__(self, in_channels=3, canonical_size=256, channels=(32, 64, 128), kernel_size=3):
        super().__init__()
        self.canonical_size = _as_hw(canonical_size)

        kernel_padding = kernel_size // 2
        blocks = []
        prev_channels = in_channels
        for out_channels in channels:
            blocks += [
                Conv2d(prev_channels, out_channels, kernel_size, padding=kernel_padding),
                BatchNorm2d(out_channels),
                ReLU(inplace=True),
                MaxPool2d(2),
            ]
            prev_channels = out_channels
        self.blocks = Sequential(*blocks)

    def forward(self, x):
        x = F.interpolate(x, size=self.canonical_size, mode="bilinear", align_corners=False)
        return self.blocks(x)


class ResidualStage(Module):
    """conv(stride) -> bn -> act -> conv(stride=1) -> bn, added to a 1x1-conv
    shortcut that reshapes the input to the output's shape, then activated."""

    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = BatchNorm2d(out_channels)
        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.activation(out + identity)


def _split_channels(total, parts):
    base, remainder = divmod(total, parts)
    return [base + (1 if i < remainder else 0) for i in range(parts)]


class InceptionStage(Module):
    """Four parallel paths (pool, 1x1, 3x3, 5x5) whose outputs are
    concatenated to out_channels, added to a 1x1-conv shortcut that reshapes
    the input to the output's shape, then activated — same contract as
    `ResidualStage`.

    The 3x3 and 5x5 paths are preceded by a 1x1 "reduce" convolution that
    bottlenecks the channel count before the spatially expensive convolution,
    as in GoogLeNet.
    """

    def __init__(self, in_channels, out_channels, stride=2, branch_channels=None, reduce_channels=None):
        super().__init__()

        pool_channels, conv1_channels, conv3_channels, conv5_channels = branch_channels or _split_channels(out_channels, 4)
        if reduce_channels is None:
            reduce_channels = max(out_channels // 4, 1)

        self.pool_branch = Sequential(
            MaxPool2d(kernel_size=3, stride=stride, padding=1),
            Conv2d(in_channels, pool_channels, kernel_size=1),
            BatchNorm2d(pool_channels),
            ReLU(inplace=True),
        )
        self.conv1_branch = Sequential(
            Conv2d(in_channels, conv1_channels, kernel_size=1, stride=stride),
            BatchNorm2d(conv1_channels),
            ReLU(inplace=True),
        )
        self.conv3_branch = Sequential(
            Conv2d(in_channels, reduce_channels, kernel_size=1),
            BatchNorm2d(reduce_channels),
            ReLU(inplace=True),
            Conv2d(reduce_channels, conv3_channels, kernel_size=3, stride=stride, padding=1),
            BatchNorm2d(conv3_channels),
            ReLU(inplace=True),
        )
        self.conv5_branch = Sequential(
            Conv2d(in_channels, reduce_channels, kernel_size=1),
            BatchNorm2d(reduce_channels),
            ReLU(inplace=True),
            Conv2d(reduce_channels, conv5_channels, kernel_size=5, stride=stride, padding=2),
            BatchNorm2d(conv5_channels),
            ReLU(inplace=True),
        )

        self.shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        branches = [self.pool_branch(x), self.conv1_branch(x), self.conv3_branch(x), self.conv5_branch(x)]
        out = torch.cat(branches, dim=1)
        return self.activation(out + identity)


class FeatureExtractor(Module):
    """Stack of stages (`ResidualStage` by default), each halving feature
    map size while increasing channels. Agnostic to which stage type is
    used, as long as it follows the `(in_channels, out_channels, stride)`
    contract shared by `ResidualStage` and `InceptionStage`.

    With the defaults, an `in_channels=128` input at 32x32 (the Stem's
    default output) becomes 512 channels at 8x8 (32 / 2**2).
    """

    def __init__(self, in_channels=128, stage_channels=(256, 512), stride=2, stage_cls=ResidualStage, stage_kwargs=None):
        super().__init__()
        stage_kwargs = stage_kwargs or {}

        stages = []
        prev_channels = in_channels
        for out_channels in stage_channels:
            stages.append(stage_cls(prev_channels, out_channels, stride=stride, **stage_kwargs))
            prev_channels = out_channels
        self.stages = Sequential(*stages)

    def forward(self, x):
        return self.stages(x)


class Classifier(Module):
    """Global average pooling into an MLP head. Returns raw logits: pair
    with `nn.CrossEntropyLoss`, which applies log-softmax internally."""

    def __init__(self, in_channels=512, hidden_dim=256, num_classes=100):
        super().__init__()
        self.pool = AdaptiveAvgPool2d(output_size=1)
        self.fc1 = Linear(in_channels, hidden_dim)
        self.activation = ReLU(inplace=True)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.activation(self.fc1(x))
        return self.fc2(x)


@dataclass
class CnnConfig:
    """Hyperparameters for `Cnn`. Defaults reproduce the module's original,
    hardcoded defaults, with a `ResidualStage`-based feature extractor."""

    in_channels: int = 3
    num_classes: int = 100

    # Stem
    canonical_size: Union[int, Tuple[int, int]] = 256
    stem_channels: Tuple[int, ...] = (32, 64, 128)
    stem_kernel_size: int = 3

    # Feature extractor
    stage_channels: Tuple[int, ...] = (256, 512)
    stage_stride: int = 2
    block_type: Literal["resnet", "inception"] = "resnet"
    resnet_kernel_size: int = 3
    inception_branch_channels: Optional[Tuple[int, int, int, int]] = None
    inception_reduce_channels: Optional[int] = None

    # Classifier
    classifier_hidden_dim: int = 256


class Cnn(Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or CnnConfig()

        self.stem = Stem(
            in_channels=config.in_channels,
            canonical_size=config.canonical_size,
            channels=config.stem_channels,
            kernel_size=config.stem_kernel_size,
        )

        if config.block_type == "resnet":
            stage_cls = ResidualStage
            stage_kwargs = {"kernel_size": config.resnet_kernel_size}
        elif config.block_type == "inception":
            stage_cls = InceptionStage
            stage_kwargs = {
                "branch_channels": config.inception_branch_channels,
                "reduce_channels": config.inception_reduce_channels,
            }
        else:
            raise ValueError(f"Unknown block_type: {config.block_type!r}")

        self.feature_extractor = FeatureExtractor(
            in_channels=config.stem_channels[-1],
            stage_channels=config.stage_channels,
            stride=config.stage_stride,
            stage_cls=stage_cls,
            stage_kwargs=stage_kwargs,
        )

        self.classifier = Classifier(
            in_channels=config.stage_channels[-1],
            hidden_dim=config.classifier_hidden_dim,
            num_classes=config.num_classes,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.feature_extractor(x)
        return self.classifier(x)
