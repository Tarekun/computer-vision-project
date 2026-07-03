from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

# the bottom strip of every image is a copyright banner burned in by the
# dataset authors, not part of the photograph
COPYRIGHT_BAR_PX = 20
# FGVC-Aircraft images aren't all the same resolution, so a resize is
# mandatory just to let the DataLoader stack samples into a batch tensor.
STD_SHAPING = T.Compose([T.Resize((256, 256)), T.ToTensor()])


def _read_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _parse_variant_file(data_dir, split):
    classes = _read_lines(data_dir / "variants.txt")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    variant_file = data_dir / f"images_variant_{split}.txt"
    samples = []
    for line in _read_lines(variant_file):
        filename, variant = line.split(" ", 1)
        samples.append((filename, class_to_idx[variant]))
    return samples


class FGVCAircraftDataset(Dataset):
    """FGVC-Aircraft dataset, labeled by variant.

    Each `images_variant_{split}.txt` file already lists exactly the images
    belonging to that split together with their variant label, so it alone
    is enough to recover both the train/val/test membership and the class
    of every image (no need to cross-reference `images_{split}.txt`).
    """

    def __init__(self, root, split="train", transform=None):
        self.transform = (
            STD_SHAPING if transform is None else T.Compose([STD_SHAPING, transform])
        )
        data_dir = Path(root) / "data"

        self.images_dir = data_dir / "images"
        self.samples = _parse_variant_file(data_dir, split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        image_path = self.images_dir / f"{filename}.jpg"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image = image.crop((0, 0, width, height - COPYRIGHT_BAR_PX))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
