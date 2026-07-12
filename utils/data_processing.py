from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class FruitImageDataset(Dataset):
    """Load an ImageFolder-style fruit dataset with stable class indices.

    Passing the training set's ``class_to_idx`` to validation/test datasets
    guarantees that labels stay aligned even if a split is missing a class.
    """

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir).expanduser()
        self.transform = transform
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {self.root_dir}")

        self.image_paths = []
        self.labels = []
        directories = sorted(path for path in self.root_dir.iterdir() if path.is_dir())

        if class_to_idx is None:
            self.class_to_idx = {path.name: index for index, path in enumerate(directories)}
        else:
            self.class_to_idx = dict(class_to_idx)
            expected = set(range(len(self.class_to_idx)))
            if set(self.class_to_idx.values()) != expected:
                raise ValueError("class_to_idx values must be contiguous indices starting at zero")

        unknown_classes = [path.name for path in directories if path.name not in self.class_to_idx]
        if unknown_classes:
            raise ValueError(f"classes are not present in class_to_idx: {unknown_classes}")

        for class_dir in directories:
            label = self.class_to_idx[class_dir.name]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.image_paths.append(str(image_path))
                    self.labels.append(label)

        if not self.image_paths:
            raise ValueError(f"no supported images found in {self.root_dir}")
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        with Image.open(img_path) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_mapping(self):
        return self.class_to_idx.copy()