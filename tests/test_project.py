import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from models import build_model
from trainer import parse_args
from utils.data_processing import FruitImageDataset


class ModelTests(unittest.TestCase):
    def test_all_models_produce_class_logits(self):
        configurations = {
            "resnet": {"num_blocks": 1, "base_channels": 8},
            "vit": {
                "image_size": 32,
                "patch_size": 8,
                "dim": 32,
                "depth": 1,
                "heads": 4,
                "mlp_dim": 64,
                "dropout": 0.0,
            },
            "mamba": {
                "image_size": 32,
                "patch_size": 8,
                "dim": 32,
                "depth": 1,
                "state_dim": 8,
            },
        }
        inputs = torch.randn(2, 3, 32, 32)
        for name, kwargs in configurations.items():
            with self.subTest(model=name):
                model = build_model(name, num_classes=5, **kwargs)
                self.assertEqual(model(inputs).shape, (2, 5))


class DatasetTests(unittest.TestCase):
    def test_labels_are_contiguous_and_reusable_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("Training", "Test"):
                for class_name in ("Apple Braeburn", "Banana"):
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True)
                    Image.new("RGB", (8, 8), "red").save(class_dir / "sample.png")

            train = FruitImageDataset(root / "Training")
            test = FruitImageDataset(root / "Test", class_to_idx=train.class_to_idx)

            self.assertEqual(
                train.class_to_idx,
                {"Apple Braeburn": 0, "Banana": 1},
            )
            self.assertEqual(test.labels, [0, 1])
            self.assertEqual(test.idx_to_class[1], "Banana")


class ArgumentTests(unittest.TestCase):
    def test_required_and_model_arguments(self):
        args = parse_args(["--data-root", "/tmp/fruits", "--model", "vit", "--epochs", "2"])
        self.assertEqual(args.data_root, Path("/tmp/fruits"))
        self.assertEqual(args.model, "vit")
        self.assertEqual(args.epochs, 2)


if __name__ == "__main__":
    unittest.main()
