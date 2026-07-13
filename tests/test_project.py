import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from demo import load_predictor
from models import build_model
from trainer import model_kwargs, parse_args, save_checkpoint, validate_args
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

    def test_invalid_vit_dimensions_are_rejected(self):
        args = parse_args(
            [
                "--data-root",
                "/tmp/fruits",
                "--model",
                "vit",
                "--model-dim",
                "30",
                "--heads",
                "8",
            ]
        )
        with self.assertRaisesRegex(ValueError, "divisible by heads"):
            validate_args(args)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_loads_in_demo_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            args = parse_args(
                [
                    "--data-root",
                    directory,
                    "--image-size",
                    "32",
                    "--depth",
                    "1",
                    "--resnet-channels",
                    "8",
                    "--checkpoint",
                    str(checkpoint_path),
                ]
            )
            model = build_model("resnet", num_classes=2, **model_kwargs(args))
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=1,
                args=args,
                class_to_idx={"Apple": 0, "Banana": 1},
            )

            predictor = load_predictor(checkpoint_path, torch.device("cpu"))
            prediction = predictor(Image.new("RGB", (32, 32), "red"))

            self.assertEqual(set(prediction), {"Apple", "Banana"})


if __name__ == "__main__":
    unittest.main()
