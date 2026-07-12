import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models import build_model
from utils.data_processing import FruitImageDataset


NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train an image classifier on Fruits-360")
    parser.add_argument("--data-root", type=Path, required=True, help="directory containing Training/ and Test/")
    parser.add_argument("--model", choices=("resnet", "vit", "mamba"), default="resnet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4, help="blocks/layers in the selected model")
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--resnet-channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8, help="attention heads for ViT")
    parser.add_argument("--state-dim", type=int, default=16, help="state size for Mamba")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--resume", action="store_true", help="resume model and optimizer from --checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def create_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(round(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    return train_transform, test_transform


def model_kwargs(args):
    if args.model == "resnet":
        return {"num_blocks": args.depth, "base_channels": args.resnet_channels}
    if args.model == "vit":
        return {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "dim": args.model_dim,
            "depth": args.depth,
            "heads": args.heads,
            "mlp_dim": args.model_dim * 2,
        }
    return {
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "dim": args.model_dim,
        "depth": args.depth,
        "state_dim": args.state_dim,
    }


def validate_args(args):
    positive = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "depth": args.depth,
        "image-size": args.image_size,
        "patch-size": args.patch_size,
        "model-dim": args.model_dim,
        "resnet-channels": args.resnet_channels,
        "heads": args.heads,
        "state-dim": args.state_dim,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f"these options must be positive: {', '.join(invalid)}")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay cannot be negative")
    if args.model in {"vit", "mamba"} and args.image_size % args.patch_size:
        raise ValueError("image-size must be divisible by patch-size for ViT and Mamba")
    if args.model == "vit" and args.model_dim % args.heads:
        raise ValueError("model-dim must be divisible by heads for ViT")


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {
        "model_state",
        "optimizer_state",
        "epoch",
        "model_name",
        "model_kwargs",
        "image_size",
        "class_to_idx",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        missing = required.difference(checkpoint if isinstance(checkpoint, dict) else ())
        raise ValueError(
            "checkpoint is not in the resumable project format; "
            f"missing fields: {', '.join(sorted(missing))}"
        )
    return checkpoint


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """Train for one epoch and return mean sample loss."""
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(dataloader, model, loss_fn, device):
    """Return mean loss and accuracy for a data loader."""
    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
    size = len(dataloader.dataset)
    return total_loss / size, correct / size


def save_checkpoint(path, model, optimizer, epoch, args, class_to_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "model_name": args.model,
            "model_kwargs": model_kwargs(args),
            "image_size": args.image_size,
            "class_to_idx": class_to_idx,
        },
        path,
    )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    train_transform, test_transform = create_transforms(args.image_size)
    train_dataset = FruitImageDataset(args.data_root / "Training", train_transform)
    test_dataset = FruitImageDataset(
        args.data_root / "Test",
        test_transform,
        class_to_idx=train_dataset.class_to_idx,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = build_model(
        args.model,
        num_classes=len(train_dataset.class_to_idx),
        **model_kwargs(args),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0

    if args.resume:
        checkpoint = load_checkpoint(args.checkpoint, device)
        if checkpoint["model_name"] != args.model:
            raise ValueError(
                f"checkpoint uses {checkpoint['model_name']!r}, not requested {args.model!r}"
            )
        if checkpoint["class_to_idx"] != train_dataset.class_to_idx:
            raise ValueError("checkpoint classes do not match the training dataset")
        if checkpoint["model_kwargs"] != model_kwargs(args):
            raise ValueError("checkpoint model settings do not match the supplied arguments")
        if checkpoint["image_size"] != args.image_size:
            raise ValueError("checkpoint image size does not match --image-size")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
            parameter_group["weight_decay"] = args.weight_decay
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")
        if start_epoch >= args.epochs:
            print(f"Checkpoint already completed {start_epoch} epochs; nothing to train")
            return

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        test_loss, accuracy = evaluate(test_loader, model, loss_fn, device)
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            epoch + 1,
            args,
            train_dataset.class_to_idx,
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train loss {train_loss:.4f}, test loss {test_loss:.4f}, "
            f"accuracy {accuracy:.2%}"
        )

    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()