# PyTorch Fruits-360 Image Classification

An end-to-end image classification project for the
[Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits). It includes
custom ResNet, Vision Transformer, and Mamba-style state-space classifiers,
training and evaluation, resumable checkpoints, and a Gradio demo.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
./data_downloader.sh data
```

The path passed to the trainer must directly contain the dataset's `Training`
and `Test` directories. Depending on the archive layout, this is typically:

```text
data/fruits-360_100x100/fruits-360/
├── Training/
└── Test/
```

## Train and evaluate

```bash
python3 trainer.py \
  --data-root data/fruits-360_100x100/fruits-360 \
  --model resnet \
  --epochs 10
```

Select `resnet`, `vit`, or `mamba` with `--model`. Run
`python3 trainer.py --help` for architecture, optimizer, data loading, device,
and checkpoint options.

Training evaluates on the test split after each epoch and writes a complete,
resumable checkpoint to `checkpoints/model.pt`. Resume a matching run with:

```bash
python3 trainer.py \
  --data-root data/fruits-360_100x100/fruits-360 \
  --model resnet \
  --epochs 20 \
  --resume
```

Architecture options must match the saved checkpoint when resuming.

## Demo

After training, launch the local interface:

```bash
python3 demo.py --checkpoint checkpoints/model.pt
```

Add `--share` only when you want Gradio to create a temporary public URL.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests exercise all three model forward passes, stable dataset label
mapping, and command-line parsing without requiring the full dataset.

## Completed project checklist

- [x] Data preprocessing and PyTorch Dataset/DataLoader
- [x] ResNet, Vision Transformer, and Mamba-style models
- [x] Configurable training loop
- [x] Evaluation
- [x] Resumable model and optimizer checkpoints
- [x] Gradio demo
- [x] Command-line argument parsing
