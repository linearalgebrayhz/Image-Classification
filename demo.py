import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from models import build_model
from trainer import create_transforms, resolve_device


def load_predictor(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    model = build_model(
        checkpoint["model_name"],
        num_classes=len(class_to_idx),
        **checkpoint["model_kwargs"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _, transform = create_transforms(checkpoint["image_size"])

    def predict(image):
        if image is None:
            return {}
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(dim=1)[0].cpu()
        return {
            idx_to_class[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }

    return predict


def main():
    parser = argparse.ArgumentParser(description="Launch the fruit classifier Gradio demo")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/model.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio URL")
    args = parser.parse_args()

    predictor = load_predictor(args.checkpoint, resolve_device(args.device))
    interface = gr.Interface(
        fn=predictor,
        inputs=gr.Image(type="pil", label="Fruit image"),
        outputs=gr.Label(num_top_classes=5, label="Prediction"),
        title="Fruits-360 classifier",
        description="Upload a fruit image to classify it with a trained checkpoint.",
    )
    interface.launch(share=args.share)


if __name__ == "__main__":
    main()
