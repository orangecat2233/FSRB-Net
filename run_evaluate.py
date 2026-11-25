from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import BearingImageDataset
from src.engine import evaluate
from src.models import TransferModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--dataset", choices=("HUST", "CWRU"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    num_classes = 9 if args.dataset == "HUST" else 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferModel(args.model, num_classes, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    dataset = BearingImageDataset(Path(args.data_dir) / args.condition / args.split, args.image_size, args.normalize)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate(model, loader, device, num_classes)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
