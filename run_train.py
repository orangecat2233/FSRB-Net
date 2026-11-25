from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.engine import DA_METHODS, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Train FSRB-Net and comparison models")
    parser.add_argument("--dataset", choices=("HUST", "CWRU"), required=True)
    parser.add_argument("--experiment", choices=("ablations", "sota", "custom"), default="custom")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-way", default="cwt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--da", choices=DA_METHODS, default="dann")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lambda-da", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--cache-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parent / config_path).resolve()
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = dict(config_data.get("training", {}))
    config.update({"image_way": args.image_way})
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_da": args.lambda_da,
        "image_size": args.image_size,
        "num_workers": args.num_workers,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["seed"] = args.seed
    config["normalize"] = bool(args.normalize or config.get("normalize", False))
    config["pretrained"] = bool(args.pretrained)

    models = None
    if args.experiment in ("custom", "sota"):
        if args.models:
            models = [item.strip() for item in args.models.split(",") if item.strip()]
        elif args.model:
            models = [args.model]
        elif args.experiment == "custom":
            parser.error("--experiment custom requires --model or --models")

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()] if args.seeds else [args.seed]

    run_experiment(
        dataset=args.dataset,
        data_root=args.data_dir,
        output_root=args.output_dir,
        experiment=args.experiment,
        config=config,
        models=models,
        da_method=args.da,
        seeds=seeds,
        overwrite=args.overwrite,
        max_tasks=args.max_tasks,
        cache_images=args.cache_images,
    )


if __name__ == "__main__":
    main()
