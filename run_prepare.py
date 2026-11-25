from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.prepare import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare bearing image datasets from HUST XLS or MAT vibration recordings")
    parser.add_argument("--dataset", choices=("HUST", "CWRU"), required=True)
    parser.add_argument("--raw-dir", "--raw-root", dest="raw_dir", required=True)
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", required=True)
    parser.add_argument("--image-way", "--representation", dest="image_way", choices=("cwt", "stft", "chirplet", "rp", "wsst", "gadf", "mssi"), default="cwt")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--axis", choices=("X", "Y", "Z"), default="X")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--max-images-per-class", type=int, default=None,
                        help="Smoke test: generate at most N images for every condition/class/split")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a complete, verified output directory without modifying it")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = Path(args.dataset_config) if args.dataset_config else root / "configs" / f"{args.dataset.lower()}.yaml"
    if not config.is_absolute():
        config = (root / config).resolve()
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    image_size = args.image_size if args.image_size is not None else int(settings.get("image_size", 128))
    prepare_dataset(
        dataset=args.dataset,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        image_way=args.image_way,
        image_size=image_size,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
        settings_override=settings,
        axis=args.axis,
        scan_only=args.scan_only,
        max_images_per_class=args.max_images_per_class,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
