from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .representations import REPRESENTATIONS, make_representation


DATASET_PREPROCESSING = {
    "HUST": {
        "sampling_rate": 25600,
        "window_size": 1024,
        "samples_per_class": 200,
        "source_test_ratio": 0.20,
        "target_val_ratio": 0.20,
        "conditions": ("65Hz", "70Hz", "75Hz", "80Hz"),
    },
    "CWRU": {
        "sampling_rate": 12000,
        "window_size": 512,
        "samples_per_class": 200,
        "source_test_ratio": 0.20,
        "target_val_ratio": 0.20,
        "conditions": ("0HP", "1HP", "2HP", "3HP"),
    },
}


def _natural_key(value):
    return [int(piece) if piece.isdigit() else piece.lower() for piece in re.split(r"(\d+)", str(value))]


def _read_hust_xls(path, axis="X", max_points=None):
    axis = str(axis or "X").upper()
    axis_columns = {"X": 2, "Y": 3, "Z": 4}
    if axis not in axis_columns:
        raise ValueError(f"HUST axis must be X, Y, or Z, got {axis}")
    column = axis_columns[axis]
    values = []
    found_data = False
    try:
        with Path(path).open("r", encoding="utf-8-sig", errors="ignore") as handle:
            for line in handle:
                if not found_data:
                    if line.strip().lower() == "data":
                        found_data = True
                    continue
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) <= column:
                    continue
                try:
                    value = float(fields[column].strip())
                except (ValueError, TypeError):
                    continue
                if np.isfinite(value):
                    values.append(value)
                    if max_points is not None and len(values) >= int(max_points):
                        break
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read HUST tab-delimited XLS file: {path}") from exc
    if not found_data:
        raise ValueError(f"HUST XLS file has no Data section: {path}. The original HUST files use tab-delimited text, not binary Excel")
    if not values:
        raise ValueError(f"No numeric {axis}-axis vibration samples found in {path}")
    return np.asarray(values, dtype=np.float64), axis


def _select_signal(candidates, path, variable=None, channel=None):
    valid = []
    for key, value in candidates:
        if variable is not None and key != variable:
            continue
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            continue
        array = np.squeeze(array)
        if array.ndim == 1 and array.size > 1:
            vector = array
        elif array.ndim == 2 and channel is not None:
            channel = int(channel)
            if array.shape[0] >= array.shape[1]:
                if not 0 <= channel < array.shape[1]:
                    raise IndexError(f"Channel {channel} is outside shape {array.shape} in {path}")
                vector = array[:, channel]
            else:
                if not 0 <= channel < array.shape[0]:
                    raise IndexError(f"Channel {channel} is outside shape {array.shape} in {path}")
                vector = array[channel, :]
        else:
            continue
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        valid.append((key, vector))
    if not valid:
        if variable is not None:
            available = [key for key, _ in candidates if re.search(r"DE_time$", key, flags=re.IGNORECASE)]
            raise ValueError(f"Expected MAT variable {variable!r} not found or invalid in {path}. Available drive-end variables: {available}")
        raise ValueError(f"No one-dimensional vibration signal found in {path}; specify variable and optionally channel with --manifest")
    if variable is None:
        drive_end = [item for item in valid if re.search(r"(?:^|_)DE_time$", item[0], flags=re.IGNORECASE)]
        if len(drive_end) == 1:
            valid = drive_end
        elif len(drive_end) > 1:
            raise ValueError(f"Multiple drive-end channels in {path}: {[item[0] for item in drive_end]}. Specify --manifest")
    valid.sort(key=lambda item: item[1].size, reverse=True)
    if variable is None and len(valid) > 1 and valid[0][1].size == valid[1][1].size:
        raise ValueError(f"Multiple equally sized MAT signals found in {path}; select the vibration variable with --manifest")
    signal = valid[0][1]
    signal = signal[np.isfinite(signal)]
    if signal.size < 2:
        raise ValueError(f"Signal is empty after removing invalid values: {path}")
    return signal, valid[0][0]


def load_raw_signal(path, variable=None, channel=None, axis="X", max_points=None):
    path = Path(path)
    if path.suffix.lower() == ".xls":
        return _read_hust_xls(path, axis=axis, max_points=max_points)
    if path.suffix.lower() != ".mat":
        raise ValueError(f"Unsupported raw signal format: {path.suffix} in {path}")
    try:
        payload = loadmat(path)
        candidates = [(key, value) for key, value in payload.items() if not key.startswith("__")]
        signal, key = _select_signal(candidates, path, variable, channel)
    except (NotImplementedError, ValueError) as exc:
        if not isinstance(exc, NotImplementedError) and not any(item in str(exc) for item in ("Unknown mat file type", "Mat file appears to be", "Please use HDF reader")):
            raise
        import h5py
        with h5py.File(path, "r") as handle:
            candidates = [(key, np.asarray(handle[key])) for key in handle if isinstance(handle[key], h5py.Dataset)]
        signal, key = _select_signal(candidates, path, variable, channel)
    if max_points is not None:
        signal = signal[:int(max_points)]
    return signal, key


def _infer_condition(path, conditions):
    for part in path.parts:
        for condition in conditions:
            if part.casefold() == condition.casefold():
                return condition
    for condition in conditions:
        if re.search(rf"(?<!\d){re.escape(condition)}(?!\w)", path.stem, flags=re.IGNORECASE):
            return condition
    return None


def _infer_class_id(path, num_classes):
    for item in reversed(path.parts[:-1]):
        match = re.fullmatch(r"(?:class|label|fault)[_-]?(\d+)", item, flags=re.IGNORECASE)
        if match:
            number = int(match.group(1))
            if 0 <= number < num_classes:
                return number
    match = re.fullmatch(r"(\d+)-(.+)", path.stem, flags=re.IGNORECASE)
    if match:
        number = int(match.group(1))
        if 1 <= number <= num_classes:
            return number - 1
    match = re.fullmatch(r"(?:class|label|fault)?[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        number = int(match.group(1))
        if 0 <= number < num_classes:
            return number
    return None


CWRU_RECORD_BASE_IDS = (97, 105, 118, 130, 169, 185, 197, 209, 222, 234)

def _cwru_expected_variable(condition, class_id):
    match = re.fullmatch(r"([0-3])HP", condition, flags=re.IGNORECASE)
    if match is None or not 0 <= class_id < len(CWRU_RECORD_BASE_IDS):
        raise ValueError(f"Cannot map CWRU condition/class to an original record: {condition}, {class_id}")
    record_id = CWRU_RECORD_BASE_IDS[class_id] + int(match.group(1))
    return f"X{record_id:03d}_DE_time"


def _cwru_fault_name(path, condition):
    match = re.fullmatch(r"(\d+)-(.+)", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    fault = match.group(2)
    suffix = re.search(r"_(\d+)(?:HP)?$", fault, flags=re.IGNORECASE)
    if suffix:
        expected = int(condition[:-2])
        if int(suffix.group(1)) != expected:
            raise ValueError(f"CWRU file condition suffix conflicts with folder {condition}: {path.name}")
        fault = fault[:suffix.start()]
    if not fault:
        raise ValueError(f"CWRU filename has no fault name: {path.name}")
    return fault.casefold()


def build_manifest(raw_dir, dataset, num_classes, manifest_path=None, settings=None):
    raw_dir = Path(raw_dir).expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data path does not exist: {raw_dir}")
    conditions = tuple((settings or DATASET_PREPROCESSING[dataset])["conditions"])
    rows = []
    if manifest_path:
        path = Path(manifest_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"condition", "class_id"}.issubset(reader.fieldnames):
                raise ValueError("Manifest must have condition,class_id,mat_file (or raw_file),variable,channel,axis columns")
            for row in reader:
                name = (row.get("raw_file") or row.get("mat_file") or "").strip()
                if not name:
                    raise ValueError("Manifest has a row without mat_file or raw_file")
                source = Path(name).expanduser()
                if not source.is_absolute():
                    source = raw_dir / source
                rows.append({
                    "condition": row["condition"].strip(),
                    "class_id": int(row["class_id"]),
                    "raw_file": source.resolve(),
                    "variable": (row.get("variable") or "").strip() or None,
                    "channel": int(row["channel"]) if (row.get("channel") or "").strip() else None,
                    "axis": (row.get("axis") or "").strip().upper() or None,
                })
    else:
        files = sorted((file for file in raw_dir.rglob("*") if file.is_file() and file.suffix.lower() in (".mat", ".xls")), key=lambda file: _natural_key(file.relative_to(raw_dir))) if raw_dir.is_dir() else [raw_dir]
        if not files:
            nearby = [file.name for file in raw_dir.iterdir()][:12] if raw_dir.is_dir() else []
            raise ValueError(f"No .mat or .xls raw files found under {raw_dir}. HUST original recordings are tab-delimited .xls files. Nearby entries: {nearby}")
        groups = defaultdict(list)
        unresolved_conditions = []
        for source in files:
            relative = source.relative_to(raw_dir) if raw_dir.is_dir() else source
            condition = _infer_condition(relative, conditions)
            if condition is None:
                unresolved_conditions.append(relative.as_posix())
                continue
            explicit_hust_class = dataset == "HUST" and any(
                re.fullmatch(r"(?:class|label)[_-]?\d+", part, flags=re.IGNORECASE)
                for part in relative.parts[:-1]
            )
            inferred_class = _infer_class_id(relative, num_classes) if dataset != "HUST" or explicit_hust_class else None
            groups[condition].append((source, relative, inferred_class))
        if unresolved_conditions:
            preview = unresolved_conditions[:8]
            raise ValueError(f"Cannot identify the operating condition for {len(unresolved_conditions)} raw files: {preview}. Place files under 65Hz/70Hz/75Hz/80Hz for HUST or 0HP/1HP/2HP/3HP for CWRU, or use --manifest")
        cwru_class_names = {}
        for condition in conditions:
            records = groups.get(condition, [])
            if not records:
                continue
            records.sort(key=lambda item: _natural_key(item[1].name))
            assigned = [item[2] for item in records]
            if all(label is not None for label in assigned):
                class_ids = assigned
            elif all(label is None for label in assigned) and dataset == "HUST" and len(records) == num_classes:
                class_ids = list(range(num_classes))
                print(f"HUST {condition}: assigned class IDs by natural filename order, matching the original XLS-to-CSV notebook. Verify the class_map.csv before publishing results", flush=True)
            else:
                files_example = [item[1].as_posix() for item in records[:8]]
                raise ValueError(f"Cannot safely map classes in {condition}. Found {len(records)} files, expected {num_classes} class-labeled files. Example: {files_example}. Provide --manifest with explicit class_id for each raw file")
            for (source, relative, _), label in zip(records, class_ids):
                if dataset == "CWRU":
                    name = _cwru_fault_name(relative, condition)
                    if name is not None:
                        previous = cwru_class_names.setdefault(int(label), name)
                        if previous != name:
                            raise ValueError(f"CWRU class {label} has conflicting fault names across conditions: {previous!r} versus {name!r} in {relative}")
                rows.append({
                    "condition": condition,
                    "class_id": int(label),
                    "raw_file": source.resolve(),
                    "variable": _cwru_expected_variable(condition, int(label)) if dataset == "CWRU" else None,
                    "channel": None,
                    "axis": None,
                })
    if not rows:
        raise ValueError(f"No vibration records mapped under {raw_dir}; HUST uses .xls files under 65Hz,70Hz,75Hz,80Hz, or supply --manifest")
    for row in rows:
        if row["condition"] not in conditions:
            raise ValueError(f"Unrecognized operating condition {row['condition']}; expected {conditions}")
        if not 0 <= row["class_id"] < num_classes:
            raise ValueError(f"class_id={row['class_id']} must be between 0 and {num_classes - 1}")
        if not row["raw_file"].is_file():
            raise FileNotFoundError(f"Raw file in manifest not found: {row['raw_file']}")
        if row["raw_file"].suffix.lower() not in (".mat", ".xls"):
            raise ValueError(f"Raw input must be a .mat or HUST tab-delimited .xls file: {row['raw_file']}")
    counts = Counter((row["condition"], row["class_id"]) for row in rows)
    collisions = {item: count for item, count in counts.items() if count != 1}
    if collisions:
        raise ValueError(f"Expected one raw file per condition/class; duplicates found: {collisions}. Aggregate recordings explicitly before image conversion")
    for condition in conditions:
        found = {row["class_id"] for row in rows if row["condition"] == condition}
        if found and found != set(range(num_classes)):
            raise ValueError(f"{condition} has class IDs {sorted(found)}, expected all IDs 0..{num_classes-1}. Check raw files or --manifest")
    return rows


def _safe_relative(path, raw_root):
    path = Path(path).resolve()
    root = Path(raw_root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _verify_output_path(raw_dir, output_dir):
    raw = Path(raw_dir).resolve()
    output = Path(output_dir).resolve()
    if output == raw or output in raw.parents or raw in output.parents:
        raise ValueError(f"Output directory cannot contain the raw dataset directory: {output}")
    return output


def _window_segments(signal, start, stop, window_size):
    if start < 0 or stop > len(signal) or (stop - start) % window_size:
        raise ValueError(f"Invalid raw signal split [{start}:{stop}] for window_size={window_size}")
    for offset in range(start, stop, window_size):
        yield offset, signal[offset:offset + window_size]


def _check_prepared_output(output_dir, dataset, image_way, image_size, expected, expected_records, max_images_per_class):
    summary_path = output_dir / "preparation_summary.json"
    manifest_path = output_dir / "split_manifest.csv"
    class_map_path = output_dir / "class_map.csv"
    if not all(path.is_file() for path in (summary_path, manifest_path, class_map_path)):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (summary.get("dataset") != dataset or summary.get("representation") != image_way
                or int(summary.get("image_size", 128)) != int(image_size)
                or int(summary.get("record_count", -1)) != int(expected_records)
                or summary.get("max_images_per_class") != max_images_per_class):
            return False
        if summary.get("counts") != expected:
            return False
        with manifest_path.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != sum(expected.values()):
            return False
        if len({row["image_file"] for row in rows}) != len(rows):
            return False
        if any(not (output_dir / row["image_file"]).is_file() for row in rows):
            return False
        return True
    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        return False


def prepare_dataset(dataset, raw_dir, output_dir, image_way="cwt", image_size=128, manifest_path=None, overwrite=False, settings_override=None, axis="X", scan_only=False, max_images_per_class=None, skip_existing=False):
    if dataset not in DATASET_PREPROCESSING:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if image_way not in REPRESENTATIONS:
        raise ValueError(f"Unsupported image representation: {image_way}")
    settings = dict(DATASET_PREPROCESSING[dataset])
    if settings_override:
        settings.update(settings_override)
    num_classes = 9 if dataset == "HUST" else 10
    if max_images_per_class is not None and int(max_images_per_class) < 1:
        raise ValueError("--max-images-per-class must be a positive integer")
    if overwrite and skip_existing:
        raise ValueError("--overwrite and --skip-existing cannot be combined")
    required_per_class = int(settings["samples_per_class"])
    window_size = int(settings["window_size"])
    source_test_count = int(round(required_per_class * float(settings["source_test_ratio"])))
    target_val_count = int(round(required_per_class * float(settings["target_val_ratio"])))
    if not 0 < source_test_count < required_per_class or not 0 < target_val_count < required_per_class:
        raise ValueError("source_test_ratio and target_val_ratio must yield between 1 and samples_per_class-1 windows")
    total_points = required_per_class * window_size
    source_test_end = source_test_count * window_size
    target_val_start = (required_per_class - target_val_count) * window_size
    raw_dir = Path(raw_dir).expanduser().resolve()
    rows = build_manifest(raw_dir, dataset, num_classes, manifest_path, settings)
    output_dir = _verify_output_path(raw_dir, output_dir)

    print(f"Raw dataset: {raw_dir}", flush=True)
    print(f"Records: {len(rows)} | Required points per class: {total_points}", flush=True)
    print(f"Source test: [0:{source_test_end}] | Source train: [{source_test_end}:{total_points}] | Target adapt/eval: [{target_val_start}:{total_points}]", flush=True)
    signals = []
    for row in rows:
        loaded, variable = load_raw_signal(row["raw_file"], row["variable"], row["channel"], (row.get("axis") or axis), max_points=total_points)
        if len(loaded) < total_points:
            raise ValueError(f"Not enough samples in {row['raw_file']}: found {len(loaded)}, need {total_points}")
        signals.append((row, loaded, variable))
        print(f"Mapped: {row['condition']} | class {row['class_id']} | {_safe_relative(row['raw_file'], raw_dir)} | variable {variable} | {len(loaded)} points", flush=True)
    if scan_only:
        print("Scan successful. No images were generated or overwritten", flush=True)
        return {"dataset": dataset, "record_count": len(rows), "scan_only": True}
    if output_dir.exists() and any(output_dir.iterdir()) and skip_existing:
        expected = {}
        per_split = {
            "train": required_per_class - source_test_count,
            "test": source_test_count,
            "val": target_val_count,
        }
        for condition in settings["conditions"]:
            for split, count in per_split.items():
                expected[f"{condition}/{split}"] = min(count, max_images_per_class) * num_classes if max_images_per_class is not None else count * num_classes
        if _check_prepared_output(output_dir, dataset, image_way, image_size, expected, len(rows), max_images_per_class):
            print(f"[SKIP VERIFIED] {dataset} {image_way}: {output_dir}", flush=True)
            return json.loads((output_dir / "preparation_summary.json").read_text(encoding="utf-8"))
        raise FileExistsError(f"Existing output is incomplete or incompatible: {output_dir}. Use a new --output-dir, or --overwrite only after checking its contents")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite to regenerate it")
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    class_map = []
    counters = defaultdict(int)
    for row, signal, variable in signals:
        class_map.append({
            "condition": row["condition"],
            "class_id": row["class_id"],
            "raw_file": _safe_relative(row["raw_file"], raw_dir),
            "variable": variable,
            "channel": row["channel"] if row["channel"] is not None else "",
            "axis": (row.get("axis") or axis) if row["raw_file"].suffix.lower() == ".xls" else "",
        })
        split_regions = (
            ("train", source_test_end, total_points),
            ("test", 0, source_test_end),
            ("val", target_val_start, total_points),
        )
        for split, start, stop in split_regions:
            for sample_index, (offset, segment) in enumerate(_window_segments(signal, start, stop, window_size)):
                if max_images_per_class is not None and sample_index >= max_images_per_class:
                    break
                key = (row["condition"], split)
                counters[key] += 1
                filename = f"{split}_{counters[key]:05d}_label{row['class_id']}.png"
                target = output_dir / row["condition"] / split / filename
                make_representation(segment, target, image_way, int(settings["sampling_rate"]), image_size)
                audit_rows.append({
                    "condition": row["condition"],
                    "class_id": row["class_id"],
                    "raw_file": _safe_relative(row["raw_file"], raw_dir),
                    "variable": variable,
                    "split": split,
                    "start_sample": offset,
                    "end_sample_exclusive": offset + window_size,
                    "image_file": target.relative_to(output_dir).as_posix(),
                })
        print(f"Generated {row['condition']} class {row['class_id']} for {image_way}", flush=True)

    with (output_dir / "class_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_map[0].keys()))
        writer.writeheader()
        writer.writerows(class_map)
    with (output_dir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)
    expected = {"train": required_per_class - source_test_count, "test": source_test_count, "val": target_val_count}
    if max_images_per_class is not None:
        expected = {split: min(count, max_images_per_class) for split, count in expected.items()}
    for condition in {row["condition"] for row in rows}:
        for split, windows_per_class in expected.items():
            actual = counters[(condition, split)]
            if actual != windows_per_class * num_classes:
                raise RuntimeError(f"Generated {actual} images for {condition}/{split}; expected {windows_per_class * num_classes}")
    summary = {
        "dataset": dataset,
        "representation": image_way,
        "image_size": int(image_size),
        "max_images_per_class": max_images_per_class,
        "generation_mode": "smoke" if max_images_per_class is not None else "full",
        "sampling_rate": int(settings["sampling_rate"]),
        "window_size": window_size,
        "samples_per_class": required_per_class,
        "source_test_count_per_class": source_test_count,
        "source_train_count_per_class": required_per_class - source_test_count,
        "target_val_count_per_class": target_val_count,
        "window_overlap": 0.0,
        "source_test_region": [0, source_test_end],
        "source_train_region": [source_test_end, total_points],
        "target_val_region": [target_val_start, total_points],
        "target_val_overlaps_the_source_train_region_within_same_condition": True,
        "transductive_target_adaptation": True,
        "record_count": len(rows),
        "counts": {"/".join(key): value for key, value in sorted(counters.items())},
    }
    (output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Prepared {dataset} {image_way} ({'smoke' if max_images_per_class is not None else 'full'}): {output_dir}", flush=True)
    print(f"Class mapping: {output_dir / 'class_map.csv'}", flush=True)
    print(f"Sample provenance: {output_dir / 'split_manifest.csv'}", flush=True)
    return summary
