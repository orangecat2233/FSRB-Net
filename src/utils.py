from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classification_metrics(y_true, y_pred, y_prob, num_classes):
    labels = np.arange(int(num_classes))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100.0),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100.0),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100.0),
    }
    if np.unique(y_true).size < 2:
        result["auc"] = float("nan")
    else:
        try:
            result["auc"] = float(roc_auc_score(y_true, y_prob, labels=labels, average="macro", multi_class="ovr") * 100.0)
        except ValueError:
            result["auc"] = float("nan")
    return result


def write_csv(records, path, append=True):
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in records for key in row))
    if not append and path.exists():
        path.unlink()
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def config_signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def upsert_csv(records, path, key_fields, prefer_more_tasks=False):
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            previous = list(csv.DictReader(stream))
    ordered = {}
    for row in [*previous, *records]:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if prefer_more_tasks and key in ordered:
            prior = ordered[key]
            if int(float(prior.get("task_count", 0))) > int(float(row.get("task_count", 0))):
                continue
        ordered[key] = row
    fields = list(dict.fromkeys(name for row in ordered.values() for name in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered.values())
    temporary.replace(path)
