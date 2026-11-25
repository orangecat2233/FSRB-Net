from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .adaptation import CORALLoss, DWDANLoss, DomainDiscriminator, MKMMDLoss, MMDLoss
from .ablations import ABLATION_PLAN
from .data import build_transfer_loaders
from .models import SOTA_MODELS, TransferModel
from .utils import classification_metrics, config_signature, save_json, set_seed, upsert_csv, write_csv

DA_METHODS = ("source_only", "mmd", "coral", "mk_mmd", "dann", "dwdan")
DATASET_CONFIG = {
    "HUST": {"num_classes": 9, "conditions": ("65Hz", "70Hz", "75Hz", "80Hz")},
    "CWRU": {"num_classes": 10, "conditions": ("0HP", "1HP", "2HP", "3HP")},
}

def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def _forward(model, images):
    features, logits = model(images)
    if features.ndim > 2:
        features = torch.flatten(features, 1)
    return features, logits


def train_one_epoch(model, loaders, optimizer, device, da_method, num_classes, lambda_da=1.0, domain_discriminator=None, da_criterion=None, epoch_index=0, total_epochs=30):
    model.train()
    if domain_discriminator is not None:
        domain_discriminator.train()
    source_loader = loaders["source_train"]
    target_loader = loaders["target_adapt"]
    target_iterator = iter(target_loader)
    classification_loss = nn.CrossEntropyLoss()
    total_loss_value = 0.0
    cls_loss_value = 0.0
    da_loss_value = 0.0
    correct = 0
    seen = 0
    steps = 0

    for source_images, source_labels in source_loader:
        source_images = source_images.to(device, non_blocking=True)
        source_labels = source_labels.to(device, non_blocking=True)
        source_features, source_logits = _forward(model, source_images)
        cls_loss = classification_loss(source_logits, source_labels)
        da_loss = source_features.new_zeros(())

        if da_method != "source_only":
            target_batch, target_iterator = _next_batch(target_iterator, target_loader)
            target_images = target_batch.to(device, non_blocking=True) if torch.is_tensor(target_batch) else target_batch[0].to(device, non_blocking=True)
            target_features, target_logits = _forward(model, target_images)

            if da_method == "mmd":
                da_loss = da_criterion(source_features, target_features)
                total_loss = cls_loss + float(lambda_da) * da_loss
            elif da_method == "coral":
                da_loss = da_criterion(source_features, target_features)
                total_loss = cls_loss + float(lambda_da) * da_loss
            elif da_method == "mk_mmd":
                da_loss = da_criterion(source_features, target_features)
                total_loss = cls_loss + float(lambda_da) * da_loss
            elif da_method == "dann":
                source_domain = domain_discriminator(source_features, grl_coeff=1.0)
                target_domain = domain_discriminator(target_features, grl_coeff=1.0)
                source_domain_labels = torch.zeros(source_domain.size(0), dtype=torch.long, device=device)
                target_domain_labels = torch.ones(target_domain.size(0), dtype=torch.long, device=device)
                domain_loss = 0.5 * (
                    nn.functional.cross_entropy(source_domain, source_domain_labels)
                    + nn.functional.cross_entropy(target_domain, target_domain_labels)
                )
                da_loss = domain_loss
                total_loss = cls_loss + domain_loss
            elif da_method == "dwdan":
                result = da_criterion(
                    source_features=source_features,
                    source_labels=source_labels,
                    target_features=target_features,
                    target_logits=target_logits,
                )
                progress = 1.0 if total_epochs <= 1 else epoch_index / max(total_epochs - 1, 1)
                ramp = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
                weight = min(max(float(lambda_da), 0.0), 0.20) * ramp * ramp
                da_loss = result["joint_loss"]
                total_loss = cls_loss + weight * da_loss
            else:
                raise ValueError(f"Unsupported DA method: {da_method}")
        else:
            total_loss = cls_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        batch_size = source_labels.size(0)
        total_loss_value += float(total_loss.detach())
        cls_loss_value += float(cls_loss.detach())
        da_loss_value += float(da_loss.detach())
        correct += int((source_logits.argmax(1) == source_labels).sum().item())
        seen += batch_size
        steps += 1

    if steps == 0:
        raise RuntimeError("No source training batches were processed")
    return {
        "loss": total_loss_value / steps,
        "classification_loss": cls_loss_value / steps,
        "adaptation_loss": da_loss_value / steps,
        "source_train_accuracy": 100.0 * correct / max(seen, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    targets, predictions, probabilities = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _, logits = _forward(model, images)
        probs = torch.softmax(logits, dim=1)
        targets.extend(labels.cpu().numpy().tolist())
        predictions.extend(logits.argmax(1).cpu().numpy().tolist())
        probabilities.extend(probs.cpu().numpy().tolist())
    return classification_metrics(np.asarray(targets), np.asarray(predictions), np.asarray(probabilities), num_classes)


def _build_optimizer(model, da_method, domain_discriminator, lr, weight_decay, epochs):
    parameters = list(model.parameters())
    if da_method == "dann" and domain_discriminator is not None:
        parameters += list(domain_discriminator.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=float(lr) * 0.01)
    return optimizer, scheduler


def _metric_text(metrics):
    return (
        f"ACC={metrics['accuracy']:.2f}% | Precision={metrics['precision']:.2f}% | "
        f"Recall={metrics['recall']:.2f}% | F1={metrics['f1']:.2f}% | AUC={metrics['auc']:.2f}%"
    )


def _print_task_result(result, skipped=False):
    source_metrics = {
        name: result[f"source_test_{key}"]
        for name, key in (("accuracy", "acc"), ("precision", "precision"),
                          ("recall", "recall"), ("f1", "f1"), ("auc", "auc"))
    }
    target_metrics = {
        name: result[f"target_{key}"]
        for name, key in (("accuracy", "acc"), ("precision", "precision"),
                          ("recall", "recall"), ("f1", "f1"), ("auc", "auc"))
    }
    print(f"[TASK {'REUSED' if skipped else 'COMPLETE'}] {result['dataset']} | "
          f"{result['model']} | DA={result['da_strategy']} | "
          f"Seed={result['seed']} | {result['task']}", flush=True)
    print(f"  Source test: {_metric_text(source_metrics)}", flush=True)
    print(f"  Target eval: {_metric_text(target_metrics)}", flush=True)
    print(f"  Checkpoint: {result['checkpoint']}", flush=True)
    print("-" * 104, flush=True)


def run_task(dataset, data_root, output_root, model_name, da_method, seed, source_condition, target_condition, config, overwrite=False, cache_images=True):
    num_classes = DATASET_CONFIG[dataset]["num_classes"]
    task_name = f"{source_condition}_to_{target_condition}"
    task_dir = Path(output_root) / dataset / config["image_way"] / model_name / da_method / f"seed_{seed}" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = task_dir / "last.pth"
    completed_path = task_dir / "completed_result.json"
    signature_payload = {
        "dataset": dataset, "data_root": str(Path(data_root).resolve()), "model": model_name,
        "da": da_method, "seed": int(seed), "source": source_condition, "target": target_condition,
        **config,
    }
    signature = config_signature(signature_payload)
    if completed_path.is_file() and not overwrite:
        previous = json.loads(completed_path.read_text(encoding="utf-8"))
        if previous.get("signature") == signature:
            result = previous["result"]
            _print_task_result(result, skipped=True)
            return result
        raise FileExistsError(f"Completed output exists with a different configuration: {task_dir}. Use --overwrite to replace it.")
    if overwrite and task_dir.exists():
        for path in task_dir.iterdir():
            if path.is_file():
                path.unlink()

    set_seed(seed)
    loaders, sizes = build_transfer_loaders(
        data_root=data_root,
        source_condition=source_condition,
        target_condition=target_condition,
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config["num_workers"],
        normalize=config["normalize"],
        seed=seed,
        cache_images=cache_images,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 104, flush=True)
    print(f"[TASK START] Dataset={dataset} | Model={model_name} | DA={da_method} | "
          f"Seed={seed} | Transfer={task_name} | Device={device}", flush=True)
    print(f"  Samples: Source train={sizes['source_train']} | Source test={sizes['source_test']} | "
          f"Target adapt={sizes['target_adapt']} | Target eval={sizes['target_eval']}", flush=True)
    print(f"  Data loading: RAM image cache={'enabled' if cache_images else 'disabled'} | "
          f"Worker processes={config['num_workers']} | "
          f"Persistent workers={'enabled' if config['num_workers'] > 0 else 'disabled'} | "
          f"Pinned memory={'enabled' if torch.cuda.is_available() else 'disabled'}", flush=True)
    print(f"  Training: Epochs={config['epochs']} | Batch size={config['batch_size']} | "
          f"LR={config['learning_rate']:.8f} | Weight decay={config['weight_decay']:.8f}", flush=True)
    model = TransferModel(model_name, num_classes, pretrained=config["pretrained"]).to(device)
    domain_discriminator = DomainDiscriminator(hidden_dim=256, dropout=0.5).to(device) if da_method == "dann" else None
    if da_method == "mmd":
        da_criterion = MMDLoss().to(device)
    elif da_method == "coral":
        da_criterion = CORALLoss().to(device)
    elif da_method == "mk_mmd":
        da_criterion = MKMMDLoss().to(device)
    elif da_method == "dwdan":
        da_criterion = DWDANLoss(num_classes=num_classes, kernel_mul=2.0, kernel_num=5, confidence_threshold=0.60, min_class_samples=2, min_dynamic_weight=0.10).to(device)
    else:
        da_criterion = None
    optimizer, scheduler = _build_optimizer(model, da_method, domain_discriminator, config["learning_rate"], config["weight_decay"], config["epochs"])
    start_epoch = 0
    history = []
    if checkpoint_path.is_file() and not overwrite:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("signature") != signature:
            raise FileExistsError(f"An incomplete checkpoint has a different configuration: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state"])
        if domain_discriminator is not None and checkpoint.get("domain_state") is not None:
            domain_discriminator.load_state_dict(checkpoint["domain_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint.get("history", [])
        print(f"[RESUME] {task_name} from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, int(config["epochs"])):
        metrics = train_one_epoch(
            model=model, loaders=loaders, optimizer=optimizer, device=device, da_method=da_method,
            num_classes=num_classes, lambda_da=config["lambda_da"], domain_discriminator=domain_discriminator,
            da_criterion=da_criterion, epoch_index=epoch, total_epochs=config["epochs"],
        )
        learning_rate_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {"epoch": epoch + 1, "learning_rate": learning_rate_used, **metrics}
        history.append(row)
        write_csv(history, task_dir / "training_history.csv", append=False)
        checkpoint_payload = {
            "signature": signature, "epoch": epoch + 1, "model_state": model.state_dict(),
            "domain_state": domain_discriminator.state_dict() if domain_discriminator is not None else None,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "history": history, "config": signature_payload,
        }
        temporary = checkpoint_path.with_suffix(".pth.tmp")
        torch.save(checkpoint_payload, temporary)
        temporary.replace(checkpoint_path)
        print(f"[EPOCH] {dataset} | {model_name} | {task_name} | "
              f"{epoch + 1:03d}/{config['epochs']:03d} | LR={learning_rate_used:.8f} | "
              f"Total Loss={metrics['loss']:.5f} | Cls Loss={metrics['classification_loss']:.5f} | "
              f"DA Loss={metrics['adaptation_loss']:.5f} | "
              f"Source Train ACC={metrics['source_train_accuracy']:.2f}%", flush=True)

    source_metrics = evaluate(model, loaders["source_test"], device, num_classes)
    target_metrics = evaluate(model, loaders["target_eval"], device, num_classes)
    result = {
        "dataset": dataset, "model": model_name, "da_strategy": da_method, "seed": int(seed),
        "task": task_name, "source": source_condition, "target": target_condition,
        "source_test_acc": source_metrics["accuracy"], "source_test_precision": source_metrics["precision"],
        "source_test_recall": source_metrics["recall"], "source_test_f1": source_metrics["f1"], "source_test_auc": source_metrics["auc"],
        "target_acc": target_metrics["accuracy"], "target_precision": target_metrics["precision"],
        "target_recall": target_metrics["recall"], "target_f1": target_metrics["f1"], "target_auc": target_metrics["auc"],
        "source_train_samples": sizes["source_train"], "source_test_samples": sizes["source_test"],
        "target_adapt_eval_samples": sizes["target_adapt"], "checkpoint": str(checkpoint_path),
    }
    save_json({"signature": signature, "config": signature_payload, "result": result}, completed_path)
    _print_task_result(result)
    return result


def _mean_results(results, prefix):
    names = ("acc", "precision", "recall", "f1", "auc")
    return {name: _safe_mean([item[f"{prefix}{name}"] for item in results]) for name in names}


def _print_seed_summary(dataset, model_name, strategy, seed, results, average):
    print("=" * 104, flush=True)
    print(f"[TRANSFER TASK SUMMARY] Dataset={dataset} | Model={model_name} | "
          f"DA={strategy} | Seed={seed} | Completed tasks={len(results)}", flush=True)
    print(f"{'Task':<5} {'Transfer':<14} {'Source ACC':>11} {'Target ACC':>11} "
          f"{'Precision':>11} {'Recall':>10} {'F1':>10} {'AUC':>10}", flush=True)
    print("-" * 104, flush=True)
    for index, item in enumerate(results, start=1):
        print(f"T{index:<4} {item['task']:<14} "
              f"{item['source_test_acc']:>10.2f}% {item['target_acc']:>10.2f}% "
              f"{item['target_precision']:>10.2f}% {item['target_recall']:>9.2f}% "
              f"{item['target_f1']:>9.2f}% {item['target_auc']:>9.2f}%", flush=True)
    print("-" * 104, flush=True)
    print(f"{'MEAN':<20} {average['average_source_test_acc']:>10.2f}% "
          f"{average['average_target_acc']:>10.2f}% "
          f"{average['average_target_precision']:>10.2f}% "
          f"{average['average_target_recall']:>9.2f}% "
          f"{average['average_target_f1']:>9.2f}% "
          f"{average['average_target_auc']:>9.2f}%", flush=True)
    print(f"[AVERAGE SOURCE TEST] ACC={average['average_source_test_acc']:.4f}% | "
          f"Precision={average['average_source_test_precision']:.4f}% | "
          f"Recall={average['average_source_test_recall']:.4f}% | "
          f"F1={average['average_source_test_f1']:.4f}% | "
          f"AUC={average['average_source_test_auc']:.4f}%", flush=True)
    print(f"[AVERAGE TARGET EVAL] ACC={average['average_target_acc']:.4f}% | "
          f"Precision={average['average_target_precision']:.4f}% | "
          f"Recall={average['average_target_recall']:.4f}% | "
          f"F1={average['average_target_f1']:.4f}% | "
          f"AUC={average['average_target_auc']:.4f}%", flush=True)
    print("=" * 104, flush=True)


def _print_multi_seed_summary(dataset, model_name, strategy, seed_averages):
    if len(seed_averages) < 2:
        return
    print(f"[MULTI-SEED SUMMARY] Dataset={dataset} | Model={model_name} | "
          f"DA={strategy} | Seed count={len(seed_averages)}", flush=True)
    for metric in ("acc", "precision", "recall", "f1", "auc"):
        values = np.asarray([item[f"average_target_{metric}"] for item in seed_averages], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            std = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
            print(f"  Target {metric.upper()}: {float(finite.mean()):.4f}% +/- {std:.4f}% "
                  f"(across {finite.size} seeds)", flush=True)
    print("-" * 104, flush=True)


def run_experiment(dataset, data_root, output_root, experiment, config, models=None, da_method=None, seeds=None, overwrite=False, max_tasks=None, cache_images=True):
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset}")
    conditions = DATASET_CONFIG[dataset]["conditions"]
    tasks = [(source, target) for source, target in itertools.permutations(conditions, 2)]
    if max_tasks is not None:
        tasks = tasks[:max(1, int(max_tasks))]
    if experiment == "ablations":
        if dataset != "HUST":
            raise ValueError("The ablation protocol is defined for HUST only")
        model_da_pairs = list(ABLATION_PLAN)
    elif experiment == "sota":
        if dataset != "CWRU":
            raise ValueError("The SOTA protocol is defined for CWRU only")
        selected = list(models) if models else list(SOTA_MODELS)
        model_da_pairs = [(name, da_method or "dann") for name in selected]
    elif experiment == "custom":
        if not models:
            raise ValueError("Custom experiments require at least one model")
        model_da_pairs = [(name, da_method or "dann") for name in models]
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    all_results = []
    all_averages = []
    for model_name, strategy in model_da_pairs:
        current_model_averages = []
        if strategy not in DA_METHODS:
            raise ValueError(f"Unsupported DA method: {strategy}")
        for seed in (seeds or [config["seed"]]):
            seed_results = []
            for source_condition, target_condition in tasks:
                result = run_task(
                    dataset, data_root, output_root, model_name, strategy, seed,
                    source_condition, target_condition, config, overwrite=overwrite,
                    cache_images=cache_images,
                )
                seed_results.append(result)
                all_results.append(result)
            source_averages = _mean_results(seed_results, "source_test_")
            target_averages = _mean_results(seed_results, "target_")
            average = {
                "dataset": dataset, "model": model_name, "da_strategy": strategy,
                "seed": int(seed), "task_count": len(seed_results),
                **{f"average_source_test_{key}": value for key, value in source_averages.items()},
                **{f"average_target_{key}": value for key, value in target_averages.items()},
            }
            all_averages.append(average)
            current_model_averages.append(average)
            _print_seed_summary(dataset, model_name, strategy, seed, seed_results, average)
        _print_multi_seed_summary(dataset, model_name, strategy, current_model_averages)
    summary_dir = Path(output_root) / dataset / config["image_way"] / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if all_results:
        upsert_csv(all_results, summary_dir / f"{experiment}_tasks.csv",
                   key_fields=("dataset", "model", "da_strategy", "seed", "task"))
    if all_averages:
        upsert_csv(all_averages, summary_dir / f"{experiment}_averages.csv",
                   key_fields=("dataset", "model", "da_strategy", "seed"),
                   prefer_more_tasks=True)
    return all_results
