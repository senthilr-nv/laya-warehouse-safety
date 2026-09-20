#!/usr/bin/env python3
"""Fine-tune Laya on frozen simulator-generated warehouse decisions."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, amp_dtype, build_model, build_sequence
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya_warehouse.controllers import LayaController
from laya_warehouse.dataset import add_horizontal_mirrors, dataset_summary, generate_cases
from laya_warehouse.model import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--output", type=Path, default=Path("results/laya-warehouse-model"))
    parser.add_argument("--max-ticks", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_checkpoint(model_id: str, device: torch.device):
    model_dir = snapshot_download(model_id)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as source:
        cfg = json.load(source)

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    model.to(device)
    return model_dir, cfg, tokenizer, model


def make_items(cases: list[dict[str, Any]], tokenizer, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for case in cases:
        action_question = LayaController.BENCHMARK_QUESTIONS["action"]
        actions = list(action_question["criteria"])
        action_label = Action(case["labels"]["action"]).value
        action_ids, action_markers = build_sequence(
            tokenizer,
            case["state"],
            {
                "t": "choice",
                "ins": action_question["instructions"],
                "crit": action_question["criteria"],
            },
            cfg["max_len"],
            cfg["head_max_len"],
        )
        if len(action_markers) != len(actions):
            raise RuntimeError("action question did not produce one marker per choice")
        action_target = [0.02] * len(actions)
        action_target[actions.index(action_label)] = 0.94
        items.append(
            {
                "ids": action_ids,
                "markers": action_markers,
                "qtype": QTYPES["choice"],
                "target": action_target,
                "case_id": case["id"],
                "question_id": "action",
                "options": actions,
                "label": action_label,
            }
        )

        blocked_question = LayaController.BENCHMARK_QUESTIONS["path_blocked"]
        blocked_ids, blocked_markers = build_sequence(
            tokenizer,
            case["state"],
            {"t": "noul", "ins": blocked_question["instructions"], "crit": {}},
            cfg["max_len"],
            cfg["head_max_len"],
        )
        if len(blocked_markers) != 2:
            raise RuntimeError("path_blocked question did not produce two markers")
        blocked = bool(case["labels"]["path_blocked"])
        items.append(
            {
                "ids": blocked_ids,
                "markers": blocked_markers,
                "qtype": QTYPES["noul"],
                "target": [0.02, 0.98] if blocked else [0.98, 0.02],
                "case_id": case["id"],
                "question_id": "path_blocked",
                "label": blocked,
            }
        )
    return items


def add_inverse_frequency_weights(items: list[dict[str, Any]]) -> None:
    """Give every label equal total weight within each typed question."""

    keys = [(item["question_id"], str(item["label"])) for item in items]
    counts = Counter(keys)
    question_totals = Counter(question_id for question_id, _label in keys)
    question_label_counts = Counter(question_id for question_id, _label in counts)
    for item, key in zip(items, keys):
        question_id, _label = key
        item["sample_weight"] = question_totals[question_id] / (
            question_label_counts[question_id] * counts[key]
        )


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    size = len(items)
    length = max(len(item["ids"]) for item in items)
    marker_count = max(len(item["markers"]) for item in items)
    input_ids = torch.full((size, length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((size, length), dtype=torch.long)
    marker_pos = torch.zeros((size, marker_count), dtype=torch.long)
    marker_mask = torch.zeros((size, marker_count), dtype=torch.bool)
    target = torch.zeros((size, marker_count), dtype=torch.float32)
    sample_weight = torch.ones(size, dtype=torch.float32)
    for index, item in enumerate(items):
        item_length = len(item["ids"])
        item_markers = len(item["markers"])
        input_ids[index, :item_length] = torch.tensor(item["ids"])
        attention_mask[index, :item_length] = 1
        marker_pos[index, :item_markers] = torch.tensor(item["markers"])
        marker_mask[index, :item_markers] = True
        target[index, :item_markers] = torch.tensor(item["target"])
        sample_weight[index] = item.get("sample_weight", 1.0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "sample_weight": sample_weight,
        "qtype": torch.tensor([item["qtype"] for item in items], dtype=torch.long),
        "meta": items,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    model,
    items: list[dict[str, Any]],
    *,
    batch_size: int,
    pad_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    action_correct = 0
    action_cases = 0
    per_label: dict[str, list[int]] = {action.value: [] for action in Action}
    blocked_correct = 0
    blocked_cases = 0
    blocked_brier = 0.0
    for offset in range(0, len(items), batch_size):
        batch = move_batch(collate(items[offset : offset + batch_size], pad_id), device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits, _ = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        for meta, item_probabilities in zip(batch["meta"], probabilities):
            if meta["question_id"] == "action":
                prediction_index = max(
                    range(len(meta["options"])),
                    key=lambda index: item_probabilities[index],
                )
                prediction = meta["options"][prediction_index]
                is_correct = int(prediction == meta["label"])
                action_correct += is_correct
                action_cases += 1
                per_label[meta["label"]].append(is_correct)
            else:
                probability_true = item_probabilities[1]
                target = int(bool(meta["label"]))
                blocked_correct += int((probability_true >= 0.5) == bool(target))
                blocked_brier += (probability_true - target) ** 2
                blocked_cases += 1
    per_label_metrics = {
        label: {
            "accuracy": round(sum(values) / len(values), 4),
            "correct": sum(values),
            "cases": len(values),
        }
        for label, values in sorted(per_label.items())
        if values
    }
    label_accuracies = [metrics["accuracy"] for metrics in per_label_metrics.values()]
    return {
        "action": {
            "accuracy": round(action_correct / action_cases, 4),
            "cases": action_cases,
            "per_label_accuracy": {
                label: metrics["accuracy"]
                for label, metrics in per_label_metrics.items()
            },
            "per_label": per_label_metrics,
            "macro_accuracy": round(sum(label_accuracies) / len(label_accuracies), 4),
            "minimum_label_accuracy": min(label_accuracies),
        },
        "path_blocked": {
            "accuracy": round(blocked_correct / blocked_cases, 4),
            "brier": round(blocked_brier / blocked_cases, 4),
            "cases": blocked_cases,
        },
    }


def train_epoch(
    model,
    items: list[dict[str, Any]],
    *,
    optimizer,
    scaler,
    batch_size: int,
    pad_id: int,
    device: torch.device,
    dtype: torch.dtype,
    rng: random.Random,
) -> float:
    model.train()
    rng.shuffle(items)
    total_loss = 0.0
    batches = 0
    for offset in range(0, len(items), batch_size):
        batch = move_batch(collate(items[offset : offset + batch_size], pad_id), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits, act = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["marker_pos"],
                batch["marker_mask"],
                batch["qtype"],
            )
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
            item_loss = -(batch["target"] * log_probabilities).sum(dim=-1)
            loss = (item_loss * batch["sample_weight"]).mean()
            loss = loss + 0.0 * act.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(1, batches)


def save_checkpoint(
    output: Path,
    *,
    cfg: dict[str, Any],
    tokenizer,
    model,
    metrics: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing model directory: {output}")
    output.mkdir(parents=True)
    weights = {
        name: parameter.detach().half().contiguous().cpu()
        for name, parameter in model.state_dict().items()
    }
    save_file(weights, output / "model.safetensors")
    model.encoder.config.save_pretrained(output / "encoder")
    tokenizer.save_pretrained(output / "tokenizer")
    saved_cfg = dict(cfg)
    saved_cfg["fine_tuned"] = True
    saved_cfg["model_name"] = "laya-warehouse-policy"
    saved_cfg["trained_questions"] = ["action", "path_blocked"]
    (output / "rl_agent_config.json").write_text(
        json.dumps(saved_cfg, indent=2) + "\n", encoding="utf-8"
    )
    (output / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for policy fine-tuning")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing model directory: {args.output}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    source_train_cases = generate_cases(split="train", max_ticks=args.max_ticks)
    train_cases = add_horizontal_mirrors(source_train_cases)
    eval_cases = generate_cases(split="validation", max_ticks=args.max_ticks)
    print("source train dataset:", dataset_summary(source_train_cases), flush=True)
    print("train dataset:", dataset_summary(train_cases), flush=True)
    print("evaluation dataset:", dataset_summary(eval_cases), flush=True)

    started = time.time()
    _, cfg, tokenizer, model = load_checkpoint(args.model, device)
    dtype = amp_dtype(cfg.get("amp_dtype", "bf16"))
    if torch.cuda.get_device_capability(device)[0] < 8:
        dtype = torch.float16
    train_items = make_items(train_cases, tokenizer, cfg)
    eval_items = make_items(eval_cases, tokenizer, cfg)
    add_inverse_frequency_weights(train_items)

    baseline = evaluate(
        model,
        eval_items,
        batch_size=args.batch_size,
        pad_id=tokenizer.pad_token_id,
        device=device,
        dtype=dtype,
    )
    print("baseline:", json.dumps(baseline, sort_keys=True), flush=True)

    encoder_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if name.startswith("encoder."):
            encoder_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    rng = random.Random(args.seed)
    epoch_metrics = []
    best_selection: tuple[float, ...] | None = None
    best_selection_metrics = None
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model,
            train_items,
            optimizer=optimizer,
            scaler=scaler,
            batch_size=args.batch_size,
            pad_id=tokenizer.pad_token_id,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        evaluation = evaluate(
            model,
            eval_items,
            batch_size=args.batch_size,
            pad_id=tokenizer.pad_token_id,
            device=device,
            dtype=dtype,
        )
        epoch_metric = {"epoch": epoch, "loss": round(loss, 5), "evaluation": evaluation}
        epoch_metrics.append(epoch_metric)
        print("epoch:", json.dumps(epoch_metric, sort_keys=True), flush=True)
        selection = (
            evaluation["action"]["minimum_label_accuracy"],
            evaluation["action"]["macro_accuracy"],
            evaluation["path_blocked"]["accuracy"],
            -evaluation["path_blocked"]["brier"],
        )
        if best_selection is None or selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            best_selection_metrics = {
                "minimum_action_label_accuracy": selection[0],
                "action_macro_accuracy": selection[1],
                "path_blocked_accuracy": selection[2],
                "path_blocked_brier": -selection[3],
            }
            best_state = {
                name: value.detach().half().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state, strict=True)

    metrics = {
        "base_model": args.model,
        "seed": args.seed,
        "source_train_dataset": dataset_summary(source_train_cases),
        "train_dataset": dataset_summary(train_cases),
        "evaluation_dataset": dataset_summary(eval_cases),
        "baseline": baseline,
        "epochs": epoch_metrics,
        "best_epoch": best_epoch,
        "best_validation_selection": best_selection_metrics,
        "checkpoint_selection_order": [
            "minimum_action_label_accuracy",
            "action_macro_accuracy",
            "path_blocked_accuracy",
            "lowest_path_blocked_brier",
        ],
        "validation_support_warning": (
            "Lateral action labels have three cases each; report them as development evidence, "
            "not stable recall estimates."
        ),
        "training_augmentation": "horizontal mirror",
        "training_loss_weighting": "inverse frequency by question and label",
        "elapsed_seconds": round(time.time() - started, 2),
    }
    save_checkpoint(args.output, cfg=cfg, tokenizer=tokenizer, model=model, metrics=metrics)
    print(f"saved fine-tuned policy to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
