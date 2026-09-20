#!/usr/bin/env python3
"""Fine-tune Laya on balanced simulator-generated warehouse decisions."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, amp_dtype, build_model, build_sequence
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya_warehouse.controllers import LayaController
from laya_warehouse.dataset import dataset_summary, generate_cases
from laya_warehouse.model import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--output", type=Path, default=Path("results/laya-warehouse-model"))
    parser.add_argument("--per-action", type=int, default=128)
    parser.add_argument("--eval-per-action", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
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
        label = Action(case["label"])
        for action, question_id in LayaController.ACTION_QUESTIONS.items():
            question = LayaController.QUESTIONS[question_id]
            ids, markers = build_sequence(
                tokenizer,
                case["state"],
                {"t": "noul", "ins": question["instructions"], "crit": {}},
                cfg["max_len"],
                cfg["head_max_len"],
            )
            if len(markers) != 2:
                raise RuntimeError(f"question {question_id} did not produce two decision markers")
            positive = action is label
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "qtype": QTYPES["noul"],
                    "target": [0.02, 0.98] if positive else [0.98, 0.02],
                    "case_id": case["id"],
                    "action": action.value,
                    "label": label.value,
                }
            )
    return items


def collate(items: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    size = len(items)
    length = max(len(item["ids"]) for item in items)
    marker_count = max(len(item["markers"]) for item in items)
    input_ids = torch.full((size, length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((size, length), dtype=torch.long)
    marker_pos = torch.zeros((size, marker_count), dtype=torch.long)
    marker_mask = torch.zeros((size, marker_count), dtype=torch.bool)
    target = torch.zeros((size, marker_count), dtype=torch.float32)
    for index, item in enumerate(items):
        item_length = len(item["ids"])
        item_markers = len(item["markers"])
        input_ids[index, :item_length] = torch.tensor(item["ids"])
        attention_mask[index, :item_length] = 1
        marker_pos[index, :item_markers] = torch.tensor(item["markers"])
        marker_mask[index, :item_markers] = True
        target[index, :item_markers] = torch.tensor(item["target"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.full((size,), QTYPES["noul"], dtype=torch.long),
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
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    labels: dict[str, str] = {}
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
        probabilities = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist()
        for meta, probability in zip(batch["meta"], probabilities):
            scores[meta["case_id"]][meta["action"]] = probability
            labels[meta["case_id"]] = meta["label"]

    correct = 0
    per_label: dict[str, list[int]] = defaultdict(list)
    for case_id, action_scores in scores.items():
        prediction = max(action_scores, key=action_scores.get)
        is_correct = int(prediction == labels[case_id])
        correct += is_correct
        per_label[labels[case_id]].append(is_correct)
    return {
        "accuracy": round(correct / len(scores), 4),
        "cases": len(scores),
        "per_label_accuracy": {
            label: round(sum(values) / len(values), 4)
            for label, values in sorted(per_label.items())
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
            loss = -(batch["target"] * log_probabilities).sum(dim=-1).mean()
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

    train_cases = generate_cases(per_action=args.per_action, seed=args.seed)
    eval_cases = generate_cases(per_action=args.eval_per_action, seed=args.seed + 1)
    print("train dataset:", dataset_summary(train_cases), flush=True)
    print("evaluation dataset:", dataset_summary(eval_cases), flush=True)

    started = time.time()
    _, cfg, tokenizer, model = load_checkpoint(args.model, device)
    dtype = amp_dtype(cfg.get("amp_dtype", "bf16"))
    if torch.cuda.get_device_capability(device)[0] < 8:
        dtype = torch.float16
    train_items = make_items(train_cases, tokenizer, cfg)
    eval_items = make_items(eval_cases, tokenizer, cfg)

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

    metrics = {
        "base_model": args.model,
        "seed": args.seed,
        "train_dataset": dataset_summary(train_cases),
        "evaluation_dataset": dataset_summary(eval_cases),
        "baseline": baseline,
        "epochs": epoch_metrics,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    save_checkpoint(args.output, cfg=cfg, tokenizer=tokenizer, model=model, metrics=metrics)
    print(f"saved fine-tuned policy to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
