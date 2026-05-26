from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from mlops.data import build_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = evaluate.load("accuracy").compute(predictions=predictions, references=labels)
    f1 = evaluate.load("f1").compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1_weighted": f1["f1"]}


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text())

    df = build_dataframe(config["data_dir"])
    labels = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    train_df, eval_df = train_test_split(
        df, train_size=config["train_size"], random_state=config["seed"], stratify=df["label_id"]
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config["max_length"])

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    eval_ds = Dataset.from_pandas(eval_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=len(labels), label2id=label2id, id2label=id2label
    )

    output_dir = Path(config["output_dir"])
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    training = config["training"]
    training_args = TrainingArguments(
        output_dir=str(model_dir),
        num_train_epochs=training["epochs"],
        learning_rate=float(training["learning_rate"]),
        per_device_train_batch_size=training["train_batch_size"],
        per_device_eval_batch_size=training["eval_batch_size"],
        weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        report_to="none",
        seed=config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    metadata = {
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "metrics": metrics,
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    df.to_csv(output_dir / "training_catalog.csv", index=False)


if __name__ == "__main__":
    main()
