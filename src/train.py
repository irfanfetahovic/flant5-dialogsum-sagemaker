"""
Training script for SageMaker jobs and local runs.
MLflow tracking is required and always enabled.
"""

import argparse
import logging
import os
import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(args: argparse.Namespace):
    """Main training entry point."""

    mlflow.start_run()
    mlflow.log_params(
        {
            "model_name": args.model_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    )
    logger.info("MLflow tracking enabled")

    model_id = args.model_name
    logger.info(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Select dataset paths depending on environment (SageMaker vs local)
    if os.path.exists("/opt/ml/input/data/train/train.jsonl"):
        train_path = "/opt/ml/input/data/train/train.jsonl"
        val_path = "/opt/ml/input/data/validation/val.jsonl"
        logger.info("Using SageMaker input paths")
    else:
        train_path = "data/jsonl/train.jsonl"
        val_path = "data/jsonl/val.jsonl"
        logger.info("Using local paths")

    logger.info(f"Loading dataset from {train_path}")
    dataset = load_dataset(
        "json", data_files={"train": train_path, "validation": val_path}
    )

    def tokenize(batch):
        start_prompt = "Summarize the following conversation:\n\n"
        end_prompt = "\n\nSummary: "
        inputs = [start_prompt + x + end_prompt for x in batch["input"]]
        tokenized_inputs = tokenizer(
            inputs, padding="max_length", truncation=True, max_length=512
        )
        tokenized_labels = tokenizer(
            batch["output"], padding="max_length", truncation=True, max_length=128
        )
        batch["input_ids"] = tokenized_inputs["input_ids"]
        batch["attention_mask"] = tokenized_inputs["attention_mask"]
        batch["labels"] = tokenized_labels["input_ids"]
        return batch

    logger.info("Tokenizing dataset")
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["instruction", "input", "output"])

    logger.info(f"Loading base model: {model_id}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)

    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} / Total params: {total_params:,}"
    )
    mlflow.log_metrics(
        {
            "trainable_params": float(trainable_params),
            "total_params": float(total_params),
            "trainable_percent": float(trainable_params / total_params * 100),
        }
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        fp16=False,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    logger.info("Starting training")
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    train_metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime", 0),
        "train_samples_per_second": train_result.metrics.get(
            "train_samples_per_second", 0
        ),
    }
    mlflow.log_metrics(train_metrics)

    logger.info("Evaluating model")
    eval_metrics = trainer.evaluate()
    mlflow.log_metrics(eval_metrics)

    if os.path.isdir(training_args.output_dir):
        mlflow.log_artifacts(training_args.output_dir)
    mlflow.end_run()

    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train FLAN-T5 dialog summarization with LoRA"
    )
    parser.add_argument("--model-name", default="google/flan-t5-base", type=str)
    parser.add_argument("--num-epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=4, type=int)
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--output-dir", default="/opt/ml/model", type=str)

    main(parser.parse_args())
