"""
Training script for SageMaker jobs.
Works both locally and in SageMaker training environment.
"""
import torch
import logging
import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(args):
    """Main training function."""
    
    # Model configuration
    model_id = args.model_name
    logger.info(f"Loading model: {model_id}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dataset paths (works in SageMaker and locally)
    if os.path.exists("/opt/ml/input/data/train/train.jsonl"):
        # SageMaker training job
        train_path = "/opt/ml/input/data/train/train.jsonl"
        val_path = "/opt/ml/input/data/validation/val.jsonl"
        logger.info("Using SageMaker input paths")
    else:
        # Local development
        train_path = "data/jsonl/train.jsonl"
        val_path = "data/jsonl/val.jsonl"
        logger.info("Using local paths")
    
    # Load dataset
    logger.info(f"Loading dataset from {train_path}")
    dataset = load_dataset("json", data_files={"train": train_path, "validation": val_path})
    
    # Tokenization function
    def tokenize(batch):
        start_prompt = "Summarize the following conversation:\n\n"
        end_prompt = "\n\nSummary: "
        inputs = [start_prompt + x + end_prompt for x in batch["input"]]
        tokenized_inputs = tokenizer(
            inputs, 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )
        tokenized_labels = tokenizer(
            batch["output"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        batch["input_ids"] = tokenized_inputs["input_ids"]
        batch["attention_mask"] = tokenized_inputs["attention_mask"]
        batch["labels"] = tokenized_labels["input_ids"]
        return batch
    
    # Apply tokenization
    logger.info("Tokenizing dataset")
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["instruction", "input", "output"])
    
    # Load model
    logger.info(f"Loading base model: {model_id}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    # LoRA configuration
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / Total params: {total_params:,}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="/opt/ml/model",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        fp16=False,  # CPU-friendly
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to=[]
    )
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info("Saving model to /opt/ml/model")
    trainer.model.save_pretrained("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/flan-t5-base", type=str)
    parser.add_argument("--num-epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--gradient-accumulation-steps", default=4, type=int)
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    
    args = parser.parse_args()
    main(args)
