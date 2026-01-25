"""
Inference functions for dialog summarization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)


def load_base_model(model_id: str = "google/flan-t5-base"):
    """
    Load base model and tokenizer.

    Args:
        model_id: HuggingFace model ID

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {model_id}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def load_peft_model(base_model, model_id: str, peft_weights_path: str):
    """
    Load base model with LoRA weights.

    Args:
        base_model: Base model instance
        model_id: HuggingFace model ID
        peft_weights_path: Path to LoRA weights

    Returns:
        PEFT model
    """
    logger.info(f"Loading PEFT model from {peft_weights_path}")
    peft_model = PeftModel.from_pretrained(
        base_model, peft_weights_path, torch_dtype=torch.float32, is_trainable=False
    )
    return peft_model


def summarize_dialogue(
    model, tokenizer, dialogue: str, max_new_tokens: int = 200, num_beams: int = 1
) -> str:
    """
    Summarize a dialogue.

    Args:
        model: Model for inference
        tokenizer: Tokenizer
        dialogue: Input dialogue
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search

    Returns:
        Summary text
    """
    prompt = f"Summarize the following conversation:\n\n{dialogue}\n\nSummary:"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"], max_new_tokens=max_new_tokens, num_beams=num_beams
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def batch_summarize(
    model, tokenizer, dialogues: list, max_new_tokens: int = 200, num_beams: int = 1
) -> list:
    """
    Summarize multiple dialogues.

    Args:
        model: Model for inference
        tokenizer: Tokenizer
        dialogues: List of dialogues
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search

    Returns:
        List of summaries
    """
    summaries = []
    for dialogue in dialogues:
        summary = summarize_dialogue(
            model, tokenizer, dialogue, max_new_tokens, num_beams
        )
        summaries.append(summary)
    return summaries


def model_fn(model_dir):
    """SageMaker model_fn for endpoint inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model, tokenizer


def predict_fn(data, model_tokenizer):
    """SageMaker predict_fn for endpoint inference."""
    model, tokenizer = model_tokenizer
    inputs = tokenizer(data["inputs"], return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
