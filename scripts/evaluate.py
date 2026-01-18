"""
Evaluate dialog summarization model using ROUGE metrics.
"""

import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Evaluate model on test set."""

    # Load dataset
    logger.info("Loading DialogSum test set")
    dataset = load_dataset("knkarthick/dialogsum")
    test_data = dataset["test"].select(
        range(min(args.num_samples, len(dataset["test"])))
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load base model
    logger.info(f"Loading base model: {args.model_id}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype="auto")

    # Load LoRA weights if provided
    if args.peft_weights:
        logger.info(f"Loading LoRA weights: {args.peft_weights}")
        model = PeftModel.from_pretrained(
            model, args.peft_weights, torch_dtype="auto", is_trainable=False
        )

    # Generate predictions
    logger.info(f"Generating predictions on {len(test_data)} samples")
    predictions = []
    references = []

    for i, example in enumerate(test_data):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(test_data)}")

        prompt = f"Summarize the following conversation:\n\n{example['dialogue']}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(**inputs, max_new_tokens=200)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(prediction)
        references.append(example["summary"])

    # Compute ROUGE
    logger.info("Computing ROUGE metrics")
    rouge = evaluate.load("rouge")
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=True,
        use_stemmer=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("ROUGE Evaluation Results")
    print("=" * 60)
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print("=" * 60 + "\n")

    # Save results
    if args.output_file:
        import json

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument(
        "--model-id", default="google/flan-t5-base", help="Base model ID"
    )
    parser.add_argument("--peft-weights", help="Path to LoRA weights")
    parser.add_argument(
        "--num-samples", default=100, type=int, help="Number of test samples"
    )
    parser.add_argument("--output-file", help="Output file for results")

    args = parser.parse_args()
    main(args)
