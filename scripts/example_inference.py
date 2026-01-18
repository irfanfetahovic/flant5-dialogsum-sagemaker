"""
Example inference script demonstrating how to use the fine-tuned model.
Run this after training to test model predictions.
"""

import argparse
import logging
from pathlib import Path
from src.inference import load_base_model, load_peft_model, summarize_dialogue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Run inference examples."""

    logger.info("Loading model...")
    model, tokenizer = load_base_model(args.model_id)

    # Load LoRA weights if provided
    if args.peft_weights:
        logger.info(f"Loading LoRA weights from {args.peft_weights}")
        model = load_peft_model(model, args.model_id, args.peft_weights)

    # Example dialogues
    examples = [
        {
            "name": "Project Discussion",
            "dialogue": """Tom: Hi Lisa, how are you doing today?
Lisa: Pretty good! I've been working on the new project all morning.
Tom: That's great! How's it going so far?
Lisa: Really well. We've made a lot of progress. The team has been fantastic.
Tom: That's wonderful! When do you think you'll be done?
Lisa: We should have the first version ready by next Friday.""",
        },
        {
            "name": "Weekend Plans",
            "dialogue": """Alice: Hey! What are you doing this weekend?
Bob: I'm thinking about going hiking. The weather should be nice.
Alice: That sounds fun! Where are you thinking of going?
Bob: Probably to the mountains about an hour away. Want to join?
Alice: I'd love to! Count me in.""",
        },
        {
            "name": "Restaurant Recommendation",
            "dialogue": """Customer: Do you know any good restaurants nearby?
Server: Yes! There's an Italian place down the street that's very popular.
Customer: How's the pasta?
Server: Excellent! People especially love their carbonara and risotto.
Customer: Great, I'll check it out. Thanks for the recommendation!""",
        },
    ]

    # Run inference on examples
    print("\n" + "=" * 70)
    print("DIALOG SUMMARIZATION EXAMPLES")
    print("=" * 70)

    for i, example in enumerate(examples, 1):
        print(f"\n[Example {i}: {example['name']}]")
        print("-" * 70)
        print(f"Dialogue:\n{example['dialogue']}")
        print("\n" + "-" * 70)

        summary = summarize_dialogue(
            model,
            tokenizer,
            example["dialogue"],
            max_new_tokens=args.max_tokens,
            num_beams=args.num_beams,
        )

        print(f"Summary: {summary}")
        print("=" * 70)

    logger.info("Inference examples completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference examples")
    parser.add_argument(
        "--model-id",
        default="google/flan-t5-base",
        help="Base model ID from HuggingFace",
    )
    parser.add_argument(
        "--peft-weights", help="Path to LoRA weights (S3 or local path)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--num-beams", type=int, default=1, help="Number of beams for beam search"
    )

    args = parser.parse_args()
    main(args)
