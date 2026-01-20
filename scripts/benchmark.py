"""
Performance benchmarking script for inference.

Measures latency, throughput, and memory usage for model inference.
"""

import argparse
import time
import torch
import psutil
import os
from typing import List, Dict
from src.inference import load_model_and_tokenizer, summarize_dialogue


def measure_memory():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def benchmark_inference(
    model_id: str,
    peft_weights: str = None,
    num_samples: int = 50,
    test_dialogues: List[str] = None,
) -> Dict[str, float]:
    """
    Run performance benchmarks on model inference.

    Args:
        model_id: Base model ID
        peft_weights: Path to LoRA weights (optional)
        num_samples: Number of inference runs
        test_dialogues: List of test dialogues (if None, uses default)

    Returns:
        Dictionary with benchmark metrics
    """
    print("=" * 60)
    print("INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {model_id}")
    if peft_weights:
        print(f"With LoRA weights: {peft_weights}")

    mem_before = measure_memory()
    model, tokenizer = load_model_and_tokenizer(model_id, peft_weights)
    mem_after = measure_memory()

    model_memory = mem_after - mem_before
    print(f"Model loaded. Memory usage: {model_memory:.1f} MB")

    # Prepare test data
    if test_dialogues is None:
        test_dialogues = [
            "Tom: Hi Lisa, how are you doing today? Lisa: Pretty good! I've been working on the new project all morning. Tom: That's great! How's it going so far? Lisa: Really well. We've made a lot of progress.",
            "Alice: Did you see the game last night? Bob: Yes! It was amazing. The final score was 3-2. Alice: I know, I couldn't believe it!",
            "John: I'm thinking about getting a new car. Sarah: Oh yeah? What kind? John: I like the Toyota, but I've also heard good things about Honda.",
            "Mike: Hey, want to grab lunch? Emma: Sure! Where should we go? Mike: How about that new Italian place? Emma: Sounds perfect!",
            "David: I finished the report you asked for. Manager: Great! Can you send it to me? David: Just sent it. Let me know if you need anything else.",
        ]

    # Extend to num_samples by cycling
    test_set = (test_dialogues * (num_samples // len(test_dialogues) + 1))[:num_samples]

    print(f"\nRunning {num_samples} inference samples...")
    print("-" * 60)

    # Warmup (first inference is usually slower)
    _ = summarize_dialogue(test_set[0], model, tokenizer)

    # Benchmark
    latencies = []
    total_tokens = 0

    start_total = time.time()

    for i, dialogue in enumerate(test_set):
        start = time.time()
        summary = summarize_dialogue(dialogue, model, tokenizer)
        end = time.time()

        latency = (end - start) * 1000  # Convert to ms
        latencies.append(latency)
        total_tokens += len(summary.split())

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} samples...")

    total_time = time.time() - start_total

    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50_latency = sorted(latencies)[len(latencies) // 2]
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    throughput = num_samples / total_time
    avg_tokens = total_tokens / num_samples
    tokens_per_second = total_tokens / total_time

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nLatency (milliseconds):")
    print(f"  Average:  {avg_latency:.1f} ms")
    print(f"  Min:      {min_latency:.1f} ms")
    print(f"  Max:      {max_latency:.1f} ms")
    print(f"  P50:      {p50_latency:.1f} ms")
    print(f"  P95:      {p95_latency:.1f} ms")

    print(f"\nThroughput:")
    print(f"  {throughput:.2f} summaries/second")
    print(f"  {tokens_per_second:.1f} tokens/second")
    print(f"  {avg_tokens:.1f} tokens/summary (avg)")

    print(f"\nMemory:")
    print(f"  Model size: {model_memory:.1f} MB")
    print(f"  Current usage: {measure_memory():.1f} MB")

    print(f"\nTotal time: {total_time:.2f} seconds")
    print("=" * 60)

    return {
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "throughput_per_sec": throughput,
        "tokens_per_second": tokens_per_second,
        "model_memory_mb": model_memory,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark model inference performance"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/flan-t5-base",
        help="Base model ID from HuggingFace",
    )
    parser.add_argument(
        "--peft-weights",
        type=str,
        default=None,
        help="Path to PEFT/LoRA weights (local path or s3://)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of inference samples to run",
    )

    args = parser.parse_args()

    results = benchmark_inference(
        model_id=args.model_id,
        peft_weights=args.peft_weights,
        num_samples=args.num_samples,
    )

    # Optionally save results
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
