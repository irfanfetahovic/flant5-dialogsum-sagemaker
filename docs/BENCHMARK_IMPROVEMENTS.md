# Benchmark Module Improvement Suggestions

> **Internal Document** - Not for version control
> 
> This document outlines professional improvements for the `scripts/benchmark.py` module to make it production-ready.

---

## Current Status

The benchmark module provides basic performance metrics but lacks several features needed for production ML operations and professional client deliverables.

## Critical Improvements

### 1. **Use Real Test Data**

**Current:** 5 hardcoded dialogues repeated in a cycle
**Issue:** Not representative of actual production data distribution
**Solution:**
```python
# Load from actual SAMSum test set
from datasets import load_dataset

dataset = load_dataset("samsung/samsum")
test_data = dataset["test"].select(range(num_samples))
test_dialogues = [item["dialogue"] for item in test_data]
```

**Benefits:**
- Realistic performance estimates
- Varied dialogue lengths and complexity
- Credible metrics for stakeholders

---

### 2. **Persist Results to JSON**

**Current:** Only prints to console
**Issue:** No historical tracking or comparison capability
**Solution:**
```python
import json
from datetime import datetime

results = {
    "timestamp": datetime.now().isoformat(),
    "model_id": model_id,
    "peft_weights": peft_weights,
    "device": str(device),
    "num_samples": num_samples,
    "metrics": {
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,  # Add P99
        "throughput_per_sec": throughput,
        "tokens_per_second": tokens_per_second,
        "model_memory_mb": model_memory,
    }
}

output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
```

**Benefits:**
- Track performance over time
- Compare model versions
- Generate reports/charts
- CI/CD integration

---

### 3. **Multiple Runs with Confidence Intervals**

**Current:** Single run only
**Issue:** No statistical reliability
**Solution:**
```python
def benchmark_with_confidence(num_runs=3):
    all_run_metrics = []
    
    for run in range(num_runs):
        print(f"\n=== Run {run + 1}/{num_runs} ===")
        metrics = run_single_benchmark()
        all_run_metrics.append(metrics)
    
    # Calculate mean and std dev
    import numpy as np
    avg_metrics = {
        key: {
            "mean": np.mean([m[key] for m in all_run_metrics]),
            "std": np.std([m[key] for m in all_run_metrics]),
            "min": np.min([m[key] for m in all_run_metrics]),
            "max": np.max([m[key] for m in all_run_metrics]),
        }
        for key in all_run_metrics[0].keys()
    }
    
    return avg_metrics
```

**Benefits:**
- Statistical confidence
- Identify variance/stability issues
- Professional reporting

---

### 4. **Batch Inference Benchmarking**

**Current:** Only single-sample inference
**Issue:** Doesn't reflect production batch processing
**Solution:**
```python
def benchmark_batch_inference(model, tokenizer, dialogues, batch_sizes=[1, 4, 8, 16]):
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nBenchmarking batch size: {batch_size}")
        
        # Create batches
        batches = [dialogues[i:i+batch_size] for i in range(0, len(dialogues), batch_size)]
        
        latencies = []
        for batch in batches:
            start = time.time()
            # Batch inference
            prompts = [f"Summarize: {d}" for d in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=200)
            
            latencies.append((time.time() - start) * 1000)
        
        results[f"batch_{batch_size}"] = {
            "avg_latency_ms": np.mean(latencies),
            "throughput_per_sec": len(dialogues) / (sum(latencies) / 1000),
        }
    
    return results
```

**Benefits:**
- Optimize batch size for cost/latency tradeoff
- Production-realistic metrics
- GPU utilization insights

---

### 5. **GPU Metrics (if CUDA available)**

**Current:** Only CPU memory tracking
**Issue:** Missing GPU utilization and memory
**Solution:**
```python
if torch.cuda.is_available():
    import pynvml
    
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Before inference
    gpu_mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
    
    # During inference
    gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    
    # After inference
    gpu_mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
    
    gpu_metrics = {
        "gpu_memory_mb": gpu_mem_after - gpu_mem_before,
        "gpu_utilization_pct": gpu_utilization,
        "gpu_name": pynvml.nvmlDeviceGetName(handle),
    }
```

**Benefits:**
- Identify GPU bottlenecks
- Justify hardware requirements
- Optimize GPU usage

---

### 6. **Baseline Comparison**

**Current:** No comparison against base model
**Issue:** Can't demonstrate fine-tuning ROI
**Solution:**
```python
def compare_models():
    # Benchmark base model
    print("\n=== Benchmarking Base Model ===")
    base_results = benchmark_inference(model_id=args.model_id, peft_weights=None)
    
    # Benchmark fine-tuned model
    print("\n=== Benchmarking Fine-tuned Model ===")
    finetuned_results = benchmark_inference(model_id=args.model_id, peft_weights=args.peft_weights)
    
    # Compare
    print("\n=== Comparison ===")
    print(f"Latency change: {((finetuned_results['avg_latency_ms'] / base_results['avg_latency_ms']) - 1) * 100:.1f}%")
    print(f"Memory overhead: {finetuned_results['model_memory_mb'] - base_results['model_memory_mb']:.1f} MB")
```

**Benefits:**
- Quantify fine-tuning cost
- Prove value proposition
- Justify deployment decisions

---

### 7. **Add --output Argument**

**Current:** Hardcoded output behavior
**Solution:**
```python
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output file for benchmark results (JSON)",
)

parser.add_argument(
    "--runs",
    type=int,
    default=1,
    help="Number of benchmark runs for statistical confidence",
)

parser.add_argument(
    "--batch-sizes",
    type=int,
    nargs="+",
    default=None,
    help="Batch sizes to benchmark (e.g., --batch-sizes 1 4 8 16)",
)
```

---

### 8. **Context-aware Inference**

**Current:** Uses `torch.no_grad()` implicitly
**Issue:** Not explicitly shown, could be forgotten
**Solution:**
```python
with torch.no_grad():
    for dialogue in test_set:
        summary = summarize_dialogue(dialogue, model, tokenizer)
```

**Benefits:**
- Explicit memory optimization
- Faster inference
- Best practice demonstration

---

## Implementation Priority

### Phase 1 (Quick Wins - 1-2 hours):
1. ✅ Use real test data from SAMSum
2. ✅ Persist results to JSON
3. ✅ Add --output argument
4. ✅ Add torch.no_grad() context

### Phase 2 (Medium Effort - 2-4 hours):
5. ✅ Multiple runs with confidence intervals
6. ✅ Baseline comparison mode
7. ✅ Add P99 latency metric

### Phase 3 (Advanced - 4-8 hours):
8. ✅ Batch inference benchmarking
9. ✅ GPU metrics (pynvml)
10. ✅ Visualization (matplotlib/plotly charts)

---

## Sample Professional Output

```json
{
  "timestamp": "2026-01-23T14:30:00",
  "model_id": "google/flan-t5-base",
  "peft_weights": "s3://bucket/model.tar.gz",
  "device": "cuda:0",
  "gpu_name": "Tesla T4",
  "num_samples": 150,
  "num_runs": 3,
  "metrics": {
    "latency_ms": {
      "mean": 245.3,
      "std": 12.4,
      "p50": 240.1,
      "p95": 280.5,
      "p99": 310.2
    },
    "throughput": {
      "mean": 4.08,
      "std": 0.21,
      "unit": "samples/sec"
    },
    "memory": {
      "model_size_mb": 892.5,
      "gpu_memory_mb": 1024.3,
      "gpu_utilization_pct": 78.5
    }
  },
  "batch_performance": {
    "batch_1": {"latency_ms": 245.3, "throughput": 4.08},
    "batch_4": {"latency_ms": 620.1, "throughput": 6.45},
    "batch_8": {"latency_ms": 1180.4, "throughput": 6.77}
  }
}
```

---

## Additional Considerations

### Cost Analysis
- Add estimated SageMaker endpoint costs based on throughput
- Calculate cost per 1000 requests
- Compare instance types (ml.g4dn.xlarge vs ml.m5.xlarge)

### Automated Reporting
- Generate HTML/PDF report with charts
- Email results to stakeholders
- Slack/Teams integration for CI/CD

### Stress Testing
- Sustained load testing (run for 5+ minutes)
- Measure thermal throttling
- Identify memory leaks

---

## References

- [MLPerf Inference Benchmark](https://mlcommons.org/en/inference-edge/)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [HuggingFace Benchmarking Best Practices](https://huggingface.co/docs/transformers/benchmarks)

---

**Status:** Draft - Review & Prioritize  
**Owner:** ML Engineering Team  
**Last Updated:** January 23, 2026
