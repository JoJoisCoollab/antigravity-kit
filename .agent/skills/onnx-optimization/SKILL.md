---
name: onnx-optimization
description: Advanced deployment patterns for ONNX Runtime. Focuses on hardware acceleration, graph optimization, quantization, and suppressing logs for Agent stdout purity.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# ONNX Optimization & Deployment Patterns

> Performance tuning and deployment principles for ONNX Runtime in Agent-First architecture (2025).
> **Maximize Speed. Minimize Memory. Keep Stdout Pure.**

---

## ⚠️ How to Use This Skill

This skill teaches **optimization and safe execution patterns for ONNX**, preventing common memory and context-window traps.

* ALWAYS implement Provider Fallbacks (e.g., TensorRT -> CUDA -> CPU).
* NEVER initialize `InferenceSession` inside an inference loop.
* MUTE all ONNX Runtime C++ logs to protect the Agent's JSON parser.
* PRE-ALLOCATE arrays and use static shapes when possible.

---

## 1. Execution Providers (Hardware Acceleration)

### The Fallback Pattern

Never hardcode a single Execution Provider. Environments change (e.g., from a local GPU to a Docker container without CUDA). Always use a cascading fallback strategy.

**✅ Pro-Pattern:**
```python
import onnxruntime as ort

def get_optimized_providers():
    available_providers = ort.get_available_providers()
    providers = []
    
    if 'TensorrtExecutionProvider' in available_providers:
        providers.append(('TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': '/tmp/trt_cache'
        }))
    if 'CUDAExecutionProvider' in available_providers:
        providers.append(('CUDAExecutionProvider', {
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'arena_extend_strategy': 'kNextPowerOfTwo'
        }))
        
    providers.append('CPUExecutionProvider') # Always include CPU as the ultimate fallback
    return providers
```

---

## 2. Session Options & Graph Optimization

### Maximizing Throughput

By default, ONNX Runtime doesn't apply all possible graph optimizations. You must explicitly configure `SessionOptions`.

```python
def get_optimized_session_options():
    sess_options = ort.SessionOptions()
    
    # Enable all graph optimizations (Operator fusion, constant folding, etc.)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Threading optimization
    sess_options.intra_op_num_threads = 4 # Tune based on target CPU cores
    sess_options.inter_op_num_threads = 1
    
    return sess_options
```

---

## 3. The "Silent Agent" Principle (Log Suppression)

### Protecting JSON Stdout

ONNX Runtime emits C++ warnings to stdout/stderr, which **will break** the Agent's ability to parse JSON outputs. You must forcefully silence it.

**✅ Pro-Pattern:**
```python
def create_silent_session(model_path: str) -> ort.InferenceSession:
    sess_options = get_optimized_session_options()
    
    # CRITICAL: Set log severity to FATAL (4) to suppress warnings/info
    sess_options.log_severity_level = 4 
    
    # Optional: Suppress Python warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    return ort.InferenceSession(
        model_path, 
        sess_options=sess_options, 
        providers=get_optimized_providers()
    )
```

---

## 4. Input / Output Tensor Handling

### Avoid Dynamic Allocation Overhead

When feeding image data into the ONNX session, ensure the NumPy array is strictly contiguous and matches the expected type (usually `float32`).

**❌ Anti-Pattern:**
```python
# Slicing creates non-contiguous memory, slowing down ONNX execution
input_tensor = image[:, :, ::-1] 
outputs = session.run(None, {"input": input_tensor})
```

**✅ Pro-Pattern:**
```python
# Ensure contiguous memory block in C-order
input_tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
# Normalize to [0, 1] if required by the model
input_tensor /= 255.0 
# Add batch dimension
input_tensor = np.expand_dims(input_tensor, axis=0)

# Fetch output names dynamically rather than hardcoding
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})
```

---

## 5. Model Quantization Strategies

When RAM constraints are specified, use quantization.

| Strategy | Size Reduction | Speedup | Accuracy Loss | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **FP32 (Base)** | 1x | 1x | None | High-precision Cloud API |
| **FP16** | 0.5x | ~2x (on GPU) | Minimal | Default for CUDA/TensorRT |
| **INT8 (Dynamic)** | 0.25x | ~2-3x (on CPU)| Moderate | Edge Devices / CPU-only Agents |

*Note: INT8 quantization should be performed as a pre-processing step using `onnxruntime.quantization`, not during runtime.*

---

## 6. Decision Checklist

Before committing ONNX pipeline code, verify:

* [ ] **Cold Start Prevented?** (Is the `InferenceSession` cached as a Singleton?)
* [ ] **Stdout Protected?** (`log_severity_level = 4` configured?)
* [ ] **Provider Fallbacks Configured?** (CPU fallback exists in case GPU fails?)
* [ ] **Graph Optimization Enabled?** (`ORT_ENABLE_ALL` applied?)
* [ ] **Contiguous Memory?** (`np.ascontiguousarray` used before inference?)

---

## 7. Anti-Patterns to Avoid

### ❌ DON'T:
* Initialize `onnxruntime.InferenceSession()` inside a `for` loop or API endpoint handler.
* Hardcode `'CUDAExecutionProvider'` without `'CPUExecutionProvider'` as a fallback.
* Leave `log_severity_level` at default (0 or 2) when writing CLI skills for Agents.
* Pass `float64` NumPy arrays to the session (always cast to `float32` or `float16`).

### ✅ DO:
* Read model input names dynamically (`session.get_inputs()[0].name`).
* Cache TensorRT engines to disk to avoid 10-minute recompilations on startup.
* Document the required input shapes (e.g., `[1, 3, 640, 640]`) in the Python Docstrings.

---

> **Remember:** A well-optimized ONNX deployment should feel completely invisible to the Agent—instantaneous execution, pure JSON responses, and zero runtime crashes.