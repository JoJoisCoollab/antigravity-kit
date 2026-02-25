---
name: agent-tool-design
description: Principles for designing robust, stateless, and LLM-friendly tools (Skills). Focuses on strict JSON contracts, stdout purity, error self-correction, and progressive disclosure.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# Agent Tool Design & API Contracts

> Tool development principles for Agent-First architecture (2025).
> **Treat the Agent as a strictly-typed API client. Never assume human intuition.**

---

## ⚠️ How to Use This Skill

This skill teaches **how to bridge the gap between Python scripts and LLM context windows**. 

* ALWAYS communicate through pure, structured JSON.
* NEVER print unstructured logs or stack traces to `stdout`.
* DESIGN tools to be entirely stateless and idempotent.
* GUIDE the agent with actionable error messages when things fail.

---

## 1. The `SKILL.md` Anatomy

Every tool must be accompanied by a `SKILL.md` file. This acts as the "OpenAPI Spec" for the Agent. It must contain:

### A. Intent (The "Why")
Briefly explain *when* the Agent should use this tool. High signal, low noise.

### B. Constraints & Anti-Patterns (The Boundaries)
Explicitly tell the Agent what NOT to do to prevent infinite loops or context window bloat.
* *Example:* "Do not pass full 4K images to OCR. Crop the ROI first."

### C. Input/Output Specifications
Define the exact parameters, types, and expected JSON output structure.

---

## 2. Stdout Purity (The JSON Contract)

In an Agentic workflow, `stdout` is the API response. If a C++ library or Python script prints random warnings, the Agent's JSON parser will crash.

**❌ Anti-Pattern (Polluted Stdout):**
```text
Warning: CUDA not found, falling back to CPU.
Processing image...
{"bbox": [10, 20, 100, 200]}
```

**✅ Pro-Pattern (Pure Stdout & Stderr for Logs):**
```python
import sys
import json
import logging

# Route all human-readable logs to STDERR
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.info("Processing image...")

# ONLY print the final JSON to STDOUT
result = {"status": "success", "data": {"bbox": [10, 20, 100, 200]}}
print(json.dumps(result, ensure_ascii=False))
```

---

## 3. Designing for Context Window Limits

Agents have limited memory (Tokens). Never return raw binary data or massive arrays. 

### The "Progressive Disclosure" Pattern
If a tool processes a large dataset (or image):
1. **First Pass:** Return high-level metadata (e.g., "Found 15 objects, saved crops to `/tmp/crops/`").
2. **Second Pass:** Let the Agent call a different tool to inspect a *specific* crop if needed.

**Rule of Thumb:** If the output JSON exceeds 1000 lines, you need to paginate or summarize.

---

## 4. Self-Correcting Error Handling

When a tool fails, do not just throw a Python `Exception` and print a raw traceback. Wrap errors in a JSON response that tells the Agent **how to fix it**.

**❌ Anti-Pattern (Useless to Agent):**
```text
Traceback (most recent call last):
FileNotFoundError: [Errno 2] No such file or directory: 'img.png'
```

**✅ Pro-Pattern (Actionable JSON Error):**
```json
{
  "status": "error",
  "error_type": "FileNotFound",
  "message": "The file 'img.png' does not exist.",
  "suggested_action": "Please verify the absolute path using the 'ls' command before retrying."
}
```

---

## 5. CLI Mapping & Argparse

When building the Python backend for a tool, use `argparse` strictly. 

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Agent Tool: Extract ROI")
    parser.add_argument("--image", required=True, help="Absolute path to source image")
    parser.add_argument("--bbox", required=True, type=json.loads, help="JSON string of [x1, y1, x2, y2]")
    
    args = parser.parse_args()
    # ... execution logic
```

---

## 6. Decision Checklist

Before publishing a new Agent Tool (Skill), verify:

* [ ] **Is `SKILL.md` written?** (Does it clearly define constraints and anti-patterns?)
* [ ] **Is Stdout strictly JSON?** (Are all library warnings suppressed or routed to stderr?)
* [ ] **Are errors actionable?** (Does a failure return a JSON with a `suggested_action`?)
* [ ] **Is it Token-Efficient?** (Does it return file paths instead of massive raw data arrays?)
* [ ] **Is it Stateless?** (Will running it twice with the same inputs yield the same/safe result?)

---

## 7. Anti-Patterns to Avoid

### ❌ DON'T:
* Assume the Agent knows the internal directory structure (always demand absolute paths).
* Return Base64 strings of images (it instantly exhausts the context window).
* Leave `print()` statements used for debugging inside the final script.
* Create tools that require interactive prompts (e.g., `input("Press Enter to continue")`).

### ✅ DO:
* Map output coordinates and data back to a logical structure the Agent can reason about.
* Return a clear `"status": "success"` or `"status": "error"` key in every response.
* Write robust type hints (`TypeHinting`) in the underlying Python code to self-document intent.

---

> **Remember:** A well-designed tool guides the Agent's Chain-of-Thought (CoT). Good outputs answer the current question; great outputs suggest the next logical step.