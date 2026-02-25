---
name: cv-best-practices
description: Computer Vision development principles for Agent-First architectures. Defensive image processing, headless execution, memory management, and spatial reasoning.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# Computer Vision Best Practices

> Computer Vision development principles for Agent-First architecture (2025).
> **Process Pixels, Output Tokens. Build defensibly.**

---

## ⚠️ How to Use This Skill

This skill teaches **decision-making principles for vision pipelines**, not just OpenCV syntax.

* ALWAYS read images defensibly (check for `None`).
* NEVER use GUI functions in agent scripts.
* SEPARATE logging from JSON standard output.
* TREAT coordinates as absolute spatial contracts.

---

## 1. Defensive Image Processing

### The Reading Routine

```text
When loading an image:
├── 1. Check if path exists (os.path.exists)
├── 2. Load with cv2.imread
├── 3. Assert image is not None (corrupted file/wrong path)
└── 4. Track original dimensions (height, width) before resizing
```

### Color Space Awareness

| Framework / Model | Expected Color Space | Note |
| :--- | :--- | :--- |
| **OpenCV** (`cv2.imread`) | BGR | Default OpenCV behavior |
| **YOLO** (Ultralytics) | RGB | Requires `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| **PaddleOCR** | BGR / RGB | Check specific model documentation |
| **Pillow** (`Image.open`) | RGB | Standard for PIL |

---

## 2. The "Headless" Principle (Server/Sandbox Safety)

### Library Selection

```text
For Agent/Server environments:
├── USE: opencv-python-headless
└── AVOID: opencv-python (requires libGL.so, breaks in Docker/Sandboxes)
```

### Banned GUI Functions

Never use these in Agent skills or API endpoints:
* ❌ `cv2.imshow()` (Will freeze or crash the container)
* ❌ `cv2.waitKey()` (Creates infinite blocking loops)
* ❌ `cv2.namedWindow()`

---

## 3. Memory & Inference Management

### The Cold Start Problem

Loading ONNX/PyTorch models takes 2-5 seconds.

**❌ Anti-Pattern:**
```python
def detect(img_path):
    # Model loaded on EVERY function call! Memory leak & slow!
    session = onnxruntime.InferenceSession("yolov8.onnx")
    # ...
```

**✅ Pro-Pattern:**
```python
# Use Singletons or global lazy initialization
_SESSION = None

def get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = onnxruntime.InferenceSession("yolov8.onnx")
    return _SESSION
```

### Batching vs Streaming

| Scenario | Strategy |
| :--- | :--- |
| **Agent CLI Tool** | Lazy-load onnx, process single image, exit. |
| **Video Processing** | Pre-allocate tensors, reuse memory buffers. |
| **High-Load API** | FastAPI daemon, model loaded at startup, queue requests. |

---

## 4. Spatial Reasoning & Coordinates

### Coordinate Systems

Always explicitly document the bbox format in your JSON output.

```text
Format 1: xyxy (Standard for Agents/Bounding Boxes)
├── [x_min, y_min, x_max, y_max]
└── Absolute pixel values (e.g., [100, 150, 300, 400])

Format 2: xywh (Common in YOLO training)
├── [center_x, center_y, width, height]
└── Requires conversion before drawing or cropping

Format 3: Polygons (OCR / Segmentation)
└── [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
```

### Crop to Global Mapping

When cropping an ROI (Region of Interest) for a secondary model (e.g., YOLO -> OCR), you MUST map the coordinates back to the original image space if the Agent needs to know where it is.

```python
# Formula for mapping local crop coordinates back to global
global_x = crop_x_min + local_x
global_y = crop_y_min + local_y
```

---

## 5. Agent-Friendly Output (Stdout Purity)

### The JSON Contract

Agents parse stdout. If stdout is polluted, the Agent hallucinates.

**❌ Bad Output:**
```text
"Loading ONNX runtime..."
"Library loaded successfully"
{"detections": [...]}  <- Agent fails to parse JSON
```

**✅ Good Output:**
```python
# Suppress all C++ and library warnings.
# Only print the final JSON.
print(json.dumps({"status": "success", "detections": [...]}, ensure_ascii=False))
```

### Handling Large Data (Masks/Images)

LLMs cannot read pixel arrays.

```text
├── Bounding Boxes → Return JSON coordinates
├── Segmentation Masks → Save to /tmp/mask_123.png, return the file path
├── Cropped Objects → Save to /tmp/crop_123.jpg, return the file path
└── Base64 Strings → AVOID (consumes too many tokens, causes lag)
```

---

## 6. Error Handling Principles

### Vision-Specific Exceptions

Return structured errors instead of crashing:

```json
// 1. Empty Image / File Not Found:
{"status": "error", "error_type": "ImageLoadError", "message": "..."}

// 2. No Detections:
{"status": "success", "detections": [], "message": "No objects found."}

// 3. Low Confidence:
// Filter out detections below the threshold BEFORE returning to the Agent.
```

---

## 7. Decision Checklist

Before committing vision code, verify:

* [ ] **Headless Safe?** (No `cv2.imshow`, using `opencv-python-headless`)
* [ ] **Defensive Loading?** (Checking if `cv2.imread` returned `None`)
* [ ] **Coordinate Format Defined?** (Is it `xyxy` or `xywh`? Is it documented?)
* [ ] **Memory Safe?** (Model loaded only once per session/execution?)
* [ ] **Stdout Pure?** (Is the output strictly parseable JSON without logging noise?)
* [ ] **Token Efficient?** (Returning paths and coordinates instead of arrays?)

---

## 8. Anti-Patterns to Avoid

### ❌ DON'T:
* Assume the image exists or is readable.
* Load the ONNX model inside a loop.
* Print debugging information using `print()` (use `logging` to stderr instead).
* Send full high-res images to OCR without finding the ROI first.
* Forget Non-Maximum Suppression (NMS), resulting in overlapping ghost boxes.

### ✅ DO:
* Filter out low-confidence detections (e.g., `< 0.5`) before outputting.
* Save intermediate crops/masks to temporary files and return their paths.
* Ensure RGB/BGR alignment between OpenCV and the specific AI model.
* Use explicit type hints for spatial data (e.g., `Tuple[int, int, int, int]`).

---

> **Remember:** The goal is not just to run inference, but to translate physical pixel space into structural logic that an LLM can flawlessly reason about.