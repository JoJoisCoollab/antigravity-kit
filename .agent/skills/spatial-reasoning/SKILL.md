---
name: spatial-reasoning
description: Mathematical principles and spatial logic for Computer Vision in Agent-First architectures. Covers Bounding Box formats, IoU, Polygon conversions, and global/local coordinate mapping.
allowed-tools: Read, Write, Edit, Glob, Grep
---

# Spatial Reasoning & Coordinate Math

> Translating the physical pixel space into structural logic for AI Agents (2025).
> **Coordinates are relationships. Group items that are physically close.**

---

## ⚠️ How to Use This Skill

This skill provides the **mathematical foundation for Agent vision tools**. Since LLMs cannot "see" images, they rely entirely on the bounding boxes and spatial relationships you provide.

* ALWAYS standardize your output to absolute pixel coordinates `[x1, y1, x2, y2]`.
* NEVER leave the Agent guessing about coordinate formats.
* PROVIDE context: if text is extracted from a crop, map its location back to the global image.

---

## 1. Bounding Box Formats (The Primitives)

There are several ways to represent a bounding box. You must explicitly know which one your model outputs and convert it to the Agent Standard (`xyxy`).

### Format Dictionary

| Format | Representation | Common Origin | Note |
| :--- | :--- | :--- | :--- |
| **`xyxy`** | `[x_min, y_min, x_max, y_max]` | Agent JSON, OpenCV | The absolute standard for Agents. |
| **`xywh`** | `[x_min, y_min, width, height]` | COCO Dataset | Needs conversion for easy drawing. |
| **`cxcywh`**| `[x_center, y_center, width, height]` | YOLO (Raw output) | Must convert to `xyxy` immediately. |

### Conversion: YOLO (`cxcywh`) to Agent (`xyxy`)

**✅ Pro-Pattern:**
```python
def cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(cx - (w / 2))
    y1 = int(cy - (h / 2))
    x2 = int(cx + (w / 2))
    y2 = int(cy + (h / 2))
    return (x1, y1, x2, y2)
```

---

## 2. The Global vs. Local Mapping Problem

When using the "Crop Before Compute" pattern (e.g., YOLO finds a license plate, you crop it, then send to OCR), the OCR tool returns coordinates relative to the *cropped* image. You must map these cropped ROI coordinates back to the original image space.

**✅ Pro-Pattern (Coordinate Mapping):**
```python
def map_local_to_global(local_bbox: list[int], crop_origin_x: int, crop_origin_y: int) -> list[int]:
    """
    Maps local crop coordinates back to the global image space.
    local_bbox: [x1, y1, x2, y2]
    """
    return [
        local_bbox[0] + crop_origin_x,
        local_bbox[1] + crop_origin_y,
        local_bbox[2] + crop_origin_x,
        local_bbox[3] + crop_origin_y
    ]
```

---

## 3. Polygon to Bounding Box Conversions

OCR models like PaddleOCR often return polygons (4 points) to account for rotated text. Agents struggle to reason with polygons. Convert them to orthogonal bounding boxes.

**✅ Pro-Pattern:**
```python
def polygon_to_xyxy(polygon: list[list[int]]) -> tuple[int, int, int, int]:
    """
    Converts a polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] to an orthogonal bbox.
    """
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return (min(xs), min(ys), max(xs), max(ys))
```

---

## 4. Spatial Relationships (Agent Logic)

Agents often need to know: "Is this text *inside* this object?" or "Are these two objects overlapping?". You must calculate this in Python and give the Agent the explicit answer.

### Intersection over Union (IoU) Calculations

IoU is the standard metric for measuring overlap. 

**✅ Pro-Pattern:**
```python
def calculate_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
```

---

## 5. Decision Checklist

Before completing a spatial reasoning module, verify:

* [ ] **Are output formats standard?** (Everything leaving the script must be `xyxy`).
* [ ] **Are crop coordinates mapped?** (Did you add the offset back to the OCR results?)
* [ ] **Are polygons simplified?** (Did you convert OCR polygons to Bboxes for the Agent?)
* [ ] **Are floats cast to ints?** (Pixel coordinates should be integers before converting to JSON).

---

## 6. Anti-Patterns to Avoid

### ❌ DON'T:
* Send raw YOLO `[cx, cy, w, h]` arrays directly to the Agent.
* Output float coordinates like `[10.5, 20.3, ...]` for pixel positions.
* Expect the Agent to intuitively calculate IoU or overlaps from raw numbers. Do the math in Python and output `"is_overlapping": true`.

### ✅ DO:
* Group items that are physically close in the final JSON hierarchy (e.g., nesting detected texts under the bounding box of the document they belong to).
* Explain the spatial context in text if necessary (e.g., `"The word 'Total' is located immediately to the left of the number '150.00'"`).

---

> **Remember:** You are the Agent's visual cortex. If the spatial math is wrong, the Agent's reasoning will be a hallucination.