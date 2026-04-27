# Spaces Skills

Image inference services: classification, quality scoring, object detection, generation, editing.

```python
from unitorch_microsoft.apps.skills.spaces import SpacesClient
spaces = SpacesClient("http://127.0.0.1:5000")
```

---

## Image Classification

### `classify_image(image_path, topk=5)` — Google product category

Returns top-k `{"category": str, "score": float}` pairs.

```python
results = spaces.classify_image("photo.jpg", topk=3)
# [{"category": "Business & Industrial > ...", "score": 0.64}, ...]
```

---

## Image Quality Scoring

All scores are float in **[0, 1]**. Higher means more of that property.

### `score_blurry(image_path)` — blurriness

```python
blur = spaces.score_blurry("photo.jpg")   # e.g. 0.62
```

### `score_background(image_path)` — background complexity

Returns `{"Complex": float, "Simple": float, "White": float}`.

```python
bg = spaces.score_background("photo.jpg")
# {"Complex": 0.93, "Simple": 0.11, "White": 0.03}
```

### `score_watermark(image_path)` — watermark presence

```python
wm = spaces.score_watermark("photo.jpg")   # e.g. 0.51
```

### `score_bad_crop(image_path)` — crop quality (higher = worse)

```python
crop = spaces.score_bad_crop("photo.jpg")   # e.g. 0.08
```

### `score_bad_padding(image_path)` — padding quality (higher = worse)

```python
pad = spaces.score_bad_padding("photo.jpg")   # e.g. 0.14
```

---

## Object Detection (returns annotated image bytes)

### `detect_objects_basnet(image_path, threshold=0.1)`

Returns PNG bytes with BASNet bounding boxes drawn.

```python
img_bytes = spaces.detect_objects_basnet("photo.jpg", threshold=0.2)
with open("out.png", "wb") as f:
    f.write(img_bytes)
```

### `detect_objects_detr(image_path, threshold=0.1)`

Returns PNG bytes with DETR bounding boxes drawn.

```python
img_bytes = spaces.detect_objects_detr("photo.jpg", threshold=0.1)
```

---

## Image Generation

`size` options: `"1024x1024"` | `"1536x1024"` | `"1024x1536"`

### `generate_image_gpt(prompt, size="1024x1024", background="transparent")` — GPT Image 1.5

```python
img_bytes = spaces.generate_image_gpt("a red house on a hill", size="1024x1024")
```

### `generate_image_gemini(prompt, size="1024x1024", background="transparent")` — Gemini

```python
img_bytes = spaces.generate_image_gemini("a red house on a hill")
```

---

## Image Editing

### `edit_image_gpt(image_paths, prompt, size="1536x1024", background="transparent", mask_path=None)` — GPT Image 1.5

```python
img_bytes = spaces.edit_image_gpt(
    ["logo.png", "background.png"],
    prompt="put the logo on the top-right corner of the background",
    size="1536x1024",
)
```

### `edit_image_gemini(image_paths, prompt, size="1536x1024", background="transparent")` — Gemini

```python
img_bytes = spaces.edit_image_gemini(
    ["logo.png", "background.png"],
    prompt="put the logo on the top-right corner of the background",
)
```
