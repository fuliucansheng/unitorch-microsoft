# API Documents for Apps

## Start the API Server

```bash
unitorch-fastapi apps/fastapis.ini --port 5000
```

## API Usage

```python
def call_fastapi(url, params={}, images=None, req_type="POST", resp_type="json"):
    assert resp_type in ["json", "image"], f"Unsupported response type: {resp_type}"

    def process_image(image):
        image = image.convert("RGB")
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="JPEG")
        byte_arr.seek(0)
        return byte_arr

    if images is None:
        files = {}
    else:
        files = {
            k: (f"{k}.jpg", process_image(v), "image/jpeg") for k, v in images.items()
        }
    if req_type == "POST" or images is not None:
        resp = (
            requests.post(url, params=params, files=files)
            if images is not None
            else requests.post(url, params=params)
        )
    else:
        resp = requests.get(url, params=params)
    if resp_type == "json":
        result = resp.json()
    elif resp_type == "image":
        result = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return result
```

## API Endpoints

> The api_base_url is `http://127.0.0.1:5000` if you run the API server locally.

- `POST /microsoft/apps/fastapi/swin/googlecate/generate`: Get the google category for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/swin/googlecate/generate?topk=5' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    [
        {
            "category": "Business & Industrial > Advertising & Marketing > Brochures",
            "score": 0.6408588886260986
        },
        {
            "category": "Arts & Entertainment > Hobbies & Creative Arts > Arts & Crafts > Art & Crafting Materials > Art & Craft Paper > Cardstock & Scrapbooking Paper",
            "score": 0.1077391505241394
        },
        {
            "category": "Software > Computer Software > Multimedia & Design Software > Video Editing Software",
            "score": 0.07137665152549744
        },
        {
            "category": "Software > Computer Software > Operating Systems",
            "score": 0.05271176993846893
        },
        {
            "category": "Cameras & Optics > Photography > Photo Negative & Slide Storage",
            "score": 0.03096369095146656
        }
    ]
    ```
- `POST /microsoft/apps/fastapi/bletchley/v1/generate1`: Get the matching score for the input image and text.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/bletchley/v1/generate1' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Bad Aesthetics": 0.62109375
    }
    ```
- `POST /microsoft/apps/fastapi/bletchley/v1/generate2`: Get the matching score for the input image and text.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/bletchley/v1/generate2' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Bad Aesthetics": 0.62109375
    }
    ```
- `POST /microsoft/apps/fastapi/bletchley/v1/generate3`: Get the matching score for the input image and text.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/bletchley/v1/generate3' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Bad Aesthetics": 0.62109375
    }
    ```
- `POST /microsoft/apps/fastapi/bletchley/v3/generate1`: Get the watermark score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/bletchley/v3/generate1' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Watermark": 0.51171875
    }
    ```
- `POST /microsoft/apps/fastapi/bletchley/v3/generate2`: Get the aesthetic score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/bletchley/v3/generate2' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Bad Aesthetics": 0.62109375
    }
    ```
- `POST /microsoft/apps/fastapi/siglip2/generate1`: Get the watermark score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/siglip2/generate1' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Watermark": 0.51171875
    }
    ```
- `POST /microsoft/apps/fastapi/siglip2/generate2`: Get the aesthetic score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/fastapi/siglip2/generate2' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Bad Aesthetics": 0.62109375
    }
    ```