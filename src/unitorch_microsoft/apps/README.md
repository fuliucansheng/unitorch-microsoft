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

> The host URL is `http://127.0.0.1:5000` if you run the API server locally.

- `POST /microsoft/apps/fastapi/swin/googlecate/generate`: Get the google category for the input image.
    * Request Body:
        * image: (file) The input image.
        * topk: (param, int, optional) The number of top categories to return. Default is 1.
    * Response Body:
        * categories: (list) The list of predicted categories with their scores. Each category is a dictionary with the following keys:
            * category: (str) The name of the category.
            * score: (float) The confidence score of the prediction.
