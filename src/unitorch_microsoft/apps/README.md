# API Documents for Apps

## Start the API Server

```bash
unitorch-fastapi apps/fastapis.ini --port 5000
```

## API Endpoints

> The api_base_url is `http://127.0.0.1:5000` if you run the API server locally.

#### Spaces
- `POST /microsoft/apps/spaces/picasso/swin/googlecate/generate`: Get the google category for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/swin/googlecate/generate?topk=5' \
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
- `POST /microsoft/apps/spaces/picasso/bletchley/v1/generate1`: Get the blurry score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/bletchley/v1/generate1' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Blurry": 0.62109375
    }
    ```
- `POST /microsoft/apps/spaces/picasso/bletchley/v1/generate2`: Get the background & scores for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/bletchley/v1/generate2' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Complex": 0.93359375,
        "Simple": 0.10693359375,
        "White": 0.02978515625
    }
    ```
- `POST /microsoft/apps/spaces/picasso/bletchley/v3/generate1`: Get the watermark score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/bletchley/v3/generate1' \
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
- `POST /microsoft/apps/spaces/picasso/siglip2/generate1`: Get the bad crop score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/siglip2/generate1' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Bad Cropped": 0.07568359375
    }
    ```
- `POST /microsoft/apps/spaces/picasso/siglip2/generate2`: Get the bad padding score for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/siglip2/generate2' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: 
    ```json
    {
        "Bad Padding": 0.140625
    }
    ```
- `POST /microsoft/apps/spaces/picasso/basnet/generate1`: Get the bounding box for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/basnet/generate1?threshold=0.1' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: The response is an image bytes with the bounding box drawn on it.
- `POST /microsoft/apps/spaces/picasso/detr/generate1`: Get the bounding box for the input image.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/picasso/detr/generate1?threshold=0.1' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@debug.png;type=image/png'
    ```
    * Response: The response is an image bytes with the bounding box drawn on it.
- `POST /microsoft/apps/spaces/gpt/image-15/generate`: Get the image generation result for the input prompt.
    * Request: size should be one of "1024x1024", "1536x1024" or "1024x1536".
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/gpt/image-15/generate?prompt=a%20house&size=1024x1024&background=transparent' \
    -H 'accept: application/json' \
    -d ''
    ```
    * Response: The response is an image bytes generated by the model.
- `POST /microsoft/apps/spaces/gpt/image-15/edit`: Get the image editing result for the input image and prompt.
    * Request: size should be one of "1024x1024", "1536x1024" or "1024x1536".
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/gpt/image-15/edit?prompt=put%20the%20first%20logo%20on%20the%20top%20right%20corner%20of%20the%20second%20image&size=1536x1024&background=transparent' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'images=@logo.png;type=image/png' \
    -F 'images=@debug.png;type=image/png' \
    -F 'mask='
    ```
    * Response: The response is an image bytes generated by the model.
- `POST /microsoft/apps/spaces/gemini/image/generate`: Get the image generation result for the input prompt.
    * Request: size has little effect on the generation result, you can set it to "1024x1024", "1536x1024" or "1024x1536" as you like.
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/gemini/image/generate?prompt=a%20house&size=1024x1024&background=transparent' \
    -H 'accept: application/json' \
    -d ''
    ```
    * Response: The response is an image bytes generated by the model.
- `POST /microsoft/apps/spaces/gemini/image/edit`: Get the image editing result for the input image and prompt.
    * Request: size has little effect on the generation result, you can set it to "1024x1024", "1536x1024" or "1024x1536" as you like.
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/spaces/gemini/image/edit?prompt=put%20the%20first%20logo%20on%20the%20top%20right%20corner%20of%20the%20second%20image&size=1536x1024&background=transparent' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'images=@logo.png;type=image/png' \
    -F 'images=@debug.png;type=image/png'
    ```
    * Response: The response is an image bytes generated by the model.

#### Studio

