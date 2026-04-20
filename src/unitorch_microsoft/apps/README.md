# API Documents for Apps

## Start the API Server

```bash
unitorch-fastapi apps/fastapis.ini --port 5000
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
- `POST /microsoft/apps/fastapi/bletchley/v1/generate1`: Get the background (complex, simple, white) & types (poster, real, logo) scores for the input image.
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
        "Complex": 0.291015625,
        "Simple": 0.7578125,
        "White": 0.0927734375,
        "Poster": 0.734375,
        "Real": 0.20703125,
        "Logo": 0.18359375
    }
    ```
- `POST /microsoft/apps/fastapi/bletchley/v1/generate2`: Get the blurry score for the input image.
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
        "Blurry": 0.62109375
    }
    ```
- `POST /microsoft/apps/fastapi/bletchley/v1/generate3`: Get the background & scores for the input image.
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
        "Complex": 0.93359375,
        "Simple": 0.10693359375,
        "White": 0.02978515625
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
- `POST /microsoft/apps/fastapi/siglip2/generate1`: Get the bad crop score for the input image.
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
        "Bad Cropped": 0.07568359375
    }
    ```
- `POST /microsoft/apps/fastapi/siglip2/generate2`: Get the bad padding score for the input image.
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
        "Bad Padding": 0.140625
    }
    ```