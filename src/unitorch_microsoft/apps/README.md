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

###### Chat

- `GET /microsoft/apps/studio/chat/commands`: Get the list of available chat commands in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/chat/commands' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "name": "get_time",
            "description": "Get the current time."
        },
        {
            "name": "get_date",
            "description": "Get the current date."
        }
    ]
    ```
- `GET /microsoft/apps/studio/chat/entities`: Get the list of available entities in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/chat/entities' \
    -H 'accept: application/json'
    ```
    * Response: 
    ```json
    [
        {
            "type": "dataset",
            "id": "data1",
            "name": "data1",
            "description": "The dataset for mma."
        },
        {
            "type": "job",
            "id": "job1",
            "name": "job1",
            "description": "The job for data1 processing."
        },
        {
            "type": "job",
            "id": "job2",
            "name": "job2",
            "description": "The job for model finetuning."
        },
        {
            "type": "label",
            "id": "label1",
            "name": "label1",
            "description": "The label task for data1."
        }
        ...
    ]
    ```
- `GET /microsoft/apps/studio/chat/models`: Get the list of available models in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/chat/models' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "GPT-4",
            "name": "GPT-4",
            "description": "The GPT-4 model for chat."
        },
        {
            "id": "GPT-3.5",
            "name": "GPT-3.5",
            "description": "The GPT-3.5 model for chat."
        }
        ...
    ]
    ```
- `POST /microsoft/apps/studio/chat/new`: Create a new chat session in the studio. Will fork a new session from the input session_id, if session_id is not provided, will create a new session from scratch.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/chat/new' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "session_id": "session1",
    }'
    ```
    * Response:
    ```json
    {
        "new_session_id": "session2"
    }
    ```
- `POST /microsoft/apps/studio/chat/completions`: Get the chat completions for the input message in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "session_id": "session1",
        "message": {
            "role": "user",
            "content": "帮我分析这个数据集 @data1"
        },
        "mode": "plan",
        "model": "gpt-4",
        "entities": [
            {
            "type": "dataset",
            "id": "data1"
            }
        ],
        "stream": true
    }'
    ```
    * Streaming response: The response will be a stream of chat completions generated by the model, each completion is a json string with the following format:
    ```json
        event: message.delta
        data: { "content": "正在分析数据..." }

        event: tool.call
        data: {
        "command": "check-dataset",
        "arguments": { "dataset_id": "data1" }
        }

        event: tool.result
        data: {
        "dataset_summary": {...}
        }

        event: message.done
        data: { "content": "分析完成..." }
    ```
- `GET /microsoft/apps/studio/chat/history`: Get the chat history for a specific session in the studio.
    ```json
    {
        "id": "session1",
        "mode": "build",
        "model": "gpt-4",
        "messages": [
            { "role": "user", "content": "..." },
            { "role": "assistant", "content": "..."}
        ]
    }
    ```
- `GET /microsoft/apps/studio/chat/sessions`: Get the list of chat sessions in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/chat/sessions' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "session1",
            "mode": "build",
            "model": "gpt-4",
            "name": "session1",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        },
        {
            "id": "session2",
            "mode": "plan",
            "model": "gpt-3.5",
            "name": "session2",
            "created_at": "2023-01-02T00:00:00Z",
            "updated_at": "2023-01-02T00:00:00Z"
        }
        ...
    ]
    ```

###### Datasets

- `GET /microsoft/apps/studio/datasets`: Get the list of available datasets in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/chat/datasets' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "data1",
            "name": "data1",
            "description": "The dataset for mma."
        }
        ...
    ]
    ```
- `POST /microsoft/apps/studio/datasets/details`: Get the details of a specific dataset in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/datasets/details' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "data1"
    }'
    ```
    * Response:
    ```json
    {
        "id": "data1",
        "name": "data1",
        "description": "The dataset for mma.",
        "rows": 10000,
        "size": "100MB",
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
        "details": {
            "splits": {
                "train": 8000,
                "validation": 1000,
                "test": 1000
            },
            "columns": [
                {
                    "name": "column1",
                    "type": "string",
                    "description": "The first column of the dataset."
                },
                {
                    "name": "column2",
                    "type": "integer",
                    "description": "The second column of the dataset."
                }
            ],
            "labels_distributions": {
                "label1": 50,
                "label2": 50
            }
        }
    }
    ```
- `POST /microsoft/apps/studio/datasets/preview`: Get the preview of a specific dataset in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/datasets/preview' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "data1",
        "split": "train",
        "start": 0,
        "limit": 5
    }'
    ```
    * Response:
    ```json
    {
        "columns": [
            "column1",
            "column2"
        ],
        "rows": [
            ["value1", 1],
            ["value2", 2],
            ["value3", 3],
            ["value4", 4],
            ["value5", 5]
        ],
        "number_columns": ["column2"],
        "image_columns": [],
        "video_columns": []
    }
    ```

###### Jobs

- `GET /microsoft/apps/studio/jobs`: Get the list of available jobs in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/jobs' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "job1",
            "name": "job1",
            "description": "The job for data1 processing."
        },
        {
            "id": "job2",
            "name": "job2",
            "description": "The job for model finetuning."
        }
        ...
    ]
    ```
- `POST /microsoft/apps/studio/jobs/details`: Get the details of a specific job in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/jobs/details' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "job1"
    }'
    ```
    * Response:
    ```json
    {
        "id": "job1",
        "name": "job1",
        "description": "The job for data1 processing.",
        "status": "running",
        "progress": 50,
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
        "logs": "Job started...\nProcessing data...\nJob completed successfully.",
    }
    ```
- `POST /microsoft/apps/studio/jobs/cancel`: Cancel a specific job in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/jobs/cancel' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "job1"
    }'
    ```
    * Response:
    ```json
    {
        "id": "job1",
        "name": "job1",
        "description": "The job for data1 processing.",
        "status": "cancelled",
        "progress": 50,
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-01T00:00:00Z",
        "logs": "Job started...\nProcessing data...\nJob was cancelled.",
    }
    ``` 
- `POST /microsoft/apps/studio/jobs/restart`: Restart a specific job in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/jobs/restart' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "job1"
    }'
    ```
    * Response:
    ```json
    {
        "id": "job1",
        "name": "job1",
        "description": "The job for data1 processing.",
        "status": "running",
        "progress": 0,
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-01-02T00:00:00Z",
        "logs": "Job restarted...\nProcessing data...\nJob completed successfully.",
    }
    ```

###### Labels

- `GET /microsoft/apps/studio/labels`: Get the list of available labels in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/labels' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "label1",
            "name": "label1",
            "description": "The label task for data1."
        }
        ...
    ]
    ```
- `POST /microsoft/apps/studio/labels/details`: Get the details of a specific label in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/labels/details' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "label1"
    }'
    ```
    * Response:
    ```json
    {
        "id": "label1",
        "name": "label1",
        "type": "classification",
        "description": "The label task for data1.",
        "stats": {
            "total": 100,
            "completed": 50,
            "pending": 50,
            "partial_completed": 30,
            "agreement": 0.8,
            "labelers": [
                {
                    "id": "labeler1",
                    "name": "labeler1",
                    "completed": 20
                },
                {
                    "id": "labeler2",
                    "name": "labeler2",
                    "completed": 30
                }
            ]
        }
    }
    ```

###### Reports

- `GET /microsoft/apps/studio/reports`: Get the list of available reports in the studio.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studio/reports' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "report1",
            "name": "report1",
            "description": "The report for data1 processing job."
        }
        ...
    ]
    ```
- `POST /microsoft/apps/studio/reports/details`: Get the details of a specific report in the studio.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studio/reports/details' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "report1"
    }'
    ```
    * Response:
    ```json
    {
        "id": "report1",
        "name": "report1",
        "description": "The report for data1 processing job.",
        "created_at": "2023-01-02T00:00:00Z",
        "updated_at": "2023-01-02T00:00:00Z",
        "content": "markdown content of the report goes here",
    }
    ```