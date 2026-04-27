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

#### Studios

###### Chats

- `GET /microsoft/apps/studios/chats/commands`: Get the list of available chat commands in the studios.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studios/chats/commands' \
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
- `GET /microsoft/apps/studios/chats/entities`: Get the list of available entities in the studios.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studios/chats/entities' \
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
- `GET /microsoft/apps/studios/chats/models`: Get the list of available models in the studios.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studios/chats/models' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "GPT-4",
            "name": "GPT-4",
            "description": "The GPT-4 model for chat.",
            "model_id": "gpt-4",
            "provider_id": "openai",
        },
        {
            "id": "GPT-3.5",
            "name": "GPT-3.5",
            "description": "The GPT-3.5 model for chat.",
            "model_id": "gpt-3.5",
            "provider_id": "openai",
        }
        ...
    ]
    ```
- `POST /microsoft/apps/studios/chats/new`: Create a new chat session. Each session is bound to a workspace folder — all agent file operations use this folder as the root. If `workspace` is omitted, a folder is automatically created at `workspace_root/<session_id>/`. If `session_id` is provided, forks from that session (inheriting its workspace).
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/chats/new' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "session_id": "session1",
        "workspace": "/path/to/my/project"
    }'
    ```
    * Response:
    ```json
    {
        "new_session_id": "session2",
        "workspace": "/path/to/my/project",
        "welcome_message": "Welcome to Ads Studio. How can I assist with your ML workflows today?"
    }
    ```
- `POST /microsoft/apps/studios/chats/name`: Rename a specific chat session in the studios.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/chats/new' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "session_id": "session1",
        "name": "New Session Name"
    }'
    ```
    * Response:
    ```json
    {
        "session_id": "session1",
        "name": "New Session Name"
    }
    ```
- `POST /microsoft/apps/studios/chats/delete`: Delete a specific chat session in the studios.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/chats/delete' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "session_id": "session1"
    }'
    ```
    * Response:
    ```json
    {
        "message": "Chat session deleted successfully."
    }
    ```
- `POST /microsoft/apps/studios/chats/completions`: Get the chat completions for the input message in the studios.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/chats/completions' \
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
        "model_id": "gpt-4",
        "provider_id": "openai",
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
- `GET /microsoft/apps/studios/chats/history`: Get the chat history for a specific session in the studios.
    ```json
    {
        "id": "session1",
        "mode": "build",
        "model": "gpt-4",
        "model_id": "gpt-4",
        "provider_id": "openai",
        "messages": [
            { "role": "user", "content": "..." },
            { "role": "assistant", "content": "..."}
        ]
    }
    ```
- `GET /microsoft/apps/studios/chats/sessions`: Get the list of chat sessions in the studios.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studios/chats/sessions' \
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

- `GET /microsoft/apps/studios/datasets`: List all datasets.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studios/datasets' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "my_dataset",
            "description": "A sample dataset",
            "type": "local",
            "rows": 1000,
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00"
        }
    ]
    ```

- `POST /microsoft/apps/studios/datasets/create`: Create a new dataset by uploading one or more TSV files. All files must share the same header. Column types are auto-inferred (integer / float / string / image / video). `splits` is a JSON object mapping split name to a list of filenames (e.g. `{"train": ["train1.tsv", "train2.tsv"], "validate": ["val.tsv"], "test": ["test.tsv"]}`). Every uploaded filename must appear in exactly one split list. If `splits` is omitted, all files are assigned to `"test"`.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/create?name=my_dataset&description=A+sample+dataset&splits=%7B%22train%22%3A%5B%22train1.tsv%22%2C%22train2.tsv%22%5D%2C%22validate%22%3A%5B%22val.tsv%22%5D%2C%22test%22%3A%5B%22test.tsv%22%5D%7D' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'files=@train1.tsv;type=text/tab-separated-values' \
    -F 'files=@train2.tsv;type=text/tab-separated-values' \
    -F 'files=@val.tsv;type=text/tab-separated-values' \
    -F 'files=@test.tsv;type=text/tab-separated-values'
    ```
    * The decoded `splits` query parameter value is:
    ```json
    {"train": ["train1.tsv", "train2.tsv"], "validate": ["val.tsv"], "test": ["test.tsv"]}
    ```
    * Response:
    ```json
    {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "my_dataset",
        "description": "A sample dataset",
        "type": "local",
        "rows": 1200,
        "size": "256.3 KB",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "meta": {
            "task_type": "",
            "metrics": [],
            "label_columns": [],
            "columns": [
                {"name": "image_url", "type": "image", "description": ""},
                {"name": "label",     "type": "string", "description": ""}
            ],
            "extra": {}
        }
    }
    ```

- `POST /microsoft/apps/studios/datasets/upload`: Append rows into an existing dataset. All uploaded files must share the same header as the existing dataset. The dataset's `rows`, `size`, and `updated_at` are updated accordingly.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/upload?dataset_id=550e8400-e29b-41d4-a716-446655440000&split=test' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'files=@test.tsv;type=text/tab-separated-values'
    ```
    * Response: updated `DatasetInfo` object (same structure as `/create`).

- `POST /microsoft/apps/studios/datasets/get`: Get the full info for a dataset.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/get' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": "550e8400-e29b-41d4-a716-446655440000"}'
    ```
    * Response: same structure as `/create`.

- `POST /microsoft/apps/studios/datasets/delete`: Delete a dataset and all its samples and sample meta.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/delete' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": "550e8400-e29b-41d4-a716-446655440000"}'
    ```
    * Response:
    ```json
    {"id": "550e8400-e29b-41d4-a716-446655440000", "deleted": true}
    ```

- `POST /microsoft/apps/studios/datasets/meta/update`: Update dataset-level metadata (name, description, task_type, metrics, label_columns, columns, extra). Only fields provided in `meta` are overwritten; the rest are preserved.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/meta/update' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "meta": {
            "name": "renamed_dataset",
            "task_type": "classification",
            "metrics": ["accuracy", "f1"],
            "label_columns": ["label"],
            "extra": {"source": "internal"}
        }
    }'
    ```
    * Response: updated `DatasetInfo` object (same structure as `/create`).

- `POST /microsoft/apps/studios/datasets/sample/list`: List samples with pagination. The optional `split` field filters by the `"split"` meta key. When `include_meta` is true, `split` is included both inside `meta` and at the top-level `split` field of each record.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/sample/list' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "split": "train",
        "start": 0,
        "limit": 20,
        "include_meta": true
    }'
    ```
    * Response:
    ```json
    {
        "total": 1000,
        "start": 0,
        "limit": 20,
        "samples": [
            {
                "sample_id": "a1b2c3d4-...",
                "split": "train",
                "data": {"image_url": "http://example.com/1.jpg", "label": "cat"},
                "meta": {"split": "train", "score": 0.95, "reviewed": true}
            }
        ]
    }
    ```

- `POST /microsoft/apps/studios/datasets/sample/get`: Get a single sample by ID.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/sample/get' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "sample_id": "a1b2c3d4-..."
    }'
    ```
    * Response: a single `SampleRecord` object (same structure as one item in `samples` above).

- `POST /microsoft/apps/studios/datasets/sample/select`: Filter samples by meta conditions using AND logic. Supported operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `exists`, `not_exists`, `contains`.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/sample/select' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "filters": [
            {"key": "score", "op": "gte", "value": 0.9},
            {"key": "reviewed", "op": "eq", "value": true}
        ],
        "split": "train",
        "start": 0,
        "limit": 100,
        "include_meta": true
    }'
    ```
    * Response: same structure as `/sample/list`.

- `POST /microsoft/apps/studios/datasets/sample/meta/set`: Upsert (insert or update) meta key-value pairs for one or more samples in a single call.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/sample/meta/set' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "records": {
            "a1b2c3d4-...": {"score": 0.95, "reviewed": true},
            "b2c3d4e5-...": {"score": 0.80, "reviewed": false}
        }
    }'
    ```
    * Response:
    ```json
    {"dataset_id": "550e8400-e29b-41d4-a716-446655440000", "updated_samples": 2}
    ```

- `POST /microsoft/apps/studios/datasets/sample/meta/get`: Get meta key-value pairs for specified samples and optional key filter.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/sample/meta/get' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "sample_ids": ["a1b2c3d4-...", "b2c3d4e5-..."],
        "keys": ["score"]
    }'
    ```
    * Response:
    ```json
    {
        "a1b2c3d4-...": {"score": 0.95},
        "b2c3d4e5-...": {"score": 0.80}
    }
    ```

- `POST /microsoft/apps/studios/datasets/sample/meta/delete`: Delete specific meta keys for given samples. If `keys` is omitted, all meta for those samples is deleted.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/sample/meta/delete' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "sample_ids": ["a1b2c3d4-..."],
        "keys": ["score"]
    }'
    ```
    * Response:
    ```json
    {"dataset_id": "550e8400-e29b-41d4-a716-446655440000", "deleted_samples": 1}
    ```

- `POST /microsoft/apps/studios/datasets/sample/meta/list`: List all distinct meta keys present in the dataset (or within a given set of samples).
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/datasets/sample/meta/list' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "sample_ids": []
    }'
    ```
    * Response:
    ```json
    {
        "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
        "keys": ["score", "reviewed", "split_override"]
    }
    ```

###### Jobs

- `GET /microsoft/apps/studios/jobs`: List all jobs.
    * Request:
    ```bash
    curl -X 'GET' \
    '{api_base_url}/microsoft/apps/studios/jobs' \
    -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "a1b2c3d4-...",
            "name": "finetune-resnet",
            "description": "Finetune ResNet on my_dataset",
            "type": "finetune",
            "status": "running",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:01:00+00:00"
        }
    ]
    ```

- `POST /microsoft/apps/studios/jobs/create`: Create and immediately launch a new job. A working directory is created under `workdir_root/<job_id>/` unless `workdir` is specified. The `command` is executed in that directory via a shell subprocess; stdout and stderr are merged and written to `job.log` inside the workdir. `dataset_ids` is a list of dataset IDs that this job operates on.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/jobs/create' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "name": "finetune-resnet",
        "description": "Finetune ResNet on my_dataset",
        "type": "finetune",
        "dataset_ids": [
            "550e8400-e29b-41d4-a716-446655440000",
            "661f9511-f30c-52e5-b827-557766551111"
        ],
        "command": "python train.py --config config.ini",
        "workdir": "/workspace/jobs/finetune-resnet",
        "extra": {"model": "resnet50", "epochs": 10}
    }'
    ```
    * Response:
    ```json
    {
        "id": "a1b2c3d4-...",
        "name": "finetune-resnet",
        "description": "Finetune ResNet on my_dataset",
        "type": "finetune",
        "dataset_ids": [
            "550e8400-e29b-41d4-a716-446655440000",
            "661f9511-f30c-52e5-b827-557766551111"
        ],
        "command": "python train.py --config config.ini",
        "workdir": "/workspace/jobs/finetune-resnet",
        "status": "running",
        "pid": 12345,
        "exit_code": null,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "started_at": "2024-01-01T00:00:01+00:00",
        "finished_at": "",
        "extra": {"model": "resnet50", "epochs": 10}
    }
    ```

- `POST /microsoft/apps/studios/jobs/get`: Get full info for a job.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/jobs/get' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": "a1b2c3d4-..."}'
    ```
    * Response: same structure as `/create`.

- `POST /microsoft/apps/studios/jobs/cancel`: Cancel a pending or running job. Sends `SIGTERM` to the subprocess and marks the job as `cancelled`.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/jobs/cancel' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": "a1b2c3d4-..."}'
    ```
    * Response: updated `JobInfo` with `status: "cancelled"`.

- `POST /microsoft/apps/studios/jobs/restart`: Restart a completed, failed, or cancelled job. Clears the old log, resets progress, and re-launches the same command in the same workdir.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/jobs/restart' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": "a1b2c3d4-..."}'
    ```
    * Response: updated `JobInfo` with `status: "running"`.

- `POST /microsoft/apps/studios/jobs/delete`: Delete a job record. The job must not be in `running` state (cancel it first).
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/jobs/delete' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": "a1b2c3d4-..."}'
    ```
    * Response:
    ```json
    {"id": "a1b2c3d4-...", "deleted": true}
    ```

- `POST /microsoft/apps/studios/jobs/logs`: Get log lines for a job. While the job is **running**, returns the latest `tail` lines (default 50); pass `"tail": null` for all lines so far. When the job is **completed / failed / cancelled**, all lines are always returned regardless of `tail`.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/jobs/logs' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"id": "a1b2c3d4-...", "tail": 100}'
    ```
    * Response:
    ```json
    {
        "id": "a1b2c3d4-...",
        "status": "running",
        "lines": [
            "[2024-01-01T00:00:01+00:00] Job started (pid=12345): python train.py ...",
            "Epoch 1/10: loss=0.342",
            "Epoch 2/10: loss=0.289"
        ],
        "total_lines": 3
    }
    ```

###### Labels

- `GET /microsoft/apps/studios/labels`: List all label tasks.
    * Request:
    ```bash
    curl -X 'GET' '{api_base_url}/microsoft/apps/studios/labels' -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "t1b2c3d4-...",
            "name": "Image Quality Labeling",
            "description": "Rate image quality for training set",
            "dataset_id": "550e8400-...",
            "status": "active",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T01:00:00+00:00"
        }
    ]
    ```

- `POST /microsoft/apps/studios/labels/create`: Create a new label task tied to a dataset. `display_fields` configures which data/meta columns are shown in the annotation UI. `label_fields` defines the fields annotators must fill in — each field specifies a `type` and, for selection fields, an `options` list of allowed values. `ui_html` is an optional custom HTML template for the annotation interface.

    **`label_fields` field schema:**
    | Field | Type | Description |
    |---|---|---|
    | `key` | string | Field key stored in the label dict |
    | `label` | string | Human-readable label shown in the UI |
    | `type` | string | `text` \| `select` \| `multiselect` \| `number` \| `bool` |
    | `options` | list[string] | Allowed values for `select` / `multiselect` types |
    | `required` | bool | Whether the field must be filled before submitting (default `true`) |
    | `default` | string \| null | Pre-filled default value in the UI |

    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/create' \
    -H 'Content-Type: application/json' \
    -d '{
        "name": "Image Quality Labeling",
        "description": "Rate image quality for training set",
        "dataset_id": "550e8400-...",
        "display_fields": [
            {"key": "image_url", "label": "Image",  "source": "data", "type": "image"},
            {"key": "split",     "label": "Split",  "source": "meta", "type": "text"}
        ],
        "label_fields": [
            {"key": "quality",  "label": "Quality",  "type": "select",      "options": ["excellent", "good", "fair", "poor"], "required": true},
            {"key": "is_valid", "label": "Is Valid", "type": "bool",         "options": [],                                    "required": true},
            {"key": "tags",     "label": "Tags",     "type": "multiselect",  "options": ["blurry", "cropped", "watermark"],    "required": false},
            {"key": "comment",  "label": "Comment",  "type": "text",         "options": [],                                    "required": false}
        ],
        "ui_html": "<div>Custom annotation UI HTML here</div>",
        "extra": {}
    }'
    ```
    * Response: `LabelTaskInfo` object (see `/get`).

- `POST /microsoft/apps/studios/labels/get`: Get full info for a label task.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/get' \
    -H 'Content-Type: application/json' \
    -d '{"id": "t1b2c3d4-..."}'
    ```
    * Response:
    ```json
    {
        "id": "t1b2c3d4-...",
        "name": "Image Quality Labeling",
        "description": "Rate image quality for training set",
        "dataset_id": "550e8400-...",
        "status": "active",
        "display_fields": [
            {"key": "image_url", "label": "Image", "source": "data", "type": "image", "width": null},
            {"key": "split",     "label": "Split", "source": "meta", "type": "text",  "width": null}
        ],
        "label_fields": [
            {"key": "quality",  "label": "Quality",  "type": "select",     "options": ["excellent", "good", "fair", "poor"], "required": true,  "default": null},
            {"key": "is_valid", "label": "Is Valid", "type": "bool",        "options": [],                                    "required": true,  "default": null},
            {"key": "tags",     "label": "Tags",     "type": "multiselect", "options": ["blurry", "cropped", "watermark"],    "required": false, "default": null},
            {"key": "comment",  "label": "Comment",  "type": "text",        "options": [],                                    "required": false, "default": null}
        ],
        "ui_html": "<div>...</div>",
        "labelers": [
            {"id": "user1", "name": "user1", "assigned": 500, "completed": 120}
        ],
        "total_samples": 1000,
        "completed_samples": 120,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T01:00:00+00:00",
        "extra": {}
    }
    ```

- `POST /microsoft/apps/studios/labels/update`: Update mutable fields of a label task. Only provided (non-null) fields are overwritten; omitted fields keep their current values. Updatable fields: `name`, `description`, `display_fields`, `label_fields`, `ui_html`, `extra`.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/update' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "t1b2c3d4-...",
        "name": "Updated Task Name",
        "description": "Revised task description",
        "ui_html": "<div>Updated annotation UI</div>",
        "display_fields": [
            {"key": "image_url", "label": "Image",  "source": "data", "type": "image"},
            {"key": "split",     "label": "Split",  "source": "meta", "type": "text"}
        ],
        "label_fields": [
            {"key": "quality",      "label": "Quality",      "type": "select", "options": ["excellent", "good", "fair", "poor"]},
            {"key": "is_valid",     "label": "Is Valid",     "type": "bool",   "options": []},
            {"key": "comment_type", "label": "Comment Type", "type": "select", "options": ["noise", "mislabeled", "other"], "required": false}
        ],
        "extra": {"reviewer": "alice"}
    }'
    ```
    * Response: updated `LabelTaskInfo`.

- `POST /microsoft/apps/studios/labels/delete`: Delete a label task and all its assignments.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/delete' \
    -H 'Content-Type: application/json' \
    -d '{"id": "t1b2c3d4-..."}'
    ```
    * Response:
    ```json
    {"id": "t1b2c3d4-...", "deleted": true}
    ```

- `POST /microsoft/apps/studios/labels/assign`: Assign samples to labelers. If `sample_ids` is empty, all samples in the dataset are distributed evenly (round-robin) across `labeler_ids`. Existing assignments for the same (labeler, sample) pair are skipped.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/assign' \
    -H 'Content-Type: application/json' \
    -d '{
        "task_id": "t1b2c3d4-...",
        "labeler_ids": ["user1", "user2", "user3"],
        "sample_ids": []
    }'
    ```
    * Response:
    ```json
    {"task_id": "t1b2c3d4-...", "new_assignments": 1000}
    ```

- `POST /microsoft/apps/studios/labels/sample/random`: Get a sample's data and meta for the annotation UI. If `sample_id` is provided, returns that specific sample; otherwise picks a random sample from the task's dataset(non-labeled samples only).
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/sample/random' \
    -H 'Content-Type: application/json' \
    -d '{
        "task_id": "t1b2c3d4-...",
        "sample_id": null
    }'
    ```
    * Response:
    ```json
    {
        "sample_id": "a1b2c3d4-...",
        "data": {"image_url": "http://example.com/1.jpg", "label": "cat"},
        "meta": {"split": "train", "score": 0.95}
    }
    ```

- `POST /microsoft/apps/studios/labels/sample/submit`: Record a labeler's annotation for a sample. Updates the assignment to `completed` and recalculates task progress.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/sample/submit' \
    -H 'Content-Type: application/json' \
    -d '{
        "task_id": "t1b2c3d4-...",
        "labeler_id": "user1",
        "sample_id": "a1b2c3d4-...",
        "label": {"quality": 4, "is_valid": true},
        "comment": "slightly blurry but usable"
    }'
    ```
    * Response:
    ```json
    {"task_id": "t1b2c3d4-...", "sample_id": "a1b2c3d4-...", "status": "completed"}
    ```

- `POST /microsoft/apps/studios/labels/progress`: Get annotation progress for a task.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/progress' \
    -H 'Content-Type: application/json' \
    -d '{"id": "t1b2c3d4-..."}'
    ```
    * Response:
    ```json
    {
        "task_id": "t1b2c3d4-...",
        "total_samples": 1000,
        "completed_samples": 350,
        "labelers": [
            {"id": "user1", "name": "user1", "assigned": 500, "completed": 200},
            {"id": "user2", "name": "user2", "assigned": 500, "completed": 150}
        ]
    }
    ```

- `POST /microsoft/apps/studios/labels/export`: Export annotations merged by sample_id. Returns a dict keyed by `sample_id`; each value contains the original sample `data`, its `meta`, and per-labeler results under `labels` — a dict keyed by `labeler_id` with `label`, `comment`, `labeled_at`, and `status` fields. Merge across labelers is automatic — no separate merge step needed.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/labels/export' \
    -H 'Content-Type: application/json' \
    -d '{
        "task_id": "t1b2c3d4-...",
        "labeler_ids": null,
        "include_unfinished": false
    }'
    ```
    * Response:
    ```json
    {
        "a1b2c3d4-...": {
            "data":  {"image_url": "http://example.com/1.jpg", "label": "cat"},
            "meta":  {"split": "train", "score": 0.95},
            "labels": {
                "user1": {"label": {"quality": 4, "is_valid": true},  "comment": "looks good", "labeled_at": "2024-01-01T02:00:00+00:00", "status": "completed"},
                "user2": {"label": {"quality": 3, "is_valid": true},  "comment": "",            "labeled_at": "2024-01-01T03:00:00+00:00", "status": "completed"}
            }
        },
        "b2c3d4e5-...": {
            "data":  {"image_url": "http://example.com/2.jpg", "label": "dog"},
            "meta":  {"split": "train", "score": 0.88},
            "labels": {
                "user1": {"label": {"quality": 2, "is_valid": false}, "comment": "blurry",      "labeled_at": "2024-01-01T02:10:00+00:00", "status": "completed"}
            }
        }
    }
    ```

###### Reports

- `GET /microsoft/apps/studios/reports`: List all reports.
    * Request:
    ```bash
    curl -X 'GET' '{api_base_url}/microsoft/apps/studios/reports' -H 'accept: application/json'
    ```
    * Response:
    ```json
    [
        {
            "id": "r1b2c3d4-...",
            "name": "Dataset Analysis Report",
            "description": "Weekly analysis of training data quality",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T01:00:00+00:00"
        }
    ]
    ```

- `POST /microsoft/apps/studios/reports/create`: Create a markdown report. `dataset_ids`, `job_ids`, and `label_task_ids` record which resources this report is associated with for tracking purposes. `content` is the full markdown text.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/reports/create' \
    -H 'Content-Type: application/json' \
    -d '{
        "name": "Dataset Analysis Report",
        "description": "Weekly analysis of training data quality",
        "content": "# Dataset Analysis\n\n## Summary\n\nTotal samples: 10000 ...",
        "dataset_ids": ["550e8400-..."],
        "job_ids": ["a1b2c3d4-..."],
        "label_task_ids": ["t1b2c3d4-..."],
        "extra": {"author": "agent", "version": "1.0"}
    }'
    ```
    * Response: `ReportInfo` object (see `/get`).

- `POST /microsoft/apps/studios/reports/get`: Get the full content of a report.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/reports/get' \
    -H 'Content-Type: application/json' \
    -d '{"id": "r1b2c3d4-..."}'
    ```
    * Response:
    ```json
    {
        "id": "r1b2c3d4-...",
        "name": "Dataset Analysis Report",
        "description": "Weekly analysis of training data quality",
        "content": "# Dataset Analysis\n\n## Summary\n\nTotal samples: 10000 ...",
        "dataset_ids": ["550e8400-..."],
        "job_ids": ["a1b2c3d4-..."],
        "label_task_ids": ["t1b2c3d4-..."],
        "extra": {"author": "agent", "version": "1.0"},
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T01:00:00+00:00"
    }
    ```

- `POST /microsoft/apps/studios/reports/update`: Update a report. Only provided fields are overwritten; `extra` is merged into the existing dict.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/reports/update' \
    -H 'Content-Type: application/json' \
    -d '{
        "id": "r1b2c3d4-...",
        "content": "# Dataset Analysis\n\n## Summary (Updated)\n\nTotal samples: 12000 ...",
        "extra": {"version": "1.1"}
    }'
    ```
    * Response: updated `ReportInfo`.

- `POST /microsoft/apps/studios/reports/delete`: Delete a report.
    * Request:
    ```bash
    curl -X 'POST' '{api_base_url}/microsoft/apps/studios/reports/delete' \
    -H 'Content-Type: application/json' \
    -d '{"id": "r1b2c3d4-..."}'
    ```
    * Response:
    ```json
    {"id": "r1b2c3d4-...", "deleted": true}
    ```

###### Utils

- `POST /microsoft/apps/studios/utils/upload`: Upload a file to the studios.
    * Request:
    ```bash
    curl -X 'POST' \
    '{api_base_url}/microsoft/apps/studios/utils/upload' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@path_to_your_file'
    ```
    * Response:
    ```json
    {
        "path": "/tmp/studios_uploads/unique_file_name.ext",
        "filename": "original_file_name.ext",
        "size": 12345
    }
    ```