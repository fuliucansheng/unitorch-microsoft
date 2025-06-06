# WebUI Tools

### Classification Labeling

```bash
unitorch-webui configs/labeling/classification.ini --data_file ./domaint_color.tsv --result_file ./domaint_color.result.tsv --names "image" --image_cols "image" --choices "good;bad" --port 13505
```

Notes:
> Add ` UNITORCH_MS_FASTAPI_ENDPOINT=http://decu-pc2:5432` if you want to record the labeling results to unitorch-microsoft data collector.
> Add ` --tags "#Tag1#Tag2"` if you want to add tags to the labeling results for recording.
