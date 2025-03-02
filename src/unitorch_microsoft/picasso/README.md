# Usage

### Run OCR for local images

```bash
pip3 install paddlepaddle-gpu paddleocr
python3 -m unitorch_microsoft.picasso.ocr --data_file ./images.50.txt --names image --image_col image --output_file ./ocr.out.txt --http_url None
```

### Run BASNet for local images

```bash
python3 -m unitorch_microsoft.picasso.basnet --data_file ./images.50.txt --names image --image_col image --output_file ./basnet.out.txt --http_url None
```