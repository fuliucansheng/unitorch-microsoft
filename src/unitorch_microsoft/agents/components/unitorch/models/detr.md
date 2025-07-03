---
config file: picasso/configs/detr.ini
description: End-to-end object detection using the DETR (DEtection TRansformers) model
  with the unitorch library.
title: DETR Model for Object Detection
---

# DETR Model Introduction


The DETR model was proposed in End-to-End Object Detection with Transformers which is a model designed for object detection tasks. It uses a transformer architecture to predict bounding boxes and class labels for objects in images. The model processes images and outputs predictions in the form of bounding boxes and associated class labels.


### Configuration for DETR model

`picasso/configs/detr.ini` in the unitorch library. Don't need to generate the file in local path. The content is as follows, You can override the parameter by `--section@option value` in comand line.

```ini
[core/cli]
task_name = core/task/supervised
depends_libraries = ['unitorch_microsoft.picasso.detr','unitorch_microsoft.models.detr']
from_ckpt_dir = ./cache
cache_dir = ./cache
train_file = ./train.tsv
dev_file = ./dev.tsv
test_file = ./test.tsv

# model
[microsoft/picasso/model/detection/detr]
pretrained_name = detr-resnet-50-picasso-roi-v2
num_classes = 2
norm_bboxes = True

# dataset
[core/dataset/ast]
names = ['image', 'bboxes', 'classes']

[core/dataset/ast/train]
data_files = ${core/cli:train_file}
preprocess_functions = ['core/process/detr/detection(core/process/image/read(image), eval(bboxes), eval(classes))']

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
preprocess_functions = ['core/process/detr/detection(core/process/image/read(image), eval(bboxes), eval(classes), do_eval=True)']

[core/dataset/ast/test]
names = ['image']
data_files = ${core/cli:test_file}
preprocess_functions = ['core/process/detr/image(core/process/image/read(image))']

# process
[core/process/detr]
pretrained_name = detr-resnet-50-picasso-roi-v2

[core/process/image]
http_url = http://0.0.0.0:11230/?file={0}

[core/optim/adamw]
learning_rate = 0.00001

[core/scheduler/linear_warmup]
num_warmup_rate = 0.001

# task
[core/task/supervised]
model = microsoft/picasso/model/detection/detr
optim = core/optim/adamw
scheduler = core/scheduler/linear_warmup
dataset = core/dataset/ast
score_fn = core/score/mAP
monitor_fns = ['core/score/mAP', 'core/score/mAP50']
output_header = ['image']
postprocess_fn = core/postprocess/detr/detection
writer = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt

epochs = 5000

train_batch_size = 8
dev_batch_size = 8
test_batch_size = 8
```

### Training

#### Dataset

The dataset for training, validation, and testing should be in TSV format with the following columns:
- `image`: URL or path to the image file.
- `bboxes`: A list of Bounding boxes in the format `[[bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2], [bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2]]` for each object in the image. 
- `classes`: A list of class labels corresponding to the bounding boxes in the format `['class1', 'class2']`. class1 and class2 should be the indexes of the classes which should be smaller than `num_classes` in model section.

> Note:
> Please set the `epochs` parameter to a suitable value based on your dataset size and training requirements. The default is set to 5000, which may be too large. You can override it in the command line like `--core/task/supervised@epochs 10`.\
> The `norm_bboxes` parameter in the model section is set to `True` by default for inference, which normalizes the bounding boxes to the range [0, 1]. During training, we have to set it to `False` to avoid normalization issues. You can set it in the command line like `--microsoft/picasso/model/detection/detr@norm_bboxes False`.
> No header in the tsv files.


#### Command

```bash
unitorch-train picasso/configs/detr.ini --train_file ./train.tsv --dev_file ./dev.tsv --cache_dir ./cache --core/process/image@http_url None --microsoft/picasso/model/detection/detr@norm_bboxes False
```

### Inference

For inference, you can use the following command. The `test_file` should contain only the `image` column with URLs or paths to the images you want to test.

> Note
> No header in the tsv files.

```bash
unitorch-infer picasso/configs/detr.ini --test_file ./test.tsv --cache_dir ./cache --core/process/image@http_url None
```

### References

- [unitorch](https://github.com/fuliucansheng/unitorch)

<!-- MACHINE_GENERATED -->

# DETR Model Documentation

## Introduction

The **DETR (DEtection TRansformers)** model is an end-to-end object detection architecture that leverages transformers to directly predict bounding boxes and class labels for objects in images. Unlike traditional object detectors, DETR eliminates the need for hand-crafted components such as anchor generation and non-maximum suppression, providing a streamlined and highly effective approach to object detection.

This documentation describes how to configure, train, and perform inference with the DETR model using the [unitorch](https://github.com/fuliucansheng/unitorch) library.

---

## Configuration

The DETR model is configured via the file:

```
picasso/configs/detr.ini
```

You do **not** need to generate this file manually; it is provided by the unitorch library. Configuration parameters can be overridden from the command line using the syntax: `--section@option value`.

### Key Configuration Sections

- **Model**: `[microsoft/picasso/model/detection/detr]`
    - `pretrained_name`: Name of the pretrained DETR model to use (default: `detr-resnet-50-picasso-roi-v2`)
    - `num_classes`: Number of object classes (default: `2`)
    - `norm_bboxes`: Whether to normalize bounding boxes to [0, 1] (default: `True`)
- **Dataset**: `[core/dataset/ast]`
    - `names`: Columns expected in the dataset (`image`, `bboxes`, `classes`)
- **Optimizer**: `[core/optim/adamw]`
    - `learning_rate`: Learning rate for training (default: `0.00001`)
- **Scheduler**: `[core/scheduler/linear_warmup]`
    - `num_warmup_rate`: Warmup rate for learning rate scheduling (default: `0.001`)
- **Task**: `[core/task/supervised]`
    - `epochs`: Number of training epochs (default: `5000`)
    - `train_batch_size`, `dev_batch_size`, `test_batch_size`: Batch sizes for training, validation, and testing (default: `8`)

---

# Instruction

## Training

### Dataset

The training and validation datasets must be in **TSV format** (tab-separated values) with **no header row**. Each row should contain:

- `image`: URL or local path to the image file.
- `bboxes`: A list of bounding boxes for objects in the image, formatted as `[[x1, y1, x2, y2], ...]` (pixel coordinates).
- `classes`: A list of class indices corresponding to each bounding box, e.g., `[0, 1, ...]`. Each index must be less than `num_classes` specified in the model configuration.

**Example row:**
```
/path/to/image.jpg	[[34, 45, 200, 300], [120, 80, 170, 130]]	[0, 1]
```

> **Note:**
> - The `epochs` parameter defaults to `5000`, which may be excessive for most datasets. Adjust as needed using the command line, e.g., `--core/task/supervised@epochs 10`.
> - The `norm_bboxes` parameter should be set to `False` during training to avoid normalization issues:
>   ```
>   --microsoft/picasso/model/detection/detr@norm_bboxes False
>   ```
> - Ensure there is **no header** in your TSV files.

### Command

To start training, use the following command:

```bash
unitorch-train picasso/configs/detr.ini \
  --train_file ./train.tsv \
  --dev_file ./dev.tsv \
  --cache_dir ./cache \
  --core/process/image@http_url None \
  --microsoft/picasso/model/detection/detr@norm_bboxes False
```

- `--train_file`: Path to your training TSV file.
- `--dev_file`: Path to your validation TSV file.
- `--cache_dir`: Directory for saving checkpoints and outputs.
- `--core/process/image@http_url None`: Disables HTTP image loading (use local files).
- `--microsoft/picasso/model/detection/detr@norm_bboxes False`: Disables bounding box normalization during training.

You can override any configuration parameter using the `--section@option value` syntax.

---

## Inference

### Dataset

For inference, the test dataset should also be in **TSV format** with **no header row**. Each row should contain only:

- `image`: URL or local path to the image file.

**Example row:**
```
/path/to/test_image.jpg
```

> **Note:** No header in the TSV file.

### Command

To run inference, use the following command:

```bash
unitorch-infer picasso/configs/detr.ini \
  --test_file ./test.tsv \
  --cache_dir ./cache \
  --core/process/image@http_url None
```

- `--test_file`: Path to your test TSV file.
- `--cache_dir`: Directory for outputs and checkpoints.
- `--core/process/image@http_url None`: Disables HTTP image loading (use local files).

---

## More Instructions

### Overriding Configuration Parameters

You can override any parameter in the configuration file directly from the command line using the format:

```
--section@option value
```

**Examples:**
- Change number of epochs: `--core/task/supervised@epochs 20`
- Set number of classes: `--microsoft/picasso/model/detection/detr@num_classes 5`

### Bounding Box Normalization

- **Training:** Set `norm_bboxes` to `False` to use raw pixel coordinates.
- **Inference:** Set `norm_bboxes` to `True` (default) to output normalized bounding boxes in the range [0, 1].

### Output

- Training and inference outputs are saved in the directory specified by `--cache_dir`.
- The output file path can be set via the `output_path` parameter in the configuration.

### References

- [unitorch GitHub Repository](https://github.com/fuliucansheng/unitorch)
- [DETR: End-to-End Object Detection with Transformers (Original Paper)](https://arxiv.org/abs/2005.12872)

---