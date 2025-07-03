---
config file: omnipixel/configs/image.siglip.lora.pos.ini, omnipixel/configs/image.siglip.lora.neg.ini
description: A LoRA-adapted Siglip model for 0/1 image classification using the unitorch
  library. Supports flexible label text and efficient training/inference workflows.
title: Siglip LORA Model for Binary Image Classification
---

# Siglip LORA Model Introduction

Siglip LORA Model is a image classifier for 0/1 label.

### Configuration for Siglip LORA model

`omnipixel/configs/image.siglip.lora.pos.ini` in the unitorch library. Don't need to generate the file in local path. The content is as follows, You can override the parameter by `--section@option value` in comand line.

```ini
[core/cli]
task_name = core/task/supervised
depends_libraries = ['unitorch_microsoft.models.siglip']
cache_dir = ./cache
from_ckpt_dir = ${core/cli:cache_dir}
train_file = ./train.tsv
dev_file = ./dev.tsv
test_file = ./test.tsv
label_text = positive
lora_r = 4

# model
[microsoft/model/matching/peft/lora/siglip]
pretrained_name = siglip2-so400m-patch14-384
lora_r = ${core/cli:lora_r}

# dataset
[core/dataset/ast]
# id, image, text
names = ['image', 'label']

[core/dataset/ast/train]
data_files = ${core/cli:train_file}
preprocess_functions = [
    'core/process/siglip/classification("${core/cli:label_text}", core/process/image/read(image))',
    'core/process/label(label)',
  ]

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
preprocess_functions = [
    'core/process/siglip/classification("${core/cli:label_text}", core/process/image/read(image))',
    'core/process/label(label)',
  ]

[core/dataset/ast/test]
names = ['image']
data_files = ${core/cli:test_file}
preprocess_functions = [
    'core/process/siglip/classification("${core/cli:label_text}", core/process/image/read(image))',
  ]

# process
[core/process/siglip]
pretrained_name = siglip2-so400m-patch14-384
max_seq_length = 48

[core/process/image]
http_url = http://0.0.0.0:11230/?file={0}

[core/process/classification]
act_fn = sigmoid

# optim
[core/optim/adamw]
learning_rate = 0.00001

# scheduler
[core/scheduler/linear_warmup]
num_warmup_rate = 0.001

# task
[core/task/supervised]
model = microsoft/model/matching/peft/lora/siglip
dataset = core/dataset/ast
optim = core/optim/adamw
scheduler = core/scheduler/linear_warmup
loss_fn = core/loss/bce
score_fn = core/score/auc
monitor_fns = ['core/score/auc', 'core/score/pr_auc']
output_header = ['image']
postprocess_fn = core/postprocess/classification/binary_score
writer = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt

num_workers = 16
epochs = 1000
log_freq = 100
save_checkpoint = best
save_optimizer = False
save_scheduler = False

train_batch_size = 4
dev_batch_size = 4
```

`omnipixel/configs/image.siglip.lora.neg.ini` in the unitorch library. Don't need to generate the file in local path. The content is as follows, You can override the parameter by `--section@option value` in comand line.

```ini
[core/cli]
task_name = core/task/supervised
depends_libraries = ['unitorch_microsoft.models.siglip']
cache_dir = ./cache
from_ckpt_dir = ${core/cli:cache_dir}
train_file = ./train.tsv
dev_file = ./dev.tsv
test_file = ./test.tsv
label_text = negative
lora_r = 4

# model
[microsoft/model/matching/peft/lora/siglip]
pretrained_name = siglip2-so400m-patch14-384
lora_r = ${core/cli:lora_r}

# dataset
[core/dataset/ast]
# id, image, text
names = ['image', 'label']

[core/dataset/ast/train]
data_files = ${core/cli:train_file}
preprocess_functions = [
    'core/process/siglip/classification("${core/cli:label_text}", core/process/image/read(image))',
    'core/process/label(1 - int(label))',
  ]

[core/dataset/ast/dev]
data_files = ${core/cli:dev_file}
preprocess_functions = [
    'core/process/siglip/classification("${core/cli:label_text}", core/process/image/read(image))',
    'core/process/label(1 - int(label))',
  ]

[core/dataset/ast/test]
names = ['image']
data_files = ${core/cli:test_file}
preprocess_functions = [
    'core/process/siglip/classification("${core/cli:label_text}", core/process/image/read(image))',
  ]

# process
[core/process/siglip]
pretrained_name = siglip2-so400m-patch14-384
max_seq_length = 48

[core/process/image]
http_url = http://0.0.0.0:11230/?file={0}

[core/process/classification]
act_fn = sigmoid

# optim
[core/optim/adamw]
learning_rate = 0.00001

# scheduler
[core/scheduler/linear_warmup]
num_warmup_rate = 0.001

# task
[core/task/supervised]
model = microsoft/model/matching/peft/lora/siglip
dataset = core/dataset/ast
optim = core/optim/adamw
scheduler = core/scheduler/linear_warmup
loss_fn = core/loss/bce
score_fn = core/score/auc
monitor_fns = ['core/score/auc', 'core/score/pr_auc']
output_header = ['image']
postprocess_fn = core/postprocess/classification/binary_score
writer = core/writer/csv

from_ckpt_dir = ${core/cli:from_ckpt_dir}
to_ckpt_dir = ${core/cli:cache_dir}
output_path = ${core/cli:cache_dir}/output.txt

num_workers = 16
epochs = 1000
log_freq = 100
save_checkpoint = best
save_optimizer = False
save_scheduler = False

train_batch_size = 4
dev_batch_size = 4
```

### Training

#### Dataset

The dataset for Siglip2 LORA Model should be in TSV (Tab-Separated Values) format with the following columns:
- `image`: path to the image file.
- `label`: a binary label (0 or 1) the image label from human.

> Note
> No header in the tsv files.

#### Command

```bash
unitorch-train omnipixel/configs/image.siglip.lora.pos.ini --label_text '{label_text}' --train_file ./train.tsv --dev_file ./dev.tsv --cache_dir ./siglip_lora --core/process/image@http_url None --lora_r 8
```

> Note
> The `label_text` should be the text for the positive label you want to train.

```bash
unitorch-train omnipixel/configs/image.siglip.lora.neg.ini --label_text '{label_text}' --train_file ./train.tsv --dev_file ./dev.tsv --cache_dir ./siglip_lora --core/process/image@http_url None --lora_r 8
```

> Note
> The `label_text` should be the text for the negative label you want to train.

### Inference

For inference, you can use the following command. The `test_file` should contain only the `image` column with URLs or paths to the images you want to test.

> Note
> No header in the tsv files.

```bash
unitorch-infer omnipixel/configs/image.siglip.lora.pos.ini --label_text '{label_text}' --test_file ./test.tsv --cache_dir ./cache --from_ckpt_dir ./siglip_lora --core/process/image@http_url None --lora_r 8
```

```bash
unitorch-infer omnipixel/configs/image.siglip.lora.neg.ini --label_text '{label_text}' --test_file ./test.tsv --cache_dir ./cache --from_ckpt_dir ./siglip_lora --core/process/image@http_url None --lora_r 8
```

### Instruction

1. Follow the `clip-interrogator` tool in **unitorch_docs** to get the best label text for training. Use training file (need both positive/negative samples) for label text search. Remove the quotes in the label text.
2. Train the classifier using the positive label text, please use the config file `omnipixel/configs/image.siglip.lora.pos.ini` and fill the text in the `label_text` in command line.
3. rain the classifier using the negative label text, please use the config file `omnipixel/configs/image.siglip.lora.neg.ini` and fill the text in the `label_text` in command line.
4. We usually train both config file `omnipixel/configs/image.siglip.lora.pos.ini`, `omnipixel/configs/image.siglip.lora.neg.ini` and finally choose the best one to use. Please remember to save ckpt to different folders.

### References

- [unitorch](https://github.com/fuliucansheng/unitorch)

<!-- MACHINE_GENERATED -->

# Siglip LORA Model

The **Siglip LORA Model** is a binary image classifier built on top of the Siglip vision-language model, enhanced with LoRA (Low-Rank Adaptation) for efficient fine-tuning. It is designed for 0/1 (binary) classification tasks, where the label can be flexibly defined by a user-supplied text prompt. The model is integrated with the [unitorch](https://github.com/fuliucansheng/unitorch) library and is configured via `.ini` files for both positive and negative label training.

---

## Functionality

- **Binary Image Classification**: Classifies images into two categories (0 or 1) based on a user-defined label text.
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using LoRA.
- **Flexible Labeling**: Supports custom label text for both positive and negative classes.
- **Easy Integration**: Uses unitorch's configuration-driven workflow for training and inference.

---

## Configuration

Two main configuration files are provided (do not generate or edit them manually; they are included in the unitorch library):

- `omnipixel/configs/image.siglip.lora.pos.ini` — for training with a **positive** label text.
- `omnipixel/configs/image.siglip.lora.neg.ini` — for training with a **negative** label text.

You can override any configuration parameter at the command line using the syntax: `--section@option value`.

### Key Configuration Options

| Section/Option                        | Description                                                                                  | Default Value (pos/neg)                |
|---------------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------|
| `[core/cli] label_text`               | The text prompt for the positive/negative label.                                             | `positive` / `negative`                |
| `[core/cli] train_file`               | Path to the training TSV file.                                                               | `./train.tsv`                          |
| `[core/cli] dev_file`                 | Path to the validation TSV file.                                                             | `./dev.tsv`                            |
| `[core/cli] test_file`                | Path to the test TSV file.                                                                   | `./test.tsv`                           |
| `[core/cli] cache_dir`                | Directory for cache and outputs.                                                             | `./cache`                              |
| `[core/cli] lora_r`                   | LoRA rank (can be overridden for different adaptation strength).                             | `4`                                    |
| `[microsoft/model/matching/peft/lora/siglip] pretrained_name` | Pretrained Siglip model name.                                                                | `siglip2-so400m-patch14-384`           |
| `[core/process/image] http_url`       | Image loading URL (set to `None` for local files).                                           | `http://0.0.0.0:11230/?file={0}`       |
| `[core/process/siglip] max_seq_length`| Maximum sequence length for text prompt.                                                     | `48`                                   |
| `[core/optim/adamw] learning_rate`    | Learning rate for AdamW optimizer.                                                           | `0.00001`                              |
| `[core/scheduler/linear_warmup] num_warmup_rate` | Warmup rate for scheduler.                                                        | `0.001`                                |
| `[core/task/supervised] epochs`       | Number of training epochs.                                                                   | `1000`                                 |
| `[core/task/supervised] train_batch_size` | Training batch size.                                                                     | `4`                                    |
| `[core/task/supervised] dev_batch_size`   | Validation batch size.                                                                   | `4`                                    |

---

## Training

### Dataset

- **Format**: TSV (Tab-Separated Values), **no header**.
- **Columns**:
  - `image`: Path to the image file (local path or URL).
  - `label`: Binary label (`0` or `1`), assigned by human annotators.

**Example:**
```
/path/to/image1.jpg	1
/path/to/image2.jpg	0
```

### Command

#### Positive Label Training

```bash
unitorch-train omnipixel/configs/image.siglip.lora.pos.ini \
  --label_text '{label_text}' \
  --train_file ./train.tsv \
  --dev_file ./dev.tsv \
  --cache_dir ./siglip_lora \
  --core/process/image@http_url None \
  --lora_r 8
```

- Replace `{label_text}` with the text prompt representing the **positive** class (e.g., `cat`, `dog`, etc.).
- `--core/process/image@http_url None` ensures images are loaded from local paths.

#### Negative Label Training

```bash
unitorch-train omnipixel/configs/image.siglip.lora.neg.ini \
  --label_text '{label_text}' \
  --train_file ./train.tsv \
  --dev_file ./dev.tsv \
  --cache_dir ./siglip_lora \
  --core/process/image@http_url None \
  --lora_r 8
```

- Replace `{label_text}` with the text prompt representing the **negative** class.

> **Note:**  
> - You can override any config parameter at the command line using `--section@option value`.
> - Save checkpoints for positive and negative models in different folders.

---

## Inference

### Dataset

- **Format**: TSV (Tab-Separated Values), **no header**.
- **Columns**:
  - `image`: Path or URL to the image file.

**Example:**
```
/path/to/image1.jpg
/path/to/image2.jpg
```

### Command

#### Positive Label Inference

```bash
unitorch-infer omnipixel/configs/image.siglip.lora.pos.ini \
  --label_text '{label_text}' \
  --test_file ./test.tsv \
  --cache_dir ./cache \
  --from_ckpt_dir ./siglip_lora \
  --core/process/image@http_url None \
  --lora_r 8
```

#### Negative Label Inference

```bash
unitorch-infer omnipixel/configs/image.siglip.lora.neg.ini \
  --label_text '{label_text}' \
  --test_file ./test.tsv \
  --cache_dir ./cache \
  --from_ckpt_dir ./siglip_lora \
  --core/process/image@http_url None \
  --lora_r 8
```

---

## Instructions & Best Practices

1. **Label Text Selection**:  
   - Use the `clip-interrogator` tool (see **unitorch_docs**) to find the most representative label text for your dataset. Use training file for optimal label text search.
   - Remove the quotes in the label text in the command line.

2. **Training Workflow**:
   - Train the classifier with the **positive** label text using `omnipixel/configs/image.siglip.lora.pos.ini`. The label_text in command line should be wrapped with single quotes.
   - Train the classifier with the **negative** label text using `omnipixel/configs/image.siglip.lora.neg.ini`. The label_text in command line should be wrapped with single quotes.
   - Save checkpoints for each model in separate directories.

3. **Model Selection**:  
   After training both positive and negative label models, evaluate their performance and select the best-performing model for deployment.

4. **Parameter Overrides**:  
   Adjust LoRA rank (`--lora_r`), batch size, learning rate, and other parameters as needed for your dataset and hardware.

---

## Input/Output Format

- **Input (Training/Validation)**: TSV file with `image` and `label` columns, no header.
- **Input (Inference)**: TSV file with only the `image` column, no header.
- **Output**:  
  - Predictions are written to a CSV file (default: `${cache_dir}/output.txt`), containing image paths and predicted scores.

---

## References

- [unitorch GitHub Repository](https://github.com/fuliucansheng/unitorch)

---