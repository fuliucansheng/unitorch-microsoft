---
config file: configs/interrogator/clip.ini
description: Analyze images using the CLIP model to determine the most representative
  prompts for binary labels, aiding in image classification and prompt engineering.
title: Clip Interrogator
---

# Clip Interrogator


The Clip Interrogator is a tool that uses the CLIP model to analyze images and the label 0/1 to find the typicial prompt for the positive(1)/negative(0) label. It is useful for understanding what's the best prompt to represent the label of the image. This tool is particularly useful for tasks like image classification, where you want to understand the features that distinguish different classes.


### Configuration for Clip Interrogator

`configs/interrogator/clip.ini` is a file in unitorch library. Don't need to generate the file in local path. The content is as follows, You can override the parameter by `--section@option value` in comand line.

```ini
[core/cli]
depends_libraries = ['unitorch_microsoft.interrogators']
script_name = microsoft/script/interrogator/clip
data_file = ./data.tsv
names = ['image', 'label']
image_col = image
label_col = label
do_reverse = False
device = cpu

[microsoft/script/interrogator/clip]
data_file = ${core/cli:data_file}
names = ${core/cli:names}
image_col = ${core/cli:image_col}
label_col = ${core/cli:label_col}
do_reverse = ${core/cli:do_reverse}

[microsoft/interrogator/clip]
pretrained_name = clip-vit-large-patch14
device = ${core/cli:device}
```

### Usage

#### Dataset

The dataset for Clip Interrogator should be in TSV (Tab-Separated Values) format with the following columns:
- `image`: path to the image file.
- `label`: a binary label (0 or 1) the image label from human.

> Note
> No header in the tsv files.

#### Command

```bash
unitorch-launch configs/interrogator/clip.ini --data_file ./path/to/data.tsv --device 0 --do_reverse True
```

> Note
> The `do_reverse` option is used to reverse the labels, i.e., if set to `True`, it will treat label 0 as the positive label and label 1 as the negative label. If set to `False`, it will treat label 1 as the positive class and label 0 as the negative class.
> The output will be in the stderr, which is formatted as following. `Best Prompt` is the best prompt to represent the positive label, and `Negative Prompt` is the prompt that represents the negative label.
> We usually run this tool twice, once with `do_reverse` set to `True` and once with it set to `False`, to get both the positive and negative prompts.
```txt
<logger> ...
<logger> Best Prompt: <best_prompt>
<logger> Negative Prompt: <negative_prompt>
```

We will get the prompts for labels.
Positive label: Best Prompt (from do_reverse=False) and Negative Prompt (from do_reverse=True)
Negative label: Best Prompt (from do_reverse=True) and Negative Prompt (from do_reverse=False)

Summerize the texts for Positive/Negative label and refine it to a better one.

### References

- [unitorch](https://github.com/fuliucansheng/unitorch)

<!-- MACHINE_GENERATED -->

# Clip Interrogator

The **Clip Interrogator** is a tool that leverages the CLIP model to analyze images and their associated binary labels (0/1), identifying the most representative prompt for each label. This is particularly useful for understanding which textual prompts best describe the distinguishing features of images in each class, making it valuable for tasks such as image classification, dataset exploration, and prompt engineering.

---

## Functionality

- **Image Analysis**: Uses the CLIP model to interpret images and associate them with descriptive prompts.
- **Label Prompt Discovery**: Determines the best prompt for both positive (1) and negative (0) labels.
- **Label Reversal**: Supports reversing label interpretation to extract both positive and negative prompts.
- **Dataset Support**: Works with TSV datasets containing image paths and binary labels.

---

## Configuration

The Clip Interrogator is configured via the `configs/interrogator/clip.ini` file (provided by the [unitorch](https://github.com/fuliucansheng/unitorch) library). You do **not** need to generate this file manually. Configuration parameters can be overridden via command-line arguments using the `--section@option value` syntax.

### Default Configuration (`configs/interrogator/clip.ini`)

```ini
[core/cli]
depends_libraries = ['unitorch_microsoft.interrogators']
script_name = microsoft/script/interrogator/clip
data_file = ./data.tsv
names = ['image', 'label']
image_col = image
label_col = label
do_reverse = False
device = cpu

[microsoft/script/interrogator/clip]
data_file = ${core/cli:data_file}
names = ${core/cli:names}
image_col = ${core/cli:image_col}
label_col = ${core/cli:label_col}
do_reverse = ${core/cli:do_reverse}

[microsoft/interrogator/clip]
pretrained_name = clip-vit-large-patch14
device = ${core/cli:device}
```

#### Key Configuration Options

| Option                | Description                                                                 | Default Value                |
|-----------------------|-----------------------------------------------------------------------------|------------------------------|
| `data_file`           | Path to the TSV dataset file                                                | `./data.tsv`                 |
| `image_col`           | Name of the column containing image paths                                   | `image`                      |
| `label_col`           | Name of the column containing binary labels                                 | `label`                      |
| `do_reverse`          | If `True`, reverses label interpretation (0 as positive, 1 as negative)     | `False`                      |
| `device`              | Device to run the model on (`cpu` or GPU index, e.g., `0`)                  | `cpu`                        |
| `pretrained_name`     | Name of the pretrained CLIP model                                           | `clip-vit-large-patch14`     |

---

## Dataset Format

- **Format**: TSV (Tab-Separated Values)
- **Columns**:
  - `image`: Path to the image file.
  - `label`: Binary label (`0` or `1`) assigned by a human annotator.
- **Header**: **No header row** should be present in the TSV file.

**Example:**
```
/path/to/image1.jpg    1
/path/to/image2.jpg    0
```

---

## Usage

### Command-Line Usage with `unitorch-launch`

```bash
unitorch-launch configs/interrogator/clip.ini --data_file ./path/to/data.tsv --device 0 --do_reverse True
```

- Override configuration options as needed using `--section@option value`.
- Set `--do_reverse True` to swap the interpretation of positive and negative labels.

#### Typical Workflow

1. **Run with `do_reverse=False`**  
   - Finds the best prompt for the positive label (`1`).
2. **Run with `do_reverse=True`**  
   - Finds the best prompt for the negative label (`0`).

> **Tip:** Run the tool twice (once with each `do_reverse` setting) to obtain both positive and negative prompts for your dataset.

### Output

- The results are printed to **stderr** in the following format:
    ```
    <logger> ...
    <logger> Best Prompt: <best_prompt>
    <logger> Negative Prompt: <negative_prompt>
    ```
- **Best Prompt**: The most representative prompt for the positive label (as defined by `do_reverse`).
- **Negative Prompt**: The most representative prompt for the negative label.

#### Interpreting Results

- **Positive Label**:
  - `Best Prompt` from `do_reverse=False`
  - `Negative Prompt` from `do_reverse=True`
- **Negative Label**:
  - `Best Prompt` from `do_reverse=True`
  - `Negative Prompt` from `do_reverse=False`

This dual-run approach ensures you obtain refined, representative prompts for both classes.

---

## Summary

The Clip Interrogator provides a systematic way to extract and refine textual prompts that best describe the distinguishing features of images in binary-labeled datasets. By leveraging the CLIP model and supporting flexible configuration, it is a powerful tool for prompt engineering, dataset analysis, and improving image classification workflows.

---

## References

- [unitorch](https://github.com/fuliucansheng/unitorch)