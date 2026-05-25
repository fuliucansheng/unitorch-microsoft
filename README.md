<div align="center">

![unitorch microsoft](unitorch.png)

[Documentation](https://fuliucansheng.github.io/unitorch) •
[Installation](https://fuliucansheng.github.io/unitorch/installation/) •
[Report Issues](https://github.com/fuliucansheng/unitorch/issues/new?assignees=&labels=&template=bug-report.yml)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unitorch_microsoft)](https://pypi.org/project/unitorch_microsoft/)
[![PyPI Version](https://badge.fury.io/py/unitorch_microsoft.svg)](https://badge.fury.io/py/unitorch_microsoft)
[![PyPI Downloads](https://pepy.tech/badge/unitorch_microsoft)](https://pepy.tech/project/unitorch_microsoft)
[![License](https://img.shields.io/github/license/fuliucansheng/unitorch?color=dfd)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-pink.svg)](https://github.com/fuliucansheng/unitorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22)

</div>

## Introduction

🔥 **unitorch_microsoft** is a Microsoft extension library for [unitorch](https://fuliucansheng.github.io/unitorch) that adds state-of-the-art models and domain-specific pipelines from Ads & Microsoft teams. It covers NLU, NLG, computer vision, CTR prediction, multimodal learning, and more — built on PyTorch with seamless integration into [transformers](https://github.com/huggingface/transformers), [peft](https://github.com/huggingface/peft), and [diffusers](https://github.com/huggingface/diffusers).

Drop in with a single import: `import unitorch_microsoft`.

## Features

| | |
|---|---|
| **Microsoft Domain Models** | Bletchley, TriBERT, TULR, MMDNN, Kolors MPS, and more |
| **Multi-domain Coverage** | Ads (AdInsights, Ads+, Product Ads), vision (Picasso, VPR), generative AI (OmniGPT) |
| **Configuration-Driven CLI** | Train, evaluate, infer, and serve via `.ini` config files inherited from unitorch |
| **Multi-GPU & Distributed** | Native `torchrun` support + optional DeepSpeed integration |
| **PEFT / LoRA** | Built-in parameter-efficient fine-tuning via unitorch |
| **Model Serving** | FastAPI-based serving with `unitorch-fastapi` |
| **AI Copilots** | Copilot flows and component tools for agent-driven workflows |

## Installation

```bash
pip install unitorch_microsoft
```

<details>
<summary>Optional extras</summary>

```bash
pip install "unitorch_microsoft[all]"          # everything
pip install "unitorch_microsoft[deepspeed]"    # distributed training
pip install "unitorch_microsoft[diffusers]"    # image generation models
pip install "unitorch_microsoft[copilots]"     # copilot tools
pip install "unitorch_microsoft[others]"       # Azure, FastAPI, Gradio, etc.
```

Requires **Python >= 3.10** and **PyTorch 2.5+**.
</details>

## Quick Start

**Python API**
```python
import unitorch_microsoft

# Use any registered microsoft model via unitorch's config system
from unitorch.cli import Config
config = Config("path/to/config.ini")
```

**Multi-GPU Training**
```bash
torchrun --no_python --nproc_per_node 4 \
    unitorch-train examples/configs/generation/bart.ini \
    --train_file path/to/train.tsv --dev_file path/to/dev.tsv
```

**Inference**
```bash
unitorch-infer examples/configs/generation/bart.ini --test_file path/to/test.tsv
```

> See the [documentation](https://fuliucansheng.github.io/unitorch) for full tutorials and examples.

## Domain Modules

<details>
<summary>View all domain modules</summary>

| Domain | Description |
|--------|-------------|
| **adinsights** | Ad Insights — generation, relevance scoring, sensitive image detection, video understanding |
| **adsplus** | Ads Plus — click prediction, image retrieval, ad selection, SLAB |
| **pa** | Product Ads — click/selection/retrieval, international, L2 ranking |
| **picasso** | Picasso Image — classifiers, image matting, video processing, MSAN |
| **omnigpt / omnilora / omnipixel** | Experimental generative and vision modules |
| **copilots** | AI copilot flows and component tools |
| **vpr** | Visual Product Recommandation |

</details>

<details>
<summary>View core models</summary>

| Domain | Models |
|--------|--------|
| **Language** | BERT, RoBERTa, DeBERTa/V2, T5, BART, PEGASUS, mT5, MBart, BLOOM, LLaMA |
| **Vision** | ViT, BEiT, Swin Transformer, CLIP, SigLIP, DINOv2 |
| **Multimodal** | BLIP, VisualBERT, LLaVA |
| **Image Generation** | FLUX, Kolors |
| **Microsoft-specific** | Bletchley, TriBERT, TULR, MMDNN, SAM, Mask2Former |

</details>

## CLI Commands

| Command | Purpose |
|---------|---------|
| `unitorch-train` | Train models (supports `torchrun`) |
| `unitorch-eval` | Evaluate models |
| `unitorch-infer` | Run batch inference |
| `unitorch-launch` | Launch a quick script defined in config |
| `unitorch-fastapi` | Start a FastAPI model server |
| `unitorch-service` | Run a background service |
| `unitorch-copilot` | unitorch-native agent (similar to Claude / OpenCode) |
| `unitorch-copilot-cli` | CLI tool for agent use — invokes registered copilot tools |

## License

Released under the [MIT License](LICENSE).
