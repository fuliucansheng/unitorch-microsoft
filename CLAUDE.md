# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`unitorch_microsoft` is a Microsoft extension library for [unitorch](https://fuliucansheng.github.io/unitorch) that adds state-of-the-art models from Ads & Microsoft domains. It covers NLU, NLG, computer vision, CTR prediction, multimodal learning, and more. Built on PyTorch, integrating with transformers, deepspeed, diffusers, and detectron2.

- **Python**: 3.10, 3.11, 3.12
- **Package source**: `src/unitorch_microsoft/`
- **Version**: defined in `src/unitorch_microsoft/__init__.py` as `VERSION`

## Build & Install

```bash
# Install (production)
pip install .

# Install without dependencies (used in CI)
pip install --no-deps .

# Install with optional extras
pip install ".[deepspeed]"    # distributed training
pip install ".[diffusers]"    # image generation
pip install ".[agents]"       # agent tools
pip install ".[others]"       # Azure, FastAPI, Gradio, etc.
pip install ".[all]"          # everything
```

Dependencies are in `requirements.txt` (core) and `pyproject.toml` (optional groups).

## CLI Entry Points

All CLI tools accept an INI config file as the first argument, with overridable params via kwargs:

```bash
unitorch-train <config.ini> [--key=value ...]
unitorch-eval <config.ini> [--key=value ...]
unitorch-infer <config.ini> [--key=value ...]
unitorch-launch <config.ini> [--key=value ...]
unitorch-webui <config.ini>
unitorch-fastapi <config.ini>
unitorch-service <config.ini>
```

Multi-GPU training uses `torchrun`:
```bash
torchrun --no_python --nproc_per_node 4 unitorch-train <config.ini> --train_file path/to/train.tsv
```

## Architecture

### Configuration System (INI-based)

The framework is configuration-driven. INI files (in `src/unitorch_microsoft/configs/`) define the full pipeline: model, dataset, preprocessing, loss, scoring, and task orchestration. Sections follow a namespaced convention:

- `[core/cli]` — task name, file paths, dependent libraries
- `[microsoft/model/...]` — model configuration with `pretrained_name`
- `[core/dataset/ast]` — dataset schema and preprocessing functions
- `[core/task/...]` — training task: model, dataset, loss, score, batch sizes, checkpointing
- Cross-references use `${section:key}` interpolation

### Module Registration Pattern

Models and processors register themselves using decorators from `unitorch.cli`:
- `@add_default_section_for_init` — binds a class to a config section name
- `@register_process` — registers a preprocessing function
- `@register_score` — registers a scoring function
- Classes use `from_core_configure(cls, config, **kwargs)` classmethods for INI-based instantiation

### Import Control

- `UNITORCH_MS_SKIP_IMPORT=True` — skips all imports (used in CI for version checks)
- `UNITORCH_DEBUG=ALL` — imports all domain modules; otherwise only core modules (scores, models, modules, scripts, services) load by default
- Domain-specific modules are imported on demand via `depends_libraries` in INI configs

### Source Layout (`src/unitorch_microsoft/`)

**Core infrastructure:**
- `consoles/` — CLI entry points (train, eval, infer, launch, webui, fastapi, service)
- `models/` — Model implementations (bletchley, bloom, llama, llava, sam, siglip, detr, tribert, tulr, mmdnn, dinov2, diffusers, kolors, mask2former) plus shared utils
- `modules/` — Utility modules (beam search, etc.)
- `configs/` — INI configuration templates
- `scores/` — Evaluation metrics
- `services/` — File hosting/mirroring

**Domain modules:**
- `adinsights/` — Ad insights Team Project (generation, relevance, sensitive images, video)
- `adsplus/` — Ads Plus Team Project (click prediction, image retrieval, selection, SLAB)
- `pa/` — Product Ads Team Project (click, selection, retrieval, intl, l2)
- `picasso/` — Picasso Image Project (classifiers, matting, video, MSAN)
- `deepgen/` — DeepGen Team Project (finetuning/inference)
- `omnigpt/`, `omnilora/`, `omnipixel/` — Experimental modules
- `agents/` — AI agents with flows and component tools
- `vpr/` — Visual Place Recognition
- `fastapis/`, `webuis/`, `spaces/` — Service and UI layers

### Key Dependencies

- `unitorch` (>=0.0.1.6) — base framework providing CLI, task runners, and model scaffolding
- `transformers` (==5.0.0) — HuggingFace model backbone
- `peft` (>=0.17.0) — parameter-efficient fine-tuning (LoRA, etc.)
- `deepspeed` (optional) — distributed training

## Code Conventions

- All files carry `# Copyright (c) MICROSOFT.` / `# Licensed under the MIT License.` headers
- Type hints used throughout
- `cached_path()` is monkey-patched in `__init__.py` to also resolve paths within the `unitorch_microsoft` package via `importlib_resources`

## CI/CD

Azure Pipelines on `master` branch (`azure-pipelines.yml`):
1. Cleans disk space
2. Installs package with `--no-deps`
3. Reads `VERSION` and creates/updates a git tag

No automated test suite in CI — testing is done via the CLI tools against config files and data.
