# unitorch_microsoft.aether

Aether is a collection of data-processing scripts for batch LLM inference and evaluation. Each script is a self-contained Python module runnable via:

```bash
python3.10 -m unitorch_microsoft.aether.<script> [--arg=value ...]
```

---

## Scripts

### `dv3` — Build DaVinci-v3 batch request JSONL

Reads a TSV data file, formats each row into an Azure OpenAI completions-style JSONL request, and writes the results to an output file.

```bash
python3.10 -m unitorch_microsoft.aether.dv3 \
    --data_file=input.tsv \
    --prompt_text="Summarize: {title}" \
    --output_file=output.jsonl
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data_file` | str | **required** | Path to the input TSV file |
| `output_file` | str | `./output.txt` | Path to the output JSONL file |
| `prompt_file` | str | `None` | Path or URL to a prompt template file (mutually exclusive with `prompt_text`) |
| `prompt_text` | str | `None` | Inline prompt template string |
| `names` | str | `None` | Comma/semicolon-separated column names, or `*` to infer from header |
| `index_col` | str | `None` | Column to use as `ConversationId`; defaults to auto-incrementing index |
| `chunksize` | int | `1000` | Rows per processing chunk |
| `max_tokens` | int | `200` | Maximum tokens in the completion |
| `temperature` | float | `0` | Sampling temperature |
| `presence_penalty` | float | `0` | Presence penalty |
| `frequency_penalty` | float | `0` | Frequency penalty |
| `top_p` | float | `1` | Top-p sampling |
| `stop` | str | `None` | Stop sequence |
| `freq` | int | `20` | Log progress every N chunks |
| `input_escapechar` | str | `None` | Escape character for input CSV |
| `output_escapechar` | str | `None` | Escape character for output CSV |
| `output_header` | bool | `False` | Write column header to output |

---

### `gpt4o` — Build GPT-4o chat batch request JSONL

Formats each row as a chat `messages` array for the Azure OpenAI chat completions API.

```bash
python3.10 -m unitorch_microsoft.aether.gpt4o \
    --data_file=input.tsv \
    --prompt_text="Classify the sentiment of: {text}" \
    --system_prompt="You are a helpful assistant." \
    --output_file=output.jsonl
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data_file` | str | **required** | Path to the input TSV file |
| `output_file` | str | `./output.txt` | Path to the output JSONL file |
| `prompt_file` | str | `None` | Path or URL to a prompt template file |
| `prompt_text` | str | `None` | Inline prompt template string |
| `system_prompt` | str | `None` | Optional system message prepended to every request |
| `names` | str | `None` | Column names or `*` to infer from header |
| `index_col` | str | `None` | Column to use as `ConversationId` |
| `chunksize` | int | `1000` | Rows per processing chunk |
| `max_tokens` | int | `200` | Maximum tokens in the completion |
| `temperature` | float | `0` | Sampling temperature |
| `top_p` | float | `1` | Top-p sampling |
| `presence_penalty` | float | `0.1` | Presence penalty |
| `frequency_penalty` | float | `0.1` | Frequency penalty |
| `input_escapechar` | str | `None` | Escape character for input CSV |
| `output_escapechar` | str | `None` | Escape character for output CSV |
| `output_header` | bool | `False` | Write column header to output |

---

### `gpt4o_v` — Build GPT-4o Vision batch request JSONL

Extends `gpt4o` with image inputs. Each image column is read from disk, URL, or base64 and embedded as a `data:image/jpeg;base64,...` URL in the request.

```bash
python3.10 -m unitorch_microsoft.aether.gpt4o_v \
    --data_file=input.tsv \
    --prompt_text="Describe this image: {caption}" \
    --images=image_col \
    --output_file=output.jsonl
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data_file` | str | **required** | Path to the input TSV file |
| `output_file` | str | `./output.txt` | Path to the output JSONL file |
| `prompt_file` | str | `None` | Path or URL to a prompt template file |
| `prompt_text` | str | `None` | Inline prompt template string |
| `system_prompt` | str | `None` | Optional system message |
| `images` | str | `None` | Comma/semicolon-separated column names containing image paths, URLs, or base64 data |
| `names` | str | `None` | Column names or `*` to infer from header |
| `index_col` | str | `None` | Column to use as `ConversationId` |
| `chunksize` | int | `1000` | Rows per processing chunk |
| `max_tokens` | int | `200` | Maximum tokens |
| `temperature` | float | `0` | Sampling temperature |
| `top_p` | float | `1` | Top-p sampling |
| `presence_penalty` | float | `0.1` | Presence penalty |
| `frequency_penalty` | float | `0.1` | Frequency penalty |
| `input_escapechar` | str | `None` | Escape character for input CSV |
| `output_escapechar` | str | `None` | Escape character for output CSV |
| `output_header` | bool | `False` | Write column header to output |

---

### `gptv` — Build GPT-V (legacy vision) batch request JSONL

Formats requests using the legacy `transcript`-style vision API (GPT-4V / Turing-MM), embedding images as raw base64 data rather than URL wrappers.

```bash
python3.10 -m unitorch_microsoft.aether.gptv \
    --data_file=input.tsv \
    --prompt_text="What is in this image?" \
    --images=image_col \
    --output_file=output.jsonl
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data_file` | str | **required** | Path to the input TSV file |
| `output_file` | str | `./output.txt` | Path to the output JSONL file |
| `prompt_file` | str | `None` | Path or URL to a prompt template file |
| `prompt_text` | str | `None` | Inline prompt template string |
| `images` | str | `None` | Comma/semicolon-separated column names for image inputs |
| `names` | str | `None` | Column names or `*` to infer from header |
| `index_col` | str | `None` | Column to use as `ConversationId` |
| `chunksize` | int | `1000` | Rows per processing chunk |
| `max_tokens` | int | `200` | Maximum tokens |
| `temperature` | float | `0` | Sampling temperature |
| `presence_penalty` | float | `0` | Presence penalty |
| `frequency_penalty` | float | `0` | Frequency penalty |
| `top_p` | float | `1` | Top-p sampling |
| `stop` | str | `None` | Stop sequence |
| `freq` | int | `20` | Log progress every N chunks |
| `input_escapechar` | str | `None` | Escape character for input CSV |
| `output_escapechar` | str | `None` | Escape character for output CSV |
| `output_header` | bool | `False` | Write column header to output |

---

### `parser` — Parse Azure OpenAI batch response JSONL

Parses the raw JSONL response file returned by the Azure OpenAI batch API. Extracts the answer text, token counts, and optionally structured tags from the response.

```bash
python3.10 -m unitorch_microsoft.aether.parser \
    --data_file=responses.jsonl \
    --result_tags="label,reason" \
    --output_file=parsed.tsv
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data_file` | str | **required** | Path to the raw batch response JSONL file |
| `output_file` | str | `./output.txt` | Path to the parsed output TSV |
| `chunksize` | int | `1000` | Rows per processing chunk |
| `freq` | int | `20` | Log progress every N chunks |
| `input_escapechar` | str | `None` | Escape character for input |
| `output_escapechar` | str | `None` | Escape character for output |
| `result_tags` | str | `None` | Comma/semicolon-separated XML tag names to extract from the answer (e.g. `label,reason`) |
| `result_sep` | str | `;` | Separator used when multiple tag matches are found |
| `output_header` | bool | `False` | Write column header on the first chunk |
| `output_tokens` | bool | `True` | Include token count columns in output |

Output columns: `index`, `model`, `answer` (plus `result_<tag>` for each tag in `result_tags`, plus token columns if `output_tokens=True`).

---

### `metrics` — Evaluate predictions against ground truth

Computes one or more evaluation metrics from a TSV file containing prediction and ground-truth columns.

```bash
python3.10 -m unitorch_microsoft.aether.metrics \
    --data_file=predictions.tsv \
    --metrics=auc,f1 \
    --y_true_col=label \
    --y_pred_col=score \
    --output_file=metrics.tsv
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `data_file` | str | **required** | Path to the predictions TSV file |
| `metrics` | str | **required** | Comma/semicolon-separated metric names |
| `y_true_col` | str | **required** | Column name for ground-truth values |
| `y_pred_col` | str | **required** | Column name for predicted values |
| `output_file` | str | `./output.txt` | Path to the metrics output TSV |
| `names` | str | `None` | Column names or `*` to infer from header |
| `escapechar` | str | `None` | Escape character for the input CSV |

**Supported metrics:**

| Name | Description |
|---|---|
| `accuracy` | Accuracy score |
| `recall` | Recall score |
| `f1` | F1 score |
| `auc` | ROC-AUC score |
| `prauc` | Precision-Recall AUC |
| `ndcg` | Normalized Discounted Cumulative Gain |
| `mae` | Mean Absolute Error |
| `mse` | Mean Squared Error |
| `mattcorr` | Matthews Correlation Coefficient |
| `pearsonr` | Pearson correlation |
| `base-bleu` | BLEU (whitespace tokenizer) |
| `base-rouge1` | ROUGE-1 (whitespace tokenizer) |
| `base-rouge2` | ROUGE-2 (whitespace tokenizer) |
| `base-rougel` | ROUGE-L (whitespace tokenizer) |
| `bert-bleu` | BLEU (bert-base-cased tokenizer) |
| `bert-rouge1` | ROUGE-1 (bert-base-cased tokenizer) |
| `bert-rouge2` | ROUGE-2 (bert-base-cased tokenizer) |
| `bert-rougel` | ROUGE-L (bert-base-cased tokenizer) |
| `mbert-bleu` | BLEU (bert-base-multilingual-cased tokenizer) |
| `mbert-rouge1` | ROUGE-1 (bert-base-multilingual-cased tokenizer) |
| `mbert-rouge2` | ROUGE-2 (bert-base-multilingual-cased tokenizer) |
| `mbert-rougel` | ROUGE-L (bert-base-multilingual-cased tokenizer) |

---

### `pandas` — Flexible TSV transformation via pandas expressions

Loads up to five input TSV tables and executes a sequence of pandas expressions to produce up to three output TSV files. Useful for joins, filters, aggregations, and feature engineering without writing a custom script.

```bash
python3.10 -m unitorch_microsoft.aether.pandas \
    --input1_file=table_a.tsv \
    --input2_file=table_b.tsv \
    --action1="Output = Input1.merge(Input2, on='id')" \
    --output1_file=merged.tsv
```

**Input tables** are available as `Input1` … `Input5` in all expressions.

| Argument | Type | Default | Description |
|---|---|---|---|
| `input1_file` … `input5_file` | str | `None` | Paths to up to five input TSV files |
| `input1_names` … `input5_names` | str | `*` | Column names or `*` to infer from header |
| `input_escapechar` | str | `None` | Escape character for input files |
| `output_escapechar` | str | `\\` | Escape character for output files |
| `output_header` | bool | `False` | Write column header to output files |
| `function1` … `function4` | str | `#` | Python function definitions (use `\n` for newlines) |
| `global_action1` … `global_action4` | str | `#` | Setup expressions run before the action pipeline |
| `action1` … `action4` | str | `#` | Transformation expressions for output 1 |
| `action_output1` | str | `#` | Final expression whose result is saved to `output1_file` |
| `action5`, `action6` | str | `#` | Transformation expressions for output 2 |
| `action_output2` | str | `#` | Final expression whose result is saved to `output2_file` |
| `action_output3` | str | `#` | Expression whose result is saved to `output3_file` |
| `output1_file` | str | `./output1.txt` | Path for the first output file |
| `output2_file` | str | `./output2.txt` | Path for the second output file |
| `output3_file` | str | `./output3.txt` | Path for the third output file |

---

## Prompt Templates

All prompt arguments support Python `str.format`-style placeholders referencing column names:

```
Translate the following text to French: {text}
```

The special token `#endl#` is replaced with a newline character `\n` at runtime.
