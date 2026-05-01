# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import ast
import logging
import fire
import numpy as np
import pandas as pd
from functools import partial
from transformers import BertTokenizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from unitorch.scores import (
    auc,
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    ndcg_score,
    matthews_corrcoef,
    pearsonr,
)
from unitorch.scores import bleu_score, rouge1_score, rouge2_score, rougel_score
from unitorch.scores import map50_score, map_score


def prauc_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


class WhiteSpaceTokenizer:
    def tokenize(self, text: str):
        return text.split()


def rouge_bleu_score(y_true, y_pred, tokenizer, score_fn):
    y_true = y_true.fillna("")
    y_pred = y_pred.fillna("")
    return score_fn(
        [[tokenizer.tokenize(t)] for t in y_true],
        [tokenizer.tokenize(t) for t in y_pred],
    )


def mae_score(y_true, y_pred):
    def convert(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return list(map(float, re.split(r"[,; ]", x)))

    if y_true.dtype == "object":
        return np.mean([mean_absolute_error(convert(yt), convert(yp)) for yt, yp in zip(y_true, y_pred)])
    return mean_absolute_error(y_true, y_pred)


def mse_score(y_true, y_pred):
    def convert(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return list(map(float, re.split(r"[,; ]", x)))

    if y_true.dtype == "object":
        return np.mean([mean_squared_error(convert(yt), convert(yp)) for yt, yp in zip(y_true, y_pred)])
    return mean_squared_error(y_true, y_pred)


def _bert_tokenizer(model: str):
    return BertTokenizer.from_pretrained(model)


METRICS = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc": roc_auc_score,
    "prauc": prauc_score,
    "ndcg": ndcg_score,
    "mae": mae_score,
    "mse": mse_score,
    "mattcorr": matthews_corrcoef,
    "pearsonr": pearsonr,
    "base-bleu": partial(rouge_bleu_score, tokenizer=WhiteSpaceTokenizer(), score_fn=bleu_score),
    "base-rouge1": partial(rouge_bleu_score, tokenizer=WhiteSpaceTokenizer(), score_fn=rouge1_score),
    "base-rouge2": partial(rouge_bleu_score, tokenizer=WhiteSpaceTokenizer(), score_fn=rouge2_score),
    "base-rougel": partial(rouge_bleu_score, tokenizer=WhiteSpaceTokenizer(), score_fn=rougel_score),
}

_BERT_MODELS = {
    "bert": "bert-base-cased",
    "mbert": "bert-base-multilingual-cased",
}
_SCORE_FNS = {
    "bleu": bleu_score,
    "rouge1": rouge1_score,
    "rouge2": rouge2_score,
    "rougel": rougel_score,
}
for _prefix, _model_name in _BERT_MODELS.items():
    _tok = _bert_tokenizer(_model_name)
    for _sfx, _fn in _SCORE_FNS.items():
        METRICS[f"{_prefix}-{_sfx}"] = partial(rouge_bleu_score, tokenizer=_tok, score_fn=_fn)


def main(
    data_file: str,
    metrics: str,
    y_true_col: str,
    y_pred_col: str,
    output_file: str = "./output.txt",
    names: str = None,
    escapechar: str = None,
):
    assert os.path.exists(data_file), f"data_file not found: {data_file}"

    metric_list = [m.strip() for m in re.split(r"[,;]", metrics)]
    unknown = [m for m in metric_list if m not in METRICS]
    assert not unknown, f"Unsupported metrics: {unknown}. Supported: {list(METRICS)}"

    if isinstance(names, str) and names.strip() == "*":
        names = None
    elif isinstance(names, str):
        names = [n.strip() for n in re.split(r"[,;]", names)]

    data = pd.read_csv(
        data_file,
        names=names,
        sep="\t",
        quoting=3,
        header="infer" if names is None else None,
        escapechar=escapechar,
    )

    cols = data.columns.tolist()
    assert y_true_col in cols, f"y_true_col '{y_true_col}' not in columns: {cols}"
    assert y_pred_col in cols, f"y_pred_col '{y_pred_col}' not in columns: {cols}"

    y_true = data[y_true_col]
    y_pred = data[y_pred_col]

    if y_true.dtype == "object":
        y_true = y_true.fillna("")
    if y_pred.dtype == "object":
        y_pred = y_pred.fillna("")

    results = []
    for metric in metric_list:
        score = METRICS[metric](y_true, y_pred)
        logging.info(f"Metric {metric}: {score}")
        results.append([metric, score])

    pd.DataFrame(results, columns=["metric", "score"]).to_csv(
        output_file, sep="\t", index=False, quoting=3,
    )


if __name__ == "__main__":
    fire.Fire(main)
