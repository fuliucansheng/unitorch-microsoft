# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import io
import re
import json
import base64
import zipfile
import logging
import hashlib
import numpy as np
import pandas as pd
from functools import partial
from transformers import BertTokenizer
from unitorch.score import (
    auc,
    accuracy_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    ndcg_score,
    matthews_corrcoef,
    pearsonr,
)
from unitorch.score import bleu_score, rouge1_score, rouge2_score, rougel_score
from unitorch.score import map50_score, map_score
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path


def prauc_score(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


def rouge_bleu_score(y_true, y_pred, tokenizer, score_fn):
    y_true = [tokenizer.tokenize(text) for text in y_true]
    y_pred = [tokenizer.tokenize(text) for text in y_pred]
    return score_fn(y_true, y_pred)


class WhiteSpaceTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return text.split(" ")


metrics_dict = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc": auc,
    "prauc": prauc_score,
    "ndcg": ndcg_score,
    "mattcorr": matthews_corrcoef,
    "pearsonr": pearsonr,
    "base-bleu": partial(
        rouge_bleu_score,
        tokenizer=WhiteSpaceTokenizer(),
        score_fn=bleu_score,
    ),
    "base-rouge1": partial(
        rouge_bleu_score,
        tokenizer=WhiteSpaceTokenizer(),
        score_fn=rouge1_score,
    ),
    "base-rouge2": partial(
        rouge_bleu_score,
        tokenizer=WhiteSpaceTokenizer(),
        score_fn=rouge2_score,
    ),
    "base-rougel": partial(
        rouge_bleu_score,
        tokenizer=WhiteSpaceTokenizer(),
        score_fn=rougel_score,
    ),
    "bert-bleu": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
        score_fn=bleu_score,
    ),
    "bert-rouge1": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
        score_fn=rouge1_score,
    ),
    "bert-rouge2": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
        score_fn=rouge2_score,
    ),
    "bert-rougel": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-cased"),
        score_fn=rougel_score,
    ),
    "mbert-bleu": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
        score_fn=bleu_score,
    ),
    "mbert-rouge1": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
        score_fn=rouge1_score,
    ),
    "mbert-rouge2": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
        score_fn=rouge2_score,
    ),
    "mbert-rougel": partial(
        rouge_bleu_score,
        tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
        score_fn=rougel_score,
    ),
}

supported_metrics = metrics_dict.keys()


@register_script("microsoft/script/aether/metrics")
class MetricsScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/aether/metrics")
        metrics = config.getoption("metrics", None)
        assert metrics is not None
        if isinstance(metrics, str):
            metrics = re.split(r"[,;]", metrics)

        assert all(metric in supported_metrics for metric in metrics)
        data_file = config.getoption("data_file", None)
        output_file = config.getoption("output_file", "./output.txt")
        assert data_file is not None and os.path.exists(data_file)

        names = config.getoption("names", None)
        y_true_col = config.getoption("y_true_col", None)
        y_pred_col = config.getoption("y_pred_col", None)
        escapechar = config.getoption("escapechar", None)
        if isinstance(names, str) and names.strip() == "*":
            names = None
        elif isinstance(names, str):
            names = re.split(r"[,;]", names)
            names = [n.strip() for n in names]

        assert y_true_col is not None and y_pred_col is not None

        y_true_col = y_true_col.strip()
        y_pred_col = y_pred_col.strip()
        assert y_true_col in names and y_pred_col in names

        data = pd.read_csv(
            data_file,
            names=names,
            sep="\t",
            quoting=3,
            header="infer" if names is None else None,
            escapechar=escapechar,
        )

        y_true = data[y_true_col]
        y_pred = data[y_pred_col]

        if y_true.dtype == "object":
            y_true.fillna("", inplace=True)

        if y_pred.dtype == "object":
            y_pred.fillna("", inplace=True)

        outputs = []
        for metric in metrics:
            score = metrics_dict.get(metric)(y_true, y_pred)
            logging.info(f"Metric {metric} : {score}")
            outputs.append([metric, score])

        outputs = pd.DataFrame(outputs, columns=["metric", "score"])
        outputs.to_csv(
            output_file,
            sep="\t",
            index=False,
            quoting=3,
        )
