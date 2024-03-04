# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import torch
from unitorch.utils import pop_value
from unitorch.models import GenericOutputs
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_process,
)
from unitorch.cli.models import TensorsInputs
from unitorch_microsoft import cached_path


# https://unitorchazureblob.blob.core.windows.net/shares/models/l3g.txt
class TriTokenizer:
    def __init__(
        self,
        vocab_file,
        max_seq_len=12,
        max_n_letters=20,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path {vocab_file}")
        self.vocab = self._load_vocab(vocab_file)
        self.max_seq_len = max_seq_len
        self.max_n_letters = max_n_letters
        self.invalid = re.compile("[^a-zA-Z0-9 ]")
        self.multispace = re.compile("  +")

    def _load_vocab(self, vocab_file):
        ret = dict()
        with open(vocab_file, "r", encoding="utf-8") as fp:
            i = 1  # note that the dictionary now increases all trigram index by 1!!!
            while True:
                s = fp.readline().strip("\n\r")
                if s == "":
                    break
                ret[s] = i
                i += 1
        return ret

    @property
    def vocab_size(self):
        return len(self.vocab)

    def tokenize(
        self,
        text: str,
        max_seq_len: int = None,
        max_n_letters: int = None,
    ):
        max_seq_len = int(pop_value(max_seq_len, self.max_seq_len))
        max_n_letters = int(pop_value(max_n_letters, self.max_n_letters))
        step1 = text.lower()
        step2 = self.invalid.sub("", step1)
        step3 = self.multispace.sub(" ", step2)
        step4 = step3.strip()
        words = step4.split(" ")
        n_seq = min(len(words), max_seq_len)
        n_letter = max_n_letters
        input_ids = [0] * (max_seq_len * max_n_letters)
        attention_mask = [0] * max_seq_len
        for i in range(n_seq):
            if words[i] == "":
                words[i] = "#"
            word = "#" + words[i] + "#"
            n_letter = min(len(word) - 2, max_n_letters)
            for j in range(n_letter):
                s = word[j : (j + 3)]
                if s in self.vocab:
                    input_ids[i * max_n_letters + j] = self.vocab[s]
            attention_mask[i] = 1
        return GenericOutputs(input_ids=input_ids, attention_mask=attention_mask)


class TribertProcessor:
    def __init__(
        self,
        vocab_file,
        max_seq_length=12,
        max_n_letters=20,
    ):
        self.max_seq_length = max_seq_length
        self.max_n_letters = max_n_letters
        self.tokenizer = TriTokenizer(vocab_file=vocab_file)

    @classmethod
    @add_default_section_for_init("microsoft/process/tribert")
    def from_core_configure(cls, config, **kwargs):
        vocab_file = config.getdefault(
            "microsoft/process/tribert",
            "vocab_file",
            "https://unitorchazureblob.blob.core.windows.net/shares/models/l3g.txt",
        )
        vocab_file = cached_path(vocab_file)

        return dict(
            {
                "vocab_file": vocab_file,
            }
        )

    @register_process("microsoft/process/tribert/classification")
    def _classification(
        self,
        text,
        max_seq_length: int = None,
        max_n_letters: int = None,
        input_ids_name: str = "input_ids",
        attention_mask_name: str = "attention_mask",
    ):
        max_seq_length = int(pop_value(max_seq_length, self.max_seq_length))

        max_n_letters = int(pop_value(max_n_letters, self.max_n_letters))

        outputs = self.tokenizer.tokenize(
            str(text),
            max_seq_length,
            max_n_letters,
        )

        return TensorsInputs(
            {
                input_ids_name: torch.tensor(outputs.input_ids, dtype=torch.long),
                attention_mask_name: torch.tensor(
                    outputs.attention_mask, dtype=torch.long
                ),
            }
        )
