# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import itertools
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class MSTTR:
    """Mean segmental type-token ratio (based on tokenized data). Segment length is
    pre-set to 100 by default, computation is done on lowercased data. Returns two variants -- with
    and without taking punctuation into account.

    This is based on Emiel van Miltenburg's scripts from:
    https://github.com/evanmiltenburg/NLG-diversity/blob/main/diversity.py
    """

    def __init__(self, window_size: int = 100):
        # use MSTTR-100 by default.
        self.rnd = random.Random(1234)
        self.window_size = window_size

    def compute(self, predictions) -> Dict:
        return round(
            self._MSTTR(predictions, self.window_size)["msttr_value"],
            5,
        )

    def _TTR(self, list_of_words: List[str]) -> float:
        "Compute type-token ratio."
        return len(set(list_of_words)) / len(list_of_words)

    def _MSTTR(self, tokenized_data: List[List[str]], window_size: int) -> Dict:
        """
        Computes Mean-Segmental Type-Token Ratio (MSTTR; Johnson, 1944)
        by dividing the concatenated texts into non-overlapping segments of equal
        size and then averaging the TTRs of the segments.
        The last segment is excluded from the computation if it is smaller than
        the window size.
        """
        ttrs = []
        concatenated = list(itertools.chain.from_iterable(tokenized_data))

        for i in range(0, len(concatenated), window_size):
            window = concatenated[i : i + window_size]
            # removes the last segment from the computation
            if len(window) < window_size:
                break
            ttrs.append(self._TTR(window))

        results = {
            "msttr_value": sum(ttrs) / len(ttrs) if ttrs else float("nan"),
            "num_ttrs": len(ttrs),
            "ttrs": ttrs,
        }
        return results


def msttr_score(y_pred, window_size=100):
    msttr = MSTTR(window_size=window_size)
    return msttr.compute(y_pred)
