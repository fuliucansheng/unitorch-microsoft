# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import os
import re
import shutil
import tempfile
import logging
import pandas as pd
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path

# check if jieba/whoosh is installed
try:
    import jieba
    from jieba.analyse import ChineseAnalyzer
    from whoosh.index import create_in
    from whoosh.fields import Schema, TEXT, ID, STORED, KEYWORD
    from whoosh.qparser import QueryParser
    from whoosh import scoring
except ImportError:
    raise ImportError(
        "jieba and whoosh are required for this script. "
        "Please install them using `pip install jieba whoosh`."
    )


@register_script("microsoft/script/china/c2s/whoosh")
class WhooshScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/china/c2s/whoosh")
        query_file = config.getoption("query_file", None)
        doc_file = config.getoption("doc_file", None)

        query_names = config.getoption("query_names", "*")
        query_col = config.getoption("query_col", None)
        doc_names = config.getoption("doc_names", "*")
        doc_index_col = config.getoption("doc_index_col", None)
        doc_search_cols = config.getoption("doc_search_cols", [])

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", None)
        output_header = config.getoption("output_header", False)
        output_file = config.getoption("output_file", "./output.txt")

        assert query_file is not None and os.path.exists(query_file)
        assert doc_file is not None and os.path.exists(doc_file)
        assert query_col is not None and doc_index_col is not None
        assert doc_search_cols is not None and len(doc_search_cols) > 0

        if isinstance(doc_search_cols, str):
            doc_search_cols = re.split(r"[,;]", doc_search_cols)
            doc_search_cols = [n.strip() for n in doc_search_cols]

        def get_input_tsv(input_file, names):
            if isinstance(names, str) and names.strip() == "*":
                names = None
            if isinstance(names, str):
                names = re.split(r"[,;]", names)
                names = [n.strip() for n in names]
            return pd.read_csv(
                input_file,
                names=names,
                sep="\t",
                quoting=3,
                header="infer" if names is None else None,
                escapechar=input_escapechar,
            )

        queries_table = get_input_tsv(query_file, query_names)
        docs_table = get_input_tsv(doc_file, doc_names)

        analyzer = ChineseAnalyzer()
        temp_dir = tempfile.mkdtemp()
        schema = Schema(
            id=ID(stored=True),
            content=TEXT(stored=True, analyzer=analyzer),
        )
        ix = create_in(temp_dir, schema)
        logging.info(f"Creating index... in {temp_dir}")

        writer = ix.writer()
        for _, row in docs_table.iterrows():
            index = row[doc_index_col]
            content = " ".join([row[col] for col in doc_search_cols if col in row])
            writer.add_document(id=str(index), content=content)
        writer.commit()

        results_queries, results_doc_ids, results_scores = [], [], []
        with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
            queries = queries_table[query_col].drop_duplicates().tolist()
            for query in queries:
                _query = QueryParser("content", ix.schema).parse(query)
                _result = searcher.search(_query, limit=None)
                results_queries += [query] * len(_result)
                results_doc_ids += [r["id"] for r in _result]
                results_scores += [r.score for r in _result]
        results = pd.DataFrame(
            {
                "_Query": results_queries,
                "_DocId": results_doc_ids,
                "IRScore": results_scores,
            }
        )
        results["_DocId"] = results["_DocId"].astype(docs_table[doc_index_col].dtype)

        results = pd.merge(queries_table, results, left_on=query_col, right_on="_Query")
        results = pd.merge(
            results, docs_table, left_on="_DocId", right_on=doc_index_col
        )
        results = results.drop(columns=["_Query", "_DocId"])
        results.insert(len(results.columns) - 1, "IRScore", results.pop("IRScore"))
        results.to_csv(
            output_file,
            sep="\t",
            index=False,
            header=output_header,
            quoting=3,
            escapechar=output_escapechar,
        )

        shutil.rmtree(temp_dir)
        logging.info(f"Results saved to {output_file}")
