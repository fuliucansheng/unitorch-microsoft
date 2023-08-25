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
import sqlparse
import sqlite3
from unitorch.cli import CoreConfigureParser, GenericScript
from unitorch.cli import register_script
from unitorch_microsoft import cached_path


@register_script("microsoft/script/aether/sql")
class SqlScript(GenericScript):
    def __init__(self, config: CoreConfigureParser):
        self.config = config

    def launch(self, **kwargs):
        config = self.config
        config.set_default_section("microsoft/script/aether/sql")
        input1_file = config.getoption("input1_file", None)
        input2_file = config.getoption("input2_file", None)
        input3_file = config.getoption("input3_file", None)
        input4_file = config.getoption("input4_file", None)
        input5_file = config.getoption("input5_file", None)

        input1_names = config.getoption("input1_names", None)
        input2_names = config.getoption("input2_names", None)
        input3_names = config.getoption("input3_names", None)
        input4_names = config.getoption("input4_names", None)
        input5_names = config.getoption("input5_names", None)

        escapechar = config.getoption("escapechar", "\\")

        def get_input_table(input_file, names):
            if isinstance(names, str):
                names = re.split(r"[,;]", names)
                names = [n.strip() for n in names]
            return pd.read_csv(
                input_file,
                names=names,
                sep="\t",
                quoting=3,
                header="infer" if names is None else None,
            )

        db = sqlite3.connect(":memory:")

        if input1_file is not None:
            input1 = get_input_table(input1_file, input1_names)
            input1.to_sql("input1", db, index=False)

        if input2_file is not None:
            input2 = get_input_table(input2_file, input2_names)
            input2.to_sql("input2", db, index=False)

        if input3_file is not None:
            input3 = get_input_table(input3_file, input3_names)
            input3.to_sql("input3", db, index=False)

        if input4_file is not None:
            input4 = get_input_table(input4_file, input4_names)
            input4.to_sql("input4", db, index=False)

        if input3_file is not None:
            input5 = get_input_table(input5_file, input5_names)
            input5.to_sql("input5", db, index=False)

        def run_sql(sql):
            sql = sqlparse.format(
                sql,
                strip_comments=True,
                keyword_case="lower",
                reindent_aligned=True,
            ).replace("\n", "")
            sql = re.sub(r"\s+", " ", sql)
            logging.info(f"Running SQL: {sql}")
            pattern = r"(\w+)\s*=\s*(select .+?)\s*(?:where|group by|order by|$)"
            match = re.match(pattern, sql)
            if match:
                table = match.group(1)
                sql = match.group(2)
            else:
                table = "__default__"
                sql = sql

            result = pd.read_sql(sql, db)

            result.to_sql(table, db, index=False)
            return result

        def run_action(action):
            sqls = sqlparse.split(action)
            result = None
            for sql in sqls:
                result = run_sql(sql)
                logging.info(f"Result Info: {result.columns} -- {result.shape}")
            return result

        output = None

        action1 = config.getoption("action1", None)
        action2 = config.getoption("action2", None)
        action3 = config.getoption("action3", None)
        action4 = config.getoption("action4", None)

        action_output1 = config.getoption("action_output1", None)
        output1_file = config.getoption("output1_file", "./output1.txt")

        for action in [action1, action2, action3, action4, action_output1]:
            if action is not None:
                output = run_action(action)

        if output is not None:
            output.to_csv(
                output1_file,
                sep="\t",
                index=False,
                header=None,
                quoting=3,
                escapechar=escapechar,
            )
            logging.info(f"Processed Output1 finish. shape is {output.shape}")

        action5 = config.getoption("action5", None)
        action6 = config.getoption("action6", None)

        action_output2 = config.getoption("action_output2", None)
        output2_file = config.getoption("output2_file", "./output2.txt")

        for action in [action5, action6, action_output2]:
            if action is not None:
                output = run_action(action)

        if output is not None:
            output.to_csv(
                output2_file,
                sep="\t",
                index=False,
                header=None,
                quoting=3,
                escapechar=escapechar,
            )
            logging.info(f"Processed Output2 finish. shape is {output.shape}")

        action_output3 = config.getoption("action_output3", None)
        output3_file = config.getoption("output3_file", "./output3.txt")

        if action_output3 is not None:
            output = run_action(action_output3)

        if output is not None:
            output.to_csv(
                output3_file,
                sep="\t",
                index=False,
                header=None,
                quoting=3,
                escapechar=escapechar,
            )
            logging.info(f"Processed Output3 finish. shape is {output.shape}")

