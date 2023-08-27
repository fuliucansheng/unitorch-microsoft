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

        input1_names = config.getoption("input1_names", "*")
        input2_names = config.getoption("input2_names", "*")
        input3_names = config.getoption("input3_names", "*")
        input4_names = config.getoption("input4_names", "*")
        input5_names = config.getoption("input5_names", "*")

        input_escapechar = config.getoption("input_escapechar", None)
        output_escapechar = config.getoption("output_escapechar", "\\")

        save_header = config.getoption("save_header", False)

        def get_input_table(input_file, names):
            names = None if names.strip() == "*" else names
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

        db = sqlite3.connect(":memory:")

        if input1_file is not None and os.path.exists(input1_file):
            input1 = get_input_table(input1_file, input1_names)
            db.execute("drop table if exists Input1;")
            input1.to_sql("Input1", db, index=False)

        if input2_file is not None and os.path.exists(input2_file):
            input2 = get_input_table(input2_file, input2_names)
            db.execute("drop table if exists Input2;")
            input2.to_sql("Input2", db, index=False)

        if input3_file is not None and os.path.exists(input3_file):
            input3 = get_input_table(input3_file, input3_names)
            db.execute("drop table if exists Input3;")
            input3.to_sql("Input3", db, index=False)

        if input4_file is not None and os.path.exists(input4_file):
            input4 = get_input_table(input4_file, input4_names)
            db.execute("drop table if exists Input4;")
            input4.to_sql("Input4", db, index=False)

        if input5_file is not None and os.path.exists(input5_file):
            input5 = get_input_table(input5_file, input5_names)
            db.execute("drop table if exists Input5;")
            input5.to_sql("Input5", db, index=False)

        def run_sql(sql):
            sql = sqlparse.format(
                sql,
                strip_comments=True,
                keyword_case="lower",
                reindent_aligned=True,
            ).replace("\n", "")
            sql = re.sub(r"\s+", " ", sql)
            if sql.count("select") == 0:
                logging.warning(f"Skip the SQL: `{sql}`. It's not a Select Statement.")
                return
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

            db.execute(f"drop table if exists {table};")
            result.to_sql(table, db, index=False)
            return result

        def run_action(action):
            sqls = sqlparse.split(action)
            result = None
            for sql in sqls:
                result = run_sql(sql)
                if result is not None:
                    logging.info(f"Result Info: {result.columns} -- {result.shape}")
            return result

        output = None

        action1 = config.getoption("action1", ";")
        action2 = config.getoption("action2", ";")
        action3 = config.getoption("action3", ";")
        action4 = config.getoption("action4", ";")

        action_output1 = config.getoption("action_output1", ";")
        output1_file = config.getoption("output1_file", "./output1.txt")

        for action in [action1, action2, action3, action4, action_output1]:
            if action is not None:
                output = run_action(action)

        if output is not None:
            output.to_csv(
                output1_file,
                sep="\t",
                index=False,
                header=save_header,
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"Processed Output1 finish. shape is {output.shape}")

        action5 = config.getoption("action5", ";")
        action6 = config.getoption("action6", ";")

        action_output2 = config.getoption("action_output2", ";")
        output2_file = config.getoption("output2_file", "./output2.txt")

        for action in [action5, action6, action_output2]:
            if action is not None:
                output = run_action(action)

        if output is not None:
            output.to_csv(
                output2_file,
                sep="\t",
                index=False,
                header=save_header,
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"Processed Output2 finish. shape is {output.shape}")

        action_output3 = config.getoption("action_output3", ";")
        output3_file = config.getoption("output3_file", "./output3.txt")

        if action_output3 is not None:
            output = run_action(action_output3)

        if output is not None:
            output.to_csv(
                output3_file,
                sep="\t",
                index=False,
                header=save_header,
                quoting=3,
                escapechar=output_escapechar,
            )
            logging.info(f"Processed Output3 finish. shape is {output.shape}")

